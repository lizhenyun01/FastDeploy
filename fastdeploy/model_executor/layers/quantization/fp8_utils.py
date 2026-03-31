"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import importlib

import paddle
import triton
from paddleformers.utils.log import logger

from fastdeploy.model_executor.ops.triton_ops import _per_token_group_quant_fp8
from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import per_token_group_fp8_quant

import numpy as np

from ..utils import get_sm_version


def try_import(modules, name=None, fail_msg=None):
    """
    try_import
    """
    if not isinstance(modules, (list, tuple)):
        modules = [modules]

    for m in modules:
        assert isinstance(m, str), m
        try:
            m = importlib.import_module(m)
        except ImportError:
            m = None

        if m is not None:
            if name is None:
                return m
            elif hasattr(m, name):
                return getattr(m, name)

    if fail_msg is not None:
        logger.warning(fail_msg)


paddlefleet_ops = try_import(["paddlefleet.ops"])


def load_deep_gemm():
    """
    Load DeepGemm module according to FastDeploy env switch.

    Returns:
        Imported deep_gemm module object.
    """

    if current_platform.is_cuda():
        if get_sm_version() == 100:
            # SM100 should use PFCC DeepGemm
            paddle.compat.enable_torch_proxy(scope={"deep_gemm"})
            try:
                import logging

                import paddlefleet.ops.deep_gemm as deep_gemm

                logging.getLogger().handlers.clear()
                logger.info("Detected sm100, use PaddleFleet DeepGEMM")
            except:
                import deep_gemm as deep_gemm

                logger.info("Detected sm100, use PFCC DeepGEMM")
        else:
            logger.info("use FastDeploy DeepGEMM")
            import fastdeploy.model_executor.ops.gpu.deep_gemm as deep_gemm
    else:
        deep_gemm = None
    return deep_gemm


deep_gemm = load_deep_gemm()


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(
    x: paddle.Tensor,
):
    """Convert FP32 tensor to TMA-aligned packed UE8M0 format tensor"""

    align = deep_gemm.utils.align
    get_tma_aligned_size = deep_gemm.utils.get_tma_aligned_size

    # Input validation: must be FP32 type 2D or 3D tensor
    assert x.dtype == paddle.float32 and x.dim() in (2, 3)

    # Step 1: Convert FP32 to UE8M0 format uint8 tensor
    # Extract FP32 exponent part through bit shift operation, convert to unsigned 8-bit integer
    ue8m0_tensor = (x.view(paddle.int) >> 23).to(paddle.uint8)

    # Step 2: Create padding and pack tensor
    # Get the last two dimensions of the input tensor
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    # If it's a 2D tensor, add batch dimension for unified processing
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    # Calculate TMA-aligned dimensions (aligned to 4-byte boundary)
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    # Create padded tensor with alignment and fill with valid data
    padded = paddle.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=paddle.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    # Pack uint8 data into int32 (pack 4 uint8 into 1 int32)
    padded = padded.view(-1).view(dtype=paddle.int).view(b, aligned_mn, aligned_k // 4)

    # Step 3: Transpose tensor to meet TMA memory access pattern requirements
    # Transpose tensor dimensions for TMA to efficiently access in MN-major order
    transposed = paddle.zeros((b, aligned_k // 4, aligned_mn), device=x.device, dtype=paddle.int).mT
    transposed[:, :, :] = padded
    # Extract original non-padded part
    aligned_x = transposed[:, :mn, :]
    # If input was 2D tensor, remove batch dimension
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def transform_scale_ue8m0(sf, mn, weight_block_size=None):
    get_mn_major_tma_aligned_packed_ue8m0_tensor = _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl
    if weight_block_size:
        assert weight_block_size == [128, 128]
        sf = sf.index_select(-2, paddle.arange(mn, device=sf.device) // 128)
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return sf


def quant_weight_ue8m0(weight_dequant, weight_block_size):
    assert weight_block_size == [128, 128]
    assert weight_dequant.dtype == paddle.bfloat16, f"{weight_dequant.dtype=} {weight_dequant.shape=}"

    *batch_dims, n, k = weight_dequant.shape

    weight_dequant_flat = weight_dequant.view((-1, k))
    out_w_flat, out_s_flat = deep_gemm.utils.math.per_block_cast_to_fp8(weight_dequant_flat, use_ue8m0=True)

    out_w = out_w_flat.view((*batch_dims, n, k))
    out_s = out_s_flat.view(
        (
            *batch_dims,
            ceil_div(n, weight_block_size[0]),
            ceil_div(k, weight_block_size[1]),
        )
    )

    return out_w, out_s


def per_token_group_quant_fp8(
    x: paddle.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: paddle.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: paddle.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dtype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
        column_major_scales: Outputs scales in column major.
        tma_aligned_scales: Outputs scales in TMA-aligned layout.
        out_q: Optional output tensor. If not provided, function will create.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """

    dtype = paddle.float8_e4m3fn  # current_platform.fp8_dtype() if dtype is None else dtype
    assert x.ndim == 2, f"per_token_group_fp8_quant only supports ndim == 2, but got shape {tuple(x.shape)}"
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible " f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    fp8_min, fp8_max = -224.0, 224.0  # get_fp8_min_max()

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = paddle.empty(x.shape, dtype=dtype)

    shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    x_s = paddle.empty(shape, dtype=paddle.float32)

    if current_platform.is_cuda():
        per_token_group_fp8_quant(x.contiguous(), x_q, x_s, group_size, eps, fp8_min, fp8_max, use_ue8m0)

    else:
        M = x.numel() // group_size
        # N: int = group_size
        BLOCK = triton.next_power_of_2
        # heuristics for number of warps
        num_warps = min(max(BLOCK // 256, 1), 8)
        num_stages = 1
        _per_token_group_quant_fp8[(M,)](
            x.contiguous(),
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=use_ue8m0,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


def fused_stack_transpose_quant(expert_weight_list, use_ue8m0=False):
    """fused_stack_transpose_quant"""
    if hasattr(paddlefleet_ops, "fuse_stack_transpose_fp8_quant"):
        # Blackwell (SM100) GPUs require pow2_scale quantization.
        # Guard with is_cuda() so non-CUDA environments do not call into
        # paddle.device.cuda.* and cause a crash.
        use_pow2_scale = current_platform.is_cuda() and get_sm_version() == 100

        w, scale = paddlefleet_ops.fuse_stack_transpose_fp8_quant(
            expert_weight_list,
            use_pow2_scale,
            use_ue8m0,
            use_ue8m0,
        )
        if use_ue8m0:
            scale = scale.T
    else:
        raise RuntimeError("'fuse_stack_transpose_fp8_quant' is not available in the current paddlefleet_ops.")

    return w, scale


def reorder_sf_to_cutlass(sf_logical, mn_dim, kb_dim):
    """
    将逻辑布局的 scale factor (UE8M0 uint8) 重排为 CUTLASS SfAtom 交错布局。

    CUTLASS Sm1xxBlockScaledConfig 要求 scale factor 按照 SfAtom 的特殊交错模式排列:
      SfAtom shape: ((32, 4), (SFVecSize, 4))
      SfAtom stride: ((16, 4), (0, 1))
    即每 128 行的 MN 元素被分成 (32, 4) 的子块，并与 K 维度的 4 个 scale 交错存储。

    参数:
      sf_logical: 逻辑布局的 scale factor tensor, shape (..., mn_dim, kb_dim), dtype=uint8
      mn_dim: M 或 N 维度大小
      kb_dim: K // block_size (scale factor 的 K 维度)
    返回:
      重排后的 scale factor tensor, 相同 shape 和 dtype
    """
    sf_np = sf_logical.numpy()
    orig_shape = sf_np.shape
    flat = sf_np.reshape(-1, mn_dim, kb_dim)

    # 向量化计算: 构建所有 (n, kb) 索引到 (n_phys, kb_phys) 的映射
    n_idx = np.arange(mn_dim)
    kb_idx = np.arange(kb_dim)
    n_grid, kb_grid = np.meshgrid(n_idx, kb_idx, indexing="ij")  # (mn_dim, kb_dim)

    n_tile = n_grid // 128
    n_local = n_grid % 128
    mn_i = n_local % 32
    mn_j = n_local // 32
    k_tile = kb_grid // 4
    sf_l = kb_grid % 4
    num_k_tiles = kb_dim // 4

    cutlass_byte = (n_tile * num_k_tiles + k_tile) * 512 + mn_i * 16 + mn_j * 4 + sf_l
    n_phys = cutlass_byte // kb_dim
    kb_phys = cutlass_byte % kb_dim

    # 用高级索引一次性完成重排
    result = np.empty_like(flat)
    result[:, n_phys, kb_phys] = flat[:, n_grid, kb_grid]

    return paddle.to_tensor(result.reshape(orig_shape), dtype=paddle.uint8)


def unpack_and_convert_scale(x, masked_m=None):
    """
    方案4：使用 tile 实现重复
    """
    if hasattr(x, "numpy"):
        x_np = x.numpy()
    else:
        x_np = np.array(x)

    e, m, n = x_np.shape

    # 确保数组是连续的
    if not x_np.flags["C_CONTIGUOUS"]:
        x_np = np.ascontiguousarray(x_np)

    # 将 int32 数组视为 uint8 数组（直接内存重解释）
    x_uint8 = x_np.view(np.uint8).reshape(e, m, n, 4)

    # 使用 repeat 重复每个值4次
    # 先增加一个维度，然后 repeat
    x_expanded = x_uint8[:, :, :, :, np.newaxis]  # [e, m, n, 4, 1]
    x_repeated = np.repeat(x_expanded, 4, axis=-1)  # [e, m, n, 4, 4]

    # 重新排列维度
    x_reshaped = x_repeated.reshape(e, m, n, 16)

    # 合并最后两维
    output_np = x_reshaped.reshape(e, m, n * 16)

    output = paddle.to_tensor(output_np, dtype=paddle.uint8)
    output = reorder_sf_to_cutlass(output, m, n * 16)

    return output
