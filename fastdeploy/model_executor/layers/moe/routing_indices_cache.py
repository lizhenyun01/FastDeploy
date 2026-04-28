"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.distributed as dist
import triton
import triton.language as tl
from paddleformers.utils.log import logger

from fastdeploy.cache_manager.routing_cache_manager import RoutingHostBufferView
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)


@enable_compat_on_triton_kernel
@triton.jit
def _save_routing_kernel_v2(
    GPU_ROUTING_BUFFER_PTR,
    TOPK_IDS_PTR,
    LAYER_IDX,
    TOKEN_NUM,
    TOP_K,
    NUM_MOE_LAYERS,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    token_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    token_mask = token_offsets < TOKEN_NUM
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offsets < TOP_K

    load_mask = token_mask[:, None] & k_mask[None, :]
    topk_vals = tl.load(
        TOPK_IDS_PTR + token_offsets[:, None] * TOP_K + k_offsets[None, :],
        mask=load_mask,
    )

    STRIDE_TOKEN = NUM_MOE_LAYERS * TOP_K
    STRIDE_LAYER = TOP_K
    output_ptrs = (
        GPU_ROUTING_BUFFER_PTR + token_offsets[:, None] * STRIDE_TOKEN + LAYER_IDX * STRIDE_LAYER + k_offsets[None, :]
    )
    tl.store(output_ptrs, topk_vals, mask=load_mask)


def save_routing_to_buffer_v2(
    gpu_routing_buffer: paddle.Tensor,
    topk_ids: paddle.Tensor,
    layer_idx: int,
    tp_size: int,
    ep_size: int,
    tp_group: dist.communication.group.Group,
    total_token_num: int = -1,
):
    token_num_per_rank = topk_ids.shape[0]
    if token_num_per_rank == 0:
        return
    if tp_size > 1 and ep_size > 1:
        topk_ids_all = paddle.zeros([token_num_per_rank * tp_size, topk_ids.shape[1]], dtype=topk_ids.dtype)
        paddle.distributed.all_gather(topk_ids_all, topk_ids, tp_group)
        assert (
            total_token_num >= token_num_per_rank
        ), f"[R3] total_token_num={total_token_num} < token_num_per_rank={token_num_per_rank}"
        topk_ids = topk_ids_all[:total_token_num, :]

    token_num, top_k = topk_ids.shape
    buf_max_tokens, num_moe_layers, buf_top_k = gpu_routing_buffer.shape

    assert (
        token_num <= buf_max_tokens
    ), f"[R3] token_num={token_num} exceeds gpu_routing_buffer capacity={buf_max_tokens}"
    assert top_k == buf_top_k, f"[R3] top_k mismatch: topk_ids.top_k={top_k} vs gpu_routing_buffer.top_k={buf_top_k}"
    assert 0 <= layer_idx < num_moe_layers, f"[R3] layer_idx={layer_idx} out of range [0, {num_moe_layers})"

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = triton.next_power_of_2(top_k)
    grid = (triton.cdiv(token_num, BLOCK_SIZE_M),)
    _save_routing_kernel_v2[grid](
        gpu_routing_buffer,
        topk_ids,
        LAYER_IDX=layer_idx,
        TOKEN_NUM=token_num,
        TOP_K=top_k,
        NUM_MOE_LAYERS=num_moe_layers,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


class RoutedExpertsCapturer:
    """
    Worker-side routing capture: manages GPU transient buffer and GPU→CPU scatter.
    Does NOT manage request lifecycle — that is handled by RoutingCacheManager on the Engine side.
    """

    def __init__(self, fd_config: FDConfig, block_table, total_block_num):
        self.fd_config = fd_config
        self.block_table = block_table
        self.max_num_seqs = fd_config.scheduler_config.max_num_seqs

        # Read routing params from centralized config
        rrc = fd_config.routing_replay_config
        self.num_moe_layers = rrc.num_moe_layers
        self.moe_top_k = rrc.moe_top_k
        self.routing_dtype = rrc.routing_dtype
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank

        logger.info(f"[R3] RoutedExpertsCapturer config: {rrc}")

        self._init_routing_cache(dtype=self.routing_dtype, total_block_num=total_block_num)
        self.pending_update_positions = None

    def _init_routing_cache(self, dtype: str, total_block_num: int):
        """Initialize GPU transient buffer and prepare lazy SharedMemory attach."""
        max_num_kv_tokens = total_block_num * self.fd_config.cache_config.block_size

        # Small GPU transient buffer: only current step's token routing
        # TODO(Chengyanfu): Use max_num_batched_tokens to replace get_max_chunk_tokens()
        max_num_batched_tokens = self.fd_config.get_max_chunk_tokens()
        self.gpu_routing_buffer = paddle.full(
            shape=[max_num_batched_tokens, self.num_moe_layers, self.moe_top_k],
            fill_value=-1,
            dtype=dtype,
        )

        # Lazy attach to SharedMemory routing_host_buffer (created by Engine after profiling)
        self.routing_host_view = None
        self._routing_host_view_attach_attempted = False
        self._routing_host_view_shm_name = (
            f"routing_host_buffer.{str(self.fd_config.parallel_config.local_engine_worker_queue_port)}"
        )
        self._routing_host_view_shape = (max_num_kv_tokens, self.num_moe_layers, self.moe_top_k)
        self._routing_host_view_dtype = dtype

        gpu_buffer_bytes = int(np.prod(self.gpu_routing_buffer.shape)) * np.dtype(dtype).itemsize
        logger.info(
            f"[R3] GPU transient routing buffer: {self.gpu_routing_buffer.shape} "
            f"({gpu_buffer_bytes / 1024:.1f} KB)"
        )

    def _try_attach_routing_host_view(self):
        """Lazily attach to SharedMemory routing_host_buffer on first use."""
        if self._routing_host_view_attach_attempted:
            return
        self._routing_host_view_attach_attempted = True
        try:
            self.routing_host_view = RoutingHostBufferView(
                shape=self._routing_host_view_shape,
                dtype=self._routing_host_view_dtype,
                shm_name=self._routing_host_view_shm_name,
            )
            logger.info(f"[R3] Attached to RoutingHostBuffer SharedMemory: {self._routing_host_view_shm_name}")
        except FileNotFoundError:
            logger.warning(
                f"[R3] RoutingHostBuffer SharedMemory {self._routing_host_view_shm_name} not found. "
                "Routing capture will be skipped."
            )

    def save_captured_routing(self, num_tokens: int, slot_mapping: np.ndarray):
        """
        After forward, scatter GPU buffer routing data to routing_host_buffer.
        Called in step gap (post_process), not during forward. CUDAGraph compatible.
        """
        assert slot_mapping.shape[0] == num_tokens
        if num_tokens == 0:
            return

        # Lazy attach to SharedMemory (Engine creates it after profiling completes)
        if self.routing_host_view is None and not self._routing_host_view_attach_attempted:
            self._try_attach_routing_host_view()

        if self.routing_host_view is None:
            return

        # D2H copy: GPU → CPU numpy, then scatter to SharedMemory
        data = self.gpu_routing_buffer[:num_tokens].cpu().numpy()
        self.routing_host_view.scatter(slot_mapping, data)

    def compute_slot_mapping_flat(self, positions) -> np.ndarray:
        """
        Compute flat slot_mapping for all tokens in the step.
        Returns a 1D numpy array of slot indices.
        """
        all_slots = []
        block_size = self.fd_config.cache_config.block_size
        for batch_id, position in enumerate(positions):
            if len(position) == 0:
                continue
            block_table_indices = position // block_size
            token_block_ids = self.block_table[batch_id, block_table_indices]
            block_offset = position % block_size
            token_cache_ids = np.array(token_block_ids) * block_size + block_offset
            all_slots.append(token_cache_ids)
        if all_slots:
            return np.concatenate(all_slots)
        return np.array([], dtype=np.int64)

    def get_token_positions(self, seq_lens_decoder, seq_lens_this_time):
        """Get token position of each sequence in a batch."""
        starts = seq_lens_decoder.numpy()
        increase_num = seq_lens_this_time.numpy()

        positions = []
        for i in range(seq_lens_this_time.shape[0]):
            if increase_num[i] == 0:
                positions.append([])
                continue
            repeated_base = np.repeat(starts[i], increase_num[i])
            positions.append(repeated_base + np.arange(0, increase_num[i]))

        return positions

    def get_gpu_routing_buffer(self) -> paddle.Tensor:
        return self.gpu_routing_buffer

    def clear(self):
        """Clear GPU buffer and pending positions. Used during RL round cleanup."""
        self.gpu_routing_buffer.fill_(-1)
        self.pending_update_positions = None


# Backward compatibility alias
RoutingReplayManager = RoutedExpertsCapturer
