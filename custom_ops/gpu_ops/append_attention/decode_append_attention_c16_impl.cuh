// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include "utils.cuh"
#include "attention_func.cuh"

template <typename T,
          uint32_t GROUP_SIZE,
          bool CAUSAL,
          uint32_t NUM_WARPS,
          uint32_t NUM_WARP_Q,
          uint32_t NUM_WARP_KV,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t num_frags_x,
          uint32_t num_frags_z,
          uint32_t num_frags_y>
__global__ void decode_append_attention_c16_kernel(
    const __grid_constant__ AttentionParams<T, T> params) {
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;

  extern __shared__ __align__(128) uint8_t smem[];
  smem_t qo_smem(smem);
  smem_t k_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T)),
      v_smem(smem + (num_frags_x * 16 + BLOCK_SIZE) * HEAD_DIM * sizeof(T));

  int total_block = params.num_blocks_ptr[0];
  int chunk_size = params.chunk_size_ptr[0];

  for (int lane_idx = blockIdx.x; lane_idx < total_block;
       lane_idx += gridDim.x) {
    int4 indices =
        reinterpret_cast<const int4 *>(params.block_indices)[lane_idx];
    int batch_idx = indices.x;
    int kv_head_idx = indices.y;
    int chunk_idx = indices.z;
    int tile_idx = indices.w;
    int q_head_idx = kv_head_idx * GROUP_SIZE;

    const uint32_t q_len = params.seq_lens_q[batch_idx];
    // q_len > 0 and kv_len > 0 are guaranteed by config_decode_attn,
    // which only generates block_indices entries for valid batches.
    const int *block_table_now =
        params.block_table + batch_idx * params.max_blocks_per_seq;

    constexpr uint32_t num_rows_per_block = num_frags_x * 16;
    const uint32_t q_end =
        min(q_len, div_up((tile_idx + 1) * num_rows_per_block, GROUP_SIZE));
    const uint32_t kv_len = params.seq_lens_kv[batch_idx] + q_len;
    const uint32_t num_chunks_this_seq = div_up(kv_len, chunk_size);

    constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();

    const uint32_t q_n_stride = params.q_num_heads * HEAD_DIM;
    const uint32_t q_ori_n_stride =
        (params.q_num_heads + params.kv_num_heads * 2) * HEAD_DIM;
    const uint32_t kv_n_stride = params.kv_num_heads * BLOCK_SIZE * HEAD_DIM;
    const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
    const uint32_t kv_b_stride = HEAD_DIM;

    float s_frag[num_frags_x][num_frags_z][8];
    float o_frag[num_frags_x][num_frags_y][8];
    float m_frag[num_frags_x][2];
    float d_frag[num_frags_x][2];

    T *o_base_ptr_T = nullptr;

    const uint32_t chunk_start = chunk_idx * chunk_size;
    const uint32_t chunk_end = min(kv_len, chunk_start + chunk_size);
    const uint32_t chunk_len = chunk_end - chunk_start;

    init_states<T, num_frags_x, num_frags_y>(o_frag, m_frag, d_frag);

    const uint32_t q_start_seq_id = params.cu_seqlens_q[batch_idx];
    const uint32_t q_base_seq_id_this_block = tile_idx * num_frags_x * 16;
    const uint32_t q_offset = q_start_seq_id * q_ori_n_stride +
                              q_head_idx * HEAD_DIM +
                              tid % 8 * num_elems_per_128b<T>();
    T *q_base_ptr = params.qkv + q_offset;

    o_base_ptr_T = params.tmp_o +
                   batch_idx * params.max_tokens_per_batch *
                       params.max_num_chunks * q_n_stride +
                   chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
                   tid % 8 * num_elems_per_128b<T>();
    const int *mask_offset_this_seq =
        params.mask_offset ? params.mask_offset + q_start_seq_id * 2 : nullptr;
    const bool *attn_mask_this_seq =
        params.attn_mask ? params.attn_mask + batch_idx * params.attn_mask_len *
                                                  params.attn_mask_len
                         : nullptr;

    uint32_t q_smem_offset_r =
        smem_t::get_permuted_offset<num_vecs_per_head>(tid % 16, tid / 16);

    load_q_global_smem_multi_warps<GROUP_SIZE,
                                   num_frags_x,
                                   num_frags_y,
                                   HEAD_DIM,
                                   T>(q_base_ptr,
                                      &qo_smem,
                                      q_base_seq_id_this_block,
                                      q_end,
                                      q_ori_n_stride,
                                      HEAD_DIM);
    commit_group();
    wait_group<0>();
    __syncthreads();

    q_smem_inplace_multiply_sm_scale_multi_warps<num_frags_x, num_frags_y, T>(
        &qo_smem, params.softmax_scale);

    const uint32_t num_iterations =
        div_up(CAUSAL ? (min(chunk_len,
                             sub_if_greater_or_zero(
                                 kv_len - q_len +
                                     div_up((tile_idx + 1) * num_rows_per_block,
                                            GROUP_SIZE),
                                 chunk_start)))
                      : chunk_len,
               BLOCK_SIZE);
    const uint32_t mask_check_iteration =
        (CAUSAL               ? (min(chunk_len,
                       sub_if_greater_or_zero(kv_len - q_len, chunk_start)))
         : params.mask_offset ? 0
                              : chunk_len) /
        (BLOCK_SIZE);

    uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
        wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

    uint32_t v_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
        wid * num_frags_z * 16 + tid % 16, tid / 16);
    uint32_t kv_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
        wid * 4 + tid / 8, tid % 8);

    uint32_t kv_idx = chunk_start;
    // block_table_idx tracks kv_idx / BLOCK_SIZE; since kv_idx increments by
    // BLOCK_SIZE each iteration, block_table_idx simply increments by 1.
    int block_table_idx = kv_idx / BLOCK_SIZE;
    int block_id = __ldg(&block_table_now[block_table_idx]);
    // Prefetch next block_id to overlap __ldg latency with compute_qk.
    int block_id_next = __ldg(&block_table_now[block_table_idx + 1]);
    if (block_id_next < 0) {
      block_id_next = 0;
    }
    const uint32_t const_offset = kv_head_idx * kv_h_stride +
                                  (wid * 4 + tid / 8) * kv_b_stride +
                                  tid % 8 * num_elems_per_128b<T>();
    T *cache_k_now = params.cache_k + block_id * kv_n_stride + const_offset;
    T *cache_v_now = params.cache_v + block_id * kv_n_stride + const_offset;

    produce_kv_blockwise<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(k_smem,
                                     &kv_smem_offset_w,
                                     &cache_k_now,
                                     kv_b_stride,
                                     kv_idx,
                                     chunk_end);
    commit_group();

    produce_kv_blockwise<SharedMemFillMode::kFillZero,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(v_smem,
                                     &kv_smem_offset_w,
                                     &cache_v_now,
                                     kv_b_stride,
                                     kv_idx,
                                     chunk_end);
    commit_group();
#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      // Prefetch block_id for next iteration's K/V loading.
      // block_table_idx increments by 1 each iteration since kv_idx
      // increments by BLOCK_SIZE, avoiding integer division in the loop.
      if (iter + 1 < num_iterations) {
        block_id_next = __ldg(&block_table_now[block_table_idx + 1]);
        if (block_id_next < 0) {
          block_id_next = 0;
        }
      }

      wait_group<1>();
      __syncthreads();

      // s = qk
      compute_qk<num_frags_x, num_frags_y, num_frags_z, T>(
          &qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

      if (iter >= mask_check_iteration || params.sliding_window > 0) {
        mask_s<T,
               CAUSAL,
               GROUP_SIZE,
               NUM_WARPS,
               num_frags_x,
               num_frags_y,
               num_frags_z>(attn_mask_this_seq,
                            q_base_seq_id_this_block,
                            kv_idx + wid * num_frags_z * 16,
                            q_len,
                            kv_len,
                            chunk_end,
                            params.attn_mask_len,
                            s_frag,
                            mask_offset_this_seq,
                            params.sliding_window);
      }

      // update m,d
      update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
          s_frag, o_frag, m_frag, d_frag);
      __syncthreads();

      // Advance kv_idx and block_table_idx for next iteration.
      kv_idx += BLOCK_SIZE;
      block_table_idx++;

      // Load K/V for next iteration using prefetched block_id
      block_id = block_id_next;
      cache_k_now = params.cache_k + block_id * kv_n_stride + const_offset;
      produce_kv_blockwise<SharedMemFillMode::kNoFill,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(k_smem,
                                       &kv_smem_offset_w,
                                       &cache_k_now,
                                       kv_b_stride,
                                       kv_idx,
                                       chunk_end);
      commit_group();
      wait_group<1>();
      __syncthreads();

      // compute sfm*v
      compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, T>(
          &v_smem, &v_smem_offset_r, s_frag, o_frag, d_frag);
      __syncthreads();

      cache_v_now = params.cache_v + block_id * kv_n_stride + const_offset;
      produce_kv_blockwise<SharedMemFillMode::kFillZero,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(v_smem,
                                       &kv_smem_offset_w,
                                       &cache_v_now,
                                       kv_b_stride,
                                       kv_idx,
                                       chunk_end);
      commit_group();
    }
    wait_group<0>();
    __syncthreads();
    merge_block_res<num_frags_x, num_frags_y, T>(
        o_frag, reinterpret_cast<float *>(smem), m_frag, d_frag, wid, tid);

    if (num_chunks_this_seq <= 1) {
      normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
    }

    // write o
    write_o_reg_gmem_multi_warps<GROUP_SIZE, num_frags_x, num_frags_y, T>(
        o_frag,
        &qo_smem,
        o_base_ptr_T,
        q_base_seq_id_this_block,
        q_head_idx,
        q_len,
        q_n_stride * params.max_num_chunks,
        HEAD_DIM);

    if (num_chunks_this_seq > 1) {
      if (wid == 0) {
#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
          for (uint32_t j = 0; j < 2; ++j) {
            const uint32_t qo_idx_now =
                q_base_seq_id_this_block + tid / 4 + j * 8 + fx * 16;
            const uint32_t qo_head_idx = q_head_idx + qo_idx_now % GROUP_SIZE;
            const uint32_t qo_idx = q_start_seq_id + qo_idx_now / GROUP_SIZE;
            if (qo_idx - q_start_seq_id < q_len) {
              uint32_t offset;
              offset = ((batch_idx * params.max_tokens_per_batch +
                         qo_idx_now / GROUP_SIZE) *
                            params.max_num_chunks +
                        chunk_idx) *
                           params.q_num_heads +
                       qo_head_idx;
              params.tmp_m[offset] = m_frag[fx][j];
              params.tmp_d[offset] = d_frag[fx][j];
            }
          }
        }
      }
    }
  }
}

template <typename T,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t Q_TILE_SIZE>
void DecodeAppendC16Attention(const AppendAttnMetaData &meta_data,
                              const paddle::Tensor &qkv,
                              const paddle::Tensor &cache_k,
                              const paddle::Tensor &cache_v,
                              const paddle::Tensor &tmp_workspace,
                              const paddle::Tensor &tmp_m,
                              const paddle::Tensor &tmp_d,
                              const paddle::optional<paddle::Tensor> &attn_mask,
                              const paddle::optional<paddle::Tensor> &sinks,
                              const paddle::Tensor &seq_lens_q,
                              const paddle::Tensor &seq_lens_kv,
                              const paddle::Tensor &seq_lens_encoder,
                              const paddle::Tensor &batch_id_per_token,
                              const paddle::Tensor &cu_seqlens_q,
                              const paddle::Tensor &block_table,
                              const paddle::Tensor &block_indices,
                              const paddle::Tensor &num_blocks,
                              const paddle::Tensor &chunk_size,
                              const int max_seq_len,
                              const int max_dec_len,
                              const int max_tokens_per_batch,
                              cudaStream_t &stream,
                              paddle::Tensor *out,
                              const int sliding_window) {
  using NV_TYPE = typename type_traits<T>::nv_type;

  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto token_num = meta_data.token_num;
  auto bsz = meta_data.batch_size;
  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;

  constexpr uint32_t NUM_WARP_Q = 1;
  constexpr uint32_t NUM_WARP_KV = NUM_WARPS_PER_BLOCK / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = Q_TILE_SIZE / (16 * NUM_WARP_Q);
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_qrow_per_block = NUM_WARP_Q * num_frags_x * 16;

  constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV;
  constexpr uint32_t smem_size_0 =
      (num_frags_x + NUM_WARP_KV * num_frags_z * 2) * 16 * HEAD_DIM *
      sizeof(NV_TYPE);
  constexpr uint32_t smem_size_1 =
      NUM_WARPS_PER_BLOCK * num_frags_x * num_frags_y * 32 * 8 * sizeof(float) +
      NUM_WARPS_PER_BLOCK * num_frags_x * 2 * 32 * 8;
  constexpr uint32_t smem_size =
      smem_size_0 > smem_size_1 ? smem_size_0 : smem_size_1;

  auto split_kv_kernel = decode_append_attention_c16_kernel<NV_TYPE,
                                                            GROUP_SIZE,
                                                            CAUSAL,
                                                            NUM_WARPS_PER_BLOCK,
                                                            NUM_WARP_Q,
                                                            NUM_WARP_KV,
                                                            HEAD_DIM,
                                                            BLOCK_SIZE,
                                                            num_frags_x,
                                                            num_frags_z,
                                                            num_frags_y>;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(split_kv_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
  }
  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  const int max_num_chunks = div_up(max_seq_len, 128);
  uint32_t attn_mask_len;
  if (attn_mask) {
    attn_mask_len = attn_mask.get().shape()[1];
  } else {
    attn_mask_len = -1;
  }

  AttentionParams<NV_TYPE, NV_TYPE> params;
  memset(&params, 0, sizeof(AttentionParams<NV_TYPE, NV_TYPE>));

  params.qkv = reinterpret_cast<NV_TYPE *>(const_cast<T *>(qkv.data<T>()));
  params.cache_k =
      reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k.data<T>()));
  params.cache_v =
      reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v.data<T>()));
  params.seq_lens_q = const_cast<int *>(seq_lens_q.data<int>());
  params.seq_lens_kv = const_cast<int *>(seq_lens_kv.data<int>());
  params.block_indices = const_cast<int *>(block_indices.data<int>());
  params.num_blocks_ptr = const_cast<int *>(num_blocks.data<int>());
  params.chunk_size_ptr = const_cast<int *>(chunk_size.data<int>());
  params.cu_seqlens_q = const_cast<int *>(cu_seqlens_q.data<int>());
  params.block_table = const_cast<int *>(block_table.data<int>());
  params.mask_offset = const_cast<int *>(meta_data.mask_offset);
  params.attn_mask =
      attn_mask ? const_cast<bool *>(attn_mask.get().data<bool>()) : nullptr;
  params.max_model_len = max_dec_len;
  params.max_kv_len = max_dec_len;
  params.max_blocks_per_seq = max_blocks_per_seq;
  params.softmax_scale = 1.f / sqrt(HEAD_DIM);
  params.tmp_o =
      reinterpret_cast<NV_TYPE *>(const_cast<T *>(tmp_workspace.data<T>()));
  params.tmp_m = const_cast<float *>(tmp_m.data<float>());
  params.tmp_d = const_cast<float *>(tmp_d.data<float>());
  params.max_tokens_per_batch = max_tokens_per_batch;
  params.attn_mask_len =
      attn_mask ? attn_mask_len = attn_mask.get().shape()[1] : -1;
  params.sliding_window = sliding_window;
  params.q_num_heads = num_heads;
  params.kv_num_heads = kv_num_heads;
  params.max_num_chunks = max_num_chunks;
  params.batch_size = meta_data.batch_size;

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  int sm_cout;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&sm_cout, cudaDevAttrMultiProcessorCount, device));

  dim3 grids(sm_cout * 8);
  dim3 blocks(32, NUM_WARPS_PER_BLOCK);

  launchWithPdlWhenEnabled(
      split_kv_kernel, grids, blocks, smem_size, stream, params);

  constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
  constexpr int blockx = HEAD_DIM / vec_size;
  constexpr int blocky = (128 + blockx - 1) / blockx;
  dim3 grids_merge(min(sm_count * 4, token_num), num_heads);
  dim3 blocks_merge(blockx, blocky);
  launchWithPdlWhenEnabled(
      merge_chunks_kernel<NV_TYPE, vec_size, blocky, HEAD_DIM>,
      grids_merge,
      blocks_merge,
      0,
      stream,
      params.tmp_o,
      params.tmp_m,
      params.tmp_d,
      seq_lens_q.data<int>(),
      seq_lens_kv.data<int>(),
      seq_lens_encoder.data<int>(),
      batch_id_per_token.data<int>(),
      cu_seqlens_q.data<int>(),
      (NV_TYPE *)nullptr,
      (NV_TYPE *)nullptr,
      sinks
          ? reinterpret_cast<NV_TYPE *>(const_cast<T *>(sinks.get().data<T>()))
          : nullptr,
      chunk_size.data<int>(),
      reinterpret_cast<NV_TYPE *>(out->data<T>()),
      0.f,
      0.f,
      -1,
      max_seq_len,
      max_num_chunks,
      num_heads,
      HEAD_DIM,
      token_num,
      max_tokens_per_batch);
}
