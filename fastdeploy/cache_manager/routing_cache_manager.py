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

import math
import multiprocessing
import multiprocessing.shared_memory
from typing import Optional

import numpy as np

from fastdeploy.utils import get_logger

logger = get_logger("routing_cache_manager", "routing_cache_manager.log")


class RoutingHostBuffer:
    """
    Manages routing_host_buffer (corresponds to KVCache GPU cache).
    Indexed by gpu_block_id * block_size + offset.
    Shared across processes via POSIX SharedMemory.
    Each DP rank creates its own instance; name includes dp_suffix.
    """

    def __init__(
        self, num_gpu_blocks: int, block_size: int, num_moe_layers: int, top_k: int, dtype: str, dp_suffix: str = ""
    ):
        max_num_gpu_tokens = num_gpu_blocks * block_size
        self.shape = (max_num_gpu_tokens, num_moe_layers, top_k)
        self.dtype = np.dtype(dtype)
        self.block_size = block_size
        total_bytes = int(np.prod(self.shape)) * self.dtype.itemsize

        self.shm_name = f"routing_host_buffer.{dp_suffix}"
        # Clean up stale SharedMemory from previous crashed process
        try:
            stale = multiprocessing.shared_memory.SharedMemory(name=self.shm_name, create=False)
            stale.close()
            stale.unlink()
            logger.warning(f"[R3] Cleaned up stale SharedMemory: {self.shm_name}")
        except FileNotFoundError:
            pass
        self.shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=max(total_bytes, 1), name=self.shm_name
        )
        self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.buffer[:] = -1  # unsigned wrap: uint8→255, uint16→65535, uint32→4294967295

        self._owner = True
        logger.info(
            f"[R3] Created RoutingHostBuffer: shape={self.shape}, "
            f"size={total_bytes / 1024:.1f} KB, name={self.shm_name}"
        )

    def close(self):
        """Close and unlink SharedMemory. Only the owner (creator) unlinks."""
        self.shm.close()
        if self._owner:
            self.shm.unlink()
            self._owner = False


class RoutingHostBufferView:
    """Read/write view of routing_host_buffer (cross-process, does not own)."""

    def __init__(self, shape, dtype: str, shm_name: str):
        self.shm = multiprocessing.shared_memory.SharedMemory(name=shm_name, create=False)
        self.dtype = np.dtype(dtype)
        self.buffer = np.ndarray(shape, dtype=self.dtype, buffer=self.shm.buf)

    def scatter(self, slot_mapping: np.ndarray, data: np.ndarray):
        """Scatter GPU buffer data to corresponding slots (Worker calls this)."""
        self.buffer[slot_mapping] = data

    def gather(self, slot_mapping: np.ndarray) -> np.ndarray:
        """Gather data from specified slots (TokenProcessor calls this)."""
        return self.buffer[slot_mapping].copy()

    def close(self):
        self.shm.close()


class RoutingSwapBuffer:
    """
    Manages routing_swap_buffer (corresponds to KVCache CPU cache).
    Indexed by cpu_block_id * block_size + offset.
    CacheTransferManager creates this; shared via SharedMemory.
    """

    def __init__(
        self, num_cpu_blocks: int, block_size: int, num_moe_layers: int, top_k: int, dtype: str, dp_suffix: str = ""
    ):
        max_num_cpu_tokens = num_cpu_blocks * block_size
        self.shape = (max_num_cpu_tokens, num_moe_layers, top_k)
        self.dtype = np.dtype(dtype)
        self.block_size = block_size
        total_bytes = int(np.prod(self.shape)) * self.dtype.itemsize

        self.shm_name = f"routing_swap_buffer.{dp_suffix}"
        # Clean up stale SharedMemory from previous crashed process
        try:
            stale = multiprocessing.shared_memory.SharedMemory(name=self.shm_name, create=False)
            stale.close()
            stale.unlink()
            logger.warning(f"[R3] Cleaned up stale SharedMemory: {self.shm_name}")
        except FileNotFoundError:
            pass
        self.shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=max(total_bytes, 1), name=self.shm_name
        )
        self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.buffer[:] = -1  # unsigned wrap: uint8→255, uint16→65535, uint32→4294967295

        self._owner = True
        logger.info(
            f"[R3] Created RoutingSwapBuffer: shape={self.shape}, "
            f"size={total_bytes / 1024:.1f} KB, name={self.shm_name}"
        )

    def close(self):
        """Close and unlink SharedMemory. Only the owner (creator) unlinks."""
        self.shm.close()
        if self._owner:
            self.shm.unlink()
            self._owner = False


class RoutingSwapBufferView:
    """Read/write view of routing_swap_buffer (cross-process, does not own)."""

    def __init__(self, shape, dtype: str, shm_name: str):
        self.shm = multiprocessing.shared_memory.SharedMemory(name=shm_name, create=False)
        self.dtype = np.dtype(dtype)
        self.buffer = np.ndarray(shape, dtype=self.dtype, buffer=self.shm.buf)

    def close(self):
        self.shm.close()


def split_request_id(request_id: str) -> str:
    """
    Split the request id to get rollout id.

    request_id: "chatcmpl-request.user-uuid"
    rollout_id: "request.user"
        example: "chatcmpl-xxx_xxx_epoch_15:2:2:1-d9f16c5c-65f6-4815-b44d-14e2c581907c_0"
                 -> "xxx_xxx_epoch_15:2:2:1"
    """
    chat_type, tmp_str = request_id.split("-", 1)
    assert (
        chat_type == "chatcmpl"
    ), "Rollout Routing Replay only supports chatcmpl. Please check request type and userid settings."
    reversed_tmp_str = tmp_str[::-1].split("-", 5)
    rollout_id = reversed_tmp_str[-1][::-1]
    return rollout_id


class RoutingCacheManager:
    """
    Engine-side stateless routing data manager.
    Does NOT maintain request mapping — request state is fully managed by Scheduler.
    Responsible for: SharedMemory creation/destruction, routing data gather, return mode dispatch.
    """

    def __init__(self, fd_config, num_gpu_blocks: int):
        routing_replay_config = fd_config.routing_replay_config
        self.num_moe_layers = routing_replay_config.num_moe_layers
        self.moe_top_k = routing_replay_config.moe_top_k
        self.routing_dtype = routing_replay_config.routing_dtype
        self.only_last_turn = routing_replay_config.only_last_turn
        self.use_fused_put = routing_replay_config.use_fused_put
        self.block_size = fd_config.cache_config.block_size
        self.return_mode = (
            routing_replay_config.routing_store_type
        )  # "local" / "rdma" → p2pstore; "response" → attach to RequestOutput

        dp_suffix = str(fd_config.parallel_config.local_engine_worker_queue_port)

        # Create SharedMemory routing_host_buffer
        self.host_buffer = RoutingHostBuffer(
            num_gpu_blocks=num_gpu_blocks,
            block_size=self.block_size,
            num_moe_layers=self.num_moe_layers,
            top_k=self.moe_top_k,
            dtype=self.routing_dtype,
            dp_suffix=dp_suffix,
        )

        # Host view for gather operations
        self.host_view = RoutingHostBufferView(
            shape=self.host_buffer.shape,
            dtype=self.routing_dtype,
            shm_name=self.host_buffer.shm_name,
        )

        # Initialize store wrapper for p2pstore mode
        self._store_wrapper = None
        if self.return_mode in ("local", "rdma"):
            from fastdeploy.cache_manager.routing_store import StoreWrapper

            self._store_wrapper = StoreWrapper(fd_config=fd_config)
            self._store_wrapper.start_store_warpper()

        logger.info(
            f"[R3] RoutingCacheManager initialized: return_mode={self.return_mode}, "
            f"host_buffer shape={self.host_buffer.shape}"
        )

    def gather_routing_for_request(self, block_table, seq_len: int) -> np.ndarray:
        """
        Gather complete routing data for a request from routing_host_buffer.

        Args:
            block_table: List of block IDs for the request
            seq_len: Total sequence length

        Returns:
            routing_data: [seq_len, num_moe_layers, top_k] numpy array
        """
        num_blocks = math.ceil(seq_len / self.block_size)
        block_ids = block_table[:num_blocks]
        positions = np.arange(seq_len)
        block_indices = positions // self.block_size
        offsets = positions % self.block_size
        slot_mapping = np.array(block_ids)[block_indices] * self.block_size + offsets
        return self.host_view.gather(slot_mapping)

    def on_request_finished(self, request_id: str, block_table, seq_len: int) -> Optional[np.ndarray]:
        """
        Unified entry point when a request finishes. Called by TokenProcessor on EOS detection.
        Scheduler/TokenProcessor passes request_id, block_table, seq_len.

        Returns:
            - "response" mode: routing_data numpy array (caller attaches to RequestOutput)
            - "local"/"rdma" mode: None (submitted to StoreWrapper internally)
        """
        routing_data = self.gather_routing_for_request(block_table, seq_len)

        if self._store_wrapper is not None:
            # P2PStore mode: submit to store
            rollout_id = split_request_id(request_id)
            # Transpose to [num_moe_layers, seq_len, top_k] for store compatibility
            # TODO(gongshaotian): Delete redundant transpose
            routing_data = np.ascontiguousarray(routing_data.transpose(1, 0, 2))

            if self.use_fused_put:
                self._store_wrapper.submit_put_task(routing_indices=routing_data, rollout_id=rollout_id)
                if self.only_last_turn:
                    self._store_wrapper.submit_clear_prefix_batch_task(rollout_id=rollout_id)
            else:
                for layer_id in range(self.num_moe_layers):
                    layer_buffer = routing_data[layer_id]
                    self._store_wrapper.submit_put_task(
                        routing_indices=layer_buffer, rollout_id=rollout_id, layer_idx=layer_id
                    )
                    if self.only_last_turn:
                        self._store_wrapper.submit_clear_prefix_batch_task(rollout_id=rollout_id, layer_idx=layer_id)
            return None
        else:
            # Response mode: return data for caller to attach to RequestOutput
            return routing_data

    def reset(self):
        """Reset SharedMemory buffer. Used during RL round cleanup."""
        self.host_buffer.buffer[:] = -1

    def close(self):
        """Clean up SharedMemory resources."""
        if self.host_view is not None:
            self.host_view.close()
            self.host_view = None
        if self.host_buffer is not None:
            self.host_buffer.close()
            self.host_buffer = None
