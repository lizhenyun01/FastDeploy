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

import asyncio
import atexit
import functools
import multiprocessing
import os
import shutil
import threading
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue
from typing import Optional, TypedDict

import numpy as np
import paddle

from fastdeploy.utils import get_logger

logger = get_logger("routing_cache_manager", "routing_cache_manager.log")

from fastdeploy.config import RoutingReplayConfig


class StoreTask(TypedDict):
    task_type: str
    key: str
    data: np.ndarray


class StoreWrapper(object):
    def __init__(self, fd_config) -> None:
        super().__init__()
        self.fd_config = fd_config

        # Initialize task queue
        moe_layer_num = fd_config.model_config.num_hidden_layers - fd_config.model_config.moe_layer_start_index
        max_num_seqs = fd_config.scheduler_config.max_num_seqs
        self.queue_max_size = moe_layer_num * max_num_seqs * 1000

        self.manager = multiprocessing.Manager()
        self._task_queue = self.manager.Queue(maxsize=self.queue_max_size)

        self._monitor_thread: threading.Thread = None
        self._stop_monitor = threading.Event()

        # Initialize consumer process
        self._routing_store_process = StoreProcess(
            task_queue=self._task_queue,
            routing_replay_config=self.fd_config.routing_replay_config,
            max_model_len=self.fd_config.model_config.max_model_len,
        )
        self._store_process_running = False

        # Register atexit handler
        atexit.register(self.shutdown)

    def shutdown(self):
        """ """
        if not self._store_process_running:
            return
        self._store_process_running = False

        # Stop the monitor thread
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)

        # Put a sentinel value to signal the consumer to stop
        if self._routing_store_process and self._routing_store_process.is_alive():
            try:
                self._task_queue.put_nowait(None)
            except Exception as e:
                logger.info(f"Could not put sentinel into queue: {e}")

        if self._routing_store_process and self._routing_store_process.is_alive():
            # Wait for all tasks to be processed
            self._routing_store_process.join(timeout=10.0)
            if self._routing_store_process.is_alive():
                self._routing_store_process.close()
                self._routing_store_process.join()

        self._task_queue.join()
        self.manager.shutdown()
        self._store_process_running = False

    def start_store_warpper(self):
        """ """
        if self._store_process_running:
            return
        self._store_process_running = True

        # Start monitor thread
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_queue_load, daemon=True)
        self._monitor_thread.start()

        # Start Routing Store Wrapper in sub process
        self._routing_store_process.start()

    def _monitor_queue_load(self):
        """ """
        while not self._stop_monitor.is_set():
            time.sleep(2.0)
            if not self._store_process_running:
                break
            qsize = self._task_queue.qsize()

            # Alarm when the task exceeds 80% of the queue capacity
            if qsize > self.queue_max_size * 0.8:
                logger.warning(
                    f"[Monitor] Queue load is HIGH: {qsize}/{self.queue_max_size}. "
                    "Consider increasing max_workers or queue_max_size."
                )
            logger.debug(f"[Monitor] Queue load: {qsize}/{self.queue_max_size}")

    def submit_put_task(self, routing_indices: np.ndarray, rollout_id: str, layer_idx: int = None) -> None:
        """Submit a put task to the task queue"""
        if not self._store_process_running:
            raise RuntimeError("Store not started.")

        start_time = time.perf_counter()
        if layer_idx is not None:
            rdma_rollout_key = f"{rollout_id}_{layer_idx}"
        else:
            rdma_rollout_key = rollout_id

        task: StoreTask = {"task_type": "put", "key": rdma_rollout_key, "data": routing_indices}

        try:
            self._task_queue.put_nowait(task)
        except Exception:
            raise RuntimeError(f"Queue is FULL. Dropping put task for key: {rdma_rollout_key}. ")
        logger.info(f"[R3] Submit put task for key: {rdma_rollout_key}, cost time: {time.perf_counter()-start_time} s")

    def submit_clear_store_task(self) -> None:
        """Submit clear store task"""
        if not self._store_process_running:
            raise RuntimeError("Store not started.")

        start_time = time.perf_counter()
        task: StoreTask = {"task_type": "clear_store", "key": None, "data": None}

        try:
            self._task_queue.put_nowait(task)
            # Wait for the task to be processed
            self._task_queue.join()
        except Exception:
            raise RuntimeError("Queue is FULL. Dropping put task for key: clear_store. ")
        logger.info(f"[R3] Submit clear task, cost time: {time.perf_counter()-start_time} s")

    def submit_clear_prefix_batch_task(self, rollout_id, layer_idx: int = None) -> None:
        """Submit clear prefix batch task"""
        if not self._store_process_running:
            raise RuntimeError("Store not started.")
        prefix_batch_id = self.get_needed_clear_ids(rollout_id)
        if prefix_batch_id is None:
            return
        start_time = time.perf_counter()
        if layer_idx is not None:
            rdma_rollout_key = f"{prefix_batch_id}_{layer_idx}"
        else:
            rdma_rollout_key = prefix_batch_id

        task: StoreTask = {"task_type": "clear_prefix_batch", "key": rdma_rollout_key, "data": None}
        try:
            self._task_queue.put_nowait(task)
        except Exception:
            raise RuntimeError("Queue is FULL. Dropping put task for key: clear_store. ")
        logger.info(
            f"[R3] Submit clear prefix batch task for key: {prefix_batch_id}, cost time: {time.perf_counter()-start_time} s"
        )

    def get_needed_clear_ids(self, rollout_id: str) -> Optional[str]:
        """
        Generate the prefix IDs for all closed multi-round tasks.
        rollout_id: "xxx_xxx_epoch_15:2:2:1"
            example: xxx_xxx_data_id:gen_id:turn_id:segment_id
        """
        reversed_segment_id, reversed_turn_id, reversed_prefix_gen_id = rollout_id[::-1].split(":", 2)
        prefix_gen_id = reversed_prefix_gen_id[::-1]
        turn_id = eval(reversed_turn_id[::-1])
        segment_id = eval(reversed_segment_id[::-1])

        assert turn_id >= 0 and segment_id >= 0
        prefix_batch = None
        if turn_id > 0:
            prefix_batch = f"{prefix_gen_id}:{(turn_id-1)}:{segment_id}"
        return prefix_batch


class StoreProcess(Process):
    def __init__(self, task_queue: Queue, routing_replay_config: RoutingReplayConfig, max_model_len: int) -> None:
        super().__init__()
        self.max_model_len = max_model_len
        self._task_queue = task_queue
        self.routing_replay_config = routing_replay_config
        self.max_workers = 5
        self._closed = False

        # Note: _routing_store and _event_loop_thread must be initialized in run()
        # because they cannot be properly inherited after fork()
        self._routing_store = None
        self._event_loop_thread = None

    def run(self):
        logger.info(f"[R3] Start Running Store Wrapper in sub process {os.getpid()}")

        # Initialize routing store in subprocess
        self._routing_store = get_routing_store(routing_replay_config=self.routing_replay_config)

        # Initialize event loop thread in subprocess
        self._event_loop_thread = AsyncEventLoopThread()
        self._event_loop_thread.start()
        if not self._event_loop_thread._started_event.wait(timeout=5.0):
            raise RuntimeError("Failed to start async event loop thread in subprocess")

        clear_store_task = StoreTask({"task_type": "clear_store", "key": None, "data": None})
        self._task_queue.put_nowait(clear_store_task)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while not self._closed:
                try:
                    task = self._task_queue.get()
                    if task is None:  # Sentinel
                        self._task_queue.task_done()
                        break

                    if task["task_type"] == "put":
                        future = executor.submit(self.process_put_task, task)
                        future.add_done_callback(lambda f: self._task_queue.task_done())
                    elif task["task_type"] == "clear_store":
                        future = executor.submit(self.process_clear_store_task, task)
                        future.add_done_callback(lambda f: self._task_queue.task_done())
                    elif task["task_type"] == "clear_prefix_batch":
                        future = executor.submit(self.process_clear_prefix_batch_task, task)
                        future.add_done_callback(lambda f: self._task_queue.task_done())
                except Exception as e:
                    self._task_queue.task_done()
                    raise RuntimeError(f"Error during processing task. {e}")

        logger.info("RoutingReplay Consumer Process Shutdown.")

    def process_put_task(self, store_task: StoreTask) -> None:
        try:
            # TODO(gongshaotian): delete this after trainer support dynamic len
            store_task["data"] = self.pad_routing_indices(store_task["data"])
            coro_obj = self._routing_store.put(routing_key=store_task["key"], routing_indices=store_task["data"])
            future = self._event_loop_thread.submit_coroutine(
                coro_obj, callback=functools.partial(self._on_async_task_completed, store_task)
            )
            return future
        except Exception as e:
            logger.error(f"Error submitting put task: {e}")
            traceback.print_exc()
            raise

    def process_clear_store_task(self, store_task: StoreTask) -> None:
        try:
            coro_obj = self._routing_store.clear_store()
            future = self._event_loop_thread.submit_coroutine(
                coro_obj, callback=functools.partial(self._on_async_task_completed, store_task)
            )
            return future
        except Exception as e:
            logger.error(f"Error during processing clear store task. {e}")
            traceback.print_exc()
            raise

    def process_clear_prefix_batch_task(self, store_task: StoreTask) -> None:
        try:
            coro_obj = self._routing_store.clear_prefix_batch(routing_prefix_key=store_task["key"])
            future = self._event_loop_thread.submit_coroutine(
                coro_obj, callback=functools.partial(self._on_async_task_completed, store_task)
            )
            return future
        except Exception as e:
            logger.error(f"Error submitting clear_prefix_batch task: {e}")
            traceback.print_exc()
            raise

    def _on_async_task_completed(self, task, future):
        """ """
        try:
            # result = future.result()
            logger.info(f"[R3] Async task completed: {task['task_type']}, key: {task['key']}")
        except Exception as e:
            logger.error(f"[R3] Async task failed: {task['task_type']}, key: {task['key']}, error: {e}")
            traceback.print_exc()
            raise

    def close(self):
        """Close the store process"""
        self._closed = True
        if hasattr(self, "_event_loop_thread"):
            self._event_loop_thread.stop()

    def pad_routing_indices(self, routing_indices: np.ndarray) -> np.ndarray:
        """Pad routing indices of the request levevl to max model len"""
        routing_shape = routing_indices.shape
        if len(routing_shape) == 2:  # [token, topk]
            pad_array = np.full(
                shape=[(self.max_model_len - routing_indices.shape[0]), routing_indices.shape[1]],
                fill_value=-1,
                dtype=routing_indices.dtype,
            )
            return np.concatenate([routing_indices, pad_array], axis=0)

        elif len(routing_shape) == 3:  # [layer, token, topk]
            pad_array = np.full(
                shape=[
                    routing_indices.shape[0],
                    (self.max_model_len - routing_indices.shape[1]),
                    routing_indices.shape[2],
                ],
                fill_value=-1,
                dtype=routing_indices.dtype,
            )
            return np.concatenate([routing_indices, pad_array], axis=1)
        else:
            raise ValueError(f"Invalid routing indices shape: {routing_shape}")


class AsyncEventLoopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._loop = None
        self._started_event = threading.Event()
        self._closed = False

    def run(self):
        """Run the async event loop"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Set the event loop to be started
        self._started_event.set()
        logger.info("[EventLoopThread] Event loop started, running forever...")

        try:
            self._loop.run_forever()
            logger.info("[EventLoopThread] Event loop stopped")
        except Exception as e:
            logger.error(f"[EventLoopThread] Event loop exception: {e}")
            traceback.print_exc()
        finally:
            logger.info("[EventLoopThread] Closing event loop")
            self._loop.close()

    def submit_coroutine(self, coro, callback=None):
        """Thread safely submit coroutine to event loop"""
        if self._closed:
            raise RuntimeError("Event loop thread is closed")
        if not self._started_event.wait(timeout=5.0):
            raise RuntimeError("Event loop failed to start within 5 seconds")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        if callback:

            def wrapped_callback(f):
                try:
                    callback(f)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
                    traceback.print_exc()

            future.add_done_callback(wrapped_callback)
        return future

    def stop(self):
        """Stop the event loop"""
        if not self._closed:
            self._closed = True
            if self._loop:
                self._loop.call_soon_threadsafe(self._loop.stop)


class RoutingStoreBase(ABC):
    """Base class for routing store"""

    def __init__(self, routing_replay_config: RoutingReplayConfig) -> None:
        self.routing_replay_config = routing_replay_config

    @abstractmethod
    async def put(self, routing_key: str, routing_indices: np.ndarray) -> None:
        """Put the routing indices into store"""
        raise NotImplementedError

    @abstractmethod
    async def clear_store(
        self,
    ):
        """Clear the routing indices store"""
        raise NotImplementedError

    @abstractmethod
    async def clear_prefix_batch(self, routing_prefix_key: str):
        """Clear the routing indices"""
        raise NotImplementedError


class RoutingStoreLocal(RoutingStoreBase):
    """Routing Store using local memory"""

    def __init__(self, routing_replay_config) -> None:
        super().__init__(routing_replay_config=routing_replay_config)
        self.local_store_dir = routing_replay_config.local_store_dir
        os.makedirs(self.local_store_dir, exist_ok=True)

    async def put(
        self,
        routing_key: str,
        routing_indices: np.ndarray,
    ) -> None:
        """Put the routing indices into store"""
        # TODO(gongshaotian) covert ./store_dir/routing_key/layer_id.pdtensor to ./store_dir/routing_key.pdtensor
        time_before_put = time.perf_counter()

        if len(routing_indices.shape) == 2:
            re_layer_id, re_rollout_id = routing_key[::-1].split("_", 1)
            rollout_id = re_rollout_id[::-1]
            layer_id = re_layer_id[::-1]
            request_path = os.path.join(self.local_store_dir, rollout_id)
            file_path = os.path.join(request_path, f"layer_{layer_id}.pdtensor")
        elif len(routing_indices.shape) == 3:
            request_path = os.path.join(self.local_store_dir, routing_key)
            file_path = os.path.join(request_path, f"{routing_key}.pdtensor")
        else:
            raise ValueError(f"Invalid routing indices shape: {routing_indices.shape}")

        paddle.save(routing_indices, file_path)
        logger.info(f"[R3] The routing key {routing_key} put cost is {time.perf_counter()-time_before_put}s")

    async def clear_store(self):
        """Clear the routing indices store"""
        if os.path.isdir(self.local_store_dir):
            shutil.rmtree(self.local_store_dir)

        logger.info("[R3] Clear routing store.")

    async def clear_prefix_batch(self, routing_prefix_key: str):
        """Clear the routing indices"""
        raise NotImplementedError


class RoutingStoreRDMA(RoutingStoreBase):
    """Routing Store using RDMA"""

    def __init__(self, routing_replay_config) -> None:
        super().__init__(routing_replay_config=routing_replay_config)
        try:
            # Only used in RLHF
            from p2pstore import P2PClient, P2PConfig
        except ModuleNotFoundError:
            raise ModuleNotFoundError(" RoutingStoreRDMA and p2pstore only support in RLHF. ")

        rdma_store_server = routing_replay_config.rdma_store_server
        p2pConfig = P2PConfig(metadata_server=rdma_store_server)
        self.p2p_client = P2PClient(p2pConfig)

    async def put(self, routing_key: str, routing_indices: np.ndarray) -> None:
        """Put the routing indices into store"""
        time_before_put = time.perf_counter()
        if len(routing_indices.shape) == 3:
            # NOTE(gongshaotian) Fused put with bytes data
            routing_bytes = routing_indices.tobytes()
            result = await self.p2p_client.put(routing_key, routing_bytes)
        else:
            result = await self.p2p_client.put(routing_key, routing_indices)
        logger.info(f"[R3] The routing key {routing_key}, put cost is {time.perf_counter()-time_before_put}s")
        return result

    async def clear_prefix_batch(self, routing_prefix_key: str):
        time_before_clear = time.perf_counter()
        result = await self.p2p_client.delete_batch([routing_prefix_key])
        logger.info(
            f"[R3] The clear routing prefix key {routing_prefix_key}, cost is {time.perf_counter()-time_before_clear}s"
        )
        return result

    async def clear_store(self):
        """Clear the routing indices store"""
        time_before_clear = time.perf_counter()
        result = await self.p2p_client.clear()
        logger.info(f"[R3] Clear routing store cost is {time.perf_counter()-time_before_clear}s.")
        return result


def get_routing_store(routing_replay_config: RoutingReplayConfig) -> RoutingStoreBase:
    if routing_replay_config.routing_store_type == "local":
        return RoutingStoreLocal(routing_replay_config=routing_replay_config)
    elif routing_replay_config.routing_store_type == "rdma":
        return RoutingStoreRDMA(routing_replay_config=routing_replay_config)
    else:
        raise ValueError(
            f"Invalid routing store type: '{routing_replay_config.routing_store_type}'. "
            "Valid types are: 'local', 'rdma'"
        )
