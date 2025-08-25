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

""" process.py """
import json
import os
import subprocess
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, List, Union

import numpy as np

from fastdeploy.input.ernie_tokenizer import ErnieBotTokenizer
from fastdeploy.input.tokenzier_client import (
    AsyncTokenizerClient,
    ImageEncodeRequest,
    VideoEncodeRequest,
)
from fastdeploy.utils import data_processor_logger

IDS_TYPE_FLAG = {"text": 0, "image": 1, "video": 2, "audio": 3}


def fancy_print(input_ids, tokenizer, image_patch_id=None):
    """
    input_ids: input_ids
    tokenizer: the tokenizer of models
    """
    i = 0
    res = ""
    text_ids = []
    real_image_token_len = 0
    while i < len(input_ids):
        if input_ids[i] == image_patch_id:
            if len(text_ids) > 0:
                res += tokenizer.decode(text_ids)
                text_ids = []

            real_image_token_len += 1
        else:
            if real_image_token_len != 0:
                res += f"<|IMAGE@{real_image_token_len}|>"
                real_image_token_len = 0

            text_ids.append(input_ids[i])

        i += 1
    if len(text_ids) > 0:

        res += tokenizer.decode(text_ids)
        text_ids = []
    return res


def get_duration(bos_url):
    result = subprocess.run(
        ["bcecmd", "bos", "gen_signed_url", bos_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    url = str(result.stdout).strip()
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "json", url]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 错误: {result.stderr}")
    output = json.loads(result.stdout)
    duration = int(float(output["format"]["duration"]))
    return duration


def get_uniform_frame_timestamps_usec(duration, frame_num):
    """
    获取均匀分布的时间戳
    Args:
        duration (int): 总时长
        frame_num (int): 帧数
    Returns:
        list: 时间戳列表
    """
    assert duration > 0 and frame_num > 0
    interval = duration / frame_num
    timestamps = []
    for i in range(frame_num):
        time_in_sec = i * interval
        td = timedelta(seconds=time_in_sec)
        # 格式化为 HH:MM:SS.ffffff
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        microseconds = int((total_seconds - int(total_seconds)) * 1e2)
        timestamps.append(f"{hours:02}:{minutes:02}:{seconds:02}.{microseconds:02}")
    return timestamps


def construct_3d_position_ids(
    in_tokens,
    pack_hw_list,
    patch_id,
    eos_id,
    eoi_id,
    frame_lengths=None,
    is_video=False,
    temporal_scale=2,
):
    """
    construct_3d_position_ids
    """
    lm_labels = np.append(in_tokens[1:], 0)
    is_vision_patch = (lm_labels == patch_id).astype(np.int64)
    is_eos_token = (in_tokens == eos_id).astype(np.int64)
    is_eoi_token = (in_tokens == eoi_id).astype(np.int64)
    assert np.sum(is_vision_patch) == sum([np.sum(np.prod(hw_list, axis=-1)) for hw_list in pack_hw_list])

    vid_idx = 0

    if is_video:
        frame_width = len(pack_hw_list[vid_idx]) // frame_lengths[vid_idx]
    else:
        frame_width = len(pack_hw_list[vid_idx])

    t, h, w = -1, -1, -1
    position_ids = []
    frame_count = 0
    # cur_frame_count = 0
    cur_frame_idx = 1
    hw_list = pack_hw_list[vid_idx]
    frame_hw_list = pack_hw_list[vid_idx][:frame_width]
    max_H, max_W = hw_list[-1]
    para = temporal_scale

    def point_via_count(c, hw_list):
        cum_grade = np.cumsum([0] + np.prod(hw_list, axis=-1).tolist())
        hit_scale_index = np.sum((c > cum_grade).astype(np.int64)) - 1
        is_last_scale = hit_scale_index == (len(hw_list) - 1)
        is_last_token = cum_grade[hit_scale_index + 1] == c
        H, W = hw_list[hit_scale_index]
        rest_i = c - cum_grade[hit_scale_index] - 1
        h = rest_i // W
        w = rest_i % W
        return h, w, H, W, is_last_token, is_last_scale

    for is_eos, is_eoi, is_vid in zip(is_eos_token.tolist(), is_eoi_token.tolist(), is_vision_patch.tolist()):
        if is_vid == 0:
            t, h, w = t + 1, h + 1, w + 1
            position_ids.append([t, h, w])
        else:
            frame_count += 1
            s_h, s_w, s_H, s_W, is_last_token, is_last_scale = point_via_count(frame_count, frame_hw_list)

            div_s_H, div_s_W = 1, 1

            if s_H > 1:
                div_s_H = s_H - 1
            if s_W > 1:
                div_s_W = s_W - 1

            h_stride = (s_h + 0) / (div_s_H) * max_H - (max_H / 2)
            w_stride = (s_w + 0) / (div_s_W) * max_W - (max_W / 2)

            position_ids.append(
                [
                    t + (para * cur_frame_idx),
                    t + (para * cur_frame_idx) + h_stride,
                    t + (para * cur_frame_idx) + w_stride,
                ]
            )

            if is_last_token and is_last_scale:
                h, w = h + max_H, w + max_W
                frame_count = 0
                cur_length = frame_lengths[vid_idx] if is_video else 1
                if cur_frame_idx == cur_length:
                    t = t + para * cur_length
                    h = t
                    w = t
                else:
                    cur_frame_idx += 1

        if is_eoi == 1:
            vid_idx += 1
            cur_frame_idx = 1

            if len(pack_hw_list) <= vid_idx:
                hw_list = None
                max_H, max_W = None, None
            else:
                hw_list = pack_hw_list[vid_idx]
                if is_video:
                    frame_width = len(pack_hw_list[vid_idx]) // frame_lengths[vid_idx]
                else:
                    frame_width = len(pack_hw_list[vid_idx])
                frame_hw_list = pack_hw_list[vid_idx][:frame_width]
                max_H, max_W = hw_list[-1]

            frame_count = 0

        # if is_eos == 1 and False:
        #     h, w = max(h, w) + 1, max(h, w) + 1

    return np.array(position_ids).astype("float32").tolist()


class MultiModalDataProcessor:
    """
    Processes multimodal chat messages into model-ready inputs,
    handling text, images, and videos with 3D positional embeddings.
    """

    CLS_TOKEN = "<|begin_of_sentence|>"
    SEP_TOKEN = "<|end_of_sentence|>"
    EOS_TOKEN = "</s>"
    IMG_START = "<image_start>"
    IMG_END = "<image_end>"
    IMG_PATCH = "<image_patch_id_und>"
    VID_START = "<video_start>"
    VID_END = "<video_end>"
    VID_PATCH = "<video_patch_id_und>"
    AUD_START = "<audio_start>"
    AUD_END = "<audio_end>"
    AUD_PATCH = "<audio_patch_id_und>"

    def __init__(
        self,
        tokenizer_name: str,
        image_merge_size: int = 2,
        video_merge_size: int = 2,
        audio_code_depth: int = 32,
        add_timestamp: bool = True,
        **kwargs,
    ) -> None:
        # Tokenizer and image preprocessor
        self.model_name_or_path = tokenizer_name
        self._load_tokenizer()
        self.tokenizer.ignored_index = -100

        base_url = "http://10.11.155.135:8201"
        self.client = AsyncTokenizerClient(base_url=base_url)

        # Pixel constraints
        self.image_merge_size = image_merge_size
        self.video_merge_size = video_merge_size
        self.audio_code_depth = audio_code_depth

        # Special tokens and IDs
        self.cls_token = self.CLS_TOKEN
        self.sep_token = self.SEP_TOKEN
        self.eos_token = self.EOS_TOKEN
        self.image_start = self.IMG_START
        self.image_patch = self.IMG_PATCH
        self.image_end = self.IMG_END
        self.video_start = self.VID_START
        self.video_patch = self.VID_PATCH
        self.video_end = self.VID_END
        self.audio_start = self.AUD_START
        self.audio_patch = self.AUD_PATCH
        self.audio_end = self.AUD_END

        self.image_start_id = self.tokenizer.convert_tokens_to_ids(self.image_start)
        self.image_patch_id = self.tokenizer.convert_tokens_to_ids(self.image_patch)
        self.image_end_id = self.tokenizer.convert_tokens_to_ids(self.image_end)

        self.video_start_id = self.tokenizer.convert_tokens_to_ids(self.video_start)
        self.video_patch_id = self.tokenizer.convert_tokens_to_ids(self.video_patch)
        self.video_end_id = self.tokenizer.convert_tokens_to_ids(self.video_end)

        self.audio_start_id = self.tokenizer.convert_tokens_to_ids(self.audio_start)
        self.audio_patch_id = self.tokenizer.convert_tokens_to_ids(self.audio_patch)
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids(self.audio_end)

        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)

        self.token_type_mapping = self._build_token_type_mapping()
        self.is_training = True
        self.role_prefixes = {
            "system": "",
            "user": "User: ",
            "bot": "Assistant: ",
            "assistant": "Assistant: ",
        }
        self.add_timestamp = add_timestamp

    async def get_image_feature_url_and_shape(self, req_id, image_url, version="v1", is_gen=False, resolution=2048):
        request = ImageEncodeRequest(
            version=version, req_id=req_id, is_gen=False, resolution=resolution, image_url=image_url
        )
        result = await self.client.encode_image(request)
        image_feature_url, image_feature_shape = result["feature_url"], result["feature_shape"]
        return image_feature_url, image_feature_shape

    async def get_video_feature_url_and_shape(
        self, req_id, video_url, version="v1", is_gen=False, max_frame=30, resolution=512
    ):
        # duration = get_duration(video_url)
        duration = 5
        frame = 10
        request = VideoEncodeRequest(
            version="v1",
            req_id=req_id,
            video_url=video_url,
            is_gen=is_gen,
            resolution=resolution,
            start_ts=0,
            end_ts=duration,
            frames=frame,
        )

        result = await self.client.encode_video(request)
        # result = {
        #     "feature_url": "fake_url",
        #     "feature_shape": [10,16,26,1536]
        # }
        video_feature_url, video_feature_shape = result["feature_url"], result["feature_shape"]
        # print(f"feature_shape:{video_feature_shape}")
        # print(f"video_feature_url:{video_feature_url}")

        return video_feature_url, video_feature_shape, duration

    def _build_token_type_mapping(self) -> Dict[Any, int]:
        mapping = defaultdict(lambda: IDS_TYPE_FLAG["text"])
        for token in (
            self.IMG_START,
            self.IMG_END,
            self.VID_START,
            self.VID_END,
        ):
            mapping[token] = IDS_TYPE_FLAG["image"]
        mapping[self.image_patch_id] = IDS_TYPE_FLAG["image"]
        return mapping

    def train(self) -> None:
        """Enable training mode (produces labels)."""
        self.is_training = True

    def eval(self) -> None:
        """Enable evaluation mode (doesn't produce labels)."""
        self.is_training = False

    # def _add_special_token(self, token: Union[str, int], outputs: Dict) -> None:
    #     token_id = token if isinstance(token, int) else self.tokenizer.convert_tokens_to_ids(token)
    #     outputs["input_ids"].append(token_id)
    #     outputs["token_type_ids"].append(self.token_type_mapping[token])
    #     pos = int(outputs["cur_position"])
    #     # outputs["position_ids"].append([pos] * 3)
    #     outputs["cur_position"] += 1

    def _add_text(self, tokens, outputs: Dict) -> None:
        if isinstance(tokens, str):
            tokens = self.tokenizer.encode(tokens, add_special_tokens=False)["input_ids"]
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * len(tokens))

        # start = outputs["cur_position"]
        token_num = int(len(tokens))
        # for i in range(token_num):
        #     outputs["position_ids"].append([int(start + i)] * 3)
        outputs["cur_position"] += token_num
        return token_num

    async def _add_image(self, req_id, image_url, outputs: Dict, feature=None) -> None:
        if feature is None:
            image_feature_url, image_feature_shape = await self.get_image_feature_url_and_shape(req_id, image_url)
        else:
            image_feature_url = feature["feature_url"]
            image_feature_shape = feature["feature_shape"]

        patches_h = image_feature_shape[1] // self.image_merge_size
        patches_w = image_feature_shape[2] // self.image_merge_size
        num_tokens = patches_h * patches_w + 1
        image_grid_thw = [0, 1, 0]  # 第一个scale
        image_grid_thw.extend([patches_w for _ in range(patches_h)])

        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)
        outputs["image_feature_urls"].append(image_feature_url)

        outputs["image_grid_thws"].append(image_grid_thw)
        outputs["image_type_ids"].append(0)
        return num_tokens

    async def _add_video(self, req_id, video_url, outputs: Dict, feature=None, is_time_stamp=True) -> None:
        if feature is None:
            video_feature_url, video_feature_shape, duration = await self.get_video_feature_url_and_shape(
                req_id, video_url
            )
        else:
            video_feature_url = feature["feature_url"]
            video_feature_shape = feature["feature_shape"]
            duration = 5
        frame = video_feature_shape[0]
        patches_h = video_feature_shape[1] // self.video_merge_size
        patches_w = video_feature_shape[2] // self.video_merge_size
        num_tokens_pre_frame = patches_h * patches_w + 1

        if is_time_stamp:
            timestamp_list = get_uniform_frame_timestamps_usec(duration, frame)
        else:
            timestamp_list = []

        video_grid_thw = []
        num_tokens = 0
        for i in range(frame):
            if is_time_stamp:
                time_stamp_token = self.tokenizer.encode(timestamp_list[i], add_special_tokens=False)["input_ids"]
                outputs["input_ids"].extend(time_stamp_token)
                num_tokens += len(time_stamp_token)

            outputs["input_ids"].extend([self.video_patch_id] * num_tokens_pre_frame)
            num_tokens += num_tokens_pre_frame

            video_grid_thw.extend([0, 1, 0])
            video_grid_thw.extend([patches_w for _ in range(patches_h)])

        outputs["video_feature_urls"].append(video_feature_url)

        outputs["video_grid_thws"].append(video_grid_thw)
        outputs["video_frame_lengths"].append(frame)

        return num_tokens

    def _compute_3d_positions(self, outputs) -> List[List[int]]:
        # Downsample time if needed
        input_ids, image_grid_thws, video_grid_thws = (
            outputs["input_ids"],
            outputs["image_grid_thws"],
            outputs["video_grid_thws"],
        )

        if len(image_grid_thws) > 0:
            pack_scale_hw_list = []
            for width_info in image_grid_thws:
                scale_hw_list = []  # 不同scale height * width
                for x in width_info:
                    if x == 0:
                        scale_hw_list.append([0, -1])  # [height, width]
                    else:
                        scale_hw_list[-1][0] += 1
                        if scale_hw_list[-1][1] == -1:
                            scale_hw_list[-1][1] = x  # 首次赋值
                        else:
                            assert scale_hw_list[-1][1] == x  # width 前后一致性检查
                # print(f"[check] scale_hw_list: {scale_hw_list} | token_num: {np.sum(np.prod(scale_hw_list, axis=-1))}")
                pack_scale_hw_list.append(scale_hw_list)
            outputs["position_ids"] = construct_3d_position_ids(
                np.array(input_ids), pack_scale_hw_list, self.image_patch_id, self.eos_token_id, self.image_end_id
            )
        elif len(video_grid_thws) > 0:
            pack_scale_hw_list = []
            for width_info in video_grid_thws:
                scale_hw_list = []  # 不同scale height * width
                for x in width_info:
                    if x == 0:
                        scale_hw_list.append([0, -1])  # [height, width]
                    else:
                        scale_hw_list[-1][0] += 1
                        if scale_hw_list[-1][1] == -1:
                            scale_hw_list[-1][1] = x  # 首次赋值
                        else:
                            assert scale_hw_list[-1][1] == x  # width 前后一致性检查
                # print(f"[check] scale_hw_list: {scale_hw_list} | token_num: {np.sum(np.prod(scale_hw_list, axis=-1))}")
                pack_scale_hw_list.append(scale_hw_list)
            outputs["position_ids"] = construct_3d_position_ids(
                np.array(input_ids),
                pack_scale_hw_list,
                self.video_patch_id,
                self.eos_token_id,
                self.video_end_id,
                frame_lengths=outputs["video_frame_lengths"],
                is_video=True,
            )
        else:
            outputs["position_ids"] = np.repeat(np.arange(0, len(input_ids)), 3, axis=-1).astype("float32").tolist()

    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        vocab_file_names = [
            "tokenizer.model",
            "spm.model",
            "ernie_token_100k.model",
        ]
        for i in range(len(vocab_file_names)):
            if os.path.exists(os.path.join(self.model_name_or_path, vocab_file_names[i])):
                ErnieBotTokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
                break
        self.tokenizer = ErnieBotTokenizer.from_pretrained(self.model_name_or_path)

    def apply_chat_template(self, request):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            messages: Either a request dict containing 'messages' field,
                                or a list of message dicts directly

        Returns:
            List of token IDs as strings (converted from token objects)
        """
        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat_template.")

        prompt_token_str = (
            self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=request.get("add_generation_prompt", True),
            )
            .replace("<|image@placeholder|>", "")
            .replace("<|video@placeholder|>", "")
            .replace("<|audio@placeholder|>", "")
        )

        # 手动添加eos_token
        # prompt_token_str = "</s><|im_start|>system\n<global_setting>\nthink_mode=False\n</global_setting><|im_end|>\n\n<|im_start|>user\n<image_start><image_end>哪个银行？<|im_end|>\n\n<|im_start|>assistant\n<think></think>"
        # prompt_token_str = "</s><|im_start|>system\n<global_setting>\nthink_mode=False\n</global_setting><|im_end|>\n\n<|im_start|>user\n<image_start><image_end>Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?\nOptions:\n(A) $6\n(B) $7\n(C) $8\n(D) $9<|im_end|>\n\n<|im_start|>assistant\n<think></think>"
        tokens = self.tokenizer.tokenize(prompt_token_str)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        data_processor_logger.info(
            f"req_id:{request.get('request_id', ''), } tokens: {tokens}, token_ids: {token_ids}"
        )
        return token_ids

    def _construct_image_mask_offset(self, input_ids, image_width_flatten):
        input_ids = np.array(input_ids)
        image_patch_num = np.sum(input_ids == self.image_patch_id)
        image_width_num = sum([np.sum(image_width) for image_width in image_width_flatten])
        assert int(image_patch_num) == int(image_width_num), (image_patch_num, image_width_num)
        max_len = input_ids.size

        # 提取每一张图片的上三角mask
        pack_up_shift = []
        for image_width in image_width_flatten:
            up_shift = []
            scale_tail_num = 0
            for x in image_width:
                if x == 0:
                    if scale_tail_num > 0:
                        up_shift.extend(list(range(scale_tail_num)))
                        scale_tail_num = 0
                else:
                    scale_tail_num += x
            if scale_tail_num > 0:
                up_shift.extend(list(range(scale_tail_num)))

            up_shift = np.array(up_shift, dtype=np.int32)
            assert up_shift.size == np.sum(image_width)
            pack_up_shift.append(up_shift)

        # 找到每一张图片的开始位置（应当按照label序列计算，等同于按照输入序列计算并往左移一位）
        pack_image_patch_start = np.where(input_ids == self.image_start_id)[0].tolist()
        pack_image_patch_end = np.where(input_ids == self.image_end_id)[0].tolist()
        assert len(pack_image_patch_start) == len(pack_image_patch_end) == len(pack_up_shift), (
            len(pack_image_patch_start),
            len(pack_image_patch_end),
            len(pack_up_shift),
        )

        # 组装
        attention_mask_offset = np.arange(max_len, dtype=np.int32)
        for image_patch_start, image_patch_end, up_shift in zip(
            pack_image_patch_start, pack_image_patch_end, pack_up_shift
        ):
            attention_mask_offset[image_patch_start : image_patch_start + up_shift.size] -= up_shift
            attention_mask_offset[image_patch_start + 1 : image_patch_end] = image_patch_end - 1

        return attention_mask_offset.tolist()

    def _construct_video_mask_offset(self, input_ids, video_width_flatten, frames=None):
        input_ids = np.array(input_ids)
        video_patch_num = np.sum((input_ids == self.video_patch_id))
        video_width_num = sum([np.sum(video_width) for video_width in video_width_flatten])
        assert int(video_patch_num) == int(video_width_num), (video_patch_num, video_width_num)
        max_len = input_ids.size

        # 提取每一张图片的上三角mask
        pack_up_shift = []
        for i, video_width in enumerate(video_width_flatten):
            frame = frames[0]
            image_patch_num = sum(video_width[: len(video_width) // frame])
            up_shift = []
            scale_tail_num = 0
            skip_first_token_timestamp = True
            for x in video_width:  # [0,1,0,9,9,9,9,0,1,0,9,9,99]
                if x == 0:
                    if scale_tail_num > 0:
                        sub_tail_num_list = list(range(scale_tail_num))
                        sub_tail_num_list = [x - image_patch_num for x in sub_tail_num_list]
                        if self.add_timestamp:
                            if skip_first_token_timestamp:
                                up_shift.extend(list(range(scale_tail_num)))
                            else:
                                up_shift.extend([0] * 11 + sub_tail_num_list)
                        else:
                            up_shift.extend(sub_tail_num_list)
                        scale_tail_num = 0

                else:
                    if self.add_timestamp:
                        if x == 1:
                            skip_first_token_timestamp = True
                        else:
                            skip_first_token_timestamp = False
                    scale_tail_num += x
            if scale_tail_num > 0:
                sub_tail_num_list = list(range(scale_tail_num))
                sub_tail_num_list = [x - image_patch_num for x in sub_tail_num_list]
                if self.add_timestamp:
                    up_shift.extend([0] * 11 + sub_tail_num_list)  # [range(1), range(16*9)] * frames
                else:
                    up_shift.extend(sub_tail_num_list)
            # log.debug(f"up_shift is {up_shift}")
            up_shift = np.array(up_shift, dtype=np.int32)
            if not self.add_timestamp:
                assert up_shift.size == np.sum(video_width)
            pack_up_shift.append(up_shift)

        # 找到每一张图片的开始位置（应当按照label序列计算，等同于按照输入序列计算并往左移一位）
        pack_video_patch_start = np.where(input_ids == self.video_start_id)[0].tolist()
        pack_video_patch_end = np.where(input_ids == self.video_end_id)[0].tolist()
        assert len(pack_video_patch_start) == len(pack_video_patch_end) == len(pack_up_shift), (
            len(pack_video_patch_start),
            len(pack_video_patch_end),
            len(pack_up_shift),
        )
        # 组装
        attention_mask_offset = np.arange(max_len, dtype=np.int32)
        for video_patch_start, video_patch_end, up_shift in zip(
            pack_video_patch_start, pack_video_patch_end, pack_up_shift
        ):
            attention_mask_offset[video_patch_start : video_patch_start + up_shift.size] -= up_shift
            # attention_mask_offset[video_patch_start + 1 : video_patch_end] -= video_patch_end - 1
        return attention_mask_offset.tolist()

    async def request2ids(
        self, request: Dict[str, Any], tgts: List[str] = None
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], None]]:
        """
        Convert chat messages into model inputs.
        Returns a dict with input_ids, token_type_ids, position_ids, images, grid_thw, image_type_ids, labels.
        """

        outputs = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "image_feature_urls": [],
            "image_grid_thws": [],
            "image_type_ids": [],
            "video_feature_urls": [],
            "video_grid_thws": [],
            "video_frame_lengths": [],
            "video_type_ids": [],
            "audio_feature_urls": [],
            "audio_grid_thws": [],
            "audio_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "pic_cnt": 0,
            "video_cnt": 0,
            "patch_idx": [],
            "patch_map": [],
            "attention_mask_offset": [],
        }

        messages = request.get("messages")
        request_id = request.get("request_id")
        mm_message_list = []
        for msg in messages:
            role = msg.get("role")
            assert role in self.role_prefixes, f"Unsupported role: {role}"
            content_items = msg.get("content")
            if not isinstance(content_items, list):
                content_items = [content_items]

            for item in content_items:
                if isinstance(item, dict) and item.get("type") in ["image_url", "video_url", "audio_url"]:
                    mm_message_list.append(item)

        prompt_token_ids = self.apply_chat_template(request)
        if len(prompt_token_ids) == 0:
            raise ValueError("Invalid input: prompt_token_ids must be a non-empty sequence of token IDs")
        start_index = 0
        mm_message_index = 0

        text_idx, image_idx, video_idx, audio_idx = 0, 0, 0, 0
        total_token_idx = 0
        total_patch_idx = 0
        outputs["patch_idx"].append(total_patch_idx)
        outputs["patch_map"].append(
            {
                "modal_id": IDS_TYPE_FLAG["text"],
                "end_idx": total_token_idx,
                "image_num": 0,
                "video_num": 0,
                "audio_num": 0,
            }
        )
        total_patch_idx += 1
        for i in range(len(prompt_token_ids)):
            if prompt_token_ids[i] in [
                self.image_start_id,
                self.video_start_id,
                self.audio_start_id,
            ]:
                if start_index < i + 1:
                    token_num = self._add_text(prompt_token_ids[start_index : i + 1], outputs)
                    text_idx += 1
                    total_token_idx += token_num
                    outputs["patch_idx"].extend([total_patch_idx for _ in range(token_num)])
                    outputs["patch_map"].append(
                        {
                            "modal_id": IDS_TYPE_FLAG["text"],
                            "end_idx": total_token_idx,
                            "image_num": image_idx,
                            "video_num": video_idx,
                            "audio_num": audio_idx,
                        }
                    )
                    total_patch_idx += 1

                start_index = i + 1
                mm_message = mm_message_list[mm_message_index]
                if mm_message["type"] == "image_url":
                    image_url = mm_message.get("image_url")["url"]
                    if image_url is None:
                        continue
                    feature = mm_message.get("feature", None)
                    outputs["pic_cnt"] += 1
                    token_num = await self._add_image(request_id, image_url, outputs, feature)
                    image_idx += 1
                    total_token_idx += token_num
                    outputs["patch_idx"].extend([total_patch_idx for _ in range(token_num)])
                    outputs["patch_map"].append(
                        {
                            "modal_id": IDS_TYPE_FLAG["image"],
                            "end_idx": total_token_idx,
                            "image_num": image_idx,
                            "video_num": video_idx,
                            "audio_num": audio_idx,
                        }
                    )
                    total_patch_idx += 1
                elif mm_message["type"] == "video_url":
                    video_url = mm_message.get("video_url")["url"]
                    if video_url is None:
                        continue
                    feature = mm_message.get("feature", None)
                    outputs["video_cnt"] += 1
                    token_num = await self._add_video(request_id, video_url, outputs, feature)
                    video_idx += 1
                    total_token_idx += token_num
                    outputs["patch_idx"].extend([total_patch_idx for _ in range(token_num)])
                    outputs["patch_map"].append(
                        {
                            "modal_id": IDS_TYPE_FLAG["video"],
                            "end_idx": total_token_idx,
                            "image_num": image_idx,
                            "video_num": video_idx,
                            "audio_num": audio_idx,
                        }
                    )
                    total_patch_idx += 1
                mm_message_index += 1
        if start_index < len(prompt_token_ids):
            token_num = self._add_text(prompt_token_ids[start_index:], outputs)
            text_idx += 1
            total_token_idx += token_num
            outputs["patch_idx"].extend([total_patch_idx for _ in range(token_num)])
            outputs["patch_map"].append(
                {
                    "modal_id": IDS_TYPE_FLAG["text"],
                    "end_idx": total_token_idx,
                    "image_num": image_idx,
                    "video_num": video_idx,
                    "audio_num": audio_idx,
                }
            )
            total_patch_idx += 1

        self._compute_3d_positions(outputs)
        input_ids, image_grid_thws, video_grid_thws = (
            outputs["input_ids"],
            outputs["image_grid_thws"],
            outputs["video_grid_thws"],
        )
        if len(image_grid_thws) > 0:
            outputs["attention_mask_offset"] = self._construct_image_mask_offset(input_ids, image_grid_thws)
        elif len(video_grid_thws) > 0:

            outputs["attention_mask_offset"] = self._construct_video_mask_offset(
                input_ids, video_grid_thws, outputs["video_frame_lengths"]
            )
        return outputs
