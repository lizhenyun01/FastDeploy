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
import os
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np

from fastdeploy.input.ernie_tokenizer import ErnieBotTokenizer
from fastdeploy.input.tokenzier_client import AsyncTokenizerClient, ImageEncodeRequest
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
        spatial_conv_size: int = 2,
        temporal_conv_size: int = 2,
        image_merge_size: int = 2,
        video_merge_size: int = 2,
        audio_code_depth: int = 32,
        **kwargs,
    ) -> None:
        # Tokenizer and image preprocessor
        self.model_name_or_path = tokenizer_name
        self._load_tokenizer()
        self.tokenizer.ignored_index = -100

        base_url = "http://10.11.155.135:8201"
        self.client = AsyncTokenizerClient(base_url=base_url)

        # Convolution sizes for patch aggregation
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size

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

    async def image_encode(self, req_id, image_url, version="v1", is_gen=False, resolution=2048):
        request = ImageEncodeRequest(
            version=version, req_id=req_id, is_gen=False, resolution=resolution, image_url=image_url
        )
        result = await self.client.encode_image(request)
        image_feature_url, image_feature_shape = result["feature_url"], result["feature_shape"]
        patches_h = image_feature_shape[1] // self.image_merge_size
        patches_w = image_feature_shape[2] // self.image_merge_size
        token_num = patches_h * patches_w + 1
        image_grid_thw = [0, 1, 0]  # 第一个scale
        image_grid_thw.extend([patches_w for _ in range(patches_h)])

        return image_feature_url, image_grid_thw, token_num, patches_h, patches_w

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
                    img_url = mm_message.get("image_url")["url"]
                    if img_url is None:
                        continue
                    outputs["pic_cnt"] += 1
                    token_num = await self._add_image(request_id, img_url, outputs)
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
                    outputs["video_cnt"] += 1
                    token_num = await self._add_video(request_id, video_url, outputs)
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
        return outputs

    def _add_special_token(self, token: Union[str, int], outputs: Dict) -> None:
        token_id = token if isinstance(token, int) else self.tokenizer.convert_tokens_to_ids(token)
        outputs["input_ids"].append(token_id)
        outputs["token_type_ids"].append(self.token_type_mapping[token])
        pos = int(outputs["cur_position"])
        outputs["position_ids"].append([pos] * 3)
        outputs["cur_position"] += 1

    def _add_text(self, tokens, outputs: Dict) -> None:
        if isinstance(tokens, str):
            tokens = self.tokenizer.encode(tokens, add_special_tokens=False)["input_ids"]
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * len(tokens))

        start = outputs["cur_position"]
        token_num = int(len(tokens))
        for i in range(token_num):
            outputs["position_ids"].append([int(start + i)] * 3)
        outputs["cur_position"] += token_num
        return token_num

    async def _add_image(self, req_id, img_url, outputs: Dict) -> None:
        image_feature_url, image_grid_thw, num_tokens, patches_h, patches_w = await self.image_encode(req_id, img_url)
        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        pos_ids = self._compute_3d_positions(1, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1
        outputs["image_feature_urls"].append(image_feature_url)

        outputs["image_grid_thws"].append(image_grid_thw)
        outputs["image_type_ids"].append(0)
        return num_tokens

    def _add_video(self, request_id, video_url, outputs: Dict) -> None:
        pass
        # image_feature_url, image_grid_thw, num_tokens, patches_h, patches_w = await self.image_encode(req_id, img_url)
        # outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        # outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        # pos_ids = self._compute_3d_positions(1, patches_h, patches_w, outputs["cur_position"])
        # outputs["position_ids"].extend(pos_ids)
        # outputs["cur_position"] = np.max(pos_ids) + 1
        # outputs["image_feature_urls"].append(image_feature_url)

        # outputs["image_grid_thws"].append(image_grid_thw)
        # outputs["image_type_ids"].append(0)

    def _compute_3d_positions(self, t: int, h: int, w: int, start_idx: int) -> List[List[int]]:
        # Downsample time if needed
        t_eff = t // self.temporal_conv_size if t != 1 else 1
        gh, gw = h // self.spatial_conv_size, w // self.spatial_conv_size
        time_idx = np.repeat(np.arange(t_eff), gh * gw)
        h_idx = np.tile(np.repeat(np.arange(gh), gw), t_eff)
        w_idx = np.tile(np.arange(gw), t_eff * gh)

        coords = list(zip(time_idx, h_idx, w_idx))
        return [[int(start_idx + ti), int(start_idx + hi), int(start_idx + wi)] for ti, hi, wi in coords]

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
        prompt_token_str = "</s>" + prompt_token_str
        tokens = self.tokenizer.tokenize(prompt_token_str)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        data_processor_logger.info(
            f"req_id:{request.get('request_id', ''), } tokens: {tokens}, token_ids: {token_ids}"
        )
        return token_ids
