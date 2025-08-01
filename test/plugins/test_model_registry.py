# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from fastdeploy import ModelRegistry
from fastdeploy.plugins import load_model_register_plugins

initial_archs = ModelRegistry.get_supported_archs()
print("Supported architectures before loading plugins:", initial_archs)

# python setup.py isntall
# need install fastdeploy-plugins
load_model_register_plugins()

final_archs = ModelRegistry.get_supported_archs()
print("Supported architectures after loading plugins:", final_archs)

added_count = len(final_archs) - len(initial_archs)

if added_count != 1:
    print(
        f"Warning: Expected exactly 1 new architecture to be registered, but got {added_count}. "
        "Plugin loading may have failed or registered unexpected number of architectures."
    )
else:
    print("Success: Exactly one new architecture was registered.")
