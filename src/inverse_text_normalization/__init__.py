# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# from taggers.tokenize_and_classify import ClassifyFst
# from taggers.tokenize_and_classify_final import ClassifyFinalFst
# from verbalizers.verbalize import VerbalizeFst
# from verbalizers.verbalize_final import VerbalizeFinalFst

# from nemo.utils import logging
import warnings

try:
    import pynini

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    # logging.warning(
    #     "`pynini` is not installed ! \n"
    #     "Please run the `nemo_text_processing/setup.sh` script"
    #     "prior to usage of this toolkit."
    # )
    warnings.warn("`pynini` is not installed ! \n"
                  "Please run the `nemo_text_processing/setup.sh` script"
                  "prior to usage of this toolkit.")

    PYNINI_AVAILABLE = False
