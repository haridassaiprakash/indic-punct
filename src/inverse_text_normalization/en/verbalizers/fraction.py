# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from inverse_text_normalization.en.graph_utils import GraphFst
from pynini.lib import pynutil, utf8
from inverse_text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False



class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction, 
        e.g. one three fourth hundred ->  { fraction { integer_part: 1 numerator: "3" denominator: "4" quantity: "thousand" } } --> 1 3/4 thousand
    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")
        
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_integer = pynini.closure(integer + delete_space, 0, 1)

        numerator = (
            pynutil.delete("numerator:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_numerator = pynini.closure( numerator + delete_space, 0, 1)

        denominator = (
            pynutil.delete("denominator:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_denominator = pynini.closure(denominator + delete_space, 0, 1)

        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_quantity = pynini.closure(pynutil.insert(" ") + quantity + delete_space, 0, 1)

        graph = ( 
            optional_integer + 
            pynutil.insert(" ") +
            optional_numerator +
            delete_space +
            pynutil.insert("/") +
            optional_denominator +
            optional_quantity )

        # graph2 = (
        #     optional_numerator +
        #     delete_space +
        #     pynutil.insert("/") +
        #     optional_denominator )

        # graph = graph1 or  graph2

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
