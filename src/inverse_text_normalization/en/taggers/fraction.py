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

from inverse_text_normalization.en.graph_utils import GraphFst
import pynini
from pynini.lib import pynutil, utf8

from inverse_text_normalization.en.data_loader_utils import get_abs_path
from inverse_text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    delete_space,
    delete_extra_space,
    insert_space
)
from inverse_text_normalization.en.utils import num_to_word

data_path = 'data/'

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


def get_quantity(frac, cardinal_graph_hundred_component_at_least_one_none_zero_digit):
    numbers = cardinal_graph_hundred_component_at_least_one_none_zero_digit @ (
        pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
    )
    with open(get_abs_path(data_path + "number_suffixes.tsv"), encoding='utf-8') as f:        
        suffixes = [line.strip() for line in f]
    
    suffix = pynini.union(*suffixes)
    res = frac + delete_extra_space + pynutil.insert("quantity: \"") + (suffix) + pynutil.insert("\"")
    return res


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        e.g. one three fourth hundred ->  { fraction { integer_part: 1 numerator: "3" denominator: "3" quantity: "hazar" } }

    cardinal: Cardinal GraphFst
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator

        cardinal_graph = cardinal.graph_no_exception
        cardinal_graph_hundred_component_at_least_one_none_zero_digit = (
            cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )
        ordinal_graph = ordinal.graph

        del_And = pynutil.delete(pynini.closure(pynini.accep("and"), 1 ,1 ))
        del_by = pynutil.delete(pynini.closure(pynini.accep("by"), 1 ,1 ))

        graph_digit = pynini.string_file(get_abs_path(data_path + "numbers/digit.tsv"))
        graph_tens = pynini.string_file(get_abs_path(data_path + "numbers/tens.tsv"))

        numerator = cardinal_graph
        denominator = graph_digit | graph_tens | ordinal_graph

        # Numerator is any cardinal number (e.g., "one", "two", "three")
        graph_numerator = pynutil.insert("numerator: \"") + numerator + pynutil.insert("\"")

        # Denominator is any ordinal number (e.g., "half", "third", "fourth")
        graph_denominator = pynutil.insert("denominator: \"") + denominator + pynutil.insert("\"")        
        
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        
        final_graph_wo_sign = graph_integer + (delete_space + del_And + delete_space | delete_space)  + graph_numerator + (delete_space + del_by + delete_space ) + graph_denominator
        final_graph = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal_graph_hundred_component_at_least_one_none_zero_digit
        )
        # single_fraction = graph_numerator + delete_space + graph_denominator
        # final_graph = final_graph | single_fraction

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
