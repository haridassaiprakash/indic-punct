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


import pynini
from pynini.lib import pynutil, utf8

from inverse_text_normalization.kn.data_loader_utils import get_abs_path
from inverse_text_normalization.kn.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from inverse_text_normalization.kn.utils import num_to_word

# from inverse_text_normalization.lang_params import LANG
# data_path = f'data/{LANG}_data/'
data_path = 'data/'


def get_alternate_spellings(text):
    return


try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted.
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        # integer, negative

        NEMO_CHAR = utf8.VALID_UTF8_CHAR
        NEMO_SIGMA = pynini.closure(NEMO_CHAR)
        NEMO_SPACE = " "
        NEMO_WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", u"\u00A0").optimize()
        NEMO_NOT_SPACE = pynini.difference(NEMO_CHAR, NEMO_WHITE_SPACE).optimize()
        # NEMO_NON_BREAKING_SPACE = u"\u00A0"

        hindi_digit_file = get_abs_path(data_path + 'numbers/digit.tsv')
        with open(hindi_digit_file, encoding='utf-8') as f:
            digits = f.readlines()
        hindi_digits = ''.join([line.split()[-1] for line in digits])
        hindi_digits_with_zero = "0" + hindi_digits
        # # print(f'hindi digits is {hindi_digits}')
        HINDI_DIGIT = pynini.union(*hindi_digits).optimize()
        HINDI_DIGIT_WITH_ZERO = pynini.union(*hindi_digits_with_zero).optimize()

        graph_zero = pynini.string_file(get_abs_path(data_path + "numbers/zero.tsv"))
        graph_tens = pynini.string_file(get_abs_path(data_path + "numbers/tens.tsv"))
        graph_digit = pynini.string_file(get_abs_path(data_path + "numbers/digit.tsv"))
        graph_chars = pynini.string_file(get_abs_path(data_path + "numbers/alphabets.tsv"))
        graph_mutiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples.tsv"))
        kannada_hundreds = pynini.string_file(get_abs_path(data_path + "numbers/kn_hundreds.tsv"))
        graph_tens_en = pynini.string_file(get_abs_path(data_path + "numbers/tens_en.tsv"))


        with open(get_abs_path(data_path + "numbers/hundred.tsv"), encoding='utf-8') as f:
            hundreds = f.readlines()

        hundred = hundreds[0].strip()
        hundred_alt = hundreds[1].strip()
        hundred_alt_2 = hundreds[2].strip()
        hundred_alt_3 = hundreds[3].strip()

        with open(get_abs_path(data_path + "numbers/thousands.tsv"), encoding="utf-8") as f:
            thousands = f.readlines()
        thousand = thousands[0].strip()
        thousand_alt = thousands[1].strip()

        with open(get_abs_path(data_path + "numbers/lakh.tsv"), encoding="utf-8") as f:
            lakhs = f.readlines()
        lakh = lakhs[0].strip()
        lakh_alt = lakhs[1].strip()

        with open(get_abs_path(data_path + "numbers/crore.tsv"), encoding="utf-8") as f:
            crores = f.readlines()
        crore = crores[0].strip()

        graph_hundred = pynini.cross(hundred, "100") | pynini.cross(hundred_alt, "100")| pynini.cross(hundred_alt_2, "100") | pynini.cross(hundred_alt_3, "100")
        graph_crore = pynini.cross(crore, "10000000")
        graph_lakh = pynini.cross(lakh, "100000")
        graph_thousand = pynini.cross(thousand, "1000") | pynini.cross(thousand_alt, "1000")

        # In kannada hundreds are said as a combined word ಮೂನ್ನೂರು "Munnooru" instead of ಮೂರು ನೂರು "mooru nooru"
        # to handle that cases kannada_hundreds_component is created
        kannada_graph_hundred = pynini.cross("ನೂರು", "100") | pynini.cross("ಇನ್ನೂರು", "200") | pynini.cross("ಇನ್ನೂರ್", "200") | pynini.cross("ಇನ್ನೂರ", "200") | pynini.cross("ಮುನ್ನೂರು", "300") | pynini.cross("ಮುನ್ನೂರ", "300") |\
                               pynini.cross("ಮೂನ್ನೂರು", "300") | pynini.cross("ಮುನ್ನೂರ್", "300") | pynini.cross("ಮೂನ್ನೂರ", "300") | pynini.cross("ನಾಲ್ಕುನೂರು", "400") | pynini.cross("ಐನೂರ್", "500") |pynini.cross("ಐನೂರ", "500") | pynini.cross("ಐನೂರು", "500") |  pynini.cross("ಆರ್ನೂರ್", "600") |\
                                pynini.cross("ಏಳ್ನೂರು", "700")
        
        kannada_hundreds_component = ( kannada_hundreds + ( (delete_space + graph_tens) |
                                                           (pynutil.insert("0") + delete_space + graph_digit) |
                                                            pynutil.insert("00") ) )
  
        graph_hundred_component = pynini.union(
            graph_digit + delete_space + (pynutil.delete(hundred) | pynutil.delete(hundred_alt) | pynutil.delete(hundred_alt_2) | pynutil.delete(hundred_alt_3)) + delete_space,
            pynutil.insert("0"))

        graph_hundred_component += pynini.union(graph_tens, pynutil.insert("0") + (graph_digit | pynutil.insert("0")))

        # handling double digit hundreds like उन्निस सौ + digit/thousand/lakh/crore etc
        graph_hundred_component_prefix_tens = pynini.union(
            graph_tens + delete_space + (pynutil.delete(hundred) | pynutil.delete(hundred_alt) | pynutil.delete(hundred_alt_2)| pynutil.delete(hundred_alt_3)) + delete_space)

        graph_hundred_component_prefix_tens += pynini.union(graph_tens,
                                                            pynutil.insert("0") + (graph_digit | pynutil.insert("0")))

        graph_hundred_component_non_hundred = pynini.union(graph_tens,
                                                           pynutil.insert("0") + (graph_digit | pynutil.insert("0")))

        graph_hundred_component = pynini.union(graph_hundred_component,
                                               graph_hundred_component_prefix_tens)

        graph_hundred_component_at_least_one_none_zero_digit = pynini.union(graph_hundred_component,
                                                                            graph_hundred_component_non_hundred)

        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        graph_thousands_component = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space +
            (pynutil.delete(thousand) | pynutil.delete(thousand_alt)),
            pynutil.insert("00", weight=0.1),
        )

        graph_lakhs_component = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space +
            (pynutil.delete(lakh) | pynutil.delete(lakh_alt)),
            pynutil.insert("00", weight=0.1)
        )

        graph_crores_component = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete(crore),
            pynutil.insert("00", weight=0.1)
        )

        # fst = graph_thousands
        fst = pynini.union(
            graph_crores_component
            + delete_space
            + graph_lakhs_component
            + delete_space
            + graph_thousands_component
            + delete_space
            + graph_hundred_component,
            graph_zero,
        )

        fst_crore = fst + graph_crore  # handles words like चार हज़ार करोड़
        fst_lakh = fst + graph_lakh  # handles words like चार हज़ार लाख
        fst = pynini.union(fst, fst_crore, fst_lakh, graph_crore, graph_lakh, graph_thousand, graph_hundred,graph_zero,graph_chars,kannada_graph_hundred,graph_mutiples,kannada_hundreds_component,graph_tens_en)

        self.graph_no_exception = fst

        self.graph = (pynini.project(fst, "input")) @ fst

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
