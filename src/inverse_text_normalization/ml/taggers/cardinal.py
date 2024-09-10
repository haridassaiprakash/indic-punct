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

from inverse_text_normalization.ml.data_loader_utils import get_abs_path
from inverse_text_normalization.ml.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from inverse_text_normalization.ml.utils import num_to_word
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
        graph_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples.tsv"))
        malayalam_hundreds = pynini.string_file(get_abs_path(data_path + "numbers/ml_hundreds.tsv"))
        graph_tens_en = pynini.string_file(get_abs_path(data_path + "numbers/tens_en.tsv"))
        graph_char_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples_alphabets.tsv"))

        cents = pynini.accep("നൂറ്") |  pynini.accep("നൂറു") | pynini.accep("നൂറ്റി") | pynini.accep("ഞ്ഞൂറ്") |  pynini.accep("ണ്ണൂറ്") | pynini.accep("ള്ളായിരം") | pynini.accep("ഞ്ഞൂറ്റി")  |  pynini.accep("ണ്ണൂറ്റി") | pynini.accep("ള്ളായിരത്തി") | pynini.accep("ഹണ്ട്രഡ്") | pynini.accep("ഹൺഡ്രഡ്")
        thousands = pynini.accep("യിരം") | pynini.accep("യിരത്തി") | pynini.accep("തൗസൻഡ്") | pynini.accep("തൌസൻഡ്") | pynini.accep("ആയിരം") | pynini.accep("ആയിരത്തി")
        lakhs = pynini.accep("ലക്ഷം") | pynini.accep("ലക്ഷത്തി") | pynini.accep("ലാഖ്സ്") | pynini.accep("ലാഖ്") | pynini.accep("ലാഖ") | pynini.accep("ലാഖസ്") | pynini.accep("ലഖ്") | pynini.accep("ലക്ഷ്")
        crores = pynini.accep("കോടി") | pynini.accep("ക്രോര്സ്")

        del_And = pynutil.delete(pynini.closure(pynini.accep("ആൻഡ്"), 1 ,1 ))
        
        graph_hundred = pynini.cross("നൂറ്", "100") | pynini.cross("നൂറു", "100") | pynini.cross("നൂറ്റി", "100") | pynini.cross("ഞ്ഞൂറ്", "100") | pynini.cross("ണ്ണൂറ്", "100") | pynini.cross("ള്ളായിരം", "100") | pynini.cross("ഞ്ഞൂറ്റി", "100") | pynini.cross("ണ്ണൂറ്റി", "100") | pynini.cross("ള്ളായിരത്തി", "100") | pynini.cross("ഹണ്ട്രഡ്", "100") | pynini.cross("ഹൺഡ്രഡ്", "100")
        graph_thousand  = pynini.cross("യിരം", "1000") | pynini.cross("യിരത്തി", "1000") | pynini.cross("തൗസൻഡ്", "1000") | pynini.cross("തൌസൻഡ്", "1000") | pynini.cross("ആയിരം", "1000") | pynini.cross("ആയിരത്തി", "1000")
        graph_lakh = pynini.cross("ലക്ഷം", "100000") | pynini.cross("ലക്ഷത്തി", "100000") | pynini.cross("ലാഖ്സ്", "100000") | pynini.cross("ലാഖ്", "100000") | pynini.cross("ലാഖ", "100000") | pynini.cross("ലാഖസ്", "100000") | pynini.cross("ലഖ്", "100000") | pynini.cross("ലക്ഷ്", "100000")
        graph_crore = pynini.cross("കോടി", "10000000") | pynini.cross("ക്രോര്സ്", "10000000")

        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)

        delete_hundreds= pynutil.delete(cents)
        delete_thousands= pynutil.delete(thousands)
        delete_lakhs= pynutil.delete(lakhs)
        delete_crores= pynutil.delete(crores)
 
        # In malayalam hundreds are said as a combined word എണ്ണൂറ് "ennur" instead of എട്ട് നൂറ് "ett nooru" --> (800)
        # to handle that cases malayalam_hundreds_component is created
        malayalam_hundreds_component = ( malayalam_hundreds + ( (delete_space + graph_tens) |
                                                           (pynutil.insert("0") + delete_space + graph_digit) |
                                                            pynutil.insert("00") ) )

        #Handles 1-999 (direct spoken)
        #Handles 1-999 (direct spoken)
        hundreds =  ( graph_tens | graph_tens_en | graph_digit | pynutil.insert("1") ) + delete_space + delete_hundreds + ( ((delete_space + del_And + delete_space | delete_space) + ( graph_tens | graph_tens_en )) |
                                                                                                                                 (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + ( graph_digit )) |
                                                                                                                                  pynutil.insert("00") )

        graph_hundred_component_at_least_one_none_zero_digit = ( malayalam_hundreds_component | hundreds | graph_hundred )

        self.graph_hundred_component_at_least_one_none_zero_digit= (graph_hundred_component_at_least_one_none_zero_digit)

        # thousands graph
        thousands_prefix_digits =  ( graph_digit | pynutil.insert("1")) + delete_space + delete_thousands + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en)) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + (graph_digit )) |
                                                                                                        pynutil.insert("000", weight= -0.1) )

        thousands_prefix_tens =  (graph_tens | graph_tens_en) + delete_space + delete_thousands + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + ( graph_tens | graph_tens_en)) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + (graph_digit)) |
                                                                                                        pynutil.insert("000", weight= -0.1) )

        graph_thousands = thousands_prefix_tens | thousands_prefix_digits | graph_thousand

        # lakhs graph
        lakhs_prefix_digits =  (graph_digit | pynutil.insert("1")) + delete_space + delete_lakhs + ( (delete_space + thousands_prefix_tens) |
                                                                                              (pynutil.insert("0")+ delete_space + thousands_prefix_digits) |
                                                                                               (pynutil.insert("00")+ delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en)) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_digit)) |
                                                                                                pynutil.insert("00000", weight= -0.1) ) 
        
        lakhs_prefix_tens =  (graph_tens | graph_tens_en) + delete_space + delete_lakhs + ( (delete_space + thousands_prefix_tens) |
                                                                                              (pynutil.insert("0")+ delete_space + thousands_prefix_digits) |
                                                                                               (pynutil.insert("00")+ delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en)) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("00000", weight= -0.1) )
 
        graph_lakhs = graph_lakh | lakhs_prefix_digits | lakhs_prefix_tens

        # crores graph
        crores = ( (graph_digit | graph_tens | graph_tens_en) + delete_space + delete_crores + ( (delete_space + lakhs_prefix_tens) |
                                                                                (pynutil.insert("0")+  delete_space + lakhs_prefix_digits) |
                                                                                (pynutil.insert("00")+  delete_space + graph_thousands) |
                                                                                (pynutil.insert("0000")+ delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                (pynutil.insert("00000") + delete_space + (graph_tens | graph_tens_en)) |
                                                                                (pynutil.insert("000000") + delete_space + (graph_digit )) |
                                                                                pynutil.insert("0000000" , weight= -0.1) ) )
        graph_crores = graph_crore | crores

        fst = (  graph_zero |
                    graph_tens_en |
                    graph_tens |
                    graph_digit |
                    graph_hundred_component_at_least_one_none_zero_digit |
                    malayalam_hundreds_component |
                    graph_thousands |
                    graph_lakhs |
                    graph_crores |
                    graph_multiples)

        fst_crore = fst + graph_crore  # handles words like चार हज़ार करोड़
        fst_lakh = fst + graph_lakh  # handles words like चार हज़ार लाख
        fst = pynini.union(fst, fst_crore, fst_lakh)
        fst = fst.optimize()

        self.graph_no_exception = fst

        self.graph = (pynini.project(fst, "input")) @ fst

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
