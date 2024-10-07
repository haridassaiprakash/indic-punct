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

        cents_data = pynini.accep("നൂറ്") |  pynini.accep("നൂറു") | pynini.accep("നൂറ്റി") | pynini.accep("ഞ്ഞൂറ്") |  pynini.accep("ണ്ണൂറ്") | pynini.accep("ള്ളായിരം") | pynini.accep("ഞ്ഞൂറ്റി")  |  pynini.accep("ണ്ണൂറ്റി") | pynini.accep("ള്ളായിരത്തി") | pynini.accep("ഹണ്ട്രഡ്") | pynini.accep("ഹൺഡ്രഡ്")
        thousands_data = pynini.accep("യിരം") | pynini.accep("യിരത്തി") | pynini.accep("തൗസൻഡ്") | pynini.accep("തൌസൻഡ്") | pynini.accep("ആയിരം") | pynini.accep("ആയിരത്തി") | pynini.accep("ആയിരത്തി") | pynini.accep("തൌസൻഡ്സ്")
        lakhs_data = pynini.accep("ലക്ഷം") | pynini.accep("ലക്ഷത്തി") | pynini.accep("ലാഖ്സ്") | pynini.accep("ലാഖ്") | pynini.accep("ലാഖ") | pynini.accep("ലാഖസ്") | pynini.accep("ലഖ്") | pynini.accep("ലക്ഷ്")
        crores_data = pynini.accep("കോടി") | pynini.accep("ക്രോര്സ്") | pynini.accep("കോടികൾ") | pynini.accep("ക്രോർസ്")
        millions_data =  pynini.accep("മില്യൻ") | pynini.accep("മില്യൻസ്") | pynini.accep("ദശലക്ഷം")
        billions_data =  pynini.accep("ബില്യൻ") | pynini.accep("ബില്ല്യൻസ്") | pynini.accep("ബില്യൺ")

        del_And = pynutil.delete(pynini.closure(pynini.accep("ആൻഡ്"), 1 ,1 ))
        
        hundred = pynini.cross("നൂറ്", "100") | pynini.cross("നൂറു", "100") | pynini.cross("നൂറ്റി", "100") | pynini.cross("ഞ്ഞൂറ്", "100") | pynini.cross("ണ്ണൂറ്", "100") | pynini.cross("ള്ളായിരം", "100") | pynini.cross("ഞ്ഞൂറ്റി", "100") | pynini.cross("ണ്ണൂറ്റി", "100") | pynini.cross("ള്ളായിരത്തി", "100") | pynini.cross("ഹണ്ട്രഡ്", "100") | pynini.cross("ഹൺഡ്രഡ്", "100")
        thousand  = pynini.cross("യിരം", "1000") | pynini.cross("യിരത്തി", "1000") | pynini.cross("തൗസൻഡ്", "1000") | pynini.cross("തൌസൻഡ്", "1000") | pynini.cross("ആയിരം", "1000") | pynini.cross("ആയിരത്തി", "1000") | pynini.cross("തൌസൻഡ്സ്", "1000")
        lakh = pynini.cross("ലക്ഷം", "100000") | pynini.cross("ലക്ഷത്തി", "100000") | pynini.cross("ലാഖ്സ്", "100000") | pynini.cross("ലാഖ്", "100000") | pynini.cross("ലാഖ", "100000") | pynini.cross("ലാഖസ്", "100000") | pynini.cross("ലഖ്", "100000") | pynini.cross("ലക്ഷ്", "100000")
        crore = pynini.cross("കോടി", "10000000") | pynini.cross("ക്രോര്സ്", "10000000") | pynini.cross("ക്രോര്സ്", "10000000") | pynini.cross("കോടികൾ", "10000000") | pynini.cross("ക്രോർസ്", "10000000")
        million =  pynini.cross("മില്യൻ", "1000000") | pynini.cross("മില്യൻസ്", "1000000") | pynini.cross("ദശലക്ഷം", "1000000")
        billion =  pynini.cross("ബില്യൻ", "1000000000") | pynini.cross("ബില്ല്യൻസ്", "1000000000") | pynini.cross("ബില്യൺ", "1000000000")

        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)
 
        # In malayalam hundreds are said as a combined word എണ്ണൂറ് "ennur" instead of എട്ട് നൂറ് "ett nooru" --> (800)
        # to handle that cases malayalam_hundreds_component is created
        malayalam_hundreds_component = ( malayalam_hundreds + ( (delete_space + graph_tens) |
                                                           (pynutil.insert("0") + delete_space + graph_digit) |
                                                            pynutil.insert("00") ) )

        malayalam_hundreds_component_2 =  ( (graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(cents_data) + ( ((delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en)) |
                                                                                                     (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                      pynutil.insert("00") ) )
        hundreds_prefix_digits = (malayalam_hundreds_component | malayalam_hundreds_component_2)

        hundreds_prefix_tens = ( (graph_tens | graph_tens_en) + delete_space + pynutil.delete(cents_data) + ( ((delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                                     (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                      pynutil.insert("00") ) )

        graph_hundred_component_at_least_one_none_zero_digit= (hundred | hundreds_prefix_digits | hundreds_prefix_tens)

        self.graph_hundred_component_at_least_one_none_zero_digit= (graph_hundred_component_at_least_one_none_zero_digit)

         
        #If thousand reference is present, then extract the before "non thousand" part and delete "thousand"
        #else, just add 0 and retrieve tens
        #else, just add 00 and retrieve digits
        #else, just add 000
        thousands_prefix_digits =  (graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(thousands_data) + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                        pynutil.insert("000", weight= -0.1) )

        thousands_prefix_tens =  (graph_tens | graph_tens_en) + delete_space + pynutil.delete(thousands_data) + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en)) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                        pynutil.insert("000", weight= -0.1) )

        thousands_prefix_hundreds =  (graph_hundred_component_at_least_one_none_zero_digit ) + delete_space + pynutil.delete(thousands_data) + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en)) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                        pynutil.insert("000", weight= -0.1) )


        graph_thousands = thousands_prefix_hundreds | thousands_prefix_tens | thousands_prefix_digits | thousand 

        # Similarly lakhs graph
        lakhs_prefix_digits =  (graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(lakhs_data) + ( (delete_space + thousands_prefix_tens) |
                                                                                               (pynutil.insert("0")+ delete_space + thousands_prefix_digits) |
                                                                                               (pynutil.insert("0")+ delete_space + hundreds_prefix_tens) |
                                                                                               (pynutil.insert("00")+ delete_space + (hundreds_prefix_digits)) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("00000", weight= -0.1) )
        
        lakhs_prefix_tens =  (graph_tens | graph_tens_en) + delete_space + pynutil.delete(lakhs_data) + ( (delete_space + thousands_prefix_tens) |
                                                                                               (pynutil.insert("0")+ delete_space + thousands_prefix_digits) |
                                                                                               (pynutil.insert("0")+ delete_space + hundreds_prefix_tens) |
                                                                                               (pynutil.insert("00")+ delete_space + (hundreds_prefix_digits)) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("00000", weight= -0.1) )
 
        lakhs_prefix_hundreds =  (graph_hundred_component_at_least_one_none_zero_digit | graph_thousands ) + delete_space + pynutil.delete(lakhs_data) + ( (delete_space + thousands_prefix_tens) |
                                                                                               (pynutil.insert("0")+ delete_space + thousands_prefix_digits) |
                                                                                               (pynutil.insert("0")+ delete_space+ hundreds_prefix_tens) |
                                                                                               (pynutil.insert("00")+ delete_space + (hundreds_prefix_digits)) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("00000", weight= -0.1) )

        graph_lakhs = lakh | lakhs_prefix_digits | lakhs_prefix_tens | lakhs_prefix_hundreds

        # crores graph
        crores_prefix_digits = ( (graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(crores_data) + ( (delete_space + lakhs_prefix_tens) |
                                                                                (pynutil.insert("0")+ delete_space + lakhs_prefix_digits) |
                                                                                (pynutil.insert("0")+  delete_space + thousands_prefix_hundreds) |
                                                                                (pynutil.insert("00")+ delete_space + thousands_prefix_tens) |
                                                                                (pynutil.insert("000")+ delete_space + thousands_prefix_digits) |
                                                                                (pynutil.insert("000")+ delete_space + hundreds_prefix_tens) |
                                                                                (pynutil.insert("0000")+ delete_space + (hundreds_prefix_digits )) |
                                                                                (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                (pynutil.insert("000000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                pynutil.insert("0000000" , weight= -0.1) ) )

        crores_prefix_tens = ( (graph_tens | graph_tens_en | graph_hundred_component_at_least_one_none_zero_digit | graph_thousands | graph_lakhs) + delete_space + pynutil.delete(crores_data) + ( (delete_space + lakhs_prefix_tens) |
                                                                                (pynutil.insert("0")+  delete_space + lakhs_prefix_digits) |
                                                                                (pynutil.insert("0")+  delete_space + thousands_prefix_hundreds) |
                                                                                (pynutil.insert("00")+ delete_space + thousands_prefix_tens) |
                                                                                (pynutil.insert("000")+ delete_space + thousands_prefix_digits) |
                                                                                (pynutil.insert("000")+ delete_space + hundreds_prefix_tens) |
                                                                                (pynutil.insert("0000")+ delete_space + (hundreds_prefix_digits)) |
                                                                                (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                (pynutil.insert("000000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                pynutil.insert("0000000" , weight= -0.1) ) )
        
        graph_crores = crore | crores_prefix_digits | crores_prefix_tens

        # millions graph
        millions_prefix_digits =  ( graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(millions_data) + ( (delete_space + thousands_prefix_hundreds) |
                                                                                              (pynutil.insert("0")+ delete_space + thousands_prefix_tens) |
                                                                                              (pynutil.insert("00")+ delete_space + thousands_prefix_digits) |
                                                                                              (pynutil.insert("00")+ delete_space + hundreds_prefix_tens) |
                                                                                              (pynutil.insert("000")+ delete_space + (hundreds_prefix_digits )) |
                                                                                              (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                              (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("000000", weight= -0.1) )
        
        millions_prefix_tens =  (graph_tens | graph_tens_en ) + delete_space + pynutil.delete(millions_data) + ( (delete_space + thousands_prefix_hundreds) |
                                                                                              (pynutil.insert("0")+ delete_space + thousands_prefix_tens) |
                                                                                              (pynutil.insert("00")+ delete_space + thousands_prefix_digits) |
                                                                                              (pynutil.insert("00")+ delete_space + hundreds_prefix_tens) |
                                                                                              (pynutil.insert("000")+ delete_space + (hundreds_prefix_digits)) |
                                                                                              (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                              (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                               pynutil.insert("000000", weight= -0.1) )

        millions_prefix_hundreds =  (graph_hundred_component_at_least_one_none_zero_digit | graph_thousands| graph_lakhs | graph_crores) + delete_space + pynutil.delete(millions_data) + ( (delete_space + thousands_prefix_hundreds) |
                                                                                              (pynutil.insert("0")+ delete_space + thousands_prefix_tens) |
                                                                                              (pynutil.insert("00")+ delete_space + thousands_prefix_digits) |
                                                                                              (pynutil.insert("00")+ delete_space + hundreds_prefix_tens) |
                                                                                              (pynutil.insert("000")+ delete_space + (hundreds_prefix_digits)) |
                                                                                              (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                              (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                               pynutil.insert("000000", weight= -0.1) )
 
        graph_millions = million | millions_prefix_digits | millions_prefix_tens | millions_prefix_hundreds

        # billions graph
        billions_all =  (graph_digit | pynutil.insert("1") | graph_tens | graph_tens_en | graph_hundred_component_at_least_one_none_zero_digit | graph_lakhs | graph_crores | graph_millions) + delete_space + pynutil.delete(billions_data) + ( (delete_space + millions_prefix_hundreds) |
                                                                                        (pynutil.insert("0")+ delete_space + millions_prefix_tens) |
                                                                                        (pynutil.insert("00")+ delete_space + millions_prefix_digits) |
                                                                                        (pynutil.insert("0")+ delete_space + lakhs_prefix_hundreds) |
                                                                                        (pynutil.insert("00")+  delete_space + lakhs_prefix_tens) |
                                                                                        (pynutil.insert("000")+  delete_space + lakhs_prefix_digits) |
                                                                                        (pynutil.insert("000") + delete_space + thousands_prefix_hundreds) |
                                                                                        (pynutil.insert("0000")+ delete_space + thousands_prefix_tens) |
                                                                                        (pynutil.insert("00000")+ delete_space + thousands_prefix_digits) |
                                                                                        (pynutil.insert("00000")+ delete_space + hundreds_prefix_tens) |
                                                                                        (pynutil.insert("000000")+ delete_space+ (hundreds_prefix_digits)) |
                                                                                        (pynutil.insert("0000000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                        (pynutil.insert("00000000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                        pynutil.insert("000000000", weight= -0.1) )
  
        graph_billions = billion | billions_all

        fst = (  graph_zero |
                    graph_tens |
                    graph_tens_en |
                    graph_digit |
                    graph_hundred_component_at_least_one_none_zero_digit |
                    graph_thousands |
                    graph_lakhs |
                    graph_crores |
                    graph_millions |
                    graph_billions |
                    # graph_chars |
                    graph_multiples
                    # graph_char_multiples 
                    )


        fst = fst.optimize()

        self.graph_no_exception = fst

        self.graph = (pynini.project(fst, "input")) @ fst

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
