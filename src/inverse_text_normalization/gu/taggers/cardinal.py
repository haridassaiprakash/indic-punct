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

from inverse_text_normalization.gu.data_loader_utils import get_abs_path
from inverse_text_normalization.gu.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from inverse_text_normalization.gu.utils import num_to_word
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
        graph_ties = pynini.string_file(get_abs_path(data_path + "numbers/ties.tsv"))
        graph_chars = pynini.string_file(get_abs_path(data_path + "numbers/alphabets.tsv"))
        graph_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples.tsv"))
        graph_tens_en = pynini.string_file(get_abs_path(data_path + "numbers/tens_en.tsv"))
        graph_char_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples_alphabets.tsv"))

        cents_data = pynini.accep("સો") | pynini.accep("સ્સો") | pynini.accep("હન્ડ્રેડ")  | pynini.accep("હુંદ્રેડ")  | pynini.accep("હંડ્રેડ")  | pynini.accep("હુંડ્રેડ")  | pynini.accep("સૌ")
        thousands_data = pynini.accep("થાઉઝન્ડ") | pynini.accep("હજાર") | pynini.accep("થૌસંદ") | pynini.accep("થૌઝન્ડ") | pynini.accep("થાૌઝન્ડ") | pynini.accep("થૌસન્ડ") | pynini.accep("થાઉસેન્ડ") | pynini.accep("થાઉઝન્ડ્સ") | pynini.accep("હજારો")
        lakhs_data = pynini.accep("લાખ") | pynini.accep("લાખસ")
        crores_data = pynini.accep("કરોડ઼") | pynini.accep("કરોડ") | pynini.accep("ક્રોર્સ")
        millions_data =  pynini.accep("મિલિયન") | pynini.accep("મિલિયન્સ")
        billions_data =  pynini.accep("બિલિયન") | pynini.accep("બિલિયન્સ") | pynini.accep("અબજો") | pynini.accep("અબજ")
        and_ = pynini.accep("એન્ડ") | pynini.accep("ને") | pynini.accep("અને")
        
        hundred = pynini.cross("સો", "100") | pynini.cross("સ્સો", "100") | pynini.cross("હન્ડ્રેડ", "100") | pynini.cross("હુંદ્રેડ", "100") | pynini.cross("હંડ્રેડ", "100") | pynini.cross("હુંડ્રેડ", "100") | pynini.cross("સૌ", "100")
        thousand  = pynini.cross("થાઉઝન્ડ", "1000") | pynini.cross("હજાર", "1000") | pynini.cross("થૌસંદ", "1000") | pynini.cross("થૌઝન્ડ", "1000") | pynini.cross("થાૌઝન્ડ", "1000") | pynini.cross("થૌસન્ડ", "1000") | pynini.cross("થાઉસેન્ડ", "1000") | pynini.cross("થાઉઝન્ડ્સ", "1000") | pynini.cross("હજારો", "1000")
        lakh = pynini.cross("લાખ", "100000") | pynini.cross("લાખસ", "100000")
        crore = pynini.cross("કરોડ઼", "10000000") | pynini.cross("કરોડ", "10000000") | pynini.cross("ક્રોર્સ", "10000000")
        million =  pynini.cross("મિલિયન", "1000000") | pynini.cross("મિલિયન્સ", "1000000")
        billion =  pynini.cross("બિલિયન", "1000000000") | pynini.cross("બિલિયન્સ", "1000000000") | pynini.cross("અબજો", "1000000000") | pynini.cross("અબજ", "1000000000")
        
        del_And = pynutil.delete(and_)  

        #hundreds graph
        hundreds_prefix_digits = ( (graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(cents_data) + ( ((delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en)) |
                                                                                                     (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                      pynutil.insert("00") ) )

        hundreds_prefix_tens = ( (graph_tens | graph_tens_en) + delete_space + pynutil.delete(cents_data) + ( ((delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                                     (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                      pynutil.insert("00") ) )

        graph_hundred_component_at_least_one_none_zero_digit= ( hundred | hundreds_prefix_digits | hundreds_prefix_tens )

        self.graph_hundred_component_at_least_one_none_zero_digit= (graph_hundred_component_at_least_one_none_zero_digit)
         
        #If thousand reference is present, then extract the before "non thousand" part, delete "thousand" and retrieve numbers
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
                                                                                               (pynutil.insert("00")+ delete_space + hundreds_prefix_digits) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("00000", weight= -0.1) )
        
        lakhs_prefix_tens =  (graph_tens | graph_tens_en) + delete_space + pynutil.delete(lakhs_data) + ( (delete_space + thousands_prefix_tens) |
                                                                                               (pynutil.insert("0")+ delete_space + thousands_prefix_digits) |
                                                                                               (pynutil.insert("0")+ delete_space + hundreds_prefix_tens) |
                                                                                               (pynutil.insert("00")+ delete_space + hundreds_prefix_digits) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("00000", weight= -0.1) )
 
        lakhs_prefix_hundreds =  (graph_hundred_component_at_least_one_none_zero_digit | graph_thousands ) + delete_space + pynutil.delete(lakhs_data) + ( (delete_space + thousands_prefix_tens) |
                                                                                               (pynutil.insert("0")+ delete_space + thousands_prefix_digits) |
                                                                                               (pynutil.insert("0")+ delete_space + hundreds_prefix_tens) |
                                                                                               (pynutil.insert("00")+ delete_space + hundreds_prefix_digits) |
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
                                                                                (pynutil.insert("0000")+ delete_space + hundreds_prefix_digits) |
                                                                                (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                (pynutil.insert("000000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                pynutil.insert("0000000" , weight= -0.1) ) )

        crores_prefix_tens = ( (graph_tens | graph_tens_en | graph_hundred_component_at_least_one_none_zero_digit | graph_thousands | graph_lakhs) + delete_space + pynutil.delete(crores_data) + ( (delete_space + lakhs_prefix_tens) |
                                                                                (pynutil.insert("0")+  delete_space + lakhs_prefix_digits) |
                                                                                (pynutil.insert("0")+ delete_space+ thousands_prefix_hundreds) |
                                                                                (pynutil.insert("00")+ delete_space + thousands_prefix_tens) |
                                                                                (pynutil.insert("000")+ delete_space + thousands_prefix_digits) |
                                                                                (pynutil.insert("000")+ delete_space + hundreds_prefix_tens) |
                                                                                (pynutil.insert("0000")+ delete_space + hundreds_prefix_digits) |
                                                                                (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                (pynutil.insert("000000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                pynutil.insert("0000000" , weight= -0.1) ) )
        
        graph_crores = crore | crores_prefix_digits | crores_prefix_tens

        # millions graph
        millions_prefix_digits =  ( graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(millions_data) + ( (delete_space + thousands_prefix_hundreds) |
                                                                                              (pynutil.insert("0")+ delete_space + thousands_prefix_tens) |
                                                                                              (pynutil.insert("00")+ delete_space + thousands_prefix_digits) |
                                                                                              (pynutil.insert("00")+ delete_space + hundreds_prefix_tens) |
                                                                                              (pynutil.insert("000")+ delete_space + hundreds_prefix_digits) |
                                                                                              (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                              (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("000000", weight= -0.1) )
        
        millions_prefix_tens =  (graph_tens | graph_tens_en ) + delete_space + pynutil.delete(millions_data) + ( (delete_space + thousands_prefix_hundreds) |
                                                                                              (pynutil.insert("0")+ delete_space + thousands_prefix_tens) |
                                                                                              (pynutil.insert("00")+ delete_space + thousands_prefix_digits) |
                                                                                              (pynutil.insert("00")+ delete_space + hundreds_prefix_tens) |
                                                                                              (pynutil.insert("000")+ delete_space + hundreds_prefix_digits) |
                                                                                              (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                              (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                               pynutil.insert("000000", weight= -0.1) )

        millions_prefix_hundreds =  (graph_hundred_component_at_least_one_none_zero_digit | graph_thousands| graph_lakhs | graph_crores) + delete_space + pynutil.delete(millions_data) + ( (delete_space + thousands_prefix_hundreds) |
                                                                                              (pynutil.insert("0")+ delete_space+ thousands_prefix_tens) |
                                                                                              (pynutil.insert("00")+ delete_space + thousands_prefix_digits) |
                                                                                              (pynutil.insert("00")+ delete_space + hundreds_prefix_tens) |
                                                                                              (pynutil.insert("000")+ delete_space + hundreds_prefix_digits) |
                                                                                              (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en )) |
                                                                                              (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                               pynutil.insert("000000", weight= -0.1) )
 
        graph_millions = million | millions_prefix_digits | millions_prefix_tens | millions_prefix_hundreds

        # billions graph
        billions_all =  (graph_digit | pynutil.insert("1") | graph_tens | graph_tens_en | graph_hundred_component_at_least_one_none_zero_digit | graph_lakhs | graph_crores | graph_millions) + delete_space + pynutil.delete(billions_data) + ( (delete_space + millions_prefix_hundreds) |
                                                                                        (pynutil.insert("0")+ delete_space + millions_prefix_tens) |
                                                                                        (pynutil.insert("00")+ delete_space + millions_prefix_digits) |
                                                                                        (pynutil.insert("0")+ delete_space + lakhs_prefix_hundreds) |
                                                                                        (pynutil.insert("00")+ delete_space + lakhs_prefix_tens) |
                                                                                        (pynutil.insert("000")+ delete_space + lakhs_prefix_digits) |
                                                                                        (pynutil.insert("000")+ delete_space + thousands_prefix_hundreds) |
                                                                                        (pynutil.insert("0000")+ delete_space + thousands_prefix_tens) |
                                                                                        (pynutil.insert("00000")+ delete_space + thousands_prefix_digits) |
                                                                                        (pynutil.insert("00000")+ delete_space + hundreds_prefix_tens) |
                                                                                        (pynutil.insert("000000")+ delete_space + hundreds_prefix_digits) |
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
