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

from inverse_text_normalization.en.data_loader_utils import get_abs_path
from inverse_text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from inverse_text_normalization.en.utils import num_to_word
# from inverse_text_normalization.lang_params import LANG
# data_path = f'data/{LANG}_data/'y
data_path = 'data/'

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

        english_digit_file = get_abs_path(data_path + 'numbers/digit.tsv')
        with open(english_digit_file, encoding='utf-8') as f:
            digits = f.readlines()
        english_digits = ''.join([line.split()[-1] for line in digits])
        english_digits_with_zero = "0" + english_digits
        # # print(f'hindi digits is {hindi_digits}')
        ENGLISH_DIGIT = pynini.union(*english_digits).optimize()
        ENGLISH_DIGIT_WITH_ZERO = pynini.union(*english_digits_with_zero).optimize()

        graph_zero = pynini.string_file(get_abs_path(data_path + "numbers/zero.tsv"))
        graph_tens = pynini.string_file(get_abs_path(data_path + "numbers/tens.tsv"))
        graph_digit = pynini.string_file(get_abs_path(data_path + "numbers/digit.tsv"))
        graph_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples.tsv"))
        graph_ties = pynini.string_file(get_abs_path(data_path + "numbers/ties.tsv"))
        # graph_chars = pynini.string_file(get_abs_path(data_path + "numbers/alphabets.tsv"))
        # graph_char_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples_alphabets.tsv"))
        #graph_tens_en = pynini.string_file(get_abs_path(data_path + "numbers/tens-en.tsv"))
        # graph_teen = pynini.string_file(get_abs_path(data_path + "numbers/teen.tsv"))
        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)

        # graph_tens = pynini.union(graph_ties + delete_space + graph_digit, graph_teen)

        cents_data = pynini.accep("hundred") | pynini.accep("hundreds") 
        thousands_data = pynini.accep("thousand") | pynini.accep("thousands") 
        lakhs_data = pynini.accep("lakh") | pynini.accep("lakhs") 
        crores_data = pynini.accep("crore") | pynini.accep("crores")
        millions_data =  pynini.accep("million") | pynini.accep("millions")
        billions_data =  pynini.accep("billion") | pynini.accep("billions")

        del_And = pynutil.delete(pynini.closure(pynini.accep("and"), 1 ,1 ))
        
        hundred = pynini.cross("hundred", "100") | pynini.cross("hundreds", "100") 
        thousand  = pynini.cross("thousand", "1000") | pynini.cross("thousands", "1000") 
        lakh = pynini.cross("lakh", "100000") | pynini.cross("lakhs", "100000") 
        crore = pynini.cross("crore", "10000000") | pynini.cross("crores", "10000000")
        million =  pynini.cross("million", "1000000") | pynini.cross("millions", "1000000")
        billion =  pynini.cross("billion", "1000000000") | pynini.cross("billions", "1000000000")
        
        #hundreds graph
        hundreds_prefix_digits = ( (graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(cents_data) + ( ((delete_space + del_And + delete_space | delete_space) + (graph_tens)) |
                                                                                                     (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                      pynutil.insert("00") ) )

        hundreds_prefix_tens = ( graph_tens + delete_space + pynutil.delete(cents_data) + ( ((delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                                     (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                      pynutil.insert("00") ) )

        graph_hundred_component_at_least_one_none_zero_digit= ( hundred | hundreds_prefix_digits | hundreds_prefix_tens )

        self.graph_hundred_component_at_least_one_none_zero_digit= (graph_hundred_component_at_least_one_none_zero_digit)
         
        #If thousand reference is present, then extract the before "non thousand" part, delete "thousand" and retrieve numbers
        #else, just add 000
        thousands_prefix_digits =  (graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(thousands_data) + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                        pynutil.insert("000", weight= -0.1) )

        thousands_prefix_tens =  (graph_tens) + delete_space + pynutil.delete(thousands_data) + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (graph_tens)) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                        pynutil.insert("000", weight= -0.1) )

        thousands_prefix_hundreds =  (graph_hundred_component_at_least_one_none_zero_digit ) + delete_space + pynutil.delete(thousands_data) + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (graph_tens)) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                        pynutil.insert("000", weight= -0.1) )


        graph_thousands = thousands_prefix_hundreds | thousands_prefix_tens | thousands_prefix_digits | thousand 

        # Similarly lakhs graph
        lakhs_prefix_digits =  (graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(lakhs_data) + ( ((delete_space + del_And + delete_space | delete_space) + thousands_prefix_tens) |
                                                                                               (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_digits) |
                                                                                               (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_tens) |
                                                                                               (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_digits) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("00000", weight= -0.1) )
        
        lakhs_prefix_tens =  (graph_tens) + delete_space + pynutil.delete(lakhs_data) + ( ((delete_space + del_And + delete_space | delete_space) + thousands_prefix_tens) |
                                                                                               (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_digits) |
                                                                                               (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_tens) |
                                                                                               (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_digits) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("00000", weight= -0.1) )
 
        lakhs_prefix_hundreds =  (graph_hundred_component_at_least_one_none_zero_digit | graph_thousands ) + delete_space + pynutil.delete(lakhs_data) + ( ((delete_space + del_And + delete_space | delete_space) + thousands_prefix_tens) |
                                                                                               (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_digits) |
                                                                                               (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_tens) |
                                                                                               (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_digits) |
                                                                                               (pynutil.insert("000")+ (delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                               (pynutil.insert("0000")+ (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("00000", weight= -0.1) )

        graph_lakhs = lakh | lakhs_prefix_digits | lakhs_prefix_tens | lakhs_prefix_hundreds

        # crores graph
        crores_prefix_digits = ( (graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(crores_data) + ( (delete_space + lakhs_prefix_tens) |
                                                                                (pynutil.insert("0")+  delete_space + lakhs_prefix_digits) |
                                                                                (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_hundreds) |
                                                                                (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_tens) |
                                                                                (pynutil.insert("000")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_digits) |
                                                                                (pynutil.insert("000")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_tens) |
                                                                                (pynutil.insert("0000")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_digits) |
                                                                                (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                (pynutil.insert("000000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                pynutil.insert("0000000" , weight= -0.1) ) )

        crores_prefix_tens = ( (graph_tens | graph_hundred_component_at_least_one_none_zero_digit | graph_thousands | graph_lakhs) + delete_space + pynutil.delete(crores_data) + ( (delete_space + lakhs_prefix_tens) |
                                                                                (pynutil.insert("0")+  delete_space + lakhs_prefix_digits) |
                                                                                (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_hundreds) |
                                                                                (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_tens) |
                                                                                (pynutil.insert("000")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_digits) |
                                                                                (pynutil.insert("000")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_tens) |
                                                                                (pynutil.insert("0000")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_digits) |
                                                                                (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                (pynutil.insert("000000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                pynutil.insert("0000000" , weight= -0.1) ) )
        
        graph_crores = crore | crores_prefix_digits | crores_prefix_tens

        # millions graph
        millions_prefix_digits =  (graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(millions_data) + ( ((delete_space + del_And + delete_space | delete_space) + thousands_prefix_hundreds) |
                                                                                              (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_tens) |
                                                                                              (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_digits) |
                                                                                              (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_tens) |
                                                                                              (pynutil.insert("000")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_digits) |
                                                                                              (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                              (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                pynutil.insert("000000", weight= -0.1) )
        
        millions_prefix_tens =  (graph_tens ) + delete_space + pynutil.delete(millions_data) + ( ((delete_space + del_And + delete_space | delete_space) + thousands_prefix_hundreds) |
                                                                                              (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_tens) |
                                                                                              (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_digits) |
                                                                                              (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_tens) |
                                                                                              (pynutil.insert("000")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_digits) |
                                                                                              (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                              (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                               pynutil.insert("000000", weight= -0.1) )

        millions_prefix_hundreds =  (graph_hundred_component_at_least_one_none_zero_digit | graph_thousands| graph_lakhs | graph_crores) + delete_space + pynutil.delete(millions_data) + ( ((delete_space + del_And + delete_space | delete_space) + thousands_prefix_hundreds) |
                                                                                              (pynutil.insert("0")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_tens) |
                                                                                              (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_digits) |
                                                                                              (pynutil.insert("00")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_tens) |
                                                                                              (pynutil.insert("000")+ (delete_space + del_And + delete_space | delete_space)+ hundreds_prefix_digits) |
                                                                                              (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                              (pynutil.insert("00000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                               pynutil.insert("000000", weight= -0.1) )
 
        graph_millions = million | millions_prefix_digits | millions_prefix_tens | millions_prefix_hundreds

        # billions graph
        billions_all =  (graph_digit | pynutil.insert("1") | graph_tens | graph_hundred_component_at_least_one_none_zero_digit | graph_lakhs | graph_crores | graph_millions) + delete_space + pynutil.delete(billions_data) + ( (delete_space + millions_prefix_hundreds) |
                                                                                        (pynutil.insert("0")+ delete_space + millions_prefix_tens) |
                                                                                        (pynutil.insert("00")+ delete_space + millions_prefix_digits) |
                                                                                        (pynutil.insert("0")+  delete_space + lakhs_prefix_hundreds) |
                                                                                        (pynutil.insert("00")+  delete_space + lakhs_prefix_tens) |
                                                                                        (pynutil.insert("000")+  delete_space + lakhs_prefix_digits) |
                                                                                        (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + thousands_prefix_hundreds) |
                                                                                        (pynutil.insert("0000")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_tens) |
                                                                                        (pynutil.insert("00000")+ (delete_space + del_And + delete_space | delete_space) + thousands_prefix_digits) |
                                                                                        (pynutil.insert("00000")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_tens) |
                                                                                        (pynutil.insert("000000")+ (delete_space + del_And + delete_space | delete_space) + hundreds_prefix_digits) |
                                                                                        (pynutil.insert("0000000") + (delete_space + del_And + delete_space | delete_space) + (graph_tens )) |
                                                                                        (pynutil.insert("00000000") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                        pynutil.insert("000000000", weight= -0.1) )
  
        graph_billions = billion | billions_all

        fst = (  graph_zero |
                    graph_tens |
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

        # labels_exception = [num_to_word(x) for x in range(1, 3)]
        # graph_exception = pynini.union(*labels_exception)

        self.graph_no_exception = fst
        self.graph = fst

        # self.graph = (pynini.project(fst, "input") - graph_exception.arcsort()) @ fst

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
