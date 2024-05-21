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

from inverse_text_normalization.te.data_loader_utils import get_abs_path
from inverse_text_normalization.te.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from inverse_text_normalization.te.utils import num_to_word
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
        graph_tens_en = pynini.string_file(get_abs_path(data_path + "numbers/tens_en.tsv"))
        graph_ties = pynini.string_file(get_abs_path(data_path + "numbers/ties.tsv"))
        graph_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples.tsv"))
        graph_digit = pynini.string_file(get_abs_path(data_path + "numbers/digit.tsv"))
        graph_chars = pynini.string_file(get_abs_path(data_path + "numbers/alphabets.tsv"))
        graph_char_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples_alphabets.tsv"))

        cents = pynini.accep("వంద") | pynini.accep("వందలు") | pynini.accep("వందల") | pynini.accep("నూట") | pynini.accep("హండ్రెడ్")
        veya = pynini.accep("వెయ్యి") | pynini.accep("వేలు") | pynini.accep("వేల") | pynini.accep("వెయ్య") | pynini.accep("థౌసండ్")
        laksha = pynini.accep("లక్ష") | pynini.accep("లక్షల") | pynini.accep("లక్షలు") | pynini.accep("ల్యాక్") | pynini.accep("ల్యాక్స్")
        koti = pynini.accep("కోటి") | pynini.accep("కోట్లు") | pynini.accep("కోట్ల") | pynini.accep("క్రోర్")

        hundred = pynini.cross("వంద", "100") | pynini.cross("వందలు", "100") | pynini.cross("వందల", "100") | pynini.cross("నూట", "100") | pynini.cross("హండ్రెడ్", "100")
        thousand  = pynini.cross("వెయ్యి", "1000") | pynini.cross("వేలు", "1000") | pynini.cross("వేల", "1000") | pynini.cross("వెయ్య", "1000") | pynini.cross("థౌసండ్", "1000")
        lakh = pynini.cross("లక్ష", "100000") | pynini.cross("లక్షల", "100000") | pynini.cross("లక్షలు", "100000") | pynini.cross("ల్యాక్", "100000") | pynini.cross("ల్యాక్స్", "100000")
        crore = pynini.cross("కోటి", "10000000") | pynini.cross("కోట్లు", "10000000") | pynini.cross("కోట్ల", "10000000")  | pynini.cross("క్రోర్", "10000000")

        delete_hundreds= pynutil.delete(cents)
        delete_thousands= pynutil.delete(veya)
        delete_lakhs= pynutil.delete(laksha)
        delete_crores= pynutil.delete(koti)
        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)
        del_And = pynutil.delete(pynini.closure(pynini.accep("అండ్"), 1 ,1 ))

        # To handles the cases from 200 to 999
        # Also handling double digit hundreds like tweleve hundred + digit/thousand/lakh/crore etc (12,456)
        hundreds= ( (graph_digit | graph_tens | graph_tens_en) + delete_space + delete_hundreds + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en) |
                                                                                                                                                           (pynutil.insert("0") + graph_digit) |
                                                                                                                                                            pynutil.insert("00") ) 
        # in telugu నూట is used for 100 so the below graph is created for 101 to 199
        nuta_graph= ( pynutil.insert("1") + delete_hundreds + (delete_space + del_And + delete_space | delete_space) + ( pynutil.insert("0") + graph_digit | graph_tens | graph_tens_en))

        graph_hundred_component_at_least_one_none_zero_digit= (hundred | hundreds | nuta_graph )

        self.graph_hundred_component_at_least_one_none_zero_digit= (graph_hundred_component_at_least_one_none_zero_digit)
        
        #If thousand reference is present, then extract the before "non thousand" part and delete "thousand"
        #else, just add 0 and retrieve tens
        #else, just add 00 and retrieve digits
        #else, just add 000
        thousands =  (graph_digit | graph_tens | graph_tens_en) + delete_space + delete_thousands + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (graph_tens | graph_tens_en)) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + graph_digit) |
                                                                                                        pynutil.insert("000", weight= -0.1) )

        graph_thousands = thousand | thousands
        
        # similarly lakhs graph
        lakhs = ( (graph_digit | graph_tens | graph_tens_en) + delete_space + delete_lakhs + ( (delete_space + graph_thousands) |
                                                                                               (pynutil.insert("00")+ delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                               (pynutil.insert("000") + delete_space + (graph_tens | graph_tens_en)) |
                                                                                               (pynutil.insert("0000") + delete_space + graph_digit) |
                                                                                                pynutil.insert("00000") ) )
        graph_lakhs = lakh | lakhs

       # similarly crores graph
        crores = ( (graph_digit | graph_tens | graph_tens_en) + delete_space + delete_crores + ( (delete_space + graph_lakhs) |
                                                                                (pynutil.insert("00")+  delete_space + graph_thousands) |
                                                                                (pynutil.insert("0000")+ delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                (pynutil.insert("00000") + delete_space + (graph_tens | graph_tens_en)) |
                                                                                (pynutil.insert("000000") + delete_space + graph_digit) |
                                                                                pynutil.insert("00000") ) )
        graph_crores = crore | crores
        # All graph components
        fst = (  graph_zero |
                    graph_tens |
                    graph_tens_en |
                    graph_digit |
                    graph_hundred_component_at_least_one_none_zero_digit |
                    graph_thousands |
                    graph_lakhs |
                    graph_crores |
                    graph_chars |
                    graph_multiples|
                    graph_char_multiples )

        self.graph_no_exception = fst

        self.graph = (pynini.project(fst, "input")) @ fst

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
