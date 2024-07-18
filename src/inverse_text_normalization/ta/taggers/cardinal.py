
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

from inverse_text_normalization.ta.data_loader_utils import get_abs_path
from inverse_text_normalization.ta.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from inverse_text_normalization.ta.utils import num_to_word
# from inverse_text_normalization.lang_params import LANG
# data_path = f'data/{LANG}_data/'
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

        tamil_digit_file = get_abs_path(data_path + "numbers/ta_digit.tsv")
        with open(tamil_digit_file, encoding='utf-8') as f:
            digits = f.readlines()
        tamil_digits = ''.join([line.split()[-1] for line in digits])
        tamil_digits_with_zero = "0" + tamil_digits

        TAMIL_DIGIT = pynini.union(*tamil_digits).optimize()
        TAMIL_DIGIT_WITH_ZERO = pynini.union(*tamil_digits_with_zero).optimize()

        tamil_graph_zero = pynini.string_file(get_abs_path(data_path + "numbers/ta_zero.tsv"))
        tamil_graph_tens = pynini.string_file(get_abs_path(data_path + "numbers/ta_tens.tsv"))
        tamil_graph_digit = pynini.string_file(get_abs_path(data_path + "numbers/ta_digit.tsv"))
        tamil_graph_hundred_digit = pynini.string_file(get_abs_path(data_path + "numbers/ta_hundred_digit.tsv"))
        tamil_graph_thousand_digit = pynini.string_file(get_abs_path(data_path + "numbers/ta_thousand_digit.tsv"))
        tamil_graph_lakh_digit = pynini.string_file(get_abs_path(data_path + "numbers/ta_lakh_digit.tsv"))
        tamil_graph_crore_digit = pynini.string_file(get_abs_path(data_path + "numbers/ta_crore_digit.tsv"))
        tamil_graph_exception_list = pynini.string_file(get_abs_path(data_path + "numbers/ta_exceptions.tsv"))
        tamil_hundreds = pynini.string_file(get_abs_path(data_path + "numbers/ta_hundreds.tsv"))
        tamil_thousands = pynini.string_file(get_abs_path(data_path + "numbers/ta_thousands.tsv"))



        graph_zero = pynini.string_file(get_abs_path(data_path + "numbers/zero.tsv"))  
        graph_digit = pynini.string_file(get_abs_path(data_path + "numbers/digit.tsv"))
        graph_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples.tsv"))
        graph_ties = pynini.string_file(get_abs_path(data_path + "numbers/ties.tsv"))
        graph_chars = pynini.string_file(get_abs_path(data_path + "numbers/alphabets.tsv"))
        graph_char_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples_alphabets.tsv"))
        graph_tens_en = pynini.string_file(get_abs_path(data_path + "numbers/tens-en.tsv"))

        cents = pynini.accep("ஹண்ட்ரட்‌") | pynini.accep("ஹண்ட்ரெட்‌") | pynini.accep("ஹன்ட்ரட்‌") | pynini.accep("ஹன்ட்ரெட்‌") | pynini.accep("ஹண்ட்ரட்") | pynini.accep("ற்று") | pynini.accep('த்தி') | pynini.accep('நூற்று') | pynini.accep('நூறு') | pynini.accep('ஒன்று நூறு') | pynini.accep('நூத்தி') | pynini.accep('நூற்றுப்') | pynini.accep('நூற்றை')
        thousands = pynini.accep("தவுசண்ட்‌") | pynini.accep("தௌசண்ட்‌") | pynini.accep("தௌசண்ட்‌") | pynini.accep("தௌசண்ட்") | pynini.accep('யிரத்து') | pynini.accep('யிரத்தி') | pynini.accep('யிரம்') | pynini.accep('ஆயிரம்') | pynini.accep('ஆயிரத்து') | pynini.accep('வாயிரம்') | pynini.accep('ஆாயிரம்') | pynini.accep('ஆயிரத்தி') | pynini.accep('ஆாயிரத்தி') | pynini.accep('ஓராயிரம்') | pynini.accep('ஆயிரத்திப்')
        lakhs = pynini.accep("லேக்‌") | pynini.accep("லாக்‌") | pynini.accep("லேக்") | pynini.accep("லேக்ஸ்") | pynini.accep('லட்சம்') | pynini.accep('லட்சத்து') | pynini.accep('ஒரு லட்சம்') | pynini.accep('இலட்சம்')
        crores = pynini.accep("க்ரோர்‌") | pynini.accep('கோடி') | pynini.accep('கோடியே') | pynini.accep('ஒரு கோடி')

        graph_hundred = pynini.cross("ஹண்ட்ரட்‌", "100") | pynini.cross("ஹண்ட்ரெட்‌", "100") | pynini.cross("ஹன்ட்ரட்‌", "100") | pynini.cross("ஹன்ட்ரெட்‌", "100") | pynini.cross("ஹண்ட்ரட்", "100") | pynini.cross("ற்று", "100") | pynini.cross("த்தி", "100") | pynini.cross("நூற்று", "100") | pynini.cross("நூறு", "100") | pynini.cross("ஒன்று நூறு", "100") | pynini.cross("நூத்தி", "100") | pynini.cross("நூற்றுப்", "100") | pynini.cross("நூற்றை", "100")
        graph_thousand  = pynini.cross("தவுசண்ட்‌", "1000") | pynini.cross("தௌசண்ட்‌", "1000") | pynini.cross("தௌஸண்ட்", "1000") | pynini.cross("தௌசண்ட்", "1000") |  pynini.cross("ஆயிரம்" , "1000") | pynini.cross("யிரம்" , "1000") | pynini.cross("யிரத்தி" , "1000") | pynini.cross("யிரத்து" , "1000") | pynini.cross("ஆயிரத்து" , "1000") | pynini.cross("வாயிரம்" , "1000") | pynini.cross("ஆாயிரம்" , "1000") | pynini.cross("ஆயிரத்தி" , "1000") | pynini.cross("ஆாயிரத்தி" , "1000") | pynini.cross("ஓராயிரம்" , "1000") | pynini.cross("ஆயிரத்திப்" , "1000")
        graph_lakh = pynini.cross("லேக்‌", "100000") | pynini.cross("லாக்‌", "100000") | pynini.cross("லேக்", "100000") | pynini.cross("லேக்ஸ்", "100000") | pynini.cross("ஒரு லட்சம்", "100000") | pynini.cross("லட்சம்", "100000") | pynini.cross("லட்சத்து", "100000") | pynini.cross("இலட்சம்", "100000")
        graph_crore = pynini.cross("க்ரோர்‌", "10000000") |  pynini.cross("ஒரு கோடி", "10000000") | pynini.cross("கோடி", "10000000") | pynini.cross("கோடியே", "10000000")
        and_ = pynini.accep("அண்ட்") | pynini.accep("மற்றும்")

        del_And = pynutil.delete(and_)
        delete_hundreds= pynutil.delete(cents)
        delete_thousands= pynutil.delete(thousands)
        delete_lakhs= pynutil.delete(lakhs)
        delete_crores= pynutil.delete(crores)
        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)


        #Handles 1-999 (direct spoken)
        # தொள்ளாயிரத்தி ஐம்பது -->900        
        tamil_graph_hundred_component_1 = ( tamil_hundreds + ( (delete_space + tamil_graph_tens) |
                                                               (pynutil.insert("0") + delete_space + tamil_graph_digit) |
                                                                pynutil.insert("00") ) )
                      
        hundreds=  ( tamil_graph_digit | tamil_graph_tens | graph_tens_en | graph_digit | pynutil.insert("1") ) + delete_space + delete_hundreds + ( ((delete_space + del_And + delete_space | delete_space) + ( tamil_graph_tens | graph_tens_en )) |
                                                                                                                                 (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (tamil_graph_digit | graph_digit )) |
                                                                                                                                  pynutil.insert("00") )

        graph_hundred_component_at_least_one_none_zero_digit= ( tamil_graph_hundred_component_1 | hundreds | graph_hundred )

        self.graph_hundred_component_at_least_one_none_zero_digit= (graph_hundred_component_at_least_one_none_zero_digit)

        # thousands graph
        tamil_graph_thousand_component = ( tamil_thousands + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                               (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (tamil_graph_tens | graph_tens_en)) |
                                                               (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + (graph_digit | tamil_graph_digit)) |
                                                                pynutil.insert("000", weight= -0.1) ) )
        
        thousands_prefix_digits =  (tamil_graph_digit | graph_digit | pynutil.insert("1")) + delete_space + delete_thousands + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (tamil_graph_tens | graph_tens_en)) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + (graph_digit | tamil_graph_digit)) |
                                                                                                        pynutil.insert("000", weight= -0.1) )

        thousands_prefix_tens =  (tamil_graph_tens | graph_tens_en) + delete_space + delete_thousands + ( (delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                                       (pynutil.insert("0") + (delete_space + del_And + delete_space | delete_space) + (tamil_graph_tens | graph_tens_en)) |
                                                                                                       (pynutil.insert("00") + (delete_space + del_And + delete_space | delete_space) + (graph_digit | tamil_graph_digit)) |
                                                                                                        pynutil.insert("000", weight= -0.1) )

        graph_thousands = thousands_prefix_tens | thousands_prefix_digits | tamil_graph_thousand_component | graph_thousand

        # lakhs graph
        lakhs_prefix_digits =  (tamil_graph_digit | graph_digit | pynutil.insert("1")) + delete_space + delete_lakhs + ( (delete_space + thousands_prefix_tens) |
                                                                                              (pynutil.insert("0")+ delete_space + thousands_prefix_digits) |
                                                                                               (pynutil.insert("00")+ delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (tamil_graph_tens | graph_tens_en)) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_digit | tamil_graph_digit)) |
                                                                                                pynutil.insert("00000", weight= -0.1) )
        
        lakhs_prefix_tens =  (tamil_graph_tens | graph_tens_en) + delete_space + delete_lakhs + ( (delete_space + thousands_prefix_tens) |
                                                                                              (pynutil.insert("0")+ delete_space + thousands_prefix_digits) |
                                                                                               (pynutil.insert("00")+ delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                               (pynutil.insert("000") + (delete_space + del_And + delete_space | delete_space) + (tamil_graph_tens | graph_tens_en)) |
                                                                                               (pynutil.insert("0000") + (delete_space + del_And + delete_space | delete_space) + (graph_digit | tamil_graph_digit)) |
                                                                                                pynutil.insert("00000", weight= -0.1) )
 
        graph_lakhs = graph_lakh | lakhs_prefix_digits | lakhs_prefix_tens

        # crores graph
        crores = ( (tamil_graph_digit | graph_digit | tamil_graph_tens | graph_tens_en | pynutil.insert("1")) + delete_space + delete_crores + ( (delete_space + lakhs_prefix_tens) |
                                                                                (pynutil.insert("0")+  delete_space + lakhs_prefix_digits) |
                                                                                (pynutil.insert("00")+  delete_space + graph_thousands) |
                                                                                (pynutil.insert("0000")+ delete_space + graph_hundred_component_at_least_one_none_zero_digit) |
                                                                                (pynutil.insert("00000") + delete_space + (tamil_graph_tens | graph_tens_en)) |
                                                                                (pynutil.insert("000000") + delete_space + (graph_digit | tamil_graph_digit)) |
                                                                                pynutil.insert("0000000" , weight= -0.1) ) )
        graph_crores = graph_crore | crores

        fst = (  graph_zero |
                    tamil_graph_tens |
                    graph_tens_en |
                    tamil_graph_digit |
                    graph_digit |
                    graph_hundred_component_at_least_one_none_zero_digit |
                    tamil_graph_hundred_component_1 |
                    graph_thousands |
                    graph_lakhs |
                    graph_crores |
                    graph_chars |
                    graph_multiples|
                    graph_char_multiples |
                    tamil_graph_exception_list )
        
        self.graph_no_exception = fst
        self.graph = fst

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

                                                      
