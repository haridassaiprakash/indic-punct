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

from inverse_text_normalization.ori.data_loader_utils import get_abs_path
from inverse_text_normalization.ori.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from inverse_text_normalization.ori.utils import num_to_word

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
        graph_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples.tsv"))
        graph_ties = pynini.string_file(get_abs_path(data_path + "numbers/ties.tsv"))
        graph_chars = pynini.string_file(get_abs_path(data_path + "numbers/alphabets.tsv"))
        graph_char_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples_alphabets.tsv"))
        graph_tens_en = pynini.string_file(get_abs_path(data_path + "numbers/tens-en.tsv"))

        cents = pynini.accep("ଶହେ") | pynini.accep("ଶହ")  | pynini.accep("ହଣ୍ଡ୍ରେଡ")  | pynini.accep("ହଣ୍ଡ୍ରେଡ୍")
        thousands = pynini.accep("ହଜାର") | pynini.accep("ଥାଉଜେଣ୍ଡ")
        lakhs = pynini.accep("ଲକ୍ଷ") | pynini.accep("ଲକ୍ଷେ") | pynini.accep("ଲାଖ୍‌") | pynini.accep("ଲ୍ୟାକ୍") | pynini.accep("ଲ୍ୟାକ୍ସ") | pynini.accep("ଲାଖ")
        crores = pynini.accep("କୋଟି") | pynini.accep("କ୍ରୋର")

        del_And = pynutil.delete(pynini.closure(pynini.accep("ଆଣ୍ଡ"), 1 ,1 ))

        graph_hundred = pynini.cross("ଶହେ", "100") | pynini.cross("ଶହ", "100") | pynini.cross("ହଣ୍ଡ୍ରେଡ", "100") | pynini.cross("ହଣ୍ଡ୍ରେଡ୍", "100")
        graph_thousand  = pynini.cross("ହଜାର", "1000") | pynini.cross("ଥାଉଜେଣ୍ଡ", "1000")
        graph_lakh = pynini.cross("ଲକ୍ଷ", "100000") | pynini.cross("ଲକ୍ଷେ", "100000") | pynini.cross("ଲାଖ୍‌", "100000") | pynini.cross("ଲ୍ୟାକ୍", "100000") | pynini.cross("ଲ୍ୟାକ୍ସ", "100000") | pynini.cross("ଲାଖ", "100000")
        graph_crore = pynini.cross("କୋଟି", "10000000") | pynini.cross("କ୍ରୋର", "10000000")

        #Handles 1-999 (direct spoken)
        graph_hundred_component = pynini.union((graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(cents) + (delete_space + del_And + delete_space | delete_space),
                                               pynutil.insert("0"))
        graph_hundred_component += pynini.union((graph_tens_en | graph_tens) , (graph_ties  | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")))
        # handling double digit hundreds like उन्निस सौ + digit/thousand/lakh/crore etc
        #graph_hundred_component_prefix_tens = pynini.union(graph_tens + delete_space + pynutil.delete(cents) + delete_space,)
        #                                                   # pynutil.insert("55"))
        graph_hundred_component_prefix_tens = pynini.union((graph_tens_en | graph_tens) + delete_space + pynutil.delete(cents) + (delete_space + del_And + delete_space | delete_space),
                                                            )

        graph_hundred_component_prefix_tens += pynini.union((graph_tens_en | graph_tens),
                                                            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")))

        # Although above two components have the capability to handle 1-99 also, but since we are combining both of them
        # later on, ambiguity creeps in. So, we define a shorter fst below to handle the cases from 1-99 exclusively.

        # graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
        #         pynini.closure(HINDI_DIGIT_WITH_ZERO) + (HINDI_DIGIT_WITH_ZERO - "०") + pynini.closure(HINDI_DIGIT_WITH_ZERO)
        # )

        #Handles 10-99 in both hi, en
        graph_hundred_component_non_hundred = pynini.union((graph_tens_en | graph_tens),
                                                            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")))

        #This thing now handles only 100-999 cases (in regular spoken form) and 1000-9999 (in hundred spoken form)
        #Because of combining these both FSTs, there comes ambiguity while dealing with 1-99 cases.
        graph_hundred_component = pynini.union(graph_hundred_component,
                                               graph_hundred_component_prefix_tens)

        graph_hundred_component_at_least_one_none_zero_digit = pynini.union(graph_hundred_component, graph_hundred_component_non_hundred)



        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        #If hazar reference is present, then extract the before "non hazar" part and delete "hazar"
        #else, just add 00
        graph_thousands_component = pynini.union(
            (graph_hundred_component_at_least_one_none_zero_digit + delete_space | pynutil.insert("1", weight=-0.1)) + pynutil.delete(thousands),
            (pynutil.insert("0") + graph_hundred_component_prefix_tens),
            pynutil.insert("00", weight=-0.1),
        )

        graph_lakhs_component = pynini.union(
            (graph_hundred_component_at_least_one_none_zero_digit + delete_space | pynutil.insert("1", weight=-0.1)) + pynutil.delete(lakhs),
            pynutil.insert("00", weight=-0.1)
        )

        graph_crores_component = pynini.union(
            (graph_hundred_component_at_least_one_none_zero_digit + delete_space | pynutil.insert("1", weight=-0.1)) + pynutil.delete(crores),
            pynutil.insert("00", weight=-0.1)
        )

        # fst = graph_thousands
        fst = pynini.union(
            graph_crores_component
            + (delete_space | delete_space + del_And + delete_space)
            + graph_lakhs_component
            + (delete_space | delete_space + del_And + delete_space)
            + (graph_thousands_component)
            + (delete_space | delete_space + del_And + delete_space)
            + (graph_hundred_component | pynutil.insert("", weight=0.1)),
            graph_zero,
        )

        fst_crore = fst+graph_crore # handles words like चार हज़ार करोड़
        fst_lakh = fst+graph_lakh # handles words like चार हज़ार लाख
        # fst = pynini.union(fst, fst_crore, fst_lakh, graph_crore, graph_lakh, graph_thousand, graph_hundred, graph_zero, graph_multiples, graph_char_multiples, graph_chars)
        fst = pynini.union(fst, fst_crore, fst_lakh, graph_crore, graph_lakh, graph_thousand, graph_hundred, graph_zero, graph_multiples,graph_chars, graph_tens_en,graph_char_multiples)

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
