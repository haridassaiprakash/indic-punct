import csv
from rapidfuzz import process, fuzz
from inverse_text_normalization.gu.data_loader_utils import get_abs_path

allowed_files = {"zero.tsv", "tens.tsv", "digit.tsv", "tens_en.tsv"}
data_path = "data/numbers/"

def load_data():
    """Load only specific TSV files from the data folder for fuzzy matching."""
    dictionary = {}

    for file_name in allowed_files:
        file_path = get_abs_path(data_path + file_name)
        print(file_path)
        
        with open(file_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")  
            for row in reader:
                if len(row) == 2:
                    alternative_spelling, correct_number = row
                    dictionary[alternative_spelling] = correct_number  

    return dictionary

# Load data into a dictionary
dictionary = load_data()

def fuzzy_match_token(token, threshold=80):
    if token.isdigit():
        return token 

    result = process.extractOne(token, dictionary.keys(), scorer=fuzz.ratio)

    if result and result[1] >= threshold:
        return dictionary[result[0]]
    return token  

def apply_fuzzy_search(text):
    """Apply fuzzy matching"""
    words = text.split()
    corrected_words = [fuzzy_match_token(word) for word in words]
    return " ".join(corrected_words)
