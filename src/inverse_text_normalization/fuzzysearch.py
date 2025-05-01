import os
import csv
import importlib
from rapidfuzz import process, fuzz

allowed_files = {"zero.tsv", "tens.tsv", "digit.tsv", "tens-en.tsv", "units.tsv"}

def load_data(lang):
    """Load language-specific fuzzy matching data dynamically."""
    try:
        data_loader_module = importlib.import_module(f"inverse_text_normalization.{lang}.data_loader_utils")
        get_abs_path = data_loader_module.get_abs_path
    except ModuleNotFoundError:
        raise ImportError(f"Data loader not found for language: {lang}")

    data_path = "data/numbers/"
    dictionary = {}

    for file_name in allowed_files:
        file_path = get_abs_path(data_path + file_name)
        if not os.path.exists(file_path):
            continue

        with open(file_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")

            if file_name == "units.tsv":
                # Store unit words as keys with themselves as values
                for row in reader:
                    if row:  # Ensure the row is not empty
                        unit_word = row[0].strip()
                        dictionary[unit_word] = unit_word
            else:
                for row in reader:
                    if len(row) == 2:
                        alternative_spelling, correct_number = row
                        dictionary[alternative_spelling.strip()] = alternative_spelling.strip()
    return dictionary

def fuzzy_match_token(token, dictionary, threshold=80):
    """Match a token using fuzzy search"""
    if token.isdigit():
        return token 

    result = process.extractOne(token, dictionary.keys(), scorer=fuzz.ratio)

    if result and result[1] >= threshold:
        return dictionary[result[0]]
    return token  

def apply_fuzzy_search(text, lang):
    """Apply fuzzy matching for a given language"""
    dictionary = load_data(lang) # Load language-specific dictionary
    words = text.split()
    corrected_words = [fuzzy_match_token(word, dictionary) for word in words]
    return " ".join(corrected_words)
