import logging
from enum import Enum
import pandas as pd
import requests
import json
import re
import os
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Set BASE_PATH to the directory containing this file
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger("main_logger")
common_word_threshold = 1000
mid_frequency_word_threshold = 50


class WordFrequency(Enum):
    RARE = 0
    MID_FREQUENCY = 1
    COMMON = 2


def analyze_word(word: str, words: dict[str, int], names: list[str]) -> WordFrequency:
    if word in names:
        return WordFrequency.COMMON

    normalized_word = word.lower()
    if normalized_word in words:
        if words[normalized_word] > common_word_threshold:
            return WordFrequency.COMMON
        if words[normalized_word] > mid_frequency_word_threshold:
            return WordFrequency.MID_FREQUENCY
    return WordFrequency.RARE


def sanitize_word(word: str) -> str:
    cleaned_word = []

    for i, char in enumerate(word):
        if char.isalpha() or (char == "'" and 0 < i < len(word) - 1):
            cleaned_word.append(char)

    return ''.join(cleaned_word)


def sanitize_words(text: str) -> list[str]:
    # Replace "'s" with an empty string
    text = text.replace("'s", "")

    # Split the text by spaces and newlines, removing empty entries
    words = re.split(r'\s+', text.strip())

    # Remove words containing "@" or "www"
    words_to_delete = [word for word in words if "@" in word or "www" in word.lower()]
    words = [word for word in words if word not in words_to_delete]

    # Clean up the text by concatenating remaining words
    cleaned_text = ' '.join(words)

    # Further split cleaned text by specified delimiters (replace sr_SplitOptions)
    split_options = [' ', os.linesep]
    for option in split_options:
        cleaned_text = cleaned_text.replace(option, ' ')
    words = cleaned_text.split()

    # Apply the CleanWord function to each word and remove empty words
    words = [sanitize_word(word) for word in words]

    words = [word for word in words if word.strip()]

    return words


def calculate_score(sanitized_words_amount: int, frequency: dict[WordFrequency, int]) -> float:
    score = frequency[WordFrequency.MID_FREQUENCY] * 0.5 + frequency[WordFrequency.COMMON] * 1
    score *= 1 / sanitized_words_amount
    return 1 - score  # Reverse the score so higher means more jargon


def print_colored_words(words, frequencies):
    if len(words) != len(frequencies):
        raise ValueError("Words and frequencies lists must be of the same length")

    color_map = {
        WordFrequency.RARE: Fore.RED,
        WordFrequency.MID_FREQUENCY: Fore.YELLOW,
        WordFrequency.COMMON: Fore.WHITE
    }

    for i, (word, freq) in enumerate(zip(words, frequencies)):
        color = color_map.get(freq, Fore.WHITE)
        print(f"{color}{word}{Style.RESET_ALL}", end=' ')
        if i % 25 == 0 and i != 0:
            print()
    print('\n\n---------------------------------------------\n\n')





def analyze_text(text: str, words: dict[str, int], names: list[str], verbose=True) -> float:
    sanitized_text: list[str] = sanitize_words(text)
    frequency = {
        WordFrequency.RARE: 0,
        WordFrequency.MID_FREQUENCY: 0,
        WordFrequency.COMMON: 0
    }
    word_grades = []
    for word in sanitized_text:
        word_frequency = analyze_word(word, words, names)
        frequency[word_frequency] += 1
        word_grades.append(word_frequency)
    if verbose:
        print_colored_words(sanitized_text, word_grades)
    return calculate_score(len(sanitized_text), frequency)


def calculate_grade(text: str):
    names = pd.read_csv(BASE_PATH + "names.csv", header=None)[0].tolist()
    # WORDS = "2024DataUKUS2020-2023.csv"
    WORDS = "DataUKUS2018-2021.csv"
    words = pd.read_csv(BASE_PATH + WORDS, header=None).set_index(0)[1].to_dict()
    return analyze_text(text, words, names)

