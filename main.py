from math import log2
import string


def remove_punctuation(text: str):
    translator = text.maketrans("", "", string.punctuation)
    cleaned_text = text.translate(translator)
    return cleaned_text


def tokenizer(text: str):
    text = text.lower()
    cleaned_text = remove_punctuation(text)
    return cleaned_text.split()


def TF_calculator(text_words: list, word: str):
    return text_words.count(word)


def IDF_calculator(document_list: list, inverted_index: dict, word: str):
    return log2(len(document_list)/len(inverted_index[word]))


def TF_IDF_calculator(document_list: list, inverted_index: dict, document_words_list: list, word: str):
    return float(TF_calculator(document_words_list, word))*float(IDF_calculator(document_list, inverted_index, word))
