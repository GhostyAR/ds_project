from math import log2, sqrt
import string
import re


class Document():
    def __init__(self):
        self.text = ""
        self.sentences_vectors = {}
        self.doc_vector = []


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


def make_inverted_index(documents_list):
    inverted_index = {}
    i = 0
    for doc in documents_list:
        doc_word_set = set(tokenizer(doc))
        for word in doc_word_set:
            if word not in inverted_index.keys():
                inverted_index[word] = []
            inverted_index[word].append(i)
        i += 1
    return inverted_index


def TF_IDF_vectorize(documents_list: list, inverted_index: dict, sentence):
    tokenized_sentence = tokenizer(sentence)
    vector = []
    for word in inverted_index.keys():
        vector.append(TF_IDF_calculator(documents_list,
                      inverted_index, tokenized_sentence, word))
    return vector


def vectorizing_documents(documents_list: dict):
    for doc in documents_list:
        doc.doc_vector = [sum(x) for x in zip(*doc.sentences_vectors.values())]

def cosine_similarity_calculator(A: list, B: list):
    sumAB, sumA2, sumB2 = 0, 0, 0
    for i in range(0, len(A)):
        sumAB += A[i]*B[i]
        sumA2 += A[i]*A[i]
        sumB2 += B[i]*B[i]
    return sumAB/(sqrt(sumA2)*sqrt(sumB2))


documents_list = []

corpus = [
    "this is the first document. second document coming next.",
    "This document is the second document. going for the third!",
    "And this is the third one.",
    "Is this the first document?",
]

for txt in corpus:
    doc = Document()
    doc.text = txt
    doc.sentences_vectors = {sentence: []
                             for sentence in re.split(r'[.!?\n] ', txt)}
    documents_list.append(doc)

inverted_index = make_inverted_index(corpus)

for doc in documents_list:
    for sentence in doc.sentences_vectors.keys():
        doc.sentences_vectors[sentence] = TF_IDF_vectorize(
            corpus, inverted_index, sentence)

vectorizing_documents(documents_list)
