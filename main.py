from math import log2, sqrt
import string
from difflib import SequenceMatcher
from sklearn.decomposition import PCA
import numpy as np


class Document():
    def __init__(self):
        self.text = ""
        self.paragraphs_vectors = {}
        self.doc_vector = []
        self.two_d_vector = [] # an attribute defined for making the 2d vectors(phaze 2)
        self.most_repeated_word = ""
        self.five_most_important_words = []


def remove_punctuation(text: str):
    translator = text.maketrans("", "", string.punctuation)
    cleaned_text = text.translate(translator)
    return cleaned_text


def tokenizer(text: str):
    text = text.lower()
    cleaned_text = remove_punctuation(text)
    return cleaned_text.split()


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


# a TF_IDF for the query which we use the same IDF function as documents but a different TF function for query
def TF_IDF_vectorize_query(documents_list: list, inverted_index: dict, sentence):
    tokenized_sentence = tokenizer(sentence)
    vector = []
    for word in inverted_index.keys():
        vector.append(TF_IDF_calculator_query(documents_list,
                      inverted_index, tokenized_sentence, word))
    return vector


# query
def TF_IDF_calculator_query(document_list: list, inverted_index: dict, document_words_list: list, word: str):
    return float(TF_calculator_query(document_words_list, word))*float(IDF_calculator(document_list, inverted_index, word))


# a TF function for only the query using sequencemmatcher incase query is not exactly like what is in documents
def TF_calculator_query(text_words: list, word: str, similarity_threshold: float = 0.8):
    exact_match_count = text_words.count(word)

    if exact_match_count > 0:
        return exact_match_count

    if word not in text_words:
        mild_similarity_count = 0
        for token in text_words:
            similarity_ratio = SequenceMatcher(None, token, word).ratio()
            if similarity_ratio >= similarity_threshold:
                mild_similarity_count += 1

        return mild_similarity_count
    return 0


def IDF_calculator(document_list: list, inverted_index: dict, word: str):
    return log2(len(document_list)/len(inverted_index[word]))


# TF calculator for documents
def TF_calculator(text_words: list, word: str):
    return text_words.count(word)


def TF_IDF_calculator(document_list: list, inverted_index: dict, document_words_list: list, word: str):
    return float(TF_calculator(document_words_list, word))*float(IDF_calculator(document_list, inverted_index, word))


def TF_IDF_vectorize(documents_list: list, inverted_index: dict, paragraph):
    tokenized_paragraph = tokenizer(paragraph)
    vector = []
    for word in inverted_index.keys():
        vector.append(TF_IDF_calculator(documents_list,
                      inverted_index, tokenized_paragraph, word))
    return vector


def vectorizing_documents(documents_list: dict):
    for doc in documents_list:
        doc.doc_vector = [sum(x)for x in zip(*doc.paragraphs_vectors.values())]


def cosine_similarity_calculator(A: list, B: list):
    sumAB, sumA2, sumB2 = 0, 0, 0
    for i in range(0, len(A)):
        sumAB += A[i]*B[i]
        sumA2 += A[i]*A[i]
        sumB2 += B[i]*B[i]

    if sumA2 == 0 or sumB2 == 0:
        return 0.0

    return sumAB/(sqrt(sumA2)*sqrt(sumB2))


def most_similar(most_similar_doc: Document, input_list: list, query_vector: list):
    cosine_distances = {}
    for item in input_list:
        if (type(item) == str):
            cosine = cosine_similarity_calculator(
                most_similar_doc.paragraphs_vectors[item], query_vector)
        else:
            cosine = cosine_similarity_calculator(
                item.doc_vector, query_vector)
        cosine_distances[item] = cosine
    cosine_distances = dict(sorted(
        cosine_distances.items(), key=lambda item: item[1]))
    return cosine_distances


# storing paragraphs of the document selected
def paragraphs(most_similar_doc):
    paragraph = most_similar_doc.text.split('\n')
    return paragraph


def most_repeated_word_setter(document_list: list, doc: Document, inverted_index: dict):
    maximum_TF = 0
    MRW = ""
    temp_dict = dict(zip(inverted_index.keys(), doc.doc_vector))
    for word, TF_IDF in temp_dict.items():
        if ((TF_IDF/(IDF_calculator(document_list, inverted_index, word)+1)) > maximum_TF):
            maximum_TF = TF_IDF / \
                (IDF_calculator(document_list, inverted_index, word)+1)
            MRW = word
    doc.most_repeated_word = MRW


def five_most_important_words_setter(doc: Document):
    temp_dict = dict(zip(inverted_index.keys(), doc.doc_vector))
    sorted_temp_dict = dict(
        sorted(temp_dict.items(), key=lambda item: item[1], reverse=True))
    doc.five_most_important_words = list(sorted_temp_dict.keys())[:5]


#2d vectorization of documents
def two_d_vectorization(documents_list: list, user_query_vector: list):

    pca = PCA(n_components=2)

    vectors_list = []
    query_two_d_vectors = []

    for doc in documents_list:
        vectors_list.append(doc.doc_vector)
    
    vectors_list.append(user_query_vector)


    transformed_data = pca.fit_transform(vectors_list)

    for i in range (len(transformed_data) - 1):
        documents_list[i].two_d_vector = transformed_data[i].tolist()

    query_two_d_vectors = transformed_data[-1].tolist()

    return query_two_d_vectors



documents_list = []

corpus = []

# getting the query
user_query = input("Please insert your query: ")

# for storing the document's id to have an access at the end
document_add_list = []
# appending documents to corpus
print("candidate_document_id: ")
for i in range(0, 20):
    document_add = int(input())
    document_add_list.append(document_add)
    file_paths = "data/document_{}.txt"
    final_path = file_paths.format(document_add)

    with open(final_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        corpus.append(file_content)

for txt in corpus:
    doc = Document()
    doc.text = txt
    doc.paragraphs_vectors = {paragraph: []for paragraph in txt.split('\n')}
    documents_list.append(doc)

inverted_index = make_inverted_index(corpus)


for doc in documents_list:
    for paragraph in doc.paragraphs_vectors.keys():
        doc.paragraphs_vectors[paragraph] = TF_IDF_vectorize(
            corpus, inverted_index, paragraph)

vectorizing_documents(documents_list)


user_query_vector = TF_IDF_vectorize_query(corpus, inverted_index, user_query)

#2d vectorization of documents and  query
two_d_query = two_d_vectorization(documents_list, user_query_vector)


most_similar_doc = list(most_similar(None,
                                     documents_list, user_query_vector).keys())[-1]

# getting access to the index that the text is stored and the document's id next
wnated_index = corpus.index(most_similar_doc.text)
print("document_id: ", document_add_list[wnated_index])

most_similar_paragraph = list(most_similar(most_similar_doc,
                                           most_similar_doc.paragraphs_vectors.keys(), user_query_vector).keys())[-1]
print("most_similar_paragraph: ", most_similar_paragraph)

# finding the most similar paragraph that query used in among all the paragraph's of the selected document
paragraph_list = paragraphs(most_similar_doc)
print("is_selected: ")

for i in paragraph_list:
    if i == most_similar_paragraph:
        print("     1")
    else:
        print("     0")

for doc in documents_list:
    most_repeated_word_setter(documents_list, doc, inverted_index)
    five_most_important_words_setter(doc)
