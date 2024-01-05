from ds_project_lib import *


documents_list = []

corpus = []

# getting the query
user_query = input("Please insert your query: ")

# for storing the document's id to have an access at the end
document_add_list = []
# appending documents to corpus
print("candidate_document_id: ")
for i in range(0, 20):
    document_id = int(input())
    corpus_filler(corpus, document_id, document_add_list)

# creating objects of Document class and appending them to documents_list
doc_object_creator(documents_list, corpus)

inverted_index = make_inverted_index(corpus)


for doc in documents_list:
    for paragraph in doc.paragraphs_vectors.keys():
        doc.paragraphs_vectors[paragraph] = TF_IDF_vectorize(
            corpus, inverted_index, paragraph)

vectorizing_documents(documents_list)

two_d_vectorization(documents_list)

user_query_vector = TF_IDF_vectorize_query(corpus, inverted_index, user_query)


most_similar_doc = list(most_similar(None,
                                     documents_list, user_query_vector).keys())[-1]

# getting access to the index that the text is stored and the document's id next
wnated_index = corpus.index(most_similar_doc.text)
print("document_id: ", document_add_list[wnated_index])

most_similar_paragraph = list(most_similar(most_similar_doc,
                                           most_similar_doc.paragraphs_vectors.keys(), user_query_vector).keys())[-1]
print("most_similar_paragraph: ", most_similar_paragraph)

#  finding the most similar paragraph that query used in among all the paragraph's of the selected document
paragraph_list = paragraphs(most_similar_doc)
print("is_selected: ")

for i in paragraph_list:
    if i == most_similar_paragraph:
        print("     1")
    else:
        print("     0")

for doc in documents_list:
    most_repeated_word_setter(documents_list, doc, inverted_index)
    five_most_important_words_setter(doc, inverted_index)

temp_list = []

for doc in documents_list:
    temp_list.append(doc.two_d_vector)

data = np.array(temp_list)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(data)

# Plot the original data and the clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k')
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
