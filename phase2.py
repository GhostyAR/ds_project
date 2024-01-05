from ds_project_lib import *

documents_list = []

corpus = []

number_of_docs = 50

document_add_list = []

for i in range(0, number_of_docs):
    document_add = i
    corpus_filler(corpus, i, document_add_list)

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

two_d_vectorization(documents_list)

temp_list = []

for doc in documents_list:
    temp_list.append(doc.two_d_vector)

data = np.array(temp_list)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(data)

# Plot the original data and the clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, marker='X', c='red', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
