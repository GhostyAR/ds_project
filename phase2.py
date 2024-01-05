from ds_project_lib import *

documents_list = []

corpus = []

number_of_docs = 1000

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
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
