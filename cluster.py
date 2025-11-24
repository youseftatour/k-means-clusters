import numpy as np
import random

# -------------------------------------------
# 1. GENERATE DOCUMENTS WITH RANDOM TF-IDF
# -------------------------------------------

words = ["team", "coach", "hockey", "baseball", "soccer", 
         "penalty", "score", "win", "loss", "season"]

num_docs = 15
num_words = len(words)

def random_document():
    vec = []
    for _ in range(num_words):
        appear = random.choice([0, 1])  # if word appears
        if appear:
            vec.append(round(random.uniform(2, 6), 2))  # random TF-IDF
        else:
            vec.append(0.0)
    return np.array(vec)

documents = np.array([random_document() for _ in range(num_docs)])


# -------------------------------------------
# 2. COSINE SIMILARITY & K-MEANS (K = 2)
# -------------------------------------------

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def assign_clusters(docs, centroids):
    assignments = []
    for doc in docs:
        sims = [cosine_similarity(doc, c) for c in centroids]
        assignments.append(int(np.argmax(sims)))
    return assignments

def compute_centroids(docs, assignments, k):
    centroids = []
    for i in range(k):
        cluster_docs = docs[np.array(assignments) == i]
        if len(cluster_docs) == 0:
            centroids.append(np.zeros(num_words))
        else:
            centroids.append(np.mean(cluster_docs, axis=0))
    return np.array(centroids)

# initialize centroids randomly
K = 2
centroids = documents[random.sample(range(num_docs), K)]

# run K-means for fixed number of iterations
for _ in range(15):
    clusters = assign_clusters(documents, centroids)
    centroids = compute_centroids(documents, clusters, K)


# -------------------------------------------
# 3. SHOW TOP 5 FEATURES IN EACH CLUSTER
# -------------------------------------------

for i in range(K):
    print(f"\n=== Cluster {i} ===")
    centroid = centroids[i]
    top_indices = centroid.argsort()[::-1][:5]
    for idx in top_indices:
        print(f"{words[idx]} -> {centroid[idx]:.3f}")
