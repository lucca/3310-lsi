# For runtime testing: Have a while loop until desired data size is reached,
# adding "a", "b", "c" etc, then when "z" is reached go to "aa" for unique words
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import string


def readData(arr, filename):
    with open(filename) as file:
        for line in file:
            arr.append(line)


def normalize(arr):
    for row in arr:
        vlen = 0
        for i in range(len(row)):
            vlen += row[i] ** 2
        vlen = np.sqrt(vlen)

        if(vlen):
            for i in range(len(row)):
                row[i] = row[i] / vlen

    return arr


def buildTDM(corpus, vectorizer, rank):
    tdm = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    tdm = np.asarray(tdm.toarray(), dtype="float")
    tdm = tdm.transpose()

    u, s, vh = np.linalg.svd(tdm, full_matrices=False)

    u = np.delete(u, np.s_[rank:], 1)

    s = np.diag(s)
    s = np.delete(s, np.s_[rank:], 1)
    s = np.delete(s, np.s_[rank:], 0)
    s = np.linalg.inv(s)

    vh = np.delete(vh, np.s_[rank:], 0)
    vh = vh.transpose()
    vh = normalize(vh)
    vh = vh.transpose()

    return (u, s, vh, words)


def buildQuery(vectorizer, words, u, s):
    q = []
    readData(q, "query.txt")

    print("Query:", q[0].rstrip())

    q = vectorizer.fit_transform(q)
    q = pd.DataFrame(q.toarray())
    q.columns = vectorizer.get_feature_names()

    resized = pd.DataFrame([0 for i in range(len(words))])
    resized = resized.transpose()
    resized.columns = words

    for column in resized:
        if column in q.columns:
            resized[column] = q[column]
    
    q = resized
    q = np.asarray(q)

    q = np.matmul(q, u)
    q = np.matmul(q, s)

    q = normalize(q)

    return q


def rankDocuments(q, vh):
    ranks = []
    for i in range(len(vh)):
        val = np.dot(q, vh[i])[0]
        ranks.append((val, i))

    ranks.sort(reverse=True)

    print()
    print("Most relevant documents")
    print("------------------------")
    for i in range(len(ranks)):
        print(corpus[ranks[i][1]].rstrip())



rank = 2
corpus = []
readData(corpus, "docs.txt")

vectorizer = TfidfVectorizer(stop_words="english")
u, s, vh, words = buildTDM(corpus, vectorizer, rank)
q = buildQuery(vectorizer, words, u, s)

rankDocuments(q, vh.transpose())