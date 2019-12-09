from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np


def readData(arr, filename):
    docs = 0
    with open(filename, encoding="latin-1") as file:
        next(file)
        for line in file:
            line = line.split(',')
            line = line[-1]
            arr.append(line)
            docs += 1
            if docs > 200000:
                print("building reduced matrix...")
                break


def buildTDM(vectorizer, svd, corpus):
    tdm = vectorizer.fit_transform(corpus)

    words = vectorizer.get_feature_names()

    tdm = svd.fit_transform(tdm)
    tdm = Normalizer().fit_transform(tdm)

    return (tdm, words)


def buildQuery(vectorizer, svd, tdm, words):
    q = input("Enter your query: ")
    q = q.split(' ')

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
    q = svd.transform(q)

    return q


def rankDocuments(q, tdm):
    ranks = []
    for i in range(len(tdm)):
        val = np.dot(q, tdm[i])[0]
        ranks.append((val, i))

    ranks.sort(reverse=True)

    print()
    print("Most relevant documents")
    print("------------------------")
    for i in range(min(len(ranks), 10)):
        print(corpus[ranks[i][1]].rstrip())



rank = 300
corpus = []
readData(corpus, "dataset.csv")

vectorizer = TfidfVectorizer("english")
svd = TruncatedSVD(n_components=rank)

tdm, words = buildTDM(vectorizer, svd, corpus)

while True:
    q = buildQuery(vectorizer, svd, tdm, words)
    rankDocuments(q, tdm)