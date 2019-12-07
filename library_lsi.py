from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np



def readData(arr, filename):
    with open(filename) as file:
        for line in file:
            arr.append(line)


def buildTDM(vectorizer, svd, corpus):
    tdm = vectorizer.fit_transform(corpus)

    words = vectorizer.get_feature_names()

    tdm = svd.fit_transform(tdm)
    tdm = Normalizer().fit_transform(tdm)

    return (tdm, words)


def buildQuery(vectorizer, svd, tdm, words):
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
    for i in range(len(ranks)):
        print(corpus[ranks[i][1]].rstrip())



corpus = []
readData(corpus, "docs.txt")

vectorizer = TfidfVectorizer(stop_words="english")
svd = TruncatedSVD()

tdm, words = buildTDM(vectorizer, svd, corpus)
q = buildQuery(vectorizer, svd, tdm, words)

rankDocuments(q, tdm)