from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import string


def readData(arr, filename):
    docs = 0
    with open(filename, encoding="latin-1") as file:
        next(file)
        for line in file:
            line = line.split(',')
            line = line[-1]
            arr.append(line)
            docs += 1
            if docs > 1000:
                print("building reduced matrix...")
                break


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

    # Decompose term-document matrix with SVD
    u, s, vh = np.linalg.svd(tdm, full_matrices=False)

    # Truncate columns of u to desired rank = k
    u = np.delete(u, np.s_[rank:], 1)

    # Truncate rows/columns of s, then take the inverse
    s = np.diag(s)
    s = np.delete(s, np.s_[rank:], 1)
    s = np.delete(s, np.s_[rank:], 0)
    s = np.linalg.inv(s)

    # Truncate rows of vh
    vh = np.delete(vh, np.s_[rank:], 0)
    vh = vh.transpose()
    vh = normalize(vh)
    vh = vh.transpose()

    return (u, s, vh, words)


def buildQuery(vectorizer, words, u, s):
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
    q = np.asarray(q)

    # q = sigma^-1 * UT * q
    # Transpose q and result to get back into correct shape
    result = np.matmul(s, u.transpose())
    result = np.matmul(result, q.transpose())

    result = normalize(result)

    return result.transpose()


def rankDocuments(q, vh):
    ranks = []
    for i in range(len(vh)):
        val = np.dot(q, vh[i])[0]
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

vectorizer = TfidfVectorizer(stop_words="english")
u, s, vh, words = buildTDM(corpus, vectorizer, rank)

while True:
    q = buildQuery(vectorizer, words, u, s)
    rankDocuments(q, vh.transpose())