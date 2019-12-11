from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np


def readData(arr, filename):
    """
    Reads document data into corpus.
    Data set has the documents separated by lines.
    """
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
    """
    Builds the term-document matrix out of the read-in data.
    """
    # Transforms the corpus document list into a tdm.
    tdm = vectorizer.fit_transform(corpus)

    words = vectorizer.get_feature_names()

    # Performs SVD on the tdm and transforms it into the vh component.
    # The u and sigma components are taken care of, abstracted within the svd class instance.
    tdm = svd.fit_transform(tdm)
    tdm = Normalizer().fit_transform(tdm)

    return (tdm, words)


def buildQuery(vectorizer, svd, tdm, words):
    """
    Builds the transformed query vector from command line input.
    """
    q = input("Enter your query: ")
    q = q.split(' ')

    # q's vectorization will only contain words inside q.
    # 
    # To fix this, replace everything into a matrix (resized) that has
    # every column from the original TDM, so it's the same dimension. 
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

    # Query vector transformation into the approximated space
    q = svd.transform(q)

    return q


def rankDocuments(q, tdm):
    """
    Prints the top 10 documents most similar to query.
    """
    ranks = []
    for i in range(len(tdm)):
        # Each ith document is contained in vh[i].
        # Dot them to compute the similarity.
        val = np.dot(q, tdm[i])[0]
        ranks.append((val, i))

    # Rank the documents by similarity.
    ranks.sort(reverse=True)

    print()
    print("Most relevant documents")
    print("------------------------")
    for i in range(min(len(ranks), 10)):
        print(corpus[ranks[i][1]].rstrip())



rank = 300 # Desired rank of approximation
corpus = [] # List of documents
readData(corpus, "dataset.csv")

vectorizer = TfidfVectorizer("english")
svd = TruncatedSVD(n_components=rank)

tdm, words = buildTDM(vectorizer, svd, corpus)

while True:
    q = buildQuery(vectorizer, svd, tdm, words)
    rankDocuments(q, tdm)