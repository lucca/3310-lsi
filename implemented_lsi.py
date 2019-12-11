from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import string


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
            if docs > 1000:
                print("building reduced matrix...")
                break
 

def normalize(arr):
    """
    Normalizes vectors by finding the 2 norm, then 
    dividing every element by it.
    """
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
    """
    Builds the term-document matrix out of the read-in data.
    """
    # Transforms the corpus document list into a tdm.
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
    q = np.asarray(q)

    # Query vector transformation into the approximated space
    # q = sigma^-1 * UT * qT
    result = np.matmul(s, u.transpose())
    result = np.matmul(result, q.transpose())

    result = normalize(result)

    return result.transpose()


def rankDocuments(q, vh):
    """
    Prints the top 10 documents most similar to query.
    """
    ranks = []
    for i in range(len(vh)):
        # Each ith document is contained in vh[i].
        # Dot them to compute the similarity.
        val = np.dot(q, vh[i])[0]
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

vectorizer = TfidfVectorizer(stop_words="english")
u, s, vh, words = buildTDM(corpus, vectorizer, rank)

while True:
    q = buildQuery(vectorizer, words, u, s)
    rankDocuments(q, vh.transpose())