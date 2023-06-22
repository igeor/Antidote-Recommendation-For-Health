import numpy as np 
from sklearn.cluster import KMeans

def extract_topics(corpus, n_topics, vectorizer=None):
    """
    Extracts topics from a given corpus using K-means clustering.

    Parameters:
    - corpus: The corpus of documents to extract topics from.
    - n_topics: The number of topics to extract.
    - vectorizer: (Optional) The vectorizer to transform the corpus into embeddings. Defaults to None.

    Returns:
    - kmeans: The trained K-means clustering model.
    """

    # Uncomment the following code to re-generate the kmeans instance
    kmeans = KMeans(n_clusters=n_topics, init='k-means++')
    if 'embedding' not in corpus.columns:
        corpus['embedding'] = [vectorizer.transform(claim) for claim in corpus['claim']]
    kmeans.fit(corpus['embedding'].to_list())
    corpus['topic'] = kmeans.labels_
    return kmeans

