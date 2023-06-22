import numpy as np

def str_to_list(string):
    """ 
    Converting a string of numbers to a list of numbers.
    Parameters
    * string: string of numbers, e.g. '[1, 2, 3, 4]'
    Returns
    * list_to_return: list of numbers, e.g. [1, 2, 3, 4] | []
    """
    if string.strip('[]'): return [int(num) for num in string.strip('[]').split(', ')]
    else: return [] 


def topics_radius(kmeans, corpus):
    # Compute the maximum distance of each centroid from its claims
    centroids_radius = []
    for topic in range(kmeans.n_clusters):
        # Find the claims that belong to the current topic
        topic_claims = corpus[corpus['topic'] == topic]
        # Find the centroid of the current topic
        topic_centroid = kmeans.cluster_centers_[topic]
        # Calculate the distance between the centroid and the claims
        distances = [ np.linalg.norm(claim_embedding - topic_centroid) for claim_embedding in topic_claims['embedding'] ]
        # Find the maximum distance
        centroids_radius += [ np.max(distances) ]
        
    return centroids_radius