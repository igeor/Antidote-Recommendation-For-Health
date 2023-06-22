import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD

def predict_next_topic(G, user_id, method='UB', topK=10):
    """
    Predicts the next topic of interest for a given user based on their interaction history.

    Parameters:
    - G (networkx.Graph): The graph representing user-topic interactions. 
    - user_id (int): The ID of the user for whom to predict the next topic.
    - method (str, optional): The method to use for prediction. Can be 'UB' (user-based) or 'SVD' (Singular Value Decomposition). Default is 'UB'.
    - topK (int, optional): The number of top similar users to consider. Default is 10.

    Returns:
    - int: The predicted topic ID for the next topic of interest.

    Functionality:
    - Retrieves the authors and items from the graph.
    - Creates a binary interaction matrix, M, based on the graph.
    - Computes cosine similarity between users using M.
    - Identifies the top similar users to the given user.
    - Finds the topics the user has not interacted with.
    - Calculates a prediction value for each topic based on similar users' interactions.
    - Returns the topic ID with the highest prediction value.
    """
    
    authors = { node: node_idx for node_idx, (node, attrs) in enumerate(G.nodes(data=True)) if attrs['bipartite'] == 0 }
    items = [ node for node, attrs in G.nodes(data=True) if attrs['bipartite'] == 1 ]

    M = np.zeros((len(authors), len(items)))
    for author_idx, topic_idx, attr in G.edges(data=True):
        M[authors[author_idx], topic_idx] = 1 # attr['n_interactions']
    
    similarities = cosine_similarity(M)
    top_similar_users = similarities[user_id].argsort()[-topK:][::-1]
    
    user_topics = M[user_id]
    user_topics = np.where(user_topics == 0)[0]
    
    if method == 'UB':
        recommendations = []
        for topic in user_topics:
            pred_value = 0
            for user in top_similar_users:
                if M[user, topic] == 1:
                    pred_value += similarities[user_id, user]
            recommendations += [(topic, pred_value)]

        # Sort recommendations by predicted value
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        next_topic = recommendations[0][0]
        return next_topic
    else:
        # SVD implementation
        interaction_data = []
        for author_idx, topic_idx, attr in G.edges(data=True):
            interaction_data.append((authors[author_idx], topic_idx, 1))  # attr['n_interactions']

        df = pd.DataFrame(interaction_data, columns=['author', 'topic', 'interaction'])
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(df[['author', 'topic', 'interaction']], reader)
        trainset = data.build_full_trainset()

        model = SVD()
        model.fit(trainset)

        predicted_ratings = []
        for topic in user_topics:
            predicted_rating = model.predict(user_id, topic).est
            predicted_ratings.append((topic, predicted_rating))

        predicted_ratings = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)
        next_topic = predicted_ratings[0][0]
        return next_topic


