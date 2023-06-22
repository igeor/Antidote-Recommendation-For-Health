import os
import json
import re 
import networkx as nx 
from tqdm.notebook import tqdm
from .vectorizers import * 
from .utils import topics_radius

def extract_graph(PATH, T=5, only_title=True, _comments=True, _posts=True):
    """
    Extracts a bipartite graph from the given dataset and applies a threshold on node degrees.

    Parameters:
    - PATH (str): The path to the dataset.
    - T (int, optional): The threshold for node degrees. Default is 5.
    - only_title (bool, optional): Whether to consider only the title when extracting authors and items. Default is True.
    - _comments (bool, optional): Whether to include comments in the graph. Default is True.
    - _posts (bool, optional): Whether to include posts in the graph. Default is True.

    Returns:
    - networkx.Graph: The extracted bipartite graph.
    """

    # Get the authors dictionary
    authors = extract_authors(PATH, only_title=only_title, _posts=_posts, _comments=_comments)

    # Initialize the graph
    G_U2I = nx.Graph()

    # Author nodes
    author_nodes = authors.keys()

    # Items nodes
    item_nodes = [(_id, text) for items in authors.values() for _id, text in items]
    
    # Add author nodes to one set
    G_U2I.add_nodes_from(author_nodes, bipartite=0)
    # Add item nodes to other set
    for subm_id, subm_text in item_nodes:
        G_U2I.add_node(subm_id, bipartite=1, text=subm_text)

    # Add edges between authors and items
    for author, items in authors.items():
        for subm_id, subm_text in items: G_U2I.add_edge(author, subm_id)

    # Apply threshold only if _comments is True
    if _comments:
        nodes_to_remove = [node for node, _ in G_U2I.nodes(data=True) if G_U2I.degree(node) <= T]
        while nodes_to_remove:
            G_U2I.remove_nodes_from(nodes_to_remove)
            nodes_to_remove = [node for node, attrs in G_U2I.nodes(data=True) if G_U2I.degree(node) <= T]
        # Get the largest connected component subgraph
        if len(list(nx.connected_components(G_U2I))) > 1:
            G_U2I = G_U2I.subgraph( max(nx.connected_components(G_U2I), key=len) )

    return G_U2I



def user_to_topic_graph(G, corpus, kmeans, vectorizer):
    """
    Builds a bipartite graph representation of users and topics based on their interactions with submissions.

    Parameters:
    - G: The user to submission graph.
    - corpus: The corpus of claims used in the model.
    - kmeans: The KMeans model trained on the corpus claims.
    - vectorizer: The vectorizer used to transform claims into embeddings.

    Returns:
    - G_U2T: The bipartite graph representing users and topics.
    """

    # Add the topic attribute to the corpus
    corpus['topic'] = kmeans.labels_

    centroids_radius = topics_radius(kmeans, corpus)
    mean_radius = np.mean(centroids_radius)

    # Iterate over the submissions
    submissions_topics = []

    # Iterate over the submissions of G (bipartite=1)
    for node_idx, node_attrs in tqdm(G.nodes(data=True), desc='Generating embeddings for submissions...'):
        # Skip the author nodes
        if node_attrs['bipartite'] != 1: continue

        # Get the submission's embedding 
        submission_embedding = vectorizer.transform(node_attrs['text'])
        # Find the closest topic from the submission
        pred_topic = kmeans.predict([submission_embedding])[0]
        # Find the centroid of the topic
        topic_centroid = kmeans.cluster_centers_[pred_topic]
        # Calculate the distance between the submission and the centroid
        distance = np.linalg.norm(submission_embedding - topic_centroid)
        # If the distance is less than the cluster radius then assign the submission to the topic
        if distance < centroids_radius[pred_topic] and distance < mean_radius:
            submissions_topics += [pred_topic]
            G.nodes[node_idx]['topic'] = pred_topic
        # otherwise assign the submission to the topic -1 (outlier)
        else:
            submissions_topics += [-1]
            G.nodes[node_idx]['topic'] = -1

    # create a new bipartite graph with the cluster centers as items
    G_U2T = nx.Graph()

    # add the authors as nodes
    authors = [ node for node, attrs in G.nodes(data=True) if attrs['bipartite'] == 0 ]
    G_U2T.add_nodes_from(authors, bipartite=0)
    # add the topics (inc. no-topic) as nodes
    G_U2T.add_nodes_from(range(len(kmeans.cluster_centers_)), bipartite=1)

    # Add the edges between authors and topics
    # Iterate over the authors
    for author in authors:
        # Iterate over the submissions that the author interacted
        for submission_id in G.neighbors(author):
            submission_topic = G.nodes[submission_id]['topic']
            # Prevent adding edges for submissions that have not been assigned to a topic
            if submission_topic != -1:
                # check if the edge already exists
                if not G_U2T.has_edge(author, submission_topic):
                    G_U2T.add_edge(author, submission_topic, n_interactions=1)
                # if the edge already exists, increase the number of interactions
                else:
                    G_U2T[author][submission_topic]['n_interactions'] += 1

    return G_U2T




def extract_authors(PATH, only_title=True, _posts=True, _comments=True):
    """
    Create Author -> [Items] Dictionary
    """
    submission_filenames = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]

    authors = dict()
    for filename in submission_filenames:
        try:
            with open(f"{PATH}/{filename}", 'r') as file:
                submission = json.load(file)

            # Construct the submission textual content
            try: 
                submission_title = submission['title']
                submission_text = ""
                if not only_title:
                    submission_text = submission['selftext']
                submission_textual_content = submission_title + " " + submission_text
                # Remove urls from submission_textual_content
                submission_textual_content = re.sub(r'http\S+', '', submission_textual_content)
                if not submission_textual_content \
                    or '[deleted' in submission_textual_content \
                    or '[removed'in submission_textual_content : continue 
                submission_id = submission['id']
            except: continue  # Skip submission  

            # Get submission author (Prevent KeyError)
            try: subm_author = submission['author']['name']
            except: pass # Skip adding author in authors

            # Prevent None 
            if subm_author:
                if _posts:
                    if subm_author in authors.keys():
                        authors[subm_author].add( (submission_id, submission_textual_content) )
                    else:
                        authors[subm_author] = { (submission_id, submission_textual_content) }
            else: pass 
            
            # Get submission comments (PRevent KeyError)
            try: comments = submission['_comments']
            except: continue

            if _comments:
                for comment in comments:
                    
                    # Get comment author (Prevent KeyError)
                    try: comment_author = comment['author']['name']
                    except: continue 

                    # Prevent None
                    if comment_author:
                        # Prevent duplicating author. 
                        # comment_author could be the submission author
                        if comment_author in authors.keys():
                            authors[comment_author].add( (submission_id, submission_textual_content) )
                        if comment_author not in authors.keys():
                            authors[comment_author] = { (submission_id, submission_textual_content) }

        except: continue

    return authors
