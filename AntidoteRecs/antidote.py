import warnings
warnings.filterwarnings('ignore')
from .similarities import cosine_similarity

def compute_relevance(input_embedding, corpus, topK=None):
    """
    Computes the relevance of claims in the corpus to the input embedding.

    Parameters:
    - input_embedding: The embedding of the input claim or text.
    - corpus (DataFrame): A DataFrame containing the claims corpus, including the 'embedding' column.
    - topK (int or None): The number of top claims to return. If None, returns all claims.

    Returns:
    - DataFrame: A DataFrame of claims from the corpus ranked by their similarity to the input embedding.
    """ 

    # rank the claims by their similarity to the input embedding
    corpus['similarity'] = corpus['embedding'].apply(lambda x: cosine_similarity(input_embedding, x))
    # sort the claims by their similarity to the input embedding and return the top 10
    corpus = corpus.sort_values(by='similarity', ascending=False)
    if topK: corpus = corpus.head(topK)
    return corpus  

def get_antidotes(fake_claim, corpus, vectorizer, topK=None, topic=None):
    """
    Retrieves antidotes for a given fake claim by computing relevance to claims in a corpus.

    Parameters:
    - fake_claim: The fake claim to find antidotes for.
    - corpus: The corpus of claims to search for antidotes.
    - vectorizer: The vectorizer used to transform claims into embeddings.
    - topK: (Optional) The maximum number of top-ranked antidotes to return. Defaults to None.
    - topic: (Optional) The topic label to filter the corpus claims by. Defaults to None.

    Returns:
    - ranking: A DataFrame containing the relevant antidotes with columns 'claim', 'review', and 'rating'.
    """
    fake_claim_embedding = vectorizer.transform(fake_claim)
    if 'embedding' not in corpus.columns:
        corpus['embedding'] = [ vectorizer.transform(claim) for claim in corpus['claim'] ]
    if 'topic' in corpus.columns:
        corpus = corpus[corpus['topic'] == topic]
    # Compute the relevance of the submission to the claims of the corpus
    ranking = compute_relevance(fake_claim_embedding, corpus, topK=topK)
    # Keep only the relevant columns
    ranking = ranking[ ['claim', 'review', 'rating'] ]
    return ranking