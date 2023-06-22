# Antidote Recommendation Tool

This repository contains code for an Antidote Recommendation Tool that utilizes vector-based and topic modeling approaches to recommend antidotes for fake claims. In addition the tool leverages the bipartite graph representation and topic modeling techniques to identify relevant antidotes from a corpus of claims.

## Contents

- `AntidoteRecs/gfc.py`: Contains functions for retrieving claims from the Google Fact Check API.
- `AntidoteRecs/vectorizers.py`: Contains the `Vectorizer` class for transforming claims into embeddings using pre-trained models.
- `AntidoteRecs/antidote.py`: Contains functions for retrieving antidotes based on vector-based and topic modeling approaches.
- `AntidoteRecs/claim_topic.py`: Contains functions for extracting topics from a corpus of claims.
- `AntidoteRecs/graphs.py`: Contains functions for building bipartite graphs based on user-submission interactions.
- `AntidoteRecs/preemptive.py`: Contains functions for predicting the next topic of interest for a given user.

## Usage

To use the Antidote Recommendation Tool, follow these steps:

### Vector-Based Approach

1. Import the necessary functions and classes from the respective files.
2. Retrieve claims from the Google Fact Check API (optional) using the `get_claims` function from `AntidoteRecs.gfc` or load the instance of claims using the `load_data` function from `AntidoteRecs.gfc` or any other suitable method.
3. Initialize a vectorizer using the `Vectorizer` class from `AntidoteRecs.vectorizers`. Choose a pre-trained model appropriate for your task.
4. Utilize the `get_antidotes` function from `AntidoteRecs.antidote` to retrieve antidotes based on the vector-based approach. Pass the fake claim, corpus, vectorizer, and the desired number of top antidotes to retrieve.


### Topic Modeling Approach

1. Import the necessary functions and classes from the respective files.
2. Load the corpus of claims using the `load_data` function from `AntidoteRecs.gfc` or any other suitable method.
3. Initialize a vectorizer using the `Vectorizer` class from `AntidoteRecs.vectorizers`. Choose a pre-trained model appropriate for your task.
4. Perform topic modeling on the corpus using the `extract_topics` function from `AntidoteRecs.claim_topic`. Specify the number of topics desired and provide the corpus and vectorizer.
5. Utilize the `get_antidotes` function from `AntidoteRecs.antidote` to retrieve antidotes based on the topic modeling approach. Pass the fake claim, corpus, vectorizer, number of top antidotes, and the topic of the fake claim obtained from topic modeling.


### Preemptive Recommendations

1. Import the necessary functions and classes from the respective files.
2. Build a bipartite graph representation of user-submission interactions using the `extract_graph` function from `AntidoteRecs.graphs`.
3. Create a user-to-topic bipartite graph using the `user_to_topic_graph` function from `AntidoteRecs.graphs`, providing the user-submission graph, corpus, topic modeling results, and vectorizer.
4. Predict the next topic of interest for a given user using the `predict_next_topic` function from `AntidoteRecs.preemptive`.
5. Retrieve the claims and their corresponding reviews that belong to the predicted next topic using the topic information.