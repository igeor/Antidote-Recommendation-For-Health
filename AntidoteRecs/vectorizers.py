import re
import contractions
import torch 
import logging
import numpy as np 
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

class TfIdfModel:
    
    def __init__(self, corpus, stop_words='english', max_features=100, max_df=0.8, min_df=4):
        """
        Initializes a TfIdfModel object.

        Parameters:
        - corpus: A list of text documents used to train the TF-IDF model.
        - stop_words: The language-specific stop words to be ignored during TF-IDF computation. Defaults to 'english'.
        - max_features: The maximum number of features (terms) to be included in the TF-IDF matrix. Defaults to 100.
        - max_df: The maximum document frequency threshold for a term to be considered in TF-IDF computation. Defaults to 0.8.
        - min_df: The minimum document frequency threshold for a term to be considered in TF-IDF computation. Defaults to 4.
        """
         
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.stop_words = stop_words
        self.corpus = [ self.__preprocess(str) for str in corpus ]
        self.model = TfidfVectorizer(stop_words=self.stop_words, 
                                    max_df=self.max_df, min_df=self.min_df, max_features=self.max_features).fit(self.corpus)
    
    def transform(self, text):
        """
        Transforms the input text into a TF-IDF embedding.

        Parameters:
        - text: The input text to be transformed.

        Returns:
        - embedding: The TF-IDF embedding of the input text as a numpy array.
        """
        text = self.__preprocess(text)
        embedding = self.model.transform([text])
        embedding = embedding.toarray().squeeze()
        return embedding 
    
    def feature_names(self):
        """
        Returns the list of feature names (terms) in the TF-IDF model.

        Returns:
        - feature_names: A list of feature names (terms) in the TF-IDF model.
        """
        return list(self.model.get_feature_names_out())
    
    def vocab(self):
        """
        Returns the vocabulary (terms and their indices) of the TF-IDF model.

        Returns:
        - vocab: A dictionary containing the terms and their corresponding indices in the TF-IDF model.
        """
        return self.model.vocabulary_

    def tokenizer(self, text):
        """
        Tokenizes the input text into a list of English words.

        Parameters:
        - text: The input text to be tokenized.

        Returns:
        - tokens: A list of English words extracted from the input text.
        """
         
        tokens = re.findall(r'\b\w+\b', text.lower())
        english_words = []
        for token in tokens:
            if all(ord(c) < 128 for c in token):
                english_words.append(token)
        return english_words

    def __preprocess(self, text):
        """
        Preprocesses the input text by removing punctuation and special characters.

        Parameters:
        - text: The input text to be preprocessed.

        Returns:
        - processed_text: The preprocessed text with punctuation and special characters removed.
        """

        """Remove punctuation and special characters from a text string."""
        text = text.lower()
        # Remove punctuation marks
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove special characters
        text = re.sub(r'(http|ftp|https)://[a-zA-Z0-9\\./]+', ' ', text)  # Remove URLs
        text = re.sub(r'@\w+', ' ', text)  # Remove mentions
        text = re.sub(r'#[\w-]+', ' ', text)  # Remove hashtags
        text = re.sub(r'[\U0001f600-\U0001f650]', ' ', text)  # Remove emojis
        # Split words with special characters in the middle
        text = re.sub(r'([^\W_]+[^\s\W_])([\W_])([^\W_]+[^\s\W_])', r'\1 \3', text)
        text = re.sub(' +', ' ', text)
        return text
    
    def __str__(self) -> str: return 'tfidf_model'


class BertVectorizer:
    def __init__(self, model_name='dmis-lab/biobert-base-cased-v1.1', max_length=512, device='cuda'):
        """
        BERT-based vectorizer for transforming text into embeddings.

        Parameters:
        - model_name (str): The name or path of the pre-trained BERT model to use.
        - max_length (int): The maximum length of the input text.
        - device (str): The device to use for computation (e.g., 'cuda', 'cpu').
        """

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        # Load the BERT model and tokenizer
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    

    def transform(self, text, preproc=True):
        """
        Transforms text into BERT embeddings.

        Parameters:
        - text (str or List[str]): The input text or a list of texts to transform.
        - preproc (bool): Whether to preprocess the text before tokenization.

        Returns:
        - numpy.ndarray: The BERT embeddings of the input text.
        """ 

        # Preprocess the text
        if preproc: text = self.__preprocess_text(text)
        # Tokenize the text
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Get the BERT model outputs
        with torch.no_grad(): outputs = self.model(**inputs)
        # Get the last hidden state
        last_hidden_state = outputs['last_hidden_state']
        # Aggregate the last hidden state
        embedding = torch.mean(last_hidden_state, dim=1)
        return embedding.cpu().numpy().squeeze()
            
    def __preprocess_text(self, text):
        """
        Preprocesses the input text by removing URLs and special characters while applying contractins.

        Parameters:
        - text: The input text to be preprocessed.

        Returns:
        - processed_text: The preprocessed text.
        """

        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Convert contractions
        text = contractions.fix(text)
        # Lowercase text
        text = text.lower()
        # Remove special characters not numbers
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        return text

    def __str__(self) -> str: return self.model_name



class ConcatVectorizer:
    def __init__(self, vectorizers):
        """
        Vectorizer that concatenates embeddings from multiple vectorizers.

        Parameters:
        - vectorizers (List[object]): A list of vectorizer objects.

        Functionality:
        - Initializes the ConcatVectorizer with the provided vectorizers.
        """
        self.vectorizers = vectorizers
    
    def transform(self, text):
        """
        Transforms text into concatenated embeddings from multiple vectorizers.

        Parameters:
        - text (str): The input text to transform.

        Returns:
        - numpy.ndarray: The concatenated embeddings of the input text.
        """
        embeddings = []
        for vectorizer in self.vectorizers:
            embeddings += [vectorizer.transform(text) ]
        return np.concatenate(embeddings, axis=0)

    def __str__(self) -> str: 
        model_name = ""
        for vectorizer in self.vectorizers:
            model_name += str(vectorizer) + "_"
        return self.model_name

