import requests 
import pandas as pd 
import json 
import requests
from tqdm import tqdm 

def query(credentials, query_params):
    """
    Queries the Google Fact Check API with the provided credentials and query parameters.

    Parameters:
    - credentials (dict): A dictionary containing the API credentials, including the 'VERSION'.
    - query_params (dict): A dictionary containing the query parameters for the API request.

    Returns:
    - list or dict: The response from the API as a list of claims, or an empty dictionary if no claims are found.
    """

    base_url = f"https://factchecktools.googleapis.com/{credentials['VERSION']}"
    response = requests.get(f"{base_url}/claims:search", params=query_params).json()
    try:
        return response['claims']
    except:
        return {}


def get_claims_from_gfc(credentials, keywords, languageCode='en-US', pageSize=10000, maxAgeDays=365*3):
    """
    Retrieves claims from the Google Fact Check API based on provided keywords.

    Parameters:
    - credentials (dict): A dictionary containing the API credentials, including the 'API_KEY'.
    - keywords (list): A list of keywords corresponding to the claim topic.
    - languageCode (str, optional): The language code for the claims. Default is 'en-US'.
    - pageSize (int, optional): The maximum number of claims to retrieve. Default is 10,000.
    - maxAgeDays (int, optional): The maximum age (in days) of the claims. Default is 3 years (365*3).

    Returns:
    - list: A list of claims retrieved from the Google Fact Check API.
    """ 

    claims_to_return = {}

    for keyword in keywords:
        
        params = {
            "query": keyword,
            "languageCode": languageCode,
            "pageSize": pageSize,
            "maxAgeDays": maxAgeDays,
            "key": credentials['API_KEY'],
        }

        response = query(credentials, params)

        ## Define text of claim as key 
        ## There is a need to remove duplicate claims 
        ## Text (str) is hashable
        for claim in response: 
            claim_text = claim['text']

            ## Add keywords attribute to all claims retrieved
            if claim_text in claims_to_return.keys():
                claims_to_return[claim_text]['keywords'] += [keyword]
            else:
                claims_to_return[claim_text] = claim
                claims_to_return[claim_text]['keywords'] = [keyword]

            ## Prevent the keyError for usefull attributes 
            if 'publisher' not in claims_to_return[claim_text]['claimReview'][0].keys():
                claims_to_return[claim_text]['claimReview'][0]['publisher'] = {'name': 'Unknown publisher'}
            if 'title' not in claims_to_return[claim_text]['claimReview'][0].keys():
                claims_to_return[claim_text]['claimReview'][0]['title'] = None
            if 'name' not in claims_to_return[claim_text]['claimReview'][0]['publisher'].keys():
                claims_to_return[claim_text]['claimReview'][0]['publisher']['name'] = 'Unknown publisher'
            if 'claimant' not in claims_to_return[claim_text].keys():
                claim['claimant'] = 'Unknown claimant'

    claims_to_return = [claim for key, claim in claims_to_return.items()]
    return claims_to_return

def load_data(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pandas.DataFrame: The loaded data.
    """
    
    corpus = pd.read_csv(file_path, index_col=0)
    return corpus 

def get_claims(credentials, keywords, languageCode='en-US', pageSize=20000, maxAgeDays=365*3, progress_bar=True):
    """
    Retrieves claims from the Google Fact Check API based on provided keywords.

    Parameters:
    - credentials (str): The path to the credentials file for accessing the API.
    - keywords (list): A list of keywords corresponding to the claim topic.
    - languageCode (str, optional): The language code for the claims. Default is 'en-US'.
    - pageSize (int, optional): The maximum number of claims to retrieve. Default is 20000.
    - maxAgeDays (int, optional): The maximum age (in days) of the claims. Default is 3 years (365*3).
    - progress_bar (bool, optional): Whether to display a progress bar. Default is True.

    Returns:
    - pandas.DataFrame: A dataframe containing the retrieved claims with columns=['claim', 'claimant', 'review', 'reviewer', 'url', 'rating'].
    """
     
    claims_to_dataframe = []

    claims = get_claims_from_gfc(credentials, keywords, languageCode=languageCode, pageSize=pageSize, maxAgeDays=maxAgeDays)
    
    for claim in tqdm(claims):    

        # Get the claim instance
        claim_instance = claim['text'] 

        # Get the claimant
        claimant = claim['claimant'] 

        # Get the reviewer 
        reviewer = claim['claimReview'][0]['publisher']['name']

        # Get the claim validity rating
        review_rating = claim['claimReview'][0]['textualRating']
        # split the claim instance into words
        complement_claim = None 
        if len(review_rating.split()) > 5:
            complement_claim = review_rating
        
        if not complement_claim:
            # Get the complement claim instance
            url = claim['claimReview'][0]['url']
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            complement_claim = soup.find("h1")
            if not complement_claim:
                complement_claim = soup.find("title")
            if complement_claim:
                complement_claim = complement_claim.text.strip()
        
        ## Append dictionary to Dataframe
        claims_to_dataframe += [{ 
            'claim': claim_instance,
            'claimant': claimant,
            'review': complement_claim,
            'reviewer': reviewer,
            'url': claim['claimReview'][0]['url'],
            'rating': review_rating,
        }]

    df = pd.DataFrame.from_records(claims_to_dataframe)
    return df
