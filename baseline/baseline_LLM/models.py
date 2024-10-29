import os
import re
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util
import re
from torch.cuda import temperature

# Retrieve the API key from an environment variable
api_key = os.getenv("HUGGINGFACE_API_KEY")
if api_key is None:
    raise ValueError("API key for Hugging Face is not set. Please set 'HUGGINGFACE_API_KEY' in your environment.")

# Initialize the API client with the retrieved API key
client = InferenceClient(api_key=api_key)

# Load the embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def get_movie_embeddings(movies_df):
    """Compute embeddings for movie titles."""
    titles = movies_df['movie_title'].tolist()
    embeddings = embedding_model.encode(titles, convert_to_tensor=True)
    return titles, embeddings


def parse_llm_output(user_id, prompt, llm_output):
    """
    Parse llm outputs. Expecting input prompt as ['movie_1', 'movie_2', 'movie3'].
    When llm incorrect

    Parameters:
    - user_id (int): user's id.
    - prompt (str): prompt for the given user.
    - llm_output (str): response given by the llm model.

    return a list of string, of movie titles.
    """
    #parsed_titles = []
    #unexpected_outputs = []

    '''
    # Convert llm_output from a string to a list using eval (or better, ast.literal_eval for safety)
    import ast
    try:
        llm_output_list = ast.literal_eval(llm_output)
    except (ValueError, SyntaxError):
        print(f"Error parsing llm_output: {llm_output}")
        return []

    for title in llm_output_list:
        if isinstance(title, str):
            match = re.match(r"(.+?)\s*\((\d{4})\)", title)

            if match:
                parsed_titles.append(match.group(1).strip())
            else:
                parsed_titles.append(title.strip())
        else:
            unexpected_outputs.append({'user_id': user_id, 'prompt': prompt, 'output': title})

    if unexpected_outputs:
        df = pd.DataFrame(unexpected_outputs)
        csv_path = f'unexpected_outputs_user_{user_id}.csv'
        df.to_csv(csv_path, index=False)
        return csv_path
    '''
    cleaned = llm_output.strip("[]")
    movie_list = [item.strip(" '\"") for item in cleaned.split(",")]

    return movie_list
    #return parsed_titles


def map_movies_to_dataset(y_hat, movie_idx_ls, movie_embeddings, pooling='mean'):
    """
    Map the LLM's output movie titles to the dataset's movie titles using cosine similarity.

    Parameters:
    - y_hat (list of str): Movie titles generated by the LLM.
    - movie_idx_ls (list of int): movie_ids in the valid recommendation space.
    - movie_embeddings (Tensor): Corresponding embeddings for the dataset movie titles.
    - pooling (str): Pooling method to apply ('mean' or 'max').

    Returns:
    - list of str: Mapped movie titles from the dataset.
    """
    # Get embeddings for LLM output titles
    y_hat_embeddings = embedding_model.encode(y_hat, convert_to_tensor=True)

    # Calculate cosine similarity
    if pooling == 'mean':
        # Mean pooling
        y_hat_embeddings = y_hat_embeddings.mean(dim=0, keepdim=True)
    elif pooling == 'max':
        # Max pooling
        y_hat_embeddings, _ = y_hat_embeddings.max(dim=0, keepdim=True)

    # Calculate cosine similarities
    cosine_scores = util.pytorch_cos_sim(y_hat_embeddings.to('cpu'), movie_embeddings[movie_idx_ls])

    # Get the top matches
    top_results = np.argpartition(-cosine_scores.cpu().numpy(), range(5))[:, :5]
    mapped_movies = [movie_idx_ls[idx] for idx in top_results.flatten()]

    return mapped_movies

def model_prediction(prompts_df, watched_dict, all_movie_indices, all_movie_embeddings, model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", max_tokens=500, top_k=5,  pooling='mean'):
    """
    Recommend top k movies for the given input user prompts.

    Parameters:
    - prompts_df (pandas dataframe): The input prompt generated for the LLM.
    - watched_dict (pandas dataframe): The movie list that user has watched before the given timestamp.
    - all_movie_indices (list of int): all movies' id.
    - movie_embeddings (Tensor, numpy array): Corresponding embeddings for the dataset movie titles.
    - model_name (str): The name of the model to use for recommendations.
    - max_tokens (int): Maximum tokens to allow in the output.
    - top_k (int): Number of movie recommendations to retrieve.
    - pooling (str): Pooling method for embedding comparison.

    Returns:
    - pandas dataframe: user_id, top_k_movies.
    """

    predictions = dict()
    num_row = 0

    for row in prompts_df.itertuples(index=False):
        num_row += 1
        if num_row % 50 == 0:
            print("num_row is", num_row)
        # get valid recommendation space for the current user
        cur_embedding_idx = [i for i, movie in enumerate(all_movie_indices) if movie not in watched_dict[row.user_id]]
        cur_predict = get_llm_recommendations(row.user_id, row.Prompt, all_movie_embeddings, cur_embedding_idx, top_k=top_k, pooling=pooling, max_tokens=max_tokens, model_name=model_name)
        predictions[row.user_id] = cur_predict

    return predictions



def get_llm_recommendations(user_id, prompt, movie_embeddings, embedding_idx, model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", max_tokens=500, top_k=5,  pooling='mean'):
    """
    Sends the prompt to the LLM and retrieves the recommended movie list.

    Parameters:
    - user_id (int): user's id.
    - prompt (str): The input prompt generated for the LLM.
    - movie_ids (list of in): movie_ids in the valid recommendation space.
    - movie_embeddings (Tensor): Corresponding embeddings for the dataset movie titles.
    - movies_df (DataFrame): DataFrame containing movie information.
    - model_name (str): The name of the model to use for recommendations.
    - max_tokens (int): Maximum tokens to allow in the output.
    - top_k (int): Number of movie recommendations to retrieve.
    - pooling (str): Pooling method for embedding comparison.

    Returns:
    - list of str: A list of recommended movie titles parsed from the LLM's output.
    """

    try:
        # Send the prompt to the LLM
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "movie recommender", "content": prompt}],
            temperature=0.5,
            max_tokens=max_tokens,
            stream=False
        )

        # Extract the response text
        llm_output = response['choices'][0]['message']['content'].strip()
        #print("llm output:", llm_output)
        # Use regex to extract the list of movie titles from the output
        llm_rec_movies_ls = parse_llm_output(user_id, prompt, llm_output)
        #print("Parsed output:", llm_rec_movies_ls)
        # map predicted movie with our item space.
        mapped_movies = map_movies_to_dataset(llm_rec_movies_ls, embedding_idx, movie_embeddings, pooling)
        #print('Mapped movies: ', mapped_movies)
        return mapped_movies
        #
        # movie_titles, movie_embeddings = get_movie_embeddings(movies_df)
        #
        # movie_list_match = re.search(r"\[.*?\]", llm_output, re.DOTALL)
        # if movie_list_match:
        #     # Evaluate the extracted string to a Python list
        #     recommendations = eval(movie_list_match.group())
        #     if isinstance(recommendations, list) and all(isinstance(item, str) for item in recommendations):
        #
        #         return mapped_movies
        #     else:
        #         # return []
        #         raise ValueError("LLM response format is incorrect")
        # else:
        #     print("No valid list of movie titles found in the output.")
        #     return []

    except Exception as e:
        print(f"Error during LLM API request or response parsing: {e}")
        return []

if __name__ == "__main__":
    from dataloader import DataLoader

    # Example usage with updated prompt
    prompt = """I am a male, aged 24 from Los Angeles, CA working as a Fashion Designer.
    I have previously watched and liked the movies: ['Inception (Sci-Fi, Thriller)', 'Wonder Woman (Action, Adventure)'].
    I have also watched and disliked the movies: ['The Shallows (Thriller)', 'Shoplifting (Crime, Drama)'].
    Please provide recommendations for movies released before April 22nd, 1998, based on my history.
    Based on my history, recommend the top 5 movies I am most likely to watch next.
    Please provide the output in a list of strings format, containing only the movie titles.
    Make sure to strictly adhere to the output format given below. Strictly do not generate any additional information other than the movie names.
    Format:  ['movie_name', 'movie_name', ... 'movie_name']
    Make sure to limit the recommendations to movies available in the MovieLens dataset."""

    # Call the function with the generated prompt
    data_loader = DataLoader()
    users_df, movies_df, ratings_df = data_loader.load_data()
    recommended_movies = get_llm_recommendations(prompt, movies_df)
    print(recommended_movies)
