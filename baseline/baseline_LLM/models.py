import os
import re
import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util

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

def map_movies_to_dataset(y_hat, movie_titles, movie_embeddings, pooling='mean'):
    """
    Map the LLM's output movie titles to the dataset's movie titles using cosine similarity.

    Parameters:
    - y_hat (list of str): Movie titles generated by the LLM.
    - movie_titles (list of str): Movie titles from the dataset.
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
    cosine_scores = util.pytorch_cos_sim(y_hat_embeddings, movie_embeddings)

    # Get the top matches
    top_results = np.argpartition(-cosine_scores.cpu().numpy(), range(5))[:, :5]
    mapped_movies = [movie_titles[idx] for idx in top_results.flatten()]

    return mapped_movies


def get_llm_recommendations(prompt, movies_df, model_name="meta-llama/Llama-3.2-1B-Instruct", max_tokens=500, top_k=5,  pooling='mean'):
    """
    Sends the prompt to the LLM and retrieves the recommended movie list.

    Parameters:
    - prompt (str): The input prompt generated for the LLM.
    - movies_df (DataFrame): DataFrame containing movie information.
    - model_name (str): The name of the model to use for recommendations.
    - max_tokens (int): Maximum tokens to allow in the output.
    - top_k (int): Number of movie recommendations to retrieve.
    - pooling (str): Pooling method for embedding comparison.

    Returns:
    - list of str: A list of recommended movie titles parsed from the LLM's output.
    """
    movie_titles, movie_embeddings = get_movie_embeddings(movies_df)
    try:
        # Send the prompt to the LLM
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "movie recommender", "content": prompt}],
            max_tokens=max_tokens,
            stream=False
        )

        # Extract the response text
        llm_output = response['choices'][0]['message']['content'].strip()
        print(llm_output)
        # Use regex to extract the list of movie titles from the output
        movie_list_match = re.search(r"\[.*?\]", llm_output, re.DOTALL)
        if movie_list_match:
            # Evaluate the extracted string to a Python list
            recommendations = eval(movie_list_match.group())
            if isinstance(recommendations, list) and all(isinstance(item, str) for item in recommendations):
                mapped_movies = map_movies_to_dataset(recommendations[:top_k], movie_titles, movie_embeddings, pooling)
                return mapped_movies
            else:
                # return []
                raise ValueError("LLM response format is incorrect")
        else:
            print("No valid list of movie titles found in the output.")
            return []

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
