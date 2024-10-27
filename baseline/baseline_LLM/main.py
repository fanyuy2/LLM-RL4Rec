import random
import pandas as pd

from baseline.baseline_LLM.preprocess import process_user_df
from dataloader import DataLoader
from preprocess import preprocess_data
from generate_prompt import generate_prompt
from models import get_llm_recommendations
from evaluation import evaluate_recommendations

# pip install huggingface_hub
# export HUGGINGFACE_API_KEY="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"

def sample_valid_timestamp(cleaned_data, user_id, threshold_positive, threshold_negative, window_size):
    """
    Sample a timestamp for a given user where there are at least 'window_size' positive and 'window_size'
    negative ratings before the timestamp.

    Parameters:
    - cleaned_data: DataFrame with user-movie interactions.
    - user_id: User ID to filter the data.
    - threshold_positive: Rating threshold for positive samples.
    - threshold_negative: Rating threshold for negative samples.
    - window_size: Required number of positive and negative samples before the timestamp.

    Returns:
    - timestamp: A valid timestamp or None if requirements are not met.
    """
    # Filter the user’s sorted data
    user_data = cleaned_data[cleaned_data['user_id'] == user_id].sort_values(by='timestamp')

    # Calculate cumulative positive and negative counts
    user_data['pos_count'] = (user_data['rating'] >= threshold_positive).cumsum()
    user_data['neg_count'] = (user_data['rating'] <= threshold_negative).cumsum()

    # Find valid timestamps where counts meet window_size requirements
    valid_points = user_data[(user_data['pos_count'] >= window_size) & (user_data['neg_count'] >= window_size)]

    # Randomly select a valid timestamp or return None if no valid point exists
    if not valid_points.empty:
        return valid_points.sample(1)['timestamp'].iloc[0]
    return None


# def get_user_history_window(cleaned_data, user_id, threshold_positive, threshold_negative, window_size):
#     """
#     Retrieve positive and negative samples for a user based on a valid sampled timestamp.
#
#     Parameters:
#     - cleaned_data: Merged DataFrame with user-movie interactions.
#     - user_id: User ID for whom to extract the history.
#     - threshold_positive: Rating threshold for positive samples.
#     - threshold_negative: Rating threshold for negative samples.
#     - window_size: Number of positive and negative samples to retrieve.
#
#     Returns:
#     - liked_movies: List of movie titles with high ratings within the window.
#     - disliked_movies: List of movie titles with low ratings within the window.
#     """
#     # Attempt to sample a valid timestamp
#     timestamp = sample_valid_timestamp(cleaned_data, user_id, threshold_positive, threshold_negative, window_size)
#
#     # Graceful handling: return empty lists if no valid timestamp is found
#     if timestamp is None:
#         print(f"Warning: No valid history found for user {user_id} with the specified window size.")
#         return [], []
#
#     # Filter for the user’s data before the sampled timestamp
#     user_data = cleaned_data[(cleaned_data['user_id'] == user_id) & (cleaned_data['timestamp'] < timestamp)]
#
#     # Sort by timestamp in descending order to get the most recent items first
#     user_data = user_data.sort_values(by='timestamp', ascending=False)
#
#     # Efficiently select the most recent 'window_size' positive and negative samples
#     liked_movies = (
#         user_data[user_data['rating'] >= threshold_positive]
#         .head(window_size*2)['movie_title']
#         .tolist()
#     )[:window_size]
#
#     disliked_movies = (
#         user_data[user_data['rating'] <= threshold_negative]
#         .head(window_size*2)['movie_title']
#         .tolist()
#     )[:window_size]
#
#     return liked_movies, disliked_movies


def get_user_history_window_v2(cleaned_data, user_id, window_size, threshold_positive=4, threshold_negative=2):
    """
    Retrieve the last 'window_size' interactions for a user.

    Parameters:
    - cleaned_data: DataFrame with user-movie interactions, including 'user_id', 'movie_title', 'rating', 'timestamp', and 'genres'.
    - user_id: User ID for whom to extract the history.
    - window_size: Number of recent interactions to retrieve.

    Returns:
    - history_window: List of tuples in the format (movie_title, genres, rating).
    """

    # Attempt to sample a valid timestamp
    timestamp = sample_valid_timestamp(cleaned_data, user_id, threshold_positive, threshold_negative, window_size)

    # Graceful handling: return empty lists if no valid timestamp is found
    if timestamp is None:
        print(f"Warning: No valid history found for user {user_id} with the specified window size.")
        return [], -1


    # Filter for the user’s data before the sampled timestamp
    user_data = cleaned_data[(cleaned_data['user_id'] == user_id) & (cleaned_data['timestamp'] < timestamp)]

    # Sort by timestamp in descending order to get the most recent items first
    user_data = user_data.sort_values(by='timestamp', ascending=False)

    # Get the most recent 'window_size' interactions
    user_data_window = user_data.head(window_size)

    # Build the history_window
    history_window = []
    for _, row in user_data_window.iterrows():
        movie_title = row['movie_title']
        #genres_str = row['genre']
        # Split genres string into a list
        genres = row['genre']
        rating = row['rating']
        history_window.append((movie_title, genres, rating))

    return history_window, timestamp


def get_ground_truth(cleaned_data, user_id, timestamp, p, threshold_positive):
    """Fetch the next p movies the user rated highly (above threshold) after timestamp."""
    future_data = cleaned_data[(cleaned_data['user_id'] == user_id) & (cleaned_data['timestamp'] > timestamp)]
    future_liked = future_data[future_data['rating'] >= threshold_positive].sort_values(by='timestamp')

    return future_liked['movie_id'].head(p).tolist()


def main():
    # # Initialize data loader and load datasets
    # data_loader = DataLoader()
    # users_df, movies_df, ratings_df = data_loader.load_data()
    #
    #
    # # Preprocess the data
    # user_profiles, cleaned_data = preprocess_data(users_df, movies_df, ratings_df)
    #
    # # users_profiles = process_user_df(users_df)
    #
    # # Define constants for evaluation
    # user_id = 123  # Replace with the target user ID
    # threshold_positive = 3  # Rating threshold for "liked" movies
    # threshold_negative = 2  # Rating threshold for "disliked" movies
    # window_size = 5  # Number of positive and negative samples to retrieve.
    # top_k = 5  # Number of recommendations to fetch
    # next_p = 5  # Number of future movies to use for ground truth
    # evaluation_k = 5  # Evaluation cutoff for metrics
    #
    # print(ratings_df[ratings_df['user_id']==user_id])
    #
    # # Sample a timestamp for the user
    # user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    # # timestamp = sample_timestamp(user_ratings)
    #
    # # Get liked and disliked movies in the history window
    # history_window, timestamp = get_user_history_window_v2(
    #     cleaned_data, user_id, window_size
    # )
    #
    # print(history_window)
    #
    # # Generate the prompt based on history window
    # prompt = generate_prompt(user_profiles[user_id], history_window, top_k,
    #                          positive_threshold=threshold_positive,
    #                          negative_threshold=threshold_negative)

    movies_df = None
    top_k = 5
    prompt = """I am a female, aged 48, from Washington, DC, working as artist.
I have previously watched and liked the movies: ['Leaving Las Vegas (1995) (drama, romance)', 'Secrets & Lies (1996) (drama)', 'Star Wars (1977) (action, adventure, romance, sci_fi, war)', 'Sense and Sensibility (1995) (drama, romance)', 'Dead Man Walking (1995) (drama)'].
Please provide recommendations for movies released before April 22nd, 1998, based on my history.
Based on my history, recommend the top 5 movies I am most likely to watch next.
Please provide the output in a list of strings format, containing only the movie titles.
Make sure to strictly adhere to the output format given below. Strictly do not generate any additional information other than the movie names.
Format:  ['movie_name', 'movie_name', ... 'movie_name']
Make sure to limit the recommendations to movies available in the MovieLens dataset."""

    # Get movie recommendations from the LLM
    recommended_movies = get_llm_recommendations(prompt, movies_df)[:top_k]

    # Fetch ground truth for evaluation
    ground_truth = get_ground_truth(cleaned_data, user_id, timestamp, next_p, threshold_positive)

    # Calculate evaluation metrics
    metrics = evaluate_recommendations(ground_truth, recommended_movies, evaluation_k)

    # Output the evaluation results
    print(f"Evaluation Metrics for User {user_id}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    main()
