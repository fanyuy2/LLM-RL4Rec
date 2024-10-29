import random
import pandas as pd
import numpy as np
import torch

from baseline.baseline_LLM.preprocess import process_user_df
from dataloader import DataLoader
from preprocess import preprocess_data
from generate_prompt import generate_prompt
from models import model_prediction
from evaluation import evaluate_recommendations_for_all_users
from sklearn.model_selection import train_test_split
from collections import defaultdict


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
        # genres_str = row['genre']
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


def read_embeddings_from_csv(input_file):
    """Read movie_id, movie_title, and embeddings from a CSV file."""
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_file)

    # Extract movie_id, movie_title, and embeddings separately
    movie_ids = df['movie_id'].tolist()
    movie_titles = df['movie_title'].tolist()

    # Extract the embeddings part as a numpy array
    embeddings = df.iloc[:, 2:].values  # All columns starting from the third column

    return movie_ids, movie_titles, embeddings


def split_user_prompt_data(input_file):
    """Read user-prompt data from a CSV and split into 60% train, 40% test."""
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_file)

    # Split the data into 60% train and 40% test
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    return train_data, test_data


def save_split_data(train_data, test_data, train_file='user_prompt_train_data.csv',
                    test_file='user_prompt_test_data.csv'):
    """Save train and test data to CSV files."""
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)


def extract_user_ids_and_prompts(df):
    """Extract user IDs and prompts into separate lists."""
    user_ids = df['user_id'].tolist()  # List of user IDs
    prompts = df['Prompt'].tolist()  # List of prompts
    timestamps = df['Timestamp'].tolist()  # List of timestamps for which user prompts are generated
    return user_ids, prompts, timestamps


def fetch_user_watch_history(filename, valid_user_ids):
    """Parse the CSV and return a dictionary with only valid user_ids."""
    # Load the CSV into a DataFrame
    df = pd.read_csv(filename)

    # Convert valid_user_ids to a set for faster lookups
    valid_user_ids_set = set(valid_user_ids)

    # Initialize a defaultdict to store lists of movie_ids for each valid user_id
    user_movie_dict = defaultdict(list)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']

        # Only add movie_id if the user_id is in the valid_user_ids set
        if user_id in valid_user_ids_set:
            user_movie_dict[user_id].append(movie_id)

    return user_movie_dict


def fetch_user_future_liked_movies(filename, valid_user_ids):
    """Parse the CSV and return a dictionary with only valid user_ids."""
    # Load the CSV into a DataFrame
    df = pd.read_csv(filename)

    # Convert valid_user_ids to a set for faster lookups
    valid_user_ids_set = set(valid_user_ids)

    # Initialize a defaultdict to store lists of movie_ids for each valid user_id
    user_movie_dict = defaultdict(list)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']

        # Only add movie_id if the user_id is in the valid_user_ids set
        if user_id in valid_user_ids_set:
            user_movie_dict[user_id].append(movie_id)

    return user_movie_dict


def evaluator():
    # # Initialize data loader and load datasets
    # data_loader = DataLoader()
    # users_df, movies_df, ratings_df = data_loader.load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preprocess the data
    # user_profiles, cleaned_data = preprocess_data(users_df, movies_df, ratings_df)
    movie_id_list, movie_titles_list, movie_embeddings_2d_list = read_embeddings_from_csv('movie_embeddings.csv')
    movie_embeddings_2d_list = torch.FloatTensor(movie_embeddings_2d_list)#.to(device)
    '''
    users_profiles = process_user_df(users_df)

    # Define constants for evaluation
    user_id = 123  # Replace with the target user ID
    threshold_positive = 3  # Rating threshold for "liked" movies
    threshold_negative = 2  # Rating threshold for "disliked" movies
    window_size = 5  # Number of positive and negative samples to retrieve.
    top_k = 5  # Number of recommendations to fetch
    next_p = 5  # Number of future movies to use for ground truth
    evaluation_k = 5  # Evaluation cutoff for metrics
    print(ratings_df[ratings_df['user_id']==user_id])

    # Sample a timestamp for the user
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    # timestamp = sample_timestamp(user_ratings)

    # Get liked and disliked movies in the history window
    history_window, timestamp = get_user_history_window_v2(
        cleaned_data, user_id, window_size
    )
    print(history_window)
    # Generate the prompt based on history window
    prompt = generate_prompt(user_profiles[user_id], history_window, top_k,
                             positive_threshold=threshold_positive,
                             negative_threshold=threshold_negative)

    prompt = """I am a female, aged 48, from Washington, DC, working as artist.
    I have previously watched and liked the movies: ['Leaving Las Vegas (1995) (drama, romance)', 'Secrets & Lies (1996) (drama)', 'Star Wars (1977) (action, adventure, romance, sci_fi, war)', 'Sense and Sensibility (1995) (drama, romance)', 'Dead Man Walking (1995) (drama)'].
    Please provide recommendations for movies released before April 22nd, 1998, based on my history.
    Based on my history, recommend the top 5 movies I am most likely to watch next.
    Please provide the output in a list of strings format, containing only the movie titles.
    Make sure to strictly adhere to the output format given below. Strictly do not generate any additional information other than the movie names.
    Format:  ['movie_name', 'movie_name', ... 'movie_name']
    Make sure to limit the recommendations to movies available in the MovieLens dataset."""


    movies_df = None
    '''
    top_k = 5  # Number of recommendations to fetch
    user_prompt_file_name = 'user_prompts_top' + str(top_k) + '.csv'
    train_user_prompt_data, test_user_prompt_data = split_user_prompt_data(user_prompt_file_name)
    save_split_data(train_user_prompt_data, test_user_prompt_data)

    # Extract user IDs and prompts from the training set
    train_user_ids, train_prompts, train_timestamps = extract_user_ids_and_prompts(train_user_prompt_data)

    # Extract user IDs and prompts from the testing set
    test_user_ids, test_prompts, test_timestamps = extract_user_ids_and_prompts(test_user_prompt_data)

    users_watch_history = fetch_user_watch_history('user_watch_history.csv', test_user_ids)
    users_future_liked_movies = fetch_user_future_liked_movies('user_liked_future_movies.csv', test_user_ids)

    # Get movie recommendations from the LLM
    #recommended_movies = get_llm_recommendations(test_user_ids, test_user_prompt_data, movie_titles_list, movie_embeddings_2d_list,
    #                                             top_k=top_k)

    recommended_movies = model_prediction(test_user_prompt_data, users_watch_history, movie_id_list, movie_embeddings_2d_list)


    # Calculate evaluation metrics
    #metrics = evaluate_recommendations(ground_truth, recommended_movies, evaluation_k)

    metrics = evaluate_recommendations_for_all_users(users_future_liked_movies, recommended_movies, top_k)

    # Output the evaluation results
    print(f"Average Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    evaluator()
