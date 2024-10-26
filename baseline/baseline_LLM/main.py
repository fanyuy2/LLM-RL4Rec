import pandas as pd
from preprocess import preprocess_data
from generate_prompt import generate_prompt
from models import get_llm_recommendations

def main():
    # Load the datasets
    users_df = pd.read_csv('path/to/users.csv')  # Update with the correct path
    movies_df = pd.read_csv('path/to/movies.csv')  # Update with the correct path
    ratings_df = pd.read_csv('path/to/ratings.csv')  # Update with the correct path

    # Preprocess the data
    cleaned_data = preprocess_data(users_df, movies_df, ratings_df)

    # Example: Assuming you have a specific user_id to recommend movies for
    user_id = 1  # Replace with the user_id you want to analyze
    threshold_positive = 4  # Define threshold for liked movies
    threshold_negative = 2  # Define threshold for disliked movies

    # Generate the prompt for the user
    prompt = generate_prompt(cleaned_data, user_id, threshold_positive, threshold_negative)

    # Get movie recommendations using the model
    recommended_movies = get_llm_recommendations(prompt)

    # Print or process the recommendations
    print(f"Recommended movies for user {user_id}: {recommended_movies}")

if __name__ == '__main__':
    main()
