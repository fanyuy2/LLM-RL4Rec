from dataloader import DataLoader
from preprocess import preprocess_data
from generate_prompt import generate_prompt
from models import get_llm_recommendations

def main():
    # Create a DataLoader instance and load the datasets once
    data_loader = DataLoader()
    users_df, movies_df, ratings_df = data_loader.load_data()

    # Preprocess the data
    cleaned_data = preprocess_data(users_df, movies_df, ratings_df)

    print(cleaned_data.head())

    # Example: Assuming you have a specific user_id to recommend movies for
    user_id = 1  # Replace with the user_id you want to analyze
    threshold_positive = 4  # Define threshold for liked movies
    threshold_negative = 2  # Define threshold for disliked movies

    # Generate the prompt for the user
    prompt = generate_prompt(cleaned_data, user_id, threshold_positive, threshold_negative)
    print(prompt)
    # Get movie recommendations using the model
    recommended_movies = get_llm_recommendations(prompt)
    print(recommended_movies)
    # Print or process the recommendations
    print(f"Recommended movies for user {user_id}: {recommended_movies}")

if __name__ == '__main__':
    main()
