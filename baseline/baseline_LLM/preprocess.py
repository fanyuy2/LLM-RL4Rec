import pandas as pd
from pyzipcode import ZipCodeDatabase
zcdb = ZipCodeDatabase()

def get_city_state(zip_code):
    try:
        zip_info = zcdb[zip_code]
        return zip_info.city, zip_info.state
    except KeyError:
        return None, None

def process_user_df(users_df):
    users_dict = {
        row["user_id"]: {
            "age": row["age"],
            "gender": "male" if row["gender"] == "M" else "female",
            "occupation": row["occupation"],
            "location": f"{row['city']}, {row['state']}"
        }
        for _, row in users_df.iterrows()
    }
    return users_dict


def preprocess_data(users_df, movies_df, ratings_df):
    """
    Cleans and formats the user, movies, and ratings DataFrames.

    Parameters:
    - users_df (pd.DataFrame): DataFrame containing user information.
    - movies_df (pd.DataFrame): DataFrame containing movie information.
    - ratings_df (pd.DataFrame): DataFrame containing user ratings.

    Returns:
    - pd.DataFrame: A DataFrame that includes user info, liked and disliked movies, and their genres.
    """
    users_df[['city', 'state']] = users_df['zip_code'].apply(lambda zip_code: pd.Series(get_city_state(zip_code)))

    # convert user_df to user_profiles (dict of dict), used for generate_prompt
    users_dict = {
        row["user_id"]: {
            "age": row["age"],
            "gender": "male" if row["gender"] == "M" else "female",
            "occupation": row["occupation"],
            "location": f"{row['city']}, {row['state']}"
        }
        for _, row in users_df.iterrows()
    }

    # Ensure that the relevant columns are present and clean
    users_df.dropna(subset=['user_id', 'age', 'gender', 'occupation', 'city', 'state'], inplace=True)
    movies_df.dropna(subset=['movie_id', 'movie_title'], inplace=True)
    ratings_df.dropna(subset=['user_id', 'movie_id', 'rating', 'timestamp'], inplace=True)

    # Merge ratings with movie titles
    merged_df = ratings_df.merge(movies_df, on='movie_id')

    # Create a 'genre' column that lists all applicable genres
    genre_columns = movies_df.columns[4:]  # Assuming genres start from the 5th column onward
    merged_df['genre'] = merged_df[genre_columns].apply(
        lambda row: [genre for genre, is_genre in zip(genre_columns, row) if is_genre == 1],
        axis=1
    )

    # Clean up the genre column to ensure it's a list
    merged_df['genre'] = merged_df['genre'].apply(lambda x: x if isinstance(x, list) else [])

    # Optional: convert timestamp to datetime if needed
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    # print(merged_df.columns)
    return users_dict, merged_df[['user_id', 'movie_id', 'rating', 'timestamp', 'movie_title','genre']]

# Example usage
# Assuming you have the DataFrames users_df, movies_df, and ratings_df as described.
# users_df = pd.read_csv('users.csv')  # Load your user DataFrame
# movies_df = pd.read_csv('movies.csv')  # Load your movie DataFrame
# ratings_df = pd.read_csv('ratings.csv')  # Load your ratings DataFrame

# cleaned_data = preprocess_data(users_df, movies_df, ratings_df)
# print(cleaned_data.head())
