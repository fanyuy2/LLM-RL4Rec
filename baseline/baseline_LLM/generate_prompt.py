def generate_prompt(user_profile, history_window, top_k, include_dislikes=False,
                    positive_threshold=4, negative_threshold=2):
    """
    Generates a structured prompt for the LLM based on user profile, history (with liked/disliked movies),
    and target movie count.

    Parameters:
    - user_profile (dict): Contains 'age', 'gender', 'occupation', and 'location' (e.g., city, state).
    - history_window (list of tuples): Each tuple contains (movie name, genres, rating) from the user's history.
    - top_k (int): The number of recommended movies to return.
    - include_dislikes (bool): Whether to include movies the user disliked.
    - positive_threshold (int): The minimum rating to consider a movie "liked" by the user.
    - negative_threshold (int): The maximum rating to consider a movie "disliked" by the user.

    Returns:
    - str: The prompt to be sent to the LLM for movie recommendation.
    """
    # Extract user details
    age = user_profile.get("age", "unknown age")
    gender = user_profile.get("gender", "unknown gender")
    occupation = user_profile.get("occupation", "an unknown occupation")
    location = user_profile.get("location", "an unknown location")

    # Filter liked and disliked movies based on thresholds
    liked_movies = [
        f"{movie} ({', '.join(genres)})"
        for movie, genres, rating in history_window if rating >= positive_threshold
    ]

    disliked_movies = [
        f"{movie} ({', '.join(genres)})"
        for movie, genres, rating in history_window if rating <= negative_threshold
    ] if include_dislikes else []

    # Build the dynamic prompt with both liked and disliked movies
    prompt = f"""I am a {gender}, aged {age}, from {location}, working as {occupation}.
I have previously watched and liked the movies: {liked_movies}."""

    if include_dislikes and disliked_movies:
        prompt += f"\nI have also watched and disliked the movies: {disliked_movies}."

    prompt += f"""
Please provide recommendations for movies released before April 22nd, 1998, based on my history.
Based on my history, recommend the top {top_k} movies I am most likely to watch next.
Please provide the output in a list of strings format, containing only the movie titles.
Make sure to strictly adhere to the output format given below. Strictly do not generate any additional information other than the movie names.
Format:  ['movie_name', 'movie_name', ... 'movie_name']
Make sure to limit the recommendations to movies available in the MovieLens dataset."""

    return prompt


if __name__ == '__main__':
    # Sample user profile and history window with ratings
    user_profile = {
        "age": 24,
        "gender": "male",
        "occupation": "Fashion Designer",
        "location": "Los Angeles, CA"
    }

    # Sample history: (movie_name, [genres], rating)
    history_window = [
        ("Inception", ["Sci-Fi", "Thriller"], 5),
        ("The Shallows", ["Thriller"], 2),
        ("Wonder Woman", ["Action", "Adventure"], 4),
        ("This Is Us", ["Drama"], 3),
        ("Shoplifting", ["Crime", "Drama"], 1)
    ]

    # Generate prompt for top 5 recommendations with dislikes included
    prompt = generate_prompt(user_profile, history_window, top_k=5,
                             include_dislikes=True, positive_threshold=4, negative_threshold=2)
    print(prompt)
