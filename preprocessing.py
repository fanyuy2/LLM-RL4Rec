import pandas as pd

# Load the data files
ratings_df = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
users_df = pd.read_csv('u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
movies_df = pd.read_csv('u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=['movie_id', 'title'], header=None)

# Parameters
top_L = 5  # Number of top positive and negative movies to select
top_k = 3  # Number of ground truth movies

# Helper functions
def get_top_L_movies(user_ratings, top_L, threshold=3):
    """Get the top L positive and negative movies for a user based on rating."""
    positive_movies = user_ratings[user_ratings['rating'] >= threshold].sort_values(by='rating', ascending=False)
    negative_movies = user_ratings[user_ratings['rating'] < threshold].sort_values(by='rating', ascending=True)
    return positive_movies['movie_id'].head(top_L).tolist(), negative_movies['movie_id'].head(top_L).tolist()

def get_ground_truth_movies(user_ratings, top_k):
    """Get the top K movies for a user as ground truth."""
    return user_ratings.sort_values(by='rating', ascending=False)['movie_id'].head(top_k).tolist()

def create_prompt(user_info, top_L_pos, top_L_neg):
    """Generate a text prompt for supervised fine-tuning."""
    prompt = (f"I am a {user_info['age']} years old {user_info['gender']} working as a {user_info['occupation']}.\n"
              f"Movies I liked: {top_L_pos}.\nMovies I disliked: {top_L_neg}.")
    return prompt

# Prepare data for training and test features
train_features = []
test_features = []
train_prompts = []
test_prompts = []

# Split based on timestamp or load specific train/test files
for user_id, user_ratings in ratings_df.groupby('user_id'):
    # Get user demographic info
    user_info = users_df[users_df['user_id'] == user_id].iloc[0].to_dict()
    
    # Split user's ratings into training and test based on timestamp
    split_timestamp = user_ratings['timestamp'].quantile(0.8)
    training_ratings = user_ratings[user_ratings['timestamp'] <= split_timestamp]
    test_ratings = user_ratings[user_ratings['timestamp'] > split_timestamp]
    
    # Prepare features and prompts for training set
    if not training_ratings.empty:
        top_L_pos, top_L_neg = get_top_L_movies(training_ratings, top_L)
        ground_truth = get_ground_truth_movies(training_ratings, top_k)
        
        train_features.append({
            'user_id': user_id,
            **user_info,
            'context': "Training context for user.",
            'top_L_positive_movie_ids': top_L_pos,
            'top_L_negative_movie_ids': top_L_neg,
            'ground_truth': ground_truth
        })
        
        prompt = create_prompt(user_info, top_L_pos, top_L_neg)
        train_prompts.append({
            'user_id': user_id,
            'prompt': prompt,
            'ground_truth': ground_truth
        })
    
    # Prepare features and prompts for test set
    if not test_ratings.empty:
        top_L_pos, top_L_neg = get_top_L_movies(test_ratings, top_L)
        ground_truth = get_ground_truth_movies(test_ratings, top_k)
        
        test_features.append({
            'user_id': user_id,
            **user_info,
            'context': "Test context for user.",
            'top_L_positive_movie_ids': top_L_pos,
            'top_L_negative_movie_ids': top_L_neg,
            'ground_truth': ground_truth
        })
        
        prompt = create_prompt(user_info, top_L_pos, top_L_neg)
        test_prompts.append({
            'user_id': user_id,
            'prompt': prompt,
            'ground_truth': ground_truth
        })

# Convert to DataFrames and save to CSV
train_features_df = pd.DataFrame(train_features)
test_features_df = pd.DataFrame(test_features)
train_prompts_df = pd.DataFrame(train_prompts)
test_prompts_df = pd.DataFrame(test_prompts)

# Save to CSV files
train_features_df.to_csv('Training_features.csv', index=False)
test_features_df.to_csv('Test_features.csv', index=False)
train_prompts_df.to_csv('Training_prompt.csv', index=False)
test_prompts_df.to_csv('Test_prompt.csv', index=False)

print("Files created: Training_features.csv, Test_features.csv, Training_prompt.csv, Test_prompt.csv")
