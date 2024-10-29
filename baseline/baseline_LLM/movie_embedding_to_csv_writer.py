import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from dataloader import DataLoader

def get_movie_embeddings(embedding_model, movies_df):
    """Compute embeddings for movie titles."""
    titles = movies_df['movie_title'].tolist()
    embeddings = embedding_model.encode(titles, convert_to_tensor=True)
    return titles, embeddings

def write_embeddings_to_csv(embedding_model, movies_df, output_file):
    """Write movie_id, movie titles, and embeddings to a CSV file."""
    titles, embeddings = get_movie_embeddings(embedding_model, movies_df)

    # Convert embeddings tensor to numpy array
    embeddings_np = embeddings.cpu().numpy()  # Move tensor to CPU if necessary

    # Create a DataFrame with movie_id, titles, and embeddings
    data = pd.DataFrame(embeddings_np)
    data.insert(0, 'movie_title', titles)  # Add titles as the first column
    data.insert(0, 'movie_id', movies_df['movie_id'])  # Add movie_id as the first column

    # Write to CSV
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Load the embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load data
    data_loader = DataLoader()
    users_df, movies_df, ratings_df = data_loader.load_data()

    # Save the movie embeddings to a CSV file
    write_embeddings_to_csv(embedding_model, movies_df, 'movie_embeddings.csv')
