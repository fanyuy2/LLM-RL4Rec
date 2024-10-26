import pandas as pd
from datetime import datetime
dateparse = lambda x: datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S')

class DataLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
            cls._instance.users_df = None
            cls._instance.movies_df = None
            cls._instance.ratings_df = None
        return cls._instance

    def load_data(self):
        """Loads user, movies, and ratings datasets only once."""
        if self.users_df is None or self.movies_df is None or self.ratings_df is None:
            print("Loading data...")
            self.users_df = pd.read_csv('../../dataset/ml-100k/ml-100k/u.user', sep='|',
                                        names=['user_id', 'age', 'gender', 'occupation',
                                               'zip_code'])  # Update with the correct path
            self.movies_df = pd.read_csv('../../dataset/ml-100k/ml-100k/u.item', sep='|', encoding='latin-1',
                                         names=['movie_id', 'movie_title', 'release_date', 'video_release_date',
                                                'imdb_url', 'unknown', 'action',
                                                'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary',
                                                'drama', 'fantasy',
                                                'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi',
                                                'thriller', 'war', 'western'])  # Update with the correct path
            self.ratings_df = pd.read_csv('../../dataset/ml-100k/ml-100k/u.data', sep='\t',
                                          names=['user_id', 'movie_id', 'rating',
                                                 'timestamp'], date_parser=dateparse)  # Update with the correct path
            print("Data loaded.")
        return self.users_df, self.movies_df, self.ratings_df
