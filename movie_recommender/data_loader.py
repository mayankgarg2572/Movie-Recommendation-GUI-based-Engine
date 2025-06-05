# movie_recommender/data_loader.py

import pandas as pd
from pathlib import Path
from typing import Tuple

from config import DATA_DIR


def load_ratings(csv_filename: str = "ratings_small.csv") -> pd.DataFrame:
    """
    Load the ratings CSV into a DataFrame.

    Parameters:
        csv_filename (str): Name of the CSV file under DATA_DIR.

    Returns:
        pd.DataFrame: Contains columns [userId, movieId, rating].
    """
    path = DATA_DIR / csv_filename
    if not path.exists():
        raise FileNotFoundError(f"Ratings file not found at {path}")
    df = pd.read_csv(path, usecols=["userId", "movieId", "rating"], low_memory=False)
    return df


def load_movies_metadata(csv_filename: str = "movies_metadata.csv") -> pd.DataFrame:
    """
    Load the movies metadata CSV.

    Parameters:
        csv_filename (str): Name of the CSV file under DATA_DIR.

    Returns:
        pd.DataFrame: Contains movie metadata (including 'id', 'genres', 'vote_average', 'vote_count', 'title', etc.).
    """
    path = DATA_DIR / csv_filename
    if not path.exists():
        raise FileNotFoundError(f"Movies metadata file not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    # since we are going to use the vote count and vote_average method for each movie 
    # we need them to be completely filled so we are going to remove them 
    # Similarly we need to remove them from other dataframe also
    df.dropna(subset= ['vote_average', 'vote_count'], inplace = True)
    df["id"] = df["id"].astype(int)
    return df


def load_keywords(csv_filename: str = "keywords.csv") -> pd.DataFrame:
    """
    Load the keywords CSV.

    Parameters:
        csv_filename (str): Name of the CSV file under DATA_DIR.

    Returns:
        pd.DataFrame: Contains columns [id, keywords] where keywords is JSON-like text.
    """
    path = DATA_DIR / csv_filename
    if not path.exists():
        raise FileNotFoundError(f"Keywords file not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


def load_credits(csv_filename: str = "credits.csv") -> pd.DataFrame:
    """
    Load the credits CSV.

    Parameters:
        csv_filename (str): Name of the CSV file under DATA_DIR.

    Returns:
        pd.DataFrame: Contains columns [id, cast, crew].
    """
    path = DATA_DIR / csv_filename
    if not path.exists():
        raise FileNotFoundError(f"Credits file not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


def load_links_small(csv_filename: str = "links_small.csv") -> pd.DataFrame:
    """
    Load the links_small CSV (for IMDB ↔ TMDB ID mapping).

    Parameters:
        csv_filename (str): Name of the CSV file under DATA_DIR.

    Returns:
        pd.DataFrame: Contains columns [movieId, imdbId, tmdbId].
    """
    path = DATA_DIR / csv_filename
    if not path.exists():
        raise FileNotFoundError(f"Links_small file not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["tmdbId"])
    df["tmdbId"] = df["tmdbId"].astype(int)
    return df


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all Dataset CSVs at once.

    Returns:
        Tuple of DataFrames: (ratings_df, movies_df, keywords_df, credits_df, links_df)
    """
    ratings_df = load_ratings()
    movies_df = load_movies_metadata()
    keywords_df = load_keywords()
    credits_df = load_credits()
    links_df = load_links_small()
    return ratings_df, movies_df, keywords_df, credits_df, links_df
