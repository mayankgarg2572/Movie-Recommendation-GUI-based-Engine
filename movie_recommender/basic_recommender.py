# movie_recommender/basic_recommender.py

from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np

from movie_recommender.data_loader import load_movies_metadata
from movie_recommender.utils import compute_imdb_score


class BasicRecommender:
    """
    Basic recommender that ranks movies by a weighted IMDb score,
    and allows filtering by genre.
    """

    def __init__(self, vote_count_quantile: float = 0.95):
        """
        Initialize by loading movie metadata and computing top_movies DataFrame.

        Parameters:
            vote_count_quantile (float): Quantile to define 'm' (minimum votes threshold).
        """
        self.movies_df = load_movies_metadata()
        self.C = float(self.movies_df["vote_average"].mean())
        # Define 'm' as the vote_count threshold to be in top quantile
        self.m = float(self.movies_df["vote_count"].quantile(vote_count_quantile))
        # Precompute weighted scores for movies with vote_count >= m
        self._prepare_top_movies()
        # Create a mapping genre → DataFrame of top movies in that genre
        self.genre_to_df = self._group_by_genre()

    def _prepare_top_movies(self) -> None:
        """
        Filter movies with vote_count >= m and compute IMDb score for each.
        Sort in descending order of score.
        """
        df = self.movies_df.dropna(subset=["vote_average", "vote_count", "genres"])
        # Filter only those with sufficient vote_count
        df_top = df[df["vote_count"] >= self.m].copy()
        df_top["imdb_score"] = df_top.apply(
            lambda row: compute_imdb_score(
                vote_count=float(row["vote_count"]),
                vote_average=float(row["vote_average"]),
                C=self.C,
                m=self.m,
            ),
            axis=1,
        )
        # Sort by descending weighted score
        self.top_movies_df = df_top.sort_values(by="imdb_score", ascending=False).reset_index(drop=True)
        # Keep only necessary columns for recommendations
        self.top_movies_df = self.top_movies_df[["id", "title", "imdb_score", "genres"]]

    def _group_by_genre(self) -> dict:
        """
        Build a mapping from each genre name to a subset DataFrame
        containing only movies of that genre, sorted by imdb_score.
        """
        genre_to_df = {}
        # Each row in self.top_movies_df has 'genres' as a string representation of list-of-dicts
        # We need to parse it to extract genre names. In movies_metadata.csv, 'genres' is JSON-like text.
        # Approach: literal_eval the 'genres' column and extract 'name' of each dict.
        from ast import literal_eval

        def parse_genre_list(genre_str: str) -> List[str]:
            try:
                genre_list = literal_eval(genre_str)
                return [g["name"] for g in genre_list]
            except Exception:
                return []

        # Add a column of parsed genre names
        self.top_movies_df["parsed_genres"] = (
            self.top_movies_df["genres"].apply(parse_genre_list)
        )

        # Initialize an empty DataFrame for each genre encountered
        genre_to_df: Dict[str, pd.DataFrame] = {}
        for genre_list in self.top_movies_df["parsed_genres"]:
            for genre in genre_list:
                if genre not in genre_to_df:
                    genre_to_df[genre] = pd.DataFrame(
                        columns=self.top_movies_df.columns
                    )

        # Iterate row by row, concatenating rows into each genre's DataFrame
        for _, row in self.top_movies_df.iterrows():
            for genre in row["parsed_genres"]:
                # Convert the Series `row` into a single-row DataFrame
                single_row_df = row.to_frame().T
                # Concatenate with the existing DataFrame for that genre
                genre_to_df[genre] = pd.concat(
                    [genre_to_df[genre], single_row_df],
                    ignore_index=True
                )

        # (Optional) Sort each genre DataFrame by imdb_score descending
        for genre, df in genre_to_df.items():
            genre_to_df[genre] = df.sort_values(
                by="imdb_score", ascending=False
            ).reset_index(drop=True)

        return genre_to_df

    def get_top_n_overall(self, n: int = 10) -> pd.DataFrame:
        """
        Return the top-n movies overall by weighted IMDb score.

        Parameters:
            n (int): Number of top movies to return.

        Returns:
            pd.DataFrame with columns [id, title, imdb_score].
        """
        return self.top_movies_df[["id", "title", "imdb_score"]].head(n).copy()

    def get_top_n_by_genre(self, genre: str, n: int = 10) -> Optional[pd.DataFrame]:
        """
        Return the top-n movies in a specific genre.

        Parameters:
            genre (str): Genre name (case-sensitive, e.g., "Comedy").
            n (int): Number of top movies to return.

        Returns:
            pd.DataFrame or None: If genre not found, returns None.
        """
        if genre not in self.genre_to_df:
            return None
        # genre_to_df[genre] is already sorted by imdb_score
        return self.genre_to_df[genre][["id", "title", "imdb_score"]].head(n).copy()
