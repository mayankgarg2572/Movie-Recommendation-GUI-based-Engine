# movie_recommender/collab_recommender.py

from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from movie_recommender.data_loader import load_ratings
from movie_recommender.data_loader import load_movies_metadata
from movie_recommender.utils import get_top_n_recommendations_from_preds


class SVDRecommender:
    """
    Collaborative filtering using Surprise’s SVD algorithm.
    Given historical ratings, trains an SVD model and recommends top-n movies for any user.
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        test_size: float = 0.2,
        random_state: int = 15,
    ):
        """
        Initialize and train the SVD model.

        Parameters:
            n_factors (int): Number of latent factors for SVD.
            n_epochs (int): Number of training epochs.
            test_size (float): Fraction of data to hold out as test set (for validation).
            random_state (int): Random seed.
        """
        self.ratings_df = load_ratings()
        self.movies_df = load_movies_metadata()[["id", "title"]]
        self.reader = Reader(rating_scale=(1, 5))
        self._train_model(n_factors, n_epochs, test_size, random_state)

    def _train_model(
        self, 
        n_factors: int, 
        n_epochs: int, 
        test_size: float, 
        random_state: int
    ) -> None:
        """
        Internal method to train the SVD model on 80% of the ratings,
        leaving 20% as a test set (only for computing RMSE/MAE if desired).
        """
        # Load into Surprise dataset
        data = Dataset.load_from_df(self.ratings_df[["userId", "movieId", "rating"]], self.reader)
        trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
        self.svd = SVD(
            n_factors=n_factors, n_epochs=n_epochs, biased=True, random_state=random_state
        )
        self.svd.fit(trainset)
        # Optionally evaluate on testset (uncomment if you wish to see metrics):
        # preds = self.svd.test(testset)
        # Compute RMSE/MAE here if desired

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float, str]]:
        """
        Given a user_id, predict ratings for all movies the user hasn’t rated,
        then return top_n (movieId, predicted_rating, title).

        Parameters:
            user_id (int): User index as per the original ratings.
            top_n (int): Number of movies to recommend.

        Returns:
            List of tuples: (movie_id, pred_rating, movie_title). Sorted by pred_rating desc.
        """
        # Identify all unique movie IDs
        unique_movie_ids = self.ratings_df["movieId"].unique()
        # Find movies this user has already rated
        user_rated = set(
            self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"].tolist()
        )
        # Build a list of (movie_id, predicted_rating) for every movie user hasn’t rated
        predictions: List[Tuple[int, float]] = []
        for mid in unique_movie_ids:
            if mid not in user_rated:
                est = self.svd.predict(user_id, mid).est
                predictions.append((mid, est))
        # Sort by predicted rating descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n_preds = predictions[:top_n]
        # Map movieId → title
        id_to_title = dict(zip(self.movies_df["id"], self.movies_df["title"]))
        recommendations = [(mid, rating, id_to_title.get(mid, "Unknown Title")) for mid, rating in top_n_preds]
        return recommendations
