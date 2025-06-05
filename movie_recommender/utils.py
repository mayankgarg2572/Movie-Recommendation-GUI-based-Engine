# movie_recommender/utils.py

import numpy as np
from typing import Any, Dict, List, Optional

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def compute_imdb_score(vote_count: int, vote_average: float, C: float, m: float) -> float:
    """
    Compute the IMDb weighted rating score:
    (v / (v + m)) * R + (m / (v + m)) * C

    Where:
      - v = number of votes for the movie
      - R = average rating for the movie
      - m = minimum votes required to be listed (e.g., 95th percentile of vote counts)
      - C = mean vote across the dataset
    """
    return ((vote_count / (vote_count + m)) * vote_average) + ((m / (m + vote_count)) * C)


def stem_and_clean_keywords(keywords_list: List[str]) -> List[str]:
    """
    Given a list of keywords (strings), stem each keyword, lowercase it,
    and remove spaces. Return only those keywords that appear more than once
    in the global keyword count (filtering is done upstream).
    """
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(kw.replace(" ", "").lower()) for kw in keywords_list]


def filter_terms_by_frequency(
    all_terms_list: List[List[str]]
) -> Dict[str, int]:
    """
    Given a list of lists (e.g., list of keyword lists or overview word lists),
    construct a term → frequency dictionary, then remove keys that have frequency 1.

    Returns:
        Dict[str, int]: terms with frequency ≥ 2.
    """
    freq: Dict[str, int] = {}
    for term_list in all_terms_list:
        for term in term_list:
            freq[term] = freq.get(term, 0) + 1
    # Remove terms that occur only once
    return {term: count for term, count in freq.items() if count > 1}


def build_cosine_similarity_matrix(text_series: List[str]) -> Any:
    """
    Given a list or pandas Series of text (“soup”), vectorize it with unigrams & bigrams,
    then compute cosine similarity. Return the cosine similarity matrix.
    """
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, stop_words="english")
    count_matrix = vectorizer.fit_transform(text_series)
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim


def get_top_n_similar_indices(
    cosine_sim_matrix: Any, 
    index: int, 
    top_n: int = 10
) -> List[int]:
    """
    Given a square cosine similarity matrix and a reference index, return
    the top_n most similar indices (excluding the index itself).

    Returns:
        List[int]: indices of the top_n most similar items.
    """
    # Pair (movie_index, similarity_score), then sort descending by score
    scores = list(enumerate(cosine_sim_matrix[index]))
    scores.sort(key=lambda x: x[1], reverse=True)
    # Exclude the first element since that's the item itself (score=1.0)
    top_indices = [i for i, sim in scores[1 : top_n + 1]]
    return top_indices


def compute_user_movie_sparse_matrix(
    ratings_df: Any, 
    num_users: int, 
    num_movies: int
) -> Any:
    """
    Construct a scipy CSR matrix from ratings DataFrame with columns [userId, movieId, rating].

    Returns:
        sparse.csr_matrix
    """
    from scipy import sparse

    row = ratings_df["userId"].values
    col = ratings_df["movieId"].values
    data = ratings_df["rating"].values
    return sparse.csr_matrix((data, (row, col)), shape=(num_users + 1, num_movies + 1))


def get_average_ratings(
    sparse_matrix: Any, 
    axis: int
) -> Dict[int, float]:
    """
    Compute average ratings for each row (axis=1) or each column (axis=0) in a sparse matrix.

    - axis=1 → average rating per user
    - axis=0 → average rating per movie

    Returns:
        Dict[int, float]: mapping index → average rating (only for indices with at least one rating)
    """
    summed = sparse_matrix.sum(axis=axis).A1
    nonzero_counts = (sparse_matrix != 0).sum(axis=axis).A1
    averages = {
        idx: (summed[idx] / nonzero_counts[idx])
        for idx in range(len(summed))
        if nonzero_counts[idx] != 0
    }
    return averages


def get_top_n_recommendations_from_preds(
    predictions: List[Any],
    n: int = 10
) -> Dict[int, List[Any]]:
    """
    Given a list of Surprise prediction tuples (uid, iid, true_r, est, _), group by user,
    sort each user’s predicted ratings in descending order, and return top-n for each user.

    Returns:
        Dict[int, List[(movieId, est_rating)]]
    """
    top_n: Dict[int, List[Any]] = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n
