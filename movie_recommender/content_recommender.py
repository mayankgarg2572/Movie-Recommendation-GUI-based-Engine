# movie_recommender/content_recommender.py

import pandas as pd
import numpy as np
from typing import List, Optional

from movie_recommender.data_loader import load_movies_metadata, load_keywords, load_credits, load_links_small
from movie_recommender.utils import stem_and_clean_keywords, filter_terms_by_frequency, build_cosine_similarity_matrix, get_top_n_similar_indices


class ContentRecommender:
    """
    Content-based recommender: Given a movie title, returns top-n similar movies
    based on metadata (keywords, cast, director, genres, tagline, overview).
    """

    def __init__(self):
        # Load raw data
        self.movies_df = load_movies_metadata()
        self.keywords_df = load_keywords()
        self.credits_df = load_credits()
        self.links_df = load_links_small()
        # Merge DataFrames to get a single DataFrame 'smd' with relevant columns
        self._prepare_dataset()
        # Build the “soup” column and cosine similarity matrix
        self._build_soup_and_similarity()

    def _prepare_dataset(self) -> None:
        """
        Merge movies_df with credits_df and keywords_df, filter to only those TMDB IDs present in links_df.
        Create new columns: cast (list of top-3 names), crew (for director), keywords (list), genres (list),
        tagline (list of words), overview (list of words).
        """
        # Keep only movies where id ∈ links_df.tmdbId
        valid_tmdb_ids = set(self.links_df["tmdbId"])
        self.movies_df = self.movies_df[self.movies_df["id"].isin(valid_tmdb_ids)].copy().reset_index(drop=True)

        # Merge credits and keywords onto movies_df
        self.movies_df = pd.merge(self.movies_df, self.credits_df[["id", "cast", "crew"]], on="id", how="left")
        self.movies_df = pd.merge(self.movies_df, self.keywords_df[["id", "keywords"]], on="id", how="left")

        # Fill NaNs in tagline/overview, cast, crew, keywords
        for col in ["tagline", "overview", "cast", "crew", "keywords", "genres"]:
            # Some movies miss tagline/overview; set to empty string / list
            self.movies_df[col] = self.movies_df[col].fillna("[]")  # for 'cast', 'crew', 'keywords', 'genres' we parse next

        # Convert JSON-like strings to Python objects
        from ast import literal_eval

        def parse_json_list(x: str) -> list:
            try:
                return literal_eval(x)
            except Exception:
                return []

        # Parse columns that are stored as JSON lists
        for col in ["genres", "keywords", "cast", "crew"]:
            self.movies_df[col] = self.movies_df[col].apply(parse_json_list)

        # Extract director from crew
        def get_director_name(crew_list: list) -> str:
            for member in crew_list:
                if member.get("job") == "Director":
                    return member.get("name", "")
            return ""

        self.movies_df["director"] = self.movies_df["crew"].apply(get_director_name)

        # Keep only top-3 cast members (by order in the list), lowercase/no spaces
        def process_cast(cast_list: list) -> List[str]:
            names = [member.get("name", "") for member in cast_list]
            top3 = names[:3] if len(names) >= 3 else names
            return [name.replace(" ", "").lower() for name in top3]

        self.movies_df["cast_proc"] = self.movies_df["cast"].apply(process_cast)

        # Process keywords: keep only those with frequency > 1, then stem & lowercase
        self.movies_df["keywords_proc"] = self.movies_df["keywords"].apply(
            lambda kw_list: [kw.get("name", "") for kw in kw_list]
        )

        # Split tagline/overview into lists of words (if available)
        self.movies_df["tagline_proc"] = self.movies_df["tagline"].apply(lambda x: x.split() if isinstance(x, str) else [])
        self.movies_df["overview_proc"] = self.movies_df["overview"].apply(lambda x: x.split() if isinstance(x, str) else [])

        # Build frequency dictionaries for filtering keywords & overview
        all_keywords = list(self.movies_df["keywords_proc"])
        keyword_freq = filter_terms_by_frequency(all_keywords)
        all_overviews = list(self.movies_df["overview_proc"])
        overview_freq = filter_terms_by_frequency(all_overviews)

        # Final processing of keywords & overview: remove single-occurrence terms, stem, lowercase/no spaces
        def process_terms(term_list: List[str], freq_map: dict) -> List[str]:
            filtered = [term for term in term_list if term in freq_map]
            return [term.replace(" ", "").lower() for term in filtered]

        self.movies_df["keywords_proc"] = self.movies_df["keywords_proc"].apply(lambda lst: process_terms(lst, keyword_freq))
        self.movies_df["overview_proc"] = self.movies_df["overview_proc"].apply(lambda lst: process_terms(lst, overview_freq))

        # Process genres: list of genre names (lowercase, no spaces)
        def process_genres(genres_list: List[dict]) -> List[str]:
            names = [g.get("name", "") for g in genres_list]
            return [name.replace(" ", "").lower() for name in names]

        self.movies_df["genres_proc"] = self.movies_df["genres"].apply(process_genres)

    def _build_soup_and_similarity(self) -> None:
        """
        Create the 'soup' column (concatenate keywords, cast, director repeated 3 times, genres,
        tagline, overview), then build the cosine similarity matrix. Also create a mapping
        from movie title (lowercase) to DataFrame index.
        """
        # Create 'soup' (string) by combining all tokens
        def make_soup(row: pd.Series) -> str:
            # Director is given more weight by repeating thrice
            director_tokens = [row["director"].replace(" ", "").lower()] * 3
            all_tokens = (
                row["keywords_proc"]
                + row["cast_proc"]
                + director_tokens
                + row["genres_proc"]
                + row["tagline_proc"]
                + row["overview_proc"]
            )
            # Return a single-space-joined string
            return " ".join(all_tokens)

        self.movies_df["soup"] = self.movies_df.apply(make_soup, axis=1)

        # Build cosine similarity matrix of size [n_movies × n_movies]
        self.cosine_sim = build_cosine_similarity_matrix(self.movies_df["soup"].tolist())

        # Create title → index mapping (lowercase)
        self.title_to_index = pd.Series(
            self.movies_df.index, index=self.movies_df["title"].str.lower()
        )

    def get_similar_movies(self, title: str, top_n: int = 10) -> Optional[List[str]]:
        """
        Given a movie title, return the top_n most similar movie titles.

        Parameters:
            title (str): Movie title (case-insensitive).
            top_n (int): Number of similar movies to return.

        Returns:
            List[str] or None: Titles of the top_n similar movies, or None if title not found.
        """
        title_lower = title.lower()
        if title_lower not in self.title_to_index:
            return None
        idx = int(self.title_to_index[title_lower])
        top_indices = get_top_n_similar_indices(self.cosine_sim, idx, top_n=top_n)
        return self.movies_df["title"].iloc[top_indices].tolist()
