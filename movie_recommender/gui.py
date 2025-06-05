# movie_recommender/gui.py

import tkinter as tk
from tkinter import StringVar, Label, Button, LabelFrame, OptionMenu, Entry, Spinbox, W
import tkinter.messagebox as messagebox
import pandas as pd
from typing import Optional, List

from movie_recommender.basic_recommender import BasicRecommender
from movie_recommender.content_recommender import ContentRecommender
from movie_recommender.collab_recommender import SVDRecommender


class MovieRecommenderGUI:
    """
    A Tkinter-based GUI for three types of movie recommendations:
      1. Basic (IMDb weighted score overall & by genre)
      2. Content-based (given a movie title)
      3. Collaborative filtering (predict for a hypothetical or real user)
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Movie Recommender System")
        self.root.geometry("1000x400")

        # Instantiate recommender backends
        self.basic_rec = BasicRecommender()
        self.content_rec = ContentRecommender()
        # For collaborative, we assume user ID = 99999 for “new user” scenario with 5 anchor ratings
        self.collab_rec = SVDRecommender()

        # Shared state for label lists
        self.result_labels: List[Optional[Label]] = [None] * 10

        # Build menu and frames
        self._build_menu()
        self._build_basic_frame()
        self._build_content_frame()
        self._build_collab_frame()

        # Start by hiding all frames, then show basic by default
        self._hide_all_frames()
        self.basic_frame.grid(row=0, column=0)

    def _build_menu(self) -> None:
        main_menu = tk.Menu(self.root)
        main_menu.add_command(label="Basic", command=self._show_basic_frame)
        main_menu.add_command(label="Content Based", command=self._show_content_frame)
        main_menu.add_command(label="Collaborative", command=self._show_collab_frame)
        self.root.config(menu=main_menu)

    def _hide_all_frames(self) -> None:
        for frame in [self.basic_frame, self.content_frame, self.collab_frame]:
            frame.grid_forget()

    def _show_basic_frame(self) -> None:
        self._hide_all_frames()
        self.basic_frame.grid(row=0, column=0)

    def _show_content_frame(self) -> None:
        self._hide_all_frames()
        self.content_frame.grid(row=0, column=0)

    def _show_collab_frame(self) -> None:
        self._hide_all_frames()
        self.collab_frame.grid(row=0, column=0)

    def _build_basic_frame(self) -> None:
        # Frame for Basic Recommendation
        self.basic_frame = LabelFrame(self.root, width=1000, height=400)
        self.basic_frame.grid_propagate(False)  # Fix size

        # Title
        Label(self.basic_frame, text="Basic Recommendation (IMDb‐style)").grid(row=0, column=0, columnspan=2, pady=5)

        # Top-N Overall Button
        Button(
            self.basic_frame,
            text="Show Top 10 Movies Overall",
            bg="green",
            fg="white",
            command=self._basic_overall_action,
        ).grid(row=1, column=0, padx=10, pady=10, sticky=W)

        # Genre Dropdown
        Label(self.basic_frame, text="Select Genre:").grid(row=2, column=0, padx=10, sticky=W)
        self.genre_var = StringVar(self.basic_frame)
        self.genre_var.set("Select Genre")
        # Build option list from keys of genre_to_df
        genre_options = sorted(list(self.basic_rec.genre_to_df.keys()))
        OptionMenu(self.basic_frame, self.genre_var, *genre_options).grid(row=2, column=1, padx=10, sticky=W)

        # Genre Submit Button
        Button(
            self.basic_frame,
            text="Show Top 10 by Genre",
            bg="green",
            fg="white",
            command=self._basic_genre_action,
        ).grid(row=3, column=0, padx=10, pady=10, sticky=W)

    def _basic_overall_action(self) -> None:
        """
        Fetch top-10 movies overall and display them in labels.
        """
        # Clear previous result labels
        self._clear_result_labels()
        df_top = self.basic_rec.get_top_n_overall(n=10)
        for idx, row in df_top.iterrows():
            lbl = Label(self.basic_frame, text=f"{idx + 1}. {row['title']} (Score: {row['imdb_score']:.2f})")
            lbl.grid(row=5 + idx, column=0, columnspan=2, sticky=W, pady=2)
            self.result_labels[idx] = lbl

    def _basic_genre_action(self) -> None:
        """
        Fetch top-10 movies for selected genre and display.
        """
        self._clear_result_labels()
        selected_genre = self.genre_var.get()
        if selected_genre == "Select Genre":
            messagebox.showerror("Invalid Input", "Please select a valid genre.")
            return
        df_genre = self.basic_rec.get_top_n_by_genre(selected_genre, n=10)
        if df_genre is None or df_genre.empty:
            messagebox.showinfo("No Results", f"No movies found for genre '{selected_genre}'.")
            return
        for idx, row in df_genre.iterrows():
            lbl = Label(self.basic_frame, text=f"{idx + 1}. {row['title']} (Score: {row['imdb_score']:.2f})")
            lbl.grid(row=5 + idx, column=0, columnspan=2, sticky=W, pady=2)
            self.result_labels[idx] = lbl

    def _build_content_frame(self) -> None:
        # Frame for Content-Based Recommendation
        self.content_frame = LabelFrame(self.root, width=1000, height=400)
        self.content_frame.grid_propagate(False)

        Label(self.content_frame, text="Content-Based Recommendation").grid(row=0, column=0, columnspan=2, pady=5)
        Label(self.content_frame, text="Enter Movie Title:").grid(row=1, column=0, sticky=W, padx=10, pady=5)
        self.content_title_var = StringVar(self.content_frame)
        Entry(self.content_frame, textvariable=self.content_title_var, width=40).grid(
            row=1, column=1, padx=10, pady=5, sticky=W
        )
        Button(
            self.content_frame,
            text="Submit",
            bg="green",
            fg="white",
            command=self._content_action,
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky=W)

    def _content_action(self) -> None:
        """
        Given a movie title, fetch top-10 similar movies.
        """
        self._clear_result_labels()
        title = self.content_title_var.get().strip()
        if not title:
            messagebox.showerror("Invalid Input", "Please enter a movie title.")
            return
        similar = self.content_rec.get_similar_movies(title, top_n=10)
        if similar is None:
            messagebox.showinfo("Not Found", f"Movie '{title}' not found in our dataset.")
            return
        for idx, movie_title in enumerate(similar):
            lbl = Label(self.content_frame, text=f"{idx + 1}. {movie_title}")
            lbl.grid(row=4 + idx, column=0, columnspan=2, sticky=W, pady=2)
            self.result_labels[idx] = lbl

    def _build_collab_frame(self) -> None:
        # Frame for Collaborative Filtering
        self.collab_frame = LabelFrame(self.root, width=1000, height=400)
        self.collab_frame.grid_propagate(False)

        Label(self.collab_frame, text="Collaborative Filtering (SVD)").grid(row=0, column=0, columnspan=4, pady=5)
        # Provide five example “anchor” movies with known IDs (hard-coded IDs from original notebook)
        anchor_movies = [
            ("The Shawshank Redemption", 278),
            ("Forrest Gump", 13),
            ("Life Is Beautiful", 637),
            ("The Lord of the Rings: Return of the King", 122),
            ("Star Wars", 11),
        ]
        Label(self.collab_frame, text="Rate these movies (0–5):").grid(row=1, column=0, sticky=W, padx=10)

        # Create Spinboxes for ratings
        self.anchor_rating_vars: List[StringVar] = []
        for idx, (title, movie_id) in enumerate(anchor_movies):
            Label(self.collab_frame, text=f"{title}").grid(row=2 + idx, column=0, sticky=W, padx=10)
            var = StringVar(self.collab_frame)
            sb = Spinbox(self.collab_frame, from_=0, to=5, textvariable=var, width=5)
            sb.grid(row=2 + idx, column=1, padx=5)
            self.anchor_rating_vars.append(var)

        Button(
            self.collab_frame,
            text="Submit Ratings & Recommend",
            bg="green",
            fg="white",
            command=self._collab_action,
        ).grid(row=8, column=0, columnspan=2, padx=10, pady=10, sticky=W)

    def _collab_action(self) -> None:
        """
        Take the ratings from the five anchor movies, create a temporary
        DataFrame as if userId=99999 rated them, retrain the model on
        the combined data, then show top-10 recommendations.
        """
        self._clear_result_labels()
        try:
            ratings_input = [int(var.get()) for var in self.anchor_rating_vars]
            if any(r < 0 or r > 5 for r in ratings_input):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter integer ratings between 0 and 5.")
            return

        # Build a small DataFrame for userId=99999
        anchor_movie_ids = [278, 13, 637, 122, 11]
        df_new = pd.DataFrame(
            {
                "userId": [99999] * 5,
                "movieId": anchor_movie_ids,
                "rating": ratings_input,
            }
        )

        # Get original ratings and append new ratings
        original_ratings = self.collab_rec.ratings_df.copy()
        combined = pd.concat([original_ratings, df_new], ignore_index=True)

        # Retrain SVD on combined data
        from surprise import Dataset, Reader
        data = Dataset.load_from_df(combined[["userId", "movieId", "rating"]], self.collab_rec.reader)
        trainset = data.build_full_trainset()
        self.collab_rec.svd.fit(trainset)

        # Get recommendations for userId=99999
        recs = self.collab_rec.recommend_for_user(user_id=99999, top_n=10)
        for idx, (mid, est_rating, title) in enumerate(recs):
            lbl = Label(self.collab_frame, text=f"{idx + 1}. {title} (Pred: {est_rating:.2f})")
            lbl.grid(row=10 + idx, column=0, columnspan=2, sticky=W, pady=2)
            self.result_labels[idx] = lbl

    def _clear_result_labels(self) -> None:
        """
        Remove any existing result labels from all frames.
        """
        for lbl in self.result_labels:
            if lbl is not None:
                lbl.grid_forget()
        self.result_labels = [None] * 10


def launch_gui() -> None:
    """
    Convenience function to create the Tk root window and launch the GUI.
    """
    root = tk.Tk()
    app = MovieRecommenderGUI(root)
    root.mainloop()
