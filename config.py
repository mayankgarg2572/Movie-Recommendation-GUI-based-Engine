# config.py

"""
Configuration file for the movie_recommender package.
Adjust the DATA_DIR constant to point to where your CSV datasets reside.
"""

from pathlib import Path

# Absolute or relative path to the folder containing all CSV data files
DATA_DIR = Path(__file__).parent.resolve() / "__data__"

# Example:
# If the project root is /home/user/movie_recommender/, then DATA_DIR resolves to
# /home/user/movie_recommender/__data__
