# Movie Recommendation GUI-Based Engine

## Overview

This project implements a comprehensive movie recommendation system with a graphical user interface (GUI). It integrates three distinct recommendation strategies:

1. **Basic Recommender** : Ranks movies by a weighted IMDb score and allows filtering by genre.
2. **Content-Based Recommender** : Computes cosine similarity between movies based on metadata (keywords, cast, director, genres, tagline, overview).
3. **Collaborative Filtering (SVD)** : Uses the [Surprise](https://surprise.readthedocs.io/) library’s SVD algorithm to predict user ratings and recommend top-N movies.

The system is organized into a modular Python package (`movie_recommender/`) plus a driver script (`run_recommender.py`). A Tkinter GUI (`movie_recommender/gui.py`) provides a user-friendly interface for all three recommendation methods.

---

## Directory Structure

```
movie-recommendation-gui-based-engine/
├── README.md
├── config.py
├── requirements.txt
├── run_recommender.py
└── movie_recommender/
    ├── basic_recommender.py
    ├── collab_recommender.py
    ├── content_recommender.py
    ├── data_loader.py
    ├── gui.py
    └── utils.py
```

* `README.md`: This file contains project description, setup instructions, and usage guide.
* `config.py`: Defines the `DATA_DIR` constant that points to the folder where all CSV datasets must reside.
* `requirements.txt`: Specifies all Python dependencies required to run the project.
* `run_recommender.py`:  Entry point: launches the Tkinter GUI for interacting with the recommendation engine.
* `movie_recommender/`: Core package containing modules for data loading, recommendation algorithms, utility functions, and the GUI implementation.

---

## Prerequisites

* **Python 3.7+**
* Operating System: platform‐agnostic (Windows/Linux/macOS)
* **Download the datasets** (CSV files) and place them into a directory named `__data__` at the project root.

  * You will need, at minimum:
    * `movies_metadata.csv`
    * `ratings_small.csv` (or `ratings.csv`)
    * `keywords.csv`
    * `credits.csv`
    * `links_small.csv`

  Data Folder Structure:

  ```
  __data__/
  ├── movies_metadata.csv
  ├── ratings_small.csv
  ├── keywords.csv
  ├── credits.csv
  └── links_small.csv
  ```

  * You can download the required CSV files from [Google Drive Link]().
* **Required Python Packages** (listed in `requirements.txt`):

  ```
  pandas
  numpy
  scikit-learn
  scikit-surprise
  xgboost
  nltk
  matplotlib
  plotly
  wordcloud
  seaborn
  tk
  scipy
  ```

  These packages handle data manipulation, machine learning algorithms, NLP preprocessing, and visualization.

---

## Installation

1. **Clone the Repository**

   ```
   git clone https://github.com/mayankgarg2572/movie-recommendation-gui-based-engine.git
   cd mayankgarg2572\movie-recommendation-gui-based-engine
   ```
2. **Create a Virtual Environment (Recommended)**

   ```**
   python3 -m venv venv
   venv/bin/activate                # Linux/macOS `<span>`
   venv\Scripts\activate.bat        # Windows `<span>`
   ```
3. **Install Dependencies**

   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Place Dataset CSVs**

   1. Download the CSV files from:

   ```
      __GDRIVE_LINK_HERE__
   ```

   2. Unzip or move all CSVs into a folder named `__data__` at the project root.

   ```
      movie-recommendation-gui-based-engine/
      └── __data__/
          ├── movies_metadata.csv
          ├── ratings_small.csv
          ├── keywords.csv
          ├── credits.csv
          └── links_small.csv
   ```
5. **Verify `config.py`**

   By default, `config.py` sets:

   ```python
   from pathlib import Path
   DATA_DIR = Path(__file__).parent.resolve() / "__data__"
   ```

   Ensure that the `__data__` directory (with CSV files) is located at the same level as `config.py`. If you choose another location, update `DATA_DIR` accordingly.

---

## Usage

Once dependencies are installed and CSV files are in place, you can launch the GUI:

```python
python run_recommender.py
```

This starts a Tkinter window titled **Movie Recommender System** with three tabs/menu options:

1. **Basic Recommendation (IMDb‐style)**:

   * **Show Top 10 Movies Overall** : Displays the ten highest weighted IMDb‐style scores across all movies.
   * **Show Top 10 by Genre** : Select a genre from the dropdown and click the button to see the top ten movies in that genre, ranked by weighted IMDb score.
2. **Content-Based Recommendation**:

   * **Enter Movie Title**: Input any movie title present in `movies_metadata.csv`.
   * **Submit** : Returns ten most similar movies based on metadata (“keywords”, “cast”, “director (×3)”, “genres”, “tagline”, and “overview”) via cosine similarity on a “soup” of tokens.
3. **Collaborative Filtering (SVD)**:

   * **Rate Five Anchor Movies** : Hard-coded anchor movies (e.g., “The Shawshank Redemption”, “Forrest Gump”, etc.) each have a Spinbox (0–5).
   * **Submit Ratings & Recommend** : Constructs a temporary ratings DataFrame for a fictitious user (userId = 99999), retrains the SVD model on combined data (original + anchors), and shows the top ten movie recommendations with predicted ratings.

### Tips

* If you enter an invalid genre or movie title that does not exist in the dataset, an error or “Not Found” message will appear.
* Ratings for collaborative filtering must be integers in the range 0–5; otherwise, an error box is shown.
* The GUI dynamically updates to show result labels in the same frame. To switch between recommendation modes, use the menu at the top of the window.

---

## Module Structure

Below is a concise description of each module within `movie_recommender/`.

### 1. `data_loader.py`

* **Purpose** : Load all required CSV files into pandas DataFrames.
* **Key Functions** :
  * `load_ratings(csv_filename: str = "ratings_small.csv") → pd.DataFrame`: Loads user‐movie rating triplets `[userId, movieId, rating]`.
  * `load_movies_metadata(csv_filename: str = "movies_metadata.csv") → pd.DataFrame`: Loads movie metadata (e.g., `id`, `genres`, `vote_average`, `vote_count`, `title`, etc.) and ensures `id` is `int`.
  * `load_keywords(csv_filename: str = "keywords.csv") → pd.DataFrame`
  * `load_credits(csv_filename: str = "credits.csv") → pd.DataFrame`
  * `load_links_small(csv_filename: str = "links_small.csv") → pd.DataFrame`: Loads TMDB/IMDb ID mappings for filtering in content‐based pipeline.
  * `load_all_data() → Tuple[pd.DataFrame, ...]`:  Convenience function to load all five CSVs at once.

### 2. `utils.py`

* **Purpose** : Provide shared utility functions for preprocessing, similarity computation, and recommendation logic.
* **Important Functions** :
  * `compute_imdb_score(vote_count: int, vote_average: float, C: float, m: float) → float`: Implements weighted IMDb rating:
  * `stem_and_clean_keywords(keywords_list: List[str]) → List[str]`: Uses `NLTK`’s `SnowballStemmer` to stem keywords, lowercase, and strip spaces.
  * `filter_terms_by_frequency(all_terms_list: List[List[str]]) → Dict[str, int]`: Builds a frequency dictionary across multiple lists and retains terms with frequency ≥ 2.
  * `build_cosine_similarity_matrix(text_series: List[str]) → np.ndarray`: Vectorizes text using `CountVectorizer` (unigrams + bigrams, English stop words) and returns the cosine similarity matrix.
  * `get_top_n_similar_indices(cosine_sim_matrix: np.ndarray, index: int, top_n: int = 10) → List[int]`: Given a row index, sorts similarity scores (excluding self) and returns top-N indices.
  * Additional functions for sparse matrix construction and average‐rating calculations (used for more advanced recommender variants).

### 3. `basic_recommender.py`

* **Class** : `BasicRecommender`
* **Goal** : Provide an IMDb-style ranking of movies, with optional genre filtering.
* **Workflow** :

1. Load full movie metadata via `load_movies_metadata()`.
2. Compute overall mean rating CC**C** and the vote count threshold mm**m** (e.g., 95th percentile).
3. Filter all movies with at least mm**m** votes and compute `imdb_score` for each.
4. Sort descending by `imdb_score` and store in `self.top_movies_df`.
5. Parse the `genres` column (JSON‐like strings) to build a mapping `genre_to_df: Dict[str, pd.DataFrame]` for quick retrieval of top movies by genre.

* **Public Methods** :
  * `get_top_n_overall(n: int = 10) → pd.DataFrame`: Returns a DataFrame of the top-N movies with columns `[id, title, imdb_score]`.
  * `get_top_n_by_genre(genre: str, n: int = 10) → Optional[pd.DataFrame]`: If the genre exists, returns its top-N movies; otherwise returns `None`.

### 4. `content_recommender.py`

* **Class** : `ContentRecommender`
* **Goal** : Recommend movies most similar to a given title based on textual metadata features.
* **Workflow** :

1. Load:
   * Movie metadata (`movies_metadata.csv`)
   * Keywords (`keywords.csv`)
   * Credits (`credits.csv`)
   * Links (`links_small.csv`)

     Filter to movies present in `links_small` (ensures TMDB ID alignment).
2. **_prepare_dataset():**
   * Merge `movies_df`, `keywords_df`, and `credits_df` on `id`.
   * Fill missing values for `tagline`, `overview`, `cast`, `crew`, `keywords`, and `genres`.
   * Parse JSON‐strings (lists) in columns: `genres`, `keywords`, `cast`, `crew`.
   * Extract the director’s name from `crew`.
   * Process top-3 cast members (remove spaces, lowercase).
   * Build frequency dictionaries for keywords & overview terms (via `filter_terms_by_frequency`).
   * Stem and clean keywords and overview tokens; remove singletons.
   * Process genres into lowercase, space‐less tokens.
3. **_build_soup_and_similarity():**
   * Create a “soup” for each movie by concatenating:
     * `keywords_proc`
     * `cast_proc`
     * `[director] * 3` (increase weight)
     * `genres_proc`
     * `tagline_proc`
     * `overview_proc`
   * Use `CountVectorizer(ngram_range=(1,2), stop_words='english')` on the “soup” column, then compute a cosine similarity matrix.
   * Build a mapping `title_to_index` (lowercase movie title → DataFrame index).
4. **get_similar_movies(title: str, top_n: int = 10) → Optional[List[str]]**
   * If `title.lower()` exists in `title_to_index`, retrieve indices of top-N most similar movies (excluding self) and return their titles; otherwise return `None`.

### 5. `collab_recommender.py`

* **Class** : `SVDRecommender`
* **Goal** : Perform collaborative filtering using the SVD algorithm (from the Surprise library) to predict unknown user ratings and recommend top-N movies.
* **Workflow** :

1. Load:
   * Ratings (`ratings_small.csv`) via `load_ratings()`
   * Movie metadata (`movies_metadata.csv`) → selected columns `[id, title]` for mapping.
   * Instantiate `Reader(rating_scale=(1,5))`.
2. **_train_model()** :
   * Convert `ratings_df[["userId", "movieId", "rating"]]` to a Surprise `Dataset`.
   * Split into train/test (default `test_size=0.2`) for optional evaluation (RMSE, MAE).
   * Train an `SVD(n_factors=100, n_epochs=20, biased=True, random_state=15)` model on the training set.
3. **recommend_for_user(user_id: int, top_n: int = 10) → List[Tuple[int, float, str]]**
   * Identify all unique movie IDs.
   * Determine which movies the user has already rated.
   * For each unrated movie, call `self.svd.predict(user_id, movie_id).est` to estimate the rating.
   * Sort predictions descending by estimated rating and pick the top-N.
   * Map movie IDs → titles (via `id_to_title`) and return a list of tuples `(movie_id, predicted_rating, title)`.

### 6. `gui.py`

* **Class** : `MovieRecommenderGUI`
* **Goal** : Provide a Tkinter GUI front-end that integrates the three recommender backends.
* **Key Components** :

1. **Initialization (`__init__`)** :

   * Create the main Tk root window (`1000×400`), set title.
   * Instantiate `BasicRecommender`, `ContentRecommender`, and `SVDRecommender`.
   * Prepare three frames (Basic, Content, Collaborative) and a top menu to switch between them.
2. **Basic Frame** :

   * Button: “Show Top 10 Movies Overall” → calls `_basic_overall_action()`.
   * Dropdown (`OptionMenu`) of available genres (obtained from `BasicRecommender.genre_to_df.keys()`).
   * Button: “Show Top 10 by Genre” → calls `_basic_genre_action()`.
   * Labels dynamically populate with `title (Score: X.XX)` for each result.
3. **Content Frame** :

   * Entry box for movie title → submit button calls `_content_action()`.
   * Displays top-10 similar titles (one label per title).
4. **Collaborative Frame** :

   * Displays five hard-coded anchor movies with Spinboxes (0–5) to rate them.

   Anchor list:

   ```python
   [
      ("The Shawshank Redemption", 278),
      ("Forrest Gump", 13),
      ("Life Is Beautiful", 637),
      ("The Lord of the Rings: Return of the King", 122),
      ("Star Wars", 11)
   ]
   ```

   * “Submit Ratings & Recommend” button → calls `_collab_action()`, which:
     1. Reads user input ratings.
     2. Constructs a temporary DataFrame for `userId=99999`.
     3. Appends it to original ratings, retrains the SVD model on the full combined dataset.
     4. Retrieves and displays top-10 recommendations for `userId=99999` with predicted ratings.
5. **Helper Methods** :

   * `_hide_all_frames()` / `_show_*_frame()`: Control which frame is visible.
   * `_clear_result_labels()`: Clears old labels before displaying new results.
6. **`launch_gui()`** : Convenience function to instantiate `MovieRecommenderGUI` and start the Tkinter main loop.

---

## Installation Details

1. **Python Version**

   Confirm you have Python 3.7 or later:

   ```python
   python --version
   ```
2. **Virtual Environment**

   It is strongly recommended to use a virtual environment (venv or conda) to isolate project dependencies:

   ```python3
   python3 -m venv venv
   source venv/bin/activate # Linux/macOS 
   venv\Scripts\activate.bat # Windows 
   ```
3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **NLTK Setup**

   * The `ContentRecommender` uses NLTK for stemming.
   * If you haven’t already downloaded the NLTK Snowball data, run a Python shell:
     ```python
     import  nltk
     nltk.download( "punkt" )
     nltk.download( "stopwords" )
     ```
   * Alternatively, ensure that the SnowballStemmer is available in your NLTK data.
5. **Downloading CSV Files**

   * Create a folder named `__data__` in the project root.
   * Download and place the following files into `__data__/`:
     * `movies_metadata.csv`
     * `ratings_small.csv` (or `ratings.csv`)
     * `keywords.csv`
     * `credits.csv`
     * `links_small.csv`

---

## Extending the Project

* **Add More Data**

  * You can point `config.py` to a larger dataset (e.g., full `ratings.csv` instead of `ratings_small.csv`) by simply placing it in `__data__` and renaming accordingly.
* **Adjust Quantile in Basic Recommender**

  * By default, `BasicRecommender` uses the 95th percentile (`vote_count_quantile=0.95`) to define a minimum threshold mm**m**. Change this in `basic_recommender.py` or when instantiating:
    basic = BasicRecommender(vote_count_quantile=0.90)
* **Fine-Tune SVD Parameters**

  * In `SVDRecommender.__init__`, parameters such as `n_factors`, `n_epochs`, and `random_state` can be modified for improved performance or faster training.
* **Add Visualization**

  * Modules such as `matplotlib`, `plotly`, or `seaborn` are already listed in `requirements.txt`—use them to create interactive charts (e.g., rating distributions, similarity heatmaps) either in the GUI or as standalone scripts.
* **Integrate a Web Front-End**

  * The core recommender classes (`BasicRecommender`, `ContentRecommender`, `SVDRecommender`) are decoupled from the GUI. You can wrap them in a Flask or FastAPI application to expose a RESTful API and build a web interface.

---

## Dependencies & Versions

| Package             | Minimum Version       | Purpose                                   |
| ------------------- | --------------------- | ----------------------------------------- |
| `pandas`          | 1.0                   | DataFrame manipulation, CSV I/O           |
| `numpy`           | 1.18                  | Array operations, numerical computations  |
| `scikit-learn`    | 0.24                  | `CountVectorizer`,`cosine_similarity` |
| `scikit-surprise` | 1.1.1                 | SVD collaborative filtering               |
| `xgboost`         | 1.5                   | (Optional) future model expansion         |
| `nltk`            | 3.5                   | Snowball stemmer, tokenization            |
| `matplotlib`      | 3.3                   | (Optional) plotting                       |
| `plotly`          | 4.14                  | (Optional) interactive visualizations     |
| `wordcloud`       | 1.8                   | (Optional) word cloud generation          |
| `seaborn`         | 0.11                  | (Optional) statistical plotting           |
| `tk`              | Built‐in with Python | GUI toolkit                               |
| `scipy`           | 1.4                   | Sparse matrix utilities                   |

Ensure you install exactly (or later) these versions to guarantee compatibility. You can enforce versions by modifying `requirements.txt` accordingly.

---

## Troubleshooting

1. **`FileNotFoundError` for CSV files**

   * Verify that all required CSVs exist in `__data__` and that `config.py` is pointing to the correct directory.
   * Filenames must match exactly (including case).
2. **NLTK Stemmer Issues**

   * Checkout the above `nltk` setup as mentioned in the installation guide.
3. **Surprise Library Import Errors**

   * Ensure you install `scikit-surprise` via `pip install scikit-surprise`. On some systems, you may need additional build tools (e.g., a C++ compiler).
4. **GUI Doesn’t Appear / Blank Window**

   * Make sure no other Tkinter instance is blocking the main thread.
   * Run `python run_recommender.py` from a standard terminal, not inside certain IDE consoles that may not support GUI loops.
5. **Performance & Memory**

   * Content-based similarity computation can be memory-intensive if your dataset is large. Consider reducing the dataset size or using a more efficient vectorization method (e.g., TF-IDF, truncated SVD).
   * Training SVD on a large ratings file may take time—consider downsampling or experimenting with fewer `n_factors`/`n_epochs`.

---

## Acknowledgements

* **IMDb Weighted Rating Formula** : Inspired by [IMDb Top 250 methodology](https://help.imdb.com/article/imdb/list-records/letterboxd-and-imdb-top-charts/GHAGPQGD1Y3Q5XE6).
* **Surprise Library** : Collaborative filtering algorithm implementations.
* **TMDB Dataset** : The MovieLens/TMDB datasets used in this project.
* **Tkinter** : Standard Python library for creating GUIs.
* **NLTK** : Natural Language Toolkit for text processing and stemming.

---

## License

This project is released under the [MIT License](). Feel free to use, modify, and distribute.
