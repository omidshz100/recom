# Dataset Setup Instructions

## MovieLens Dataset Download

This movie recommender system uses the MovieLens dataset, which is not included in this repository due to file size limitations (591 MB).

### Steps to Download and Setup:

1. **Download the MovieLens 25M Dataset:**
   - Visit: https://grouplens.org/datasets/movielens/
   - Download the "ml-25m.zip" file (size: ~265 MB)

2. **Extract the Dataset:**
   ```bash
   unzip ml-25m.zip
   mv ml-25m ml-latest
   ```

3. **Verify File Structure:**
   Your project directory should look like this:
   ```
   recom/
   ├── ml-latest/
   │   ├── movies.csv
   │   ├── ratings.csv
   │   ├── tags.csv
   │   ├── links.csv
   │   └── README.txt
   ├── movie_recommender.py
   ├── requirements.txt
   └── README.md
   ```

4. **Run the Application:**
   ```bash
   pip install -r requirements.txt
   streamlit run movie_recommender.py
   ```

### Dataset Information:
- **Movies:** ~62,000 movies
- **Ratings:** ~25 million ratings
- **Users:** ~162,000 users
- **Time Period:** 1995-2019

### Note:
The `ml-latest/` directory is included in `.gitignore` to prevent accidentally committing large files to the repository.

### Alternative: Smaller Dataset
If you prefer a smaller dataset for testing:
- Download "ml-latest-small.zip" instead (1 MB)
- Contains ~100,000 ratings and ~9,000 movies
- Follow the same extraction steps above