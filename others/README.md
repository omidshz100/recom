# Movie Recommender System

A content-based movie recommendation system built with Python, scikit-learn's NMF algorithm, and Streamlit.

## Features

- **Dataset Loading**: Loads and displays MovieLens dataset (movies.csv and ratings.csv)
- **Content Vectorization**: Uses TF-IDF on movie genres to create content vectors
- **NMF Algorithm**: Employs scikit-learn's Non-negative Matrix Factorization to learn latent features
- **Interactive UI**: Clean and intuitive Streamlit interface
- **Movie Browsing**: Browse movies with search and genre filtering
- **Rating System**: Rate movies on a 0.5-5.0 scale
- **Dynamic Recommendations**: Get personalized recommendations that update based on your ratings

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run movie_recommender.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the application:
   - **Browse Movies**: Use the search bar and genre filter to find movies
   - **Rate Movies**: Select ratings from 0.5 to 5.0 stars for movies you've seen
   - **Get Recommendations**: Click "Get Recommendations" to see personalized movie suggestions
   - **View Dataset Info**: Check the sidebar for dataset statistics

## How It Works

1. **Content Features**: The system uses TF-IDF vectorization on movie genres to create content-based features
2. **NMF Training**: Non-negative Matrix Factorization learns latent features from the content data
3. **User Profile**: Your ratings create a user profile in the latent feature space
4. **Similarity Calculation**: Cosine similarity finds movies most similar to your preferences
5. **Recommendations**: Top-N most similar movies (excluding already rated ones) are recommended

## Dataset

This application uses the MovieLens dataset which includes:
- **movies.csv**: Movie information with titles and genres
- **ratings.csv**: User ratings data (sampled for performance)

## Technical Details

- **Framework**: Streamlit for the web interface
- **ML Library**: scikit-learn for TF-IDF and NMF
- **Data Processing**: pandas and numpy
- **Algorithm**: Content-based filtering with NMF dimensionality reduction
- **Similarity Metric**: Cosine similarity for recommendation generation

## Performance Notes

- The application samples the first 100,000 ratings for better performance
- TF-IDF is limited to 100 features to optimize processing speed
- NMF uses 20 components by default for latent feature learning

## Future Enhancements

- Add collaborative filtering for hybrid recommendations
- Include movie descriptions and tags for richer content features
- Implement user authentication and rating persistence
- Add movie posters and additional metadata
- Include explanation of why movies were recommended