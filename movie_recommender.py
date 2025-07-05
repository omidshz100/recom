import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Set page config
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess the MovieLens dataset"""
    try:
        # Load movies and ratings data
        movies = pd.read_csv('ml-latest/movies.csv')
        ratings = pd.read_csv('ml-latest/ratings.csv')
        
        # Clean and preprocess movies data
        movies['genres'] = movies['genres'].fillna('Unknown')
        movies['genres_processed'] = movies['genres'].str.replace('|', ' ')
        
        # Sample ratings for performance (use first 100k ratings)
        ratings_sample = ratings.head(100000)
        
        return movies, ratings_sample
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def create_content_features(_movies):
    """Create content-based features using TF-IDF on genres"""
    # Use TF-IDF on genres
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(_movies['genres_processed'])
    
    return tfidf_matrix, tfidf

def train_nmf_model(tfidf_matrix, n_components=20):
    """Train NMF model on content features"""
    nmf = NMF(n_components=n_components, random_state=42, max_iter=100)
    nmf_features = nmf.fit_transform(tfidf_matrix)
    
    return nmf, nmf_features

def get_recommendations(user_ratings, movies, nmf_features, top_n=10):
    """Generate recommendations based on user ratings and NMF features"""
    if not user_ratings:
        return []
    
    # Create user profile based on rated movies
    user_profile = np.zeros(nmf_features.shape[1])
    total_weight = 0
    user_avg_rating = np.mean(list(user_ratings.values()))
    
    for movie_id, rating in user_ratings.items():
        movie_idx = movies[movies['movieId'] == movie_id].index
        if len(movie_idx) > 0:
            movie_idx = movie_idx[0]
            weight = rating / 5.0  # Normalize rating to 0-1
            user_profile += weight * nmf_features[movie_idx]
            total_weight += weight
    
    if total_weight > 0:
        user_profile /= total_weight
    
    # Calculate similarity with all movies
    similarities = cosine_similarity([user_profile], nmf_features)[0]
    
    # Get movie indices sorted by similarity
    movie_indices = np.argsort(similarities)[::-1]
    
    # Filter out already rated movies
    rated_movie_ids = set(user_ratings.keys())
    recommendations = []
    
    for idx in movie_indices:
        movie_id = movies.iloc[idx]['movieId']
        if movie_id not in rated_movie_ids and len(recommendations) < top_n:
            # Predict rating based on similarity and user's average rating
            similarity_score = similarities[idx]
            # Scale similarity to rating range: use user's average as baseline
            # Higher similarity = higher predicted rating
            predicted_rating = user_avg_rating + (similarity_score - 0.5) * 2
            predicted_rating = max(0.5, min(5.0, predicted_rating))  # Clamp to valid range
            
            recommendations.append({
                'movieId': movie_id,
                'title': movies.iloc[idx]['title'],
                'genres': movies.iloc[idx]['genres'],
                'similarity_score': similarity_score,
                'predicted_rating': round(predicted_rating, 1)
            })
    
    return recommendations

def create_dataset_analytics(movies, ratings):
    """Create comprehensive dataset analytics and visualizations"""
    analytics = {}
    
    # Basic statistics
    analytics['total_movies'] = len(movies)
    analytics['total_ratings'] = len(ratings)
    analytics['unique_users'] = ratings['userId'].nunique()
    analytics['avg_ratings_per_user'] = ratings.groupby('userId').size().mean()
    analytics['avg_ratings_per_movie'] = ratings.groupby('movieId').size().mean()
    
    # Genre analysis
    all_genres = []
    for genres in movies['genres']:
        if pd.notna(genres) and genres != '(no genres listed)':
            all_genres.extend(genres.split('|'))
    
    genre_counts = pd.Series(all_genres).value_counts()
    analytics['genre_distribution'] = genre_counts
    
    # Rating distribution
    analytics['rating_distribution'] = ratings['rating'].value_counts().sort_index()
    analytics['avg_rating'] = ratings['rating'].mean()
    analytics['rating_std'] = ratings['rating'].std()
    
    # Movie popularity (number of ratings)
    movie_popularity = ratings.groupby('movieId').size().reset_index(name='num_ratings')
    movie_popularity = movie_popularity.merge(movies[['movieId', 'title']], on='movieId')
    analytics['top_movies'] = movie_popularity.nlargest(10, 'num_ratings')
    
    # User activity distribution
    user_activity = ratings.groupby('userId').size()
    analytics['user_activity_stats'] = {
        'mean': user_activity.mean(),
        'median': user_activity.median(),
        'std': user_activity.std(),
        'min': user_activity.min(),
        'max': user_activity.max()
    }
    
    return analytics

def create_nmf_analysis(nmf_features, movies):
    """Analyze NMF components and create visualizations"""
    analysis = {}
    
    # Component statistics
    analysis['n_components'] = nmf_features.shape[1]
    analysis['component_means'] = nmf_features.mean(axis=0)
    analysis['component_stds'] = nmf_features.std(axis=0)
    
    # Feature importance per component
    analysis['feature_variance'] = np.var(nmf_features, axis=0)
    analysis['total_variance_explained'] = np.sum(analysis['feature_variance'])
    
    return analysis

def plot_genre_distribution(genre_counts):
    """Create genre distribution visualization"""
    fig = px.bar(
        x=genre_counts.head(15).values,
        y=genre_counts.head(15).index,
        orientation='h',
        title='Top 15 Movie Genres Distribution',
        labels={'x': 'Number of Movies', 'y': 'Genre'},
        color=genre_counts.head(15).values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500, showlegend=False)
    return fig

def plot_rating_distribution(rating_dist):
    """Create rating distribution visualization"""
    fig = px.bar(
        x=rating_dist.index,
        y=rating_dist.values,
        title='Rating Distribution Across All Movies',
        labels={'x': 'Rating', 'y': 'Number of Ratings'},
        color=rating_dist.values,
        color_continuous_scale='blues'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_nmf_components(nmf_features):
    """Visualize NMF component analysis"""
    # Component variance
    component_variance = np.var(nmf_features, axis=0)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Component Variance', 'Component Distribution', 
                       'Feature Correlation Heatmap', 'Component Statistics'),
        specs=[[{"type": "bar"}, {"type": "box"}],
               [{"type": "heatmap", "colspan": 2}, None]]
    )
    
    # Component variance bar plot
    fig.add_trace(
        go.Bar(x=list(range(len(component_variance))), y=component_variance,
               name='Variance', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Component distribution box plot
    for i in range(min(5, nmf_features.shape[1])):
        fig.add_trace(
            go.Box(y=nmf_features[:, i], name=f'Comp {i+1}'),
            row=1, col=2
        )
    
    # Correlation heatmap
    if nmf_features.shape[1] <= 20:  # Only for reasonable number of components
        corr_matrix = np.corrcoef(nmf_features.T)
        fig.add_trace(
            go.Heatmap(z=corr_matrix, colorscale='RdBu', zmid=0),
            row=2, col=1
        )
    
    fig.update_layout(height=800, title_text="NMF Model Analysis")
    return fig

def plot_user_behavior(user_ratings, analytics):
    """Analyze and visualize user rating behavior"""
    if not user_ratings:
        return None
    
    user_df = pd.DataFrame(list(user_ratings.items()), columns=['movieId', 'rating'])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Your Rating Distribution', 'Your vs Average Ratings')
    )
    
    # User rating distribution
    rating_counts = user_df['rating'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=rating_counts.index, y=rating_counts.values, 
               name='Your Ratings', marker_color='lightgreen'),
        row=1, col=1
    )
    
    # Comparison with average
    avg_rating = analytics['avg_rating']
    user_avg = user_df['rating'].mean()
    
    fig.add_trace(
        go.Bar(x=['Dataset Average', 'Your Average'], 
               y=[avg_rating, user_avg],
               name='Rating Comparison', 
               marker_color=['lightcoral', 'lightgreen']),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Your Rating Behavior Analysis")
    return fig

def main():
    st.title("🎬 Movie Recommender System")
    st.markdown("### Content-Based Recommendations using NMF")
    
    # Load data
    with st.spinner("Loading MovieLens dataset..."):
        movies, ratings = load_data()
    
    if movies is None or ratings is None:
        st.error("Failed to load data. Please check if the ml-latest folder exists.")
        return
    
    # Create content features and train NMF
    with st.spinner("Training NMF model..."):
        tfidf_matrix, tfidf = create_content_features(movies)
        nmf, nmf_features = train_nmf_model(tfidf_matrix)
    
    # Create analytics
    with st.spinner("Generating analytics..."):
        analytics = create_dataset_analytics(movies, ratings)
        nmf_analysis = create_nmf_analysis(nmf_features, movies)
    
    # Initialize session state for user ratings
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.selectbox("Choose a page:", [
            "🎭 Browse & Rate Movies", 
            "⭐ Your Recommendations", 
            "📊 Dataset Analytics",
            "🔬 Model Analysis"
        ])
        
        st.header("📊 Dataset Info")
        st.metric("Total Movies", len(movies))
        st.metric("Total Ratings", len(ratings))
        st.metric("Unique Users", ratings['userId'].nunique())
        st.metric("Your Ratings", len(st.session_state.user_ratings))
        
        if st.button("🔄 Clear All Ratings"):
            st.session_state.user_ratings = {}
            st.rerun()
    
    if page == "🎭 Browse & Rate Movies":
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("🎭 Browse & Rate Movies")
            
            # Search and filter options
            search_term = st.text_input("🔍 Search movies by title:")
            genre_filter = st.selectbox(
                "Filter by genre:",
                ['All'] + sorted(set([genre for genres in movies['genres'] for genre in genres.split('|') if genre != '(no genres listed)']))
            )
            
            # Filter movies based on search and genre
            filtered_movies = movies.copy()
            
            if search_term:
                filtered_movies = filtered_movies[filtered_movies['title'].str.contains(search_term, case=False, na=False)]
            
            if genre_filter != 'All':
                filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(genre_filter, na=False)]
            
            # Display movies with rating interface
            st.subheader(f"Movies ({len(filtered_movies)} found)")
            
            # Pagination
            movies_per_page = 10
            total_pages = (len(filtered_movies) - 1) // movies_per_page + 1
            page_num = st.selectbox("Page:", range(1, total_pages + 1))
            
            start_idx = (page_num - 1) * movies_per_page
            end_idx = start_idx + movies_per_page
            page_movies = filtered_movies.iloc[start_idx:end_idx]
            
            for _, movie in page_movies.iterrows():
                with st.container():
                    col_title, col_rating = st.columns([3, 1])
                    
                    with col_title:
                        st.write(f"**{movie['title']}**")
                        st.write(f"*Genres: {movie['genres']}*")
                    
                    with col_rating:
                        current_rating = st.session_state.user_ratings.get(movie['movieId'], 0.0)
                        new_rating = st.selectbox(
                            "Rate:",
                            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                            index=int(current_rating * 2),
                            key=f"rating_{movie['movieId']}"
                        )
                        
                        if new_rating != current_rating:
                            if new_rating > 0:
                                st.session_state.user_ratings[movie['movieId']] = new_rating
                            elif movie['movieId'] in st.session_state.user_ratings:
                                del st.session_state.user_ratings[movie['movieId']]
                            st.rerun()
                    
                    st.divider()
        
        with col2:
             st.header("⭐ Your Current Ratings")
             
             # Display current ratings
             if st.session_state.user_ratings:
                 st.subheader("Your Current Ratings:")
                 for movie_id, rating in st.session_state.user_ratings.items():
                     movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
                     st.write(f"• {movie_title}: {'⭐' * int(rating)} ({rating}/5.0)")
                 
                 st.divider()
                 
                 # Add recommendation button and display
                 st.subheader("🎯 Get Recommendations")
                 if st.button("🎬 Get Movie Recommendations", type="primary", use_container_width=True):
                     with st.spinner("Generating personalized recommendations..."):
                         recommendations = get_recommendations(
                             st.session_state.user_ratings, 
                             movies, 
                             nmf_features
                         )
                     
                     if recommendations:
                         st.success("🌟 Recommended Movies for You:")
                         for i, rec in enumerate(recommendations[:5], 1):
                             with st.container():
                                 st.write(f"**{i}. {rec['title']}**")
                                 st.write(f"*{rec['genres']}*")
                                 col_rating, col_sim = st.columns(2)
                                 with col_rating:
                                     st.write(f"⭐ Predicted: {rec['predicted_rating']:.1f}/5.0")
                                 with col_sim:
                                     st.write(f"📊 Similarity: {rec['similarity_score']:.3f}")
                                 st.write("---")
                         
                         st.info("💡 See more recommendations in the 'Your Recommendations' page!")
                     else:
                         st.warning("No recommendations found. Try rating more movies!")
             else:
                 st.info("👆 Start by rating some movies to get personalized recommendations!")
                 
                 # Show some popular movies to get started
                 st.subheader("🔥 Popular Movies to Get Started:")
                 popular_movies = movies.head(10)
                 for _, movie in popular_movies.iterrows():
                     st.write(f"• **{movie['title']}** - *{movie['genres']}*")
    
    elif page == "⭐ Your Recommendations":
        st.header("⭐ Your Personalized Recommendations & Analytics")
        
        if not st.session_state.user_ratings:
            st.warning("Please rate some movies first to get recommendations!")
            st.info("Go to 'Browse & Rate Movies' to start rating.")
            
            # Show dataset overview even without ratings
            st.subheader("📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Movies", f"{analytics['total_movies']:,}")
            with col2:
                st.metric("Total Ratings", f"{analytics['total_ratings']:,}")
            with col3:
                st.metric("Unique Users", f"{analytics['unique_users']:,}")
            with col4:
                st.metric("Avg Rating", f"{analytics['avg_rating']:.2f}")
        else:
            # Create tabs for organized view
            tab1, tab2, tab3 = st.tabs(["🎯 Your Recommendations", "📈 Your Analytics", "📊 Dataset Stats"])
            
            with tab1:
                st.success(f"Based on your {len(st.session_state.user_ratings)} ratings:")
                
                # Your current ratings summary
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("📝 Your Current Ratings")
                    ratings_df = pd.DataFrame([
                        {
                            'Movie': movies[movies['movieId'] == movie_id]['title'].iloc[0],
                            'Your Rating': f"{'⭐' * int(rating)} ({rating}/5.0)",
                            'Genres': movies[movies['movieId'] == movie_id]['genres'].iloc[0]
                        }
                        for movie_id, rating in st.session_state.user_ratings.items()
                    ])
                    st.dataframe(ratings_df, use_container_width=True, hide_index=True)
                
                with col2:
                    user_avg = sum(st.session_state.user_ratings.values()) / len(st.session_state.user_ratings)
                    st.metric("Your Avg Rating", f"{user_avg:.2f}")
                    st.metric("Movies Rated", len(st.session_state.user_ratings))
                    st.metric("vs Dataset Avg", f"{user_avg - analytics['avg_rating']:+.2f}")
                
                st.divider()
                
                # Generate recommendations
                if st.button("🎯 Get Recommendations", type="primary"):
                    with st.spinner("Generating recommendations..."):
                        recommendations = get_recommendations(
                            st.session_state.user_ratings, 
                            movies, 
                            nmf_features, 
                            top_n=10
                        )
                    
                    if recommendations:
                        st.subheader("🎬 Recommended Movies for You:")
                        for i, rec in enumerate(recommendations, 1):
                            with st.container():
                                col_movie, col_rating = st.columns([3, 1])
                                
                                with col_movie:
                                    st.write(f"**{i}. {rec['title']}**")
                                    st.write(f"*Genres: {rec['genres']}*")
                                    st.write(f"*Similarity Score: {rec['similarity_score']:.3f}*")
                                
                                with col_rating:
                                    st.metric(
                                        "Predicted Rating",
                                        f"{rec['predicted_rating']}/5.0",
                                        delta=None
                                    )
                                    st.write(f"{'⭐' * int(rec['predicted_rating'])}")
                                
                                st.divider()
                    else:
                        st.warning("No recommendations found. Try rating more movies!")
            
            with tab2:
                st.subheader("📈 Your Rating Behavior Analysis")
                
                # User behavior analysis
                user_behavior_fig = plot_user_behavior(st.session_state.user_ratings, analytics)
                if user_behavior_fig:
                    st.plotly_chart(user_behavior_fig, use_container_width=True)
                    
                    # Analysis text
                    user_avg = sum(st.session_state.user_ratings.values()) / len(st.session_state.user_ratings)
                    dataset_avg = analytics['avg_rating']
                    rating_diff = user_avg - dataset_avg
                    
                    st.markdown(f"""
                    **📊 Analysis:** Your average rating of {user_avg:.2f} is {'**higher**' if rating_diff > 0 else '**lower**' if rating_diff < 0 else 'equal to'} the dataset average of {dataset_avg:.2f}. 
                    {'You tend to be more generous with ratings than typical users.' if rating_diff > 0.3 else 'You tend to be more critical than typical users.' if rating_diff < -0.3 else 'Your rating behavior aligns closely with the general user base.'}
                    This suggests {'you enjoy most movies you watch' if user_avg > 3.5 else 'you are selective about the movies you rate highly' if user_avg < 3.0 else 'you have balanced rating preferences'}.
                    """)
                
                # Detailed user statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🎭 Your Genre Preferences")
                    user_genres = []
                    for movie_id in st.session_state.user_ratings.keys():
                        movie_genres = movies[movies['movieId'] == movie_id]['genres'].iloc[0]
                        if pd.notna(movie_genres) and movie_genres != '(no genres listed)':
                            user_genres.extend(movie_genres.split('|'))
                    
                    if user_genres:
                        user_genre_counts = pd.Series(user_genres).value_counts().head(10)
                        genre_pref_fig = px.bar(
                            x=user_genre_counts.values,
                            y=user_genre_counts.index,
                            orientation='h',
                            title='Your Top Genres by Movies Rated',
                            labels={'x': 'Movies Rated', 'y': 'Genre'}
                        )
                        st.plotly_chart(genre_pref_fig, use_container_width=True)
                        
                        # Analysis text for genre preferences
                        top_genre = user_genre_counts.index[0]
                        top_count = user_genre_counts.iloc[0]
                        total_rated = len(st.session_state.user_ratings)
                        genre_percentage = (top_count / total_rated) * 100
                        
                        st.markdown(f"""
                        **🎭 Analysis:** Your favorite genre is **{top_genre}** ({top_count} movies, {genre_percentage:.1f}% of your ratings). 
                        {'You have a strong preference for this genre.' if genre_percentage > 40 else 'You enjoy diverse genres with a slight preference for this one.' if genre_percentage > 25 else 'You have very diverse taste across multiple genres.'}
                        {f'You\'ve also shown interest in {user_genre_counts.index[1]} and {user_genre_counts.index[2]}.' if len(user_genre_counts) > 2 else ''}
                        """)
                
                with col2:
                    st.subheader("📊 Rating Distribution")
                    user_ratings_series = pd.Series(list(st.session_state.user_ratings.values()))
                    rating_dist_fig = px.histogram(
                        x=user_ratings_series,
                        nbins=10,
                        title='Your Rating Distribution',
                        labels={'x': 'Rating', 'y': 'Count'}
                    )
                    st.plotly_chart(rating_dist_fig, use_container_width=True)
                    
                    # Analysis text for rating distribution
                    most_common_rating = user_ratings_series.mode().iloc[0]
                    rating_counts = user_ratings_series.value_counts()
                    high_ratings = len(user_ratings_series[user_ratings_series >= 4])
                    low_ratings = len(user_ratings_series[user_ratings_series <= 2])
                    
                    st.markdown(f"""
                    **📊 Analysis:** You most frequently give **{most_common_rating}** star ratings. 
                    {f'You give high ratings (4-5 stars) to {high_ratings}/{len(user_ratings_series)} movies ({(high_ratings/len(user_ratings_series)*100):.1f}%).' if high_ratings > 0 else ''}
                    {'You tend to rate movies you enjoy highly.' if high_ratings > len(user_ratings_series)*0.6 else 'You have a balanced rating approach.' if high_ratings > len(user_ratings_series)*0.3 else 'You are quite selective with high ratings.'}
                    """)
            
            with tab3:
                st.subheader("📊 Dataset Statistics")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Movies", f"{analytics['total_movies']:,}")
                with col2:
                    st.metric("Total Ratings", f"{analytics['total_ratings']:,}")
                with col3:
                    st.metric("Unique Users", f"{analytics['unique_users']:,}")
                with col4:
                    st.metric("Avg Rating", f"{analytics['avg_rating']:.2f}")
                
                # Quick visualizations
                col1, col2 = st.columns(2)
                with col1:
                    # Top genres
                    top_genres_fig = px.bar(
                        x=analytics['genre_distribution'].head(10).values,
                        y=analytics['genre_distribution'].head(10).index,
                        orientation='h',
                        title='Top 10 Movie Genres',
                        labels={'x': 'Number of Movies', 'y': 'Genre'}
                    )
                    st.plotly_chart(top_genres_fig, use_container_width=True)
                    
                    # Analysis for genre distribution
                    top_dataset_genre = analytics['genre_distribution'].index[0]
                    top_genre_count = analytics['genre_distribution'].iloc[0]
                    total_movies = analytics['total_movies']
                    
                    st.markdown(f"""
                    **🎭 Analysis:** **{top_dataset_genre}** dominates the dataset with {top_genre_count:,} movies ({(top_genre_count/total_movies*100):.1f}% of all movies). 
                    The top 3 genres represent a significant portion of the catalog, showing the dataset's focus on mainstream entertainment genres.
                    """)
                
                with col2:
                    # Rating distribution
                    rating_dist_fig = px.bar(
                        x=analytics['rating_distribution'].index,
                        y=analytics['rating_distribution'].values,
                        title='Overall Rating Distribution',
                        labels={'x': 'Rating', 'y': 'Number of Ratings'}
                    )
                    st.plotly_chart(rating_dist_fig, use_container_width=True)
                    
                    # Analysis for rating distribution
                    most_common_dataset_rating = analytics['rating_distribution'].idxmax()
                    high_ratings_dataset = analytics['rating_distribution'][analytics['rating_distribution'].index >= 4].sum()
                    total_ratings = analytics['total_ratings']
                    
                    st.markdown(f"""
                    **⭐ Analysis:** Users most commonly give **{most_common_dataset_rating}** star ratings. 
                    {(high_ratings_dataset/total_ratings*100):.1f}% of all ratings are 4+ stars, indicating users generally rate movies they choose to watch quite positively. 
                    The average rating of {analytics['avg_rating']:.2f} suggests a positive bias in the dataset.
                    """)
                
                # Top movies table
                st.subheader("🏆 Most Popular Movies")
                st.dataframe(analytics['top_movies'][['title', 'num_ratings']].head(15), use_container_width=True, hide_index=True)
    
    elif page == "📊 Dataset Analytics":
        st.header("📊 Dataset Analytics & Statistics")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Movies", f"{analytics['total_movies']:,}")
        with col2:
            st.metric("Total Ratings", f"{analytics['total_ratings']:,}")
        with col3:
            st.metric("Unique Users", f"{analytics['unique_users']:,}")
        with col4:
            st.metric("Avg Rating", f"{analytics['avg_rating']:.2f}")
        
        # Genre distribution
        st.subheader("🎭 Genre Distribution")
        genre_fig = plot_genre_distribution(analytics['genre_distribution'])
        st.plotly_chart(genre_fig, use_container_width=True)
        
        # Analysis for main genre chart
        top_3_genres = analytics['genre_distribution'].head(3)
        total_movies = analytics['total_movies']
        top_3_percentage = (top_3_genres.sum() / total_movies) * 100
        
        st.markdown(f"""
        **📊 Genre Analysis:** The top 3 genres (**{', '.join(top_3_genres.index)}**) account for {top_3_percentage:.1f}% of all movies in the dataset. 
        **{top_3_genres.index[0]}** leads with {top_3_genres.iloc[0]:,} movies, followed by **{top_3_genres.index[1]}** ({top_3_genres.iloc[1]:,}) and **{top_3_genres.index[2]}** ({top_3_genres.iloc[2]:,}). 
        This distribution reflects the commercial movie industry's focus on popular, mainstream genres that appeal to broad audiences.
        """)
        
        # Rating distribution
        st.subheader("⭐ Rating Distribution")
        rating_fig = plot_rating_distribution(analytics['rating_distribution'])
        st.plotly_chart(rating_fig, use_container_width=True)
        
        # Analysis for main rating chart
        most_common_rating = analytics['rating_distribution'].idxmax()
        high_ratings_count = analytics['rating_distribution'][analytics['rating_distribution'].index >= 4].sum()
        low_ratings_count = analytics['rating_distribution'][analytics['rating_distribution'].index <= 2].sum()
        total_ratings = analytics['total_ratings']
        
        st.markdown(f"""
        **⭐ Rating Analysis:** The most common rating is **{most_common_rating} stars** with {analytics['rating_distribution'][most_common_rating]:,} ratings. 
        **{(high_ratings_count/total_ratings*100):.1f}%** of all ratings are 4+ stars, while only **{(low_ratings_count/total_ratings*100):.1f}%** are 2 stars or below. 
        This positive skew (average: {analytics['avg_rating']:.2f}) suggests users tend to rate movies they choose to watch favorably, indicating a selection bias where people generally watch movies they expect to enjoy.
        """)
        
        # Top movies
        st.subheader("🏆 Most Rated Movies")
        st.dataframe(analytics['top_movies'][['title', 'num_ratings']], use_container_width=True)
        
        # User activity statistics
        st.subheader("👥 User Activity Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Ratings/User", f"{analytics['user_activity_stats']['mean']:.1f}")
        with col2:
            st.metric("Median Ratings/User", f"{analytics['user_activity_stats']['median']:.1f}")
        with col3:
            st.metric("Max Ratings/User", f"{analytics['user_activity_stats']['max']:,}")
    
    elif page == "🔬 Model Analysis":
        st.header("🔬 NMF Model Analysis")
        
        # Model metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("NMF Components", nmf_analysis['n_components'])
        with col2:
            st.metric("Total Variance", f"{nmf_analysis['total_variance_explained']:.2f}")
        with col3:
            st.metric("Avg Component Value", f"{nmf_analysis['component_means'].mean():.3f}")
        
        # NMF component analysis
        st.subheader("🧮 Component Analysis")
        nmf_fig = plot_nmf_components(nmf_features)
        st.plotly_chart(nmf_fig, use_container_width=True)
        
        # Analysis for NMF components
        highest_variance_component = np.argmax(nmf_analysis['feature_variance'])
        total_variance = nmf_analysis['total_variance_explained']
        avg_component_value = nmf_analysis['component_means'].mean()
        
        st.markdown(f"""
        **🧮 NMF Model Analysis:** The model has decomposed movie features into **{nmf_analysis['n_components']} latent components**. 
        **Component {highest_variance_component + 1}** shows the highest variance ({nmf_analysis['feature_variance'][highest_variance_component]:.3f}), indicating it captures the most distinctive patterns in movie characteristics. 
        The total variance explained is {total_variance:.2f}, with an average component value of {avg_component_value:.3f}. 
        These components represent hidden patterns in movie genres and characteristics that the recommendation system uses to find similar movies based on your preferences.
        """)
        
        # Component statistics table
        st.subheader("📈 Component Statistics")
        component_stats = pd.DataFrame({
            'Component': range(1, len(nmf_analysis['component_means']) + 1),
            'Mean': nmf_analysis['component_means'],
            'Std Dev': nmf_analysis['component_stds'],
            'Variance': nmf_analysis['feature_variance']
        })
        st.dataframe(component_stats, use_container_width=True)
        
        # Model explanation
        st.subheader("🤖 How the Model Works")
        st.write(f"""
        **Non-negative Matrix Factorization (NMF)** decomposes the movie-genre matrix into latent factors:
        
        - **Components**: Each component represents a hidden pattern in movie genres
        - **Variance**: Higher variance components capture more information
        - **Similarity**: Movies are compared using these latent features
        - **Recommendations**: Based on content similarity in the latent space
        
        The model learns {nmf_analysis['n_components']} latent factors from {analytics['total_movies']} movies to create personalized recommendations.
        """)

if __name__ == "__main__":
    main()