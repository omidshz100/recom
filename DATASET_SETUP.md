# Dataset Setup Instructions

## MovieLens Dataset

This movie recommender system uses the MovieLens dataset, which is **now included in this repository** using Git Large File Storage (LFS) to handle the large CSV files.

### Quick Start:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/omidshz100/recom.git
   cd recom
   ```

2. **Install Git LFS (if not already installed):**
   ```bash
   # On macOS with Homebrew
   brew install git-lfs
   
   # On Ubuntu/Debian
   sudo apt install git-lfs
   
   # On Windows, download from: https://git-lfs.github.io/
   ```

3. **Pull LFS Files:**
   ```bash
   git lfs pull
   ```

4. **Verify File Structure:**
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

5. **Run the Application:**
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
Large CSV files are managed using Git LFS (Large File Storage) to efficiently handle files over GitHub's 100MB limit. The `.gitattributes` file configures `*.csv` files to be tracked by LFS.

### Alternative: Smaller Dataset
If you prefer a smaller dataset for testing:
- Download "ml-latest-small.zip" instead (1 MB)
- Contains ~100,000 ratings and ~9,000 movies
- Follow the same extraction steps above