import pandas as pd
from surprise import Dataset, Reader, accuracy
from surprise.prediction_algorithms.matrix_factorization import SVD
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Load datasets
ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

# Merge ratings with movie genres
df = pd.merge(ratings_df, movies_df[['movieId', 'genres']], on='movieId', how='left')

# Initialize encoders
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
mlb = MultiLabelBinarizer()

# Encode userId and movieId
df['userId'] = user_encoder.fit_transform(df['userId'])
df['movieId'] = movie_encoder.fit_transform(df['movieId'])

# One-hot encode movie genres
df = df.join(pd.DataFrame(
    mlb.fit_transform(df.pop('genres').str.split('|')),
    columns=mlb.classes_,
    index=df.index
))

# Drop the '(no genres listed)' column if present
df.drop(columns="(no genres listed)", inplace=True)

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Prepare data for Surprise library
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Build and train the SVD model
model_svd = SVD()
model_svd.fit(trainset)

# Evaluate model using RMSE
predictions_svd = model_svd.test(trainset.build_anti_testset())
print(f"RMSE: {accuracy.rmse(predictions_svd)}")

# Function to get top N recommendations for a user
def get_top_n_recommendations(user_id, n=5):
    """
    Get top N movie recommendations for a given user based on SVD model.

    Args:
    user_id (int): The user ID for which recommendations are made.
    n (int): The number of recommendations to return.

    Returns:
    list: A list of movie titles recommended for the user.
    """
    # Get movies the user has already rated
    user_movies = df[df['userId'] == user_id]['movieId'].unique()
    all_movies = df['movieId'].unique()

    # Movies to predict (not already rated by the user)
    movies_to_predict = list(set(all_movies) - set(user_movies))

    # Generate prediction for each movie not yet rated by the user
    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]
    predictions_cf = model_svd.test(user_movie_pairs)

    # Sort predictions by estimated rating
    top_n_recommendations = sorted(predictions_cf, key=lambda x: x.est, reverse=True)[:n]

    # Print the predicted ratings (optional)
    for pred in top_n_recommendations:
        print(f"Predicted rating: {pred.est}")

    # Get movie IDs and map them back to movie titles
    top_n_movie_ids = [int(pred.iid) for pred in top_n_recommendations]
    top_n_movies = movie_encoder.inverse_transform(top_n_movie_ids)

    return top_n_movies

# Example usage: Get recommendations for a specific user
user_id = 221
recommendations = get_top_n_recommendations(user_id)

# Fetch movie titles corresponding to the recommended movie IDs
top_n_movies_titles = movies_df[movies_df['movieId'].isin(recommendations)]['title'].tolist()

# Output the top N recommended movies
print(f"Top {len(recommendations)} Recommended Movies for User {user_id}:")
for i, title in enumerate(top_n_movies_titles, 1):
    print(f"{i}. {title}")
