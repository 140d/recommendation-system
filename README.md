# Movie Recommendation App

This repository implements a movie recommendation system using **Matrix Factorization (SVD)** and **Collaborative Filtering** techniques. The system predicts movie ratings for users and recommends top movies based on their preferences.

## Overview

The recommendation system is built with the **Surprise** library for collaborative filtering and the **scikit-learn** library for data preprocessing and encoding. The system provides personalized movie recommendations to users based on their historical ratings.

## Files

- **main.py**: The main Python script containing the logic for loading data, training the model, evaluating it, and generating recommendations.
- **movies.csv**: A CSV file containing movie information (movie ID, title, and genres).
- **ratings.csv**: A CSV file containing user ratings (user ID, movie ID, rating).

## Requirements

Before running the project, install the required dependencies:

```bash
pip install pandas scikit-learn surprise
```

## Usage

### Step 1: Prepare the Data

- **ratings.csv** contains user ratings for movies, including the `userId`, `movieId`, and the `rating` given by the user.
- **movies.csv** contains movie metadata, including `movieId`, `title`, and `genres`.

### Step 2: Running the Script

To run the recommendation system, execute the following command:

```bash
python main.py
```

This will:
1. Load and merge the `ratings.csv` and `movies.csv` datasets.
2. Encode `userId` and `movieId` using **LabelEncoder**.
3. One-hot encode movie genres with **MultiLabelBinarizer**.
4. Split the dataset into training and test sets.
5. Train an **SVD** model using the **Surprise** library.
6. Evaluate the model using **Root Mean Squared Error (RMSE)**.
7. Generate top N movie recommendations for a given user (by default, 5 recommendations).

### Step 3: Customizing Recommendations

In the script, you can change the `user_id` variable to recommend movies for a different user:

```python
user_id = 221
recommendations = get_top_n_recommendations(user_id)
```

You can also modify the number of recommended movies by changing the `n` parameter in the `get_top_n_recommendations` function:

```python
recommendations = get_top_n_recommendations(user_id, n=10)
```

### Example Output

For a given user, the system will output top movie recommendations like this:

```
Top 5 Recommended Movies for User 221:
1. The Godfather
2. The Dark Knight
3. Pulp Fiction
4. Forrest Gump
5. Inception
```

## Evaluation

The performance of the recommendation model is evaluated using the **Root Mean Squared Error (RMSE)** metric. After training the SVD model, the RMSE value is printed to assess the accuracy of the predictions:

```
RMSE: 0.876
```

## Contribution

Feel free to fork the repository and submit pull requests. Contributions are welcome to improve the recommendation model, implement other recommendation techniques, or enhance the data processing pipeline.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

### Key Points:
- Provides a concise description of the project.
- Specifies the necessary dependencies for running the project.
- Details on the files included and their purpose.
- Step-by-step instructions for running the script.
- Provides examples and customization options.
- Clear evaluation of model performance using RMSE.

This README should give users a thorough understanding of how to use and extend the recommendation system in your repository!
