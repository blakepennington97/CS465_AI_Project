# Import AL/ML libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# pd.set_option("display.max_rows", None, "display.max_columns", None)

# Import the movies dataset
movies = pd.read_csv('ml-latest-small/movies.csv')
#print(movies.head())

# Import the movie ratings dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')
#print(ratings.head())

# How many movies and ratings are there?
print('The dataset contains', len(ratings), 'ratings of', len(movies), 'movies.')

# Compare preferences of romance vs. comedy movies
def compare_genre_tastes(movies, ratings, genres, columns):
    gt = pd.DataFrame()
    for genre in genres:
        genre_movies = movies[movies['genres'].str.contains(genre)]
        avg_genre_rating = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
        
        gt = pd.concat([gt, avg_genre_rating], axis=1)
        
    gt.columns = columns
    return gt

gt = compare_genre_tastes(movies, ratings, ['Romance', 'Comedy'], ['avg_romance_rating', 'avg_comedy_rating'])
# print(gt.head())

# Curate the genre tastes so that the only users within it like 
# either romance or comedy movies (rating for either is above 3.0)
gt = gt.drop(gt[(gt.avg_romance_rating <= 3.0) & (gt.avg_comedy_rating <= 3.0)].index)
# Remove rows with null ratings
gt = gt.dropna(how='any', axis=0)
gt.info()

# Visualize the data
def draw_scatterplot(x_data, x_label, y_data, y_label):
    plot = plt.figure(figsize=(8,8))
    ax = plot.add_subplot(111)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x_data, y_data, s=25)
    plt.show()

# draw_scatterplot(gt['avg_romance_rating'], 'Avg romance rating', gt['avg_comedy_rating'], 'Avg comedy rating')

# Use K-means to break the data down into hopefully 3 groups
# 1. People who like romance, but not comedy
# 2. People who like comedy, but not romance
# 3. People who like both romance and comedy

k_means = KMeans(init="random", n_clusters=3, random_state=99)
labels = k_means.fit_predict(gt)

def plot_kmeans(gt, labels):
    plot = plt.figure(figsize=(8,8))
    ax = plot.add_subplot(111)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.set_xlabel('Avg romance rating')
    ax.set_ylabel('Avg comedy rating')

    ax.scatter(x=gt['avg_romance_rating'], y=gt['avg_comedy_rating'], c=labels, s=25, cmap='viridis')

    plt.show()

plot_kmeans(gt, labels)
