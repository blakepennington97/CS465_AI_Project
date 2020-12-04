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

genre1 = 'Romance'
xlabel = 'Avg ' + genre1 + ' rating'
genre2 = 'Horror'
ylabel = 'Avg ' + genre2 + ' rating'

# Compare preferences of genre1 vs. genre2 movies
def compare_genre_tastes(movies, ratings, genres, columns):
    gt = pd.DataFrame()
    for genre in genres:
        genre_movies = movies[movies['genres'].str.contains(genre)]
        avg_genre_rating = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
        
        gt = pd.concat([gt, avg_genre_rating], axis=1)
        
    gt.columns = columns
    return gt

gt = compare_genre_tastes(movies, ratings, [genre1, genre2], ['avg_genre1_rating', 'avg_genre2_rating'])
# print(gt.head())

# Curate the genre tastes so that the only users within it like 
# either genre1 or genre2 movies (rating for either is above 3.0)
gt = gt.drop(gt[(gt.avg_genre1_rating <= 2.5) & (gt.avg_genre2_rating <= 2.5)].index)
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

draw_scatterplot(gt['avg_genre1_rating'], xlabel, gt['avg_genre2_rating'], ylabel)

# Use K-means to break the data down into hopefully 3 groups
# 1. People who like genre1, but not genre2
# 2. People who like genre2, but not genre1 
# 3. People who like both genre1 and genre2

k_means = KMeans(init="random", n_clusters=3, random_state=99)
labels = k_means.fit_predict(gt)

def plot_kmeans(gt, labels):
    plot = plt.figure(figsize=(8,8))
    ax = plot.add_subplot(111)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.scatter(x=gt['avg_genre1_rating'], y=gt['avg_genre2_rating'], c=labels, s=25, cmap='viridis')

    plt.show()

plot_kmeans(gt, labels)
