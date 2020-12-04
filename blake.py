# Import AL/ML libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import csr_matrix


# Compare preferences of romance vs. comedy movies
def compare_genre_tastes(movies, ratings, genres, columns):
    gt = pd.DataFrame()
    for genre in genres:
        genre_movies = movies[movies['genres'].str.contains(genre)]
        avg_genre_rating = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
        gt = pd.concat([gt, avg_genre_rating], axis=1)
        
    gt.columns = columns
    return gt


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


def plot_kmeans(gt, labels):
    plot = plt.figure(figsize=(8,8))
    ax = plot.add_subplot(111)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.set_xlabel('Avg romance rating')
    ax.set_ylabel('Avg comedy rating')
    ax.scatter(x=gt['avg_romance_rating'], y=gt['avg_comedy_rating'], c=labels, s=25, cmap='viridis')
    plt.show()


# Gets movies who have a high volume of ratings
def get_most_rated_movies(user_ratings, max_number_movies):
    # Count occurences and add to table
    user_ratings = user_ratings.append(user_ratings.count(), ignore_index=True)
    # Sort occurences
    sorted_user_ratings = user_ratings.sort_values((len(user_ratings) - 1), axis=1, ascending=False)
    sorted_user_ratings = sorted_user_ratings.drop(sorted_user_ratings.tail(1).index)
    # Return subset of these movies, exluding movies with low volume of ratings
    most_rated_movies = sorted_user_ratings.iloc[:, :max_number_movies]
    return most_rated_movies


# Gets users who have a high volume of ratings from the movies who have been rated the most
def get_ideal_users(most_rated_movies, max_number_users):
    # Count occurences of user ratings
    most_rated_movies['count'] = pd.Series(most_rated_movies.count(axis=1))
    # Sort counts in descending order
    most_ideal_users = most_rated_movies.sort_values('count', ascending=False)
    # Grab only top portion of these users
    selected_ideal_users = most_ideal_users.iloc[:max_number_users, :]
    selected_ideal_users = selected_ideal_users.drop(['count'], axis=1)
    return selected_ideal_users


# Filters the movie db based on most popular movies and users who have rated the most
def get_dense_dataset(user_movie_ratings, max_number_movies, max_number_users):
    most_rated_movies = get_most_rated_movies(user_movie_ratings, max_number_movies)
    most_rated_movies = get_ideal_users(most_rated_movies, max_number_users)
    return most_rated_movies


# Draws heatmap based on data
def draw_heatmap(data, axis_labels=True):
    fig = plt.figure(figsize=(15,4))
    ax = plt.gca()
    heatmap = ax.imshow(data, interpolation='nearest', vmin=0, vmax=5, aspect='auto')

    if axis_labels:
        ax.set_yticks(np.arange(data.shape[0]) , minor=False)
        ax.set_xticks(np.arange(data.shape[1]) , minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        labels = data.columns.str[:40]
        ax.set_xticklabels(labels, minor=False)
        ax.set_yticklabels(data.index, minor=False)
        plt.setp(ax.get_xticklabels(), rotation=90)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    ax.grid(False)
    ax.set_ylabel('UserID')

    # Separate heatmap from color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Color bar
    cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
    cbar.ax.set_yticklabels(['5 stars', '4 stars', '3 stars', '2 stars', '1 stars', '0 stars'])
    plt.show()


# Visualize multiple clusters as a heatmap
def draw_clusters(data, max_users, max_movies):
    c=1
    for cluster_id in data.group.unique():
        # To improve visibility, we're showing at most max_users users and max_movies movies per cluster.
        # You can change these values to see more users & movies per cluster
        d = data[data.group == cluster_id].drop(['index', 'group'], axis=1)
        n_users_in_cluster = d.shape[0]
        d = get_dense_dataset(d, max_movies, max_users)
        d = d.reindex(d.mean().sort_values(ascending=False).index, axis=1)
        d = d.reindex(d.count(axis=1).sort_values(ascending=False).index)
        d = d.iloc[:max_users, :max_movies]
        n_users_in_plot = d.shape[0]
        
        # Show clusters with more than 9 users, to avoid crowding
        if len(d) > 9:
            print('cluster # {}'.format(cluster_id))
            print('# of users in cluster: {}.'.format(n_users_in_cluster), '# of users in plot: {}'.format(n_users_in_plot))
            fig = plt.figure(figsize=(15,4))
            ax = plt.gca()

            ax.invert_yaxis()
            ax.xaxis.tick_top()
            labels = d.columns.str[:40]

            ax.set_yticks(np.arange(d.shape[0]) , minor=False)
            ax.set_xticks(np.arange(d.shape[1]) , minor=False)
            ax.set_xticklabels(labels, minor=False)            
            ax.get_yaxis().set_visible(False)

            heatmap = plt.imshow(d, vmin=0, vmax=5, aspect='auto')

            ax.set_xlabel('movies')
            ax.set_ylabel('User id')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # Color bar
            cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
            cbar.ax.set_yticklabels(['5 stars', '4 stars', '3 stars', '2 stars', '1 stars', '0 stars'])

            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=9)
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', labelbottom='off', labelleft='off') 
            #print('cluster # {} \n(Showing at most {} users and {} movies)'.format(cluster_id, max_users, max_movies))

            plt.show()


if __name__ == "__main__":
    # pd.set_option("display.max_rows", None, "display.max_columns", None)

    # Import the movies dataset
    movies = pd.read_csv('ml-latest-small/movies.csv')
    #print(movies.head())

    # Import the movie ratings dataset
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    #print(ratings.head())

    # How many movies and ratings are there?
    print('The dataset contains', len(ratings), 'ratings of', len(movies), 'movies.')

    gt = compare_genre_tastes(movies, ratings, ['Romance', 'Comedy'], ['avg_romance_rating', 'avg_comedy_rating'])
    # print(gt.head())

    # Curate the genre tastes so that the only users within it like 
    # either romance or comedy movies (rating for either is above 3.0)
    gt = gt.drop(gt[(gt.avg_romance_rating <= 3.0) & (gt.avg_comedy_rating <= 3.0)].index)
    # Remove rows with null ratings
    gt = gt.dropna(how='any', axis=0)
    gt.info()

    # draw_scatterplot(gt['avg_romance_rating'], 'Avg romance rating', gt['avg_comedy_rating'], 'Avg comedy rating')

    # Use K-means to break the data down into hopefully 3 groups
    # 1. People who like romance, but not comedy
    # 2. People who like comedy, but not romance
    # 3. People who like both romance and comedy

    #TODO: should we determine ideal number of clusters? maybe you already did this and that's how you got 3?
    k_means = KMeans(init="random", n_clusters=7, random_state=99)
    labels = k_means.fit_predict(gt)
    #plot_kmeans(gt, labels)

    # ----------------BLAKE CODE BELOW------------------

    # Shape a new dataset in the form userID vs user rating for each movie
    ratings_vs_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')

    # Convert to a pivot table based on userID
    user_movie_ratings = pd.pivot_table(ratings_vs_title, index='userId', columns='title', values='rating')

    # This pivot table has a lot of NULL values due to lack of data on certain movies
    # Therefore, repeat process above, but with only the top rated movies and the users who rated the most
    max_number_movies = 30
    max_number_users = 18
    most_rated_movies_and_user_ratings = get_dense_dataset(user_movie_ratings, max_number_movies, max_number_users)

    # Draw heatmap based on the formatted data gathered above to show distribution
    draw_heatmap(most_rated_movies_and_user_ratings)

    # Apply spare csr matrix to this to help with NULL values
    most_rated_movies_1k = get_most_rated_movies(user_movie_ratings, 1000)
    sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())

    # Cluster based on the sparse matrix
    predictions = KMeans(n_clusters=20, algorithm='full').fit_predict(sparse_ratings)

    # Now visualize the clusters as a heatmap
    max_number_movies = 50
    max_number_users = 70
    clusters = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group': predictions})], axis=1)
    draw_clusters(clusters, max_number_users, max_number_movies)

    # Pick a cluster, then display the highest density portion of that in a heatmap
    cluster_no = 0
    max_number_users = 75
    max_number_movies = 300
    cluster = clusters[clusters.group == cluster_no].drop(['index', 'group'], axis=1)
    cluster = get_dense_dataset(cluster, max_number_movies, max_number_users)
    draw_heatmap(cluster, axis_labels=False)

    # Show the top 20 most-liked movies in the cluster
    cluster.fillna('').head(20)

    # Suppose a user is interested in watching Forrest Gump, so print the predicted value of how much she would like it
    #movie_name = "Forrest Gump (1994)"
    #cluster[movie_name].mean()

    # Now predict User 10's results based on their cluster
    userID = 10
    users_ratings = cluster.loc[userID, :]
    users_unrated_movies = users_ratings[users_ratings.isnull()]
    avg_ratings = pd.concat([users_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:, 0]

    # Display user's top 20 picks
    avg_ratings.sort_values(ascending=False)[:20]
    print(avg_ratings)

