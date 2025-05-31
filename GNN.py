import networkx as nx
import matplotlib.pyplot as plt

## Basic graph representation

## Undirected graph
# Create a graph
G = nx.Graph()

# Add nodes (representing people)
G.add_node("Yubin")
G.add_node("Youngho")
G.add_node("Yumin")
G.add_node("Woojeong")
G.add_node("Rey")
G.add_node("Coin")

# Add edges (representing friendship)
G.add_edges_from([("Yubin", "Youngho"), ("Yubin", "Yumin"), ("Yubin", "Woojeong"), ("Yubin", "Rey"), ("Youngho", "Coin"), ("Yumin", "Woojeong"), ("Yumin", "Rey")])

# Make a plot
plt.axis('off')
nx.draw_networkx(G,
                 pos=nx.spring_layout(G, seed=0),
                 node_size=600,
                 cmap='coolwarm',
                 font_size=6,
                 font_color='white'
                 )


## Directed graph
# Create a graph
DG = nx.DiGraph() 
DG.add_edges_from([("Yubin", "Youngho"), ("Yubin", "Yumin"), ("Yubin", "Woojeong"), ("Yubin", "Rey"), ("Youngho", "Coin"), ("Yumin", "Woojeong"), ("Yumin", "Rey")])

# Make a plot
plt.axis('off')
nx.draw_networkx(DG,
                 pos=nx.spring_layout(G, seed=0),
                 node_size=600,
                 cmap='coolwarm',
                 font_size=6,
                 font_color='white'
                 )


## GNN example

# 1. Plot the overall graph structure with node features
def plot_graph_structure_detailed():
    G = nx.DiGraph()  # Create a directed graph

    # Add nodes with feature information
    G.add_nodes_from([
        ("A", {"feature": "Feature_A"}),
        ("B", {"feature": "Feature_B"}),
        ("C", {"feature": "Feature_C"}),
        ("D", {"feature": "Feature_D"}),
        ("E", {"feature": "Feature_E"}),
        ("F", {"feature": "Feature_F"}),
        ("G", {"feature": "Feature_G"}),
        ("H", {"feature": "Feature_H"})
    ])

    # Add directed edges representing message flow direction
    G.add_edges_from([
        ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("C", "E"),
        ("D", "F"), ("E", "F"), ("E", "G"), ("F", "H"), ("G", "H")
    ])
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout for consistent positioning
    nx.draw(G, pos, with_labels=True, node_color='lightpink', node_size=2000, font_size=12, arrows=True)
    
    # Annotate each node with its feature
    labels = {node: f"{node}\n{data['feature']}" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    plt.title("1. Graph Structure: Nodes with Features and Directed Edges")
    plt.show()



## Example of movie recommendation

import pandas as pd
import networkx as nx
from io import BytesIO
from zipfile import ZipFile
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from urllib.request import urlopen
from collections import defaultdict

# source: https://ysg2997.tistory.com/27?category=650584
# Download and extract the MovieLens 100k dataset
url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
with urlopen(url) as zurl:
    with ZipFile(BytesIO(zurl.read())) as zfile:
        zfile.extractall('.')

# Load ratings and movie titles
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', usecols=range(2), names=['movie_id', 'title'], encoding='latin-1')

# Use only ratings of 4 or higher
ratings = ratings[ratings.rating >= 4]

# Initialize a dictionary to count how often each pair of movies is liked together
# Uses defaultdict to automatically assign 0 to any new key
pairs = defaultdict(int)

# Loop through the entire list of users
for group in ratings.groupby("user_id"):
    # List of movie IDs rated by the current user
    user_movies = list(group[1]["movie_id"])

    # Count every time two movies are liked together
    for i in range(len(user_movies)):
        for j in range(i + 1, len(user_movies)):
            pairs[(user_movies[i], user_movies[j])] += 1

# Create a networkx graph
G = nx.Graph()

# Create an edge between movies that are liked together
for pair in pairs:
    movie1, movie2 = pair
    score = pairs[pair]

    # Only create the edge if the score is 20 or higher
    if score >= 20:
        G.add_edge(movie1, movie2, weight=score)

# Print the total number of nodes and edges in the graph
print("Total number of graph nodes:", G.number_of_nodes())
print("Total number of graph edges:", G.number_of_edges())

"""
# Initialize Node2Vec on graph G with specified parameters
# dimensions: size of embedding vectors
# walk_length: length of each random walk
# num_walks: number of walks per node
# p, q: Node2Vec hyperparameters controlling the walk behavior
# workers: number of parallel processes
"""
node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=200, p=2, q=1, workers=1)

"""
# Fit the Node2Vec model to generate embeddings
# window: context window size for Word2Vec
# min_count: minimum count of nodes to consider
# batch_words: number of words to process in each batch
"""
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Function to recommend similar movies
def recommend(movie):
    # Get the movie_id corresponding to the input movie title
    movie_id = str(movies[movies.title == movie].movie_id.values[0])

    # Find top 5 most similar movies based on vector similarity
    for id in model.wv.most_similar(movie_id)[:5]:
        # Get the movie title corresponding to the recommended movie_id
        title = movies[movies.movie_id == int(id[0])].title.values[0]
        # Print the recommended movie title and its similarity score
        print(f'{title}: {id[1]:.2f}')

# Call the recommend function for the movie 'Star Wars (1977)'
recommend('Star Wars (1977)')
"""
Return of the Jedi (1983): 0.61
Raiders of the Lost Ark (1981): 0.56
Monty Python and the Holy Grail (1974): 0.51
Toy Story (1995): 0.44
Terminator 2: Judgment Day (1991): 0.44
"""
