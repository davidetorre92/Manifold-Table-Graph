from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import networkx as nx
from scipy.spatial import distance_matrix
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from datetime import datetime

# Function to append date to filenames if required
def append_date_to_filenames(path):
    current_datetime = datetime.now().strftime("_%Y%m%d%H%M")
    dirname, basename = os.path.split(path)
    name, ext = os.path.splitext(basename)
    updated_name = f"{name}{current_datetime}{ext}"
    updated_path = os.path.join(dirname, updated_name)
    return updated_path

def preprocess_data(df, categorical_cols=None, scaler='standard'):
    """
    Preprocess the dataset: Normalize numerical variables and encode categorical variables.

    Parameters:
    - df: pandas DataFrame, the dataset to preprocess.
    - numerical_cols: list of strings, names of the numerical columns in the dataset.
    - scaler: 'standard' for StandardScaler, 'minmax' for MinMaxScaler. Applied to numerical columns.
    - categorical_cols: list of strings, names of the categorical columns to be encoded. Can be None.

    Returns:
    - df_transformed: pandas DataFrame, the preprocessed dataset.
    """

    # Initialize transformers for numerical and categorical data
    transformers = []
    columns = df.columns
    numerical_cols = None
    # Add encoder for categorical columns
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols))
        numerical_cols = [col for col in columns if col not in categorical_cols]

    # Add scaler for numerical columns
    if numerical_cols:
        if scaler == 'standard':
            transformers.append(('num', StandardScaler(), numerical_cols))
        elif scaler == 'minmax':
            transformers.append(('num', MinMaxScaler(), numerical_cols))

    # Create a ColumnTransformer to apply transformations
    preprocessor = ColumnTransformer(transformers, remainder='passthrough')

    # Apply transformations
    df_transformed = preprocessor.fit_transform(df)

    # Convert back to DataFrame
    # Get feature names for categorical variables after encoding
    if categorical_cols:
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        features = numerical_cols + list(cat_features)
    else:
        features = numerical_cols

    df_transformed = pd.DataFrame(df_transformed, columns=features)

    return df_transformed


def infer_and_preprocess_data(df, scaler='standard', unique_threshold=0.05):
    """
    Automatically detect numerical and categorical columns, then preprocess the dataset.

    Parameters:
    - df: pandas DataFrame, the dataset to preprocess.
    - scaler: 'standard' for StandardScaler, 'minmax' for MinMaxScaler. Applied to numerical columns.
    - unique_threshold: float, the threshold (as a fraction of total rows) below which a column is considered categorical.

    Returns:
    - df_transformed: pandas DataFrame, the preprocessed dataset.
    """

    numerical_cols = []
    categorical_cols = []

    for col in df.columns:
        # Consider column numerical if its dtype is int or float
        if pd.api.types.is_numeric_dtype(df[col]):
            # Apply heuristic based on unique value count to distinguish between categorical and numerical
            if df[col].nunique() / df.shape[0] > unique_threshold:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            categorical_cols.append(col)

    # Now preprocess the data using the inferred column types
    return preprocess_data(df, scaler=scaler, categorical_cols=categorical_cols)

def apply_manifold_learning(df, technique='TSNE', params=None):
    """
    Applies a specified manifold learning technique to a preprocessed pandas DataFrame.

    Parameters:
    - df: pandas DataFrame, the preprocessed dataset.
    - technique: string, name of the manifold learning technique to apply.
                 Options are 'TSNE', 'Isomap', 'MDS', 'LLE'.
    - params: dict, parameters to pass to the manifold learning algorithm.

    Returns:
    - transformed_df: pandas DataFrame, the dataset transformed by the manifold learning technique.
    """
    # Default parameters if none provided
    if params is None:
        params = {}

    # Initialize the manifold learning algorithm based on the technique
    if technique == 'TSNE':
        model = TSNE(**params)
    elif technique == 'Isomap':
        model = Isomap(**params)
    elif technique == 'MDS':
        model = MDS(**params)
    elif technique == 'LLE':
        model = LocallyLinearEmbedding(**params)
    else:
        raise ValueError(f"Unsupported technique: {technique}. Choose from 'TSNE', 'Isomap', 'MDS', 'LLE'.")

    # Fit and transform the data
    transformed_data = model.fit_transform(df)

    # Convert to DataFrame for easier handling
    transformed_df = pd.DataFrame(transformed_data, columns=[f'component_{i}' for i in range(transformed_data.shape[1])])

    return transformed_df

# Define a custom similarity function, for example:
def cos_sim(x, y):
    # An example similarity function that computes cosine similarity
    # Note: In real cases, ensure this function returns values in the range [0, 1]
    cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return cos_sim

def create_similarity_threshold_graph(original_df, relationship_df, similarity_function, similarity_threshold, classification_attribute = None, classification_attribute_name = None):
    """
    Creates a graph where nodes represent entries with original features from original_df,
    and edges exist if the similarity between points in relationship_df is above a similarity_threshold.

    Parameters:
    - original_df: pandas DataFrame, the original dataset.
    - relationship_df: pandas DataFrame, the dataset to learn reationships.
    - similarity_function: function, a custom function to compute the similarity between two points.
                           It should return a value in the range [0, 1].
    - similarity_threshold: float, the threshold for adding an edge between two nodes based on similarity.

    Returns:
    - G: networkx graph, with nodes representing data points and edges based on similarity threshold.
    """

    # Initialize the graph
    G = nx.Graph()
    if classification_attribute is not None:
        if classification_attribute_name is None:
            raise TypeError("If a classification label is passed, a name must be passed too.")
    G.graph['classification_attribute'] = classification_attribute_name

    # Add nodes with attributes being the original features
    for idx, (_, features) in enumerate(original_df.iterrows()):
        G.add_node(idx, **features.to_dict())

    # Compute and add edges based on the similarity threshold
    for i in range(len(relationship_df)):
        for j in range(i+1, len(relationship_df)):  # Avoid duplicating edges and self-loops
            # Calculate similarity between points i and j
            similarity = similarity_function(relationship_df.iloc[i], relationship_df.iloc[j])
            if similarity >= similarity_threshold:
                G.add_edge(i, j, sim = similarity)

    return G

def create_similarity_graph(original_df, relationship_df, n_neighbors=5, distance_threshold = None, classification_attribute = None, classification_attribute_name = None):
    """
    Creates a graph of relations based on a dataset.

    Parameters:
    - original_df: pandas DataFrame, the original dataset before preprocessing.
    - relationship_df: pandas DataFrame, the dataset to learn reationships.
    - n_neighbors: int, the number of nearest neighbors to consider for creating edges.
    - distance_threshold: float, maximum distance threshold for creating edges between nodes. 
    If specified, an edge is only added between two nodes if the distance between them (based on their features) is less than or equal to this threshold.

    Returns:
    - G: networkx graph, where nodes represent entries with original features and
         edges represent similarity relations evaluated by the manifold learning technique.
    """

    # Initialize the graph
    G = nx.Graph()
    if classification_attribute is not None:
        if classification_attribute_name is None:
            raise TypeError("If a classification label is passed, a name must be passed too.")
    G.graph['classification_attribute'] = classification_attribute_name

    # Add nodes with attributes being the original features
    for idx, (_, features) in enumerate(original_df.iterrows()):
        G.add_node(idx, **features.to_dict())
        if classification_attribute is not None: G.nodes[idx][classification_attribute_name] = classification_attribute[idx]

    # Use NearestNeighbors to find edges based on the transformed dataset
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(relationship_df)
    distances, indices = nbrs.kneighbors(relationship_df)

    # Add edges based on nearest neighbors
    for i in range(relationship_df.shape[0]):
        for j in range(1, n_neighbors): # Start from 1 to avoid self-loop
            if distance_threshold is None or distances[i, j] <= distance_threshold:
                G.add_edge(i, indices[i, j], d = distances[i,j], d_m1 = 1 / distances[i,j])

    return G

def create_distance_threshold_graph(original_df, relationship_df, p, distance_threshold, classification_attribute = None, classification_attribute_name = None):
    """
    Creates a graph where nodes represent entries with original features from original_df,
    and edges exist if the distance between points in processed_df is under distance_threshold.

    Parameters:
    - original_df: pandas DataFrame, the original dataset.
    - relationship_df: pandas DataFrame, the dataset to learn reationships.
    - p: int which Minkowski p-norm to use.
    - distance_threshold: float, the threshold for adding an edge between two nodes.

    Returns:
    - G: networkx graph, with nodes representing data points and edges based on distance threshold.
    """

    # Initialize the graph
    G = nx.Graph()
    if classification_attribute is not None:
        if classification_attribute_name is None:
            raise TypeError("If a classification label is passed, a name must be passed too.")

    G.graph['classification_attribute'] = classification_attribute_name
    # Calculate the distance matrix
    distances = distance_matrix(relationship_df, relationship_df, p=p)
    # Add nodes with attributes being the original features
    for idx, (_, features) in enumerate(original_df.iterrows()):
        G.add_node(idx, **features.to_dict())
        if classification_attribute is not None: G.nodes[idx][classification_attribute_name] = classification_attribute[idx]

    # Add edges based on the distance threshold
    for i in range(len(relationship_df)):
        for j in range(i+1, len(relationship_df)):  # Avoid duplicating edges and self-loops
            if distances[i, j] <= distance_threshold:
                G.add_edge(i, j, d = distances[i,j], d_m1 = 1 / distances[i,j])

    return G

# Draw the graph
def matplotlib_graph_visualization(G, graph_visualization, palette = 'viridis', pos = None):
    palette = 'viridis'
    plt.figure(figsize=(10, 10))
    # Get node positions using a layout
    if pos is None:
      pos = nx.spring_layout(G, seed=2112)

    node_color = []
    if G.graph['classification_attribute'] is not None:
        classification_attribute_name = G.graph['classification_attribute']
        y = np.array([G.nodes[node][classification_attribute_name] for node in G.nodes()])
        unique = np.unique(y)
        unique_dict = {key: index for index, key in enumerate(unique)}
        colrs = colors = np.linspace(0, 1, len(unique))
        cmap = cm.get_cmap(palette, len(unique))
        color_array = cmap(colors)
        node_color = [color_array[unique_dict[key]] for key in y]
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, node_color = node_color)
    plt.title("Graph of Relations Based on Manifold Learning Transformed Data")
    plt.savefig(graph_visualization)
    print(f'Graph saved in {graph_visualization}')

def analyze_neighborhood_attributes(graph, attribute_name, return_probs=False):
    """
    Analyzes attributes in the neighborhoods of each node in a graph, optionally returning probabilities.

    Parameters:
    - graph (networkx.Graph): The input graph.
    - attribute_name (str): The name of the node attribute to analyze.
    - return_probs (bool): If True, returns the probability of each attribute in the neighborhood.

    Returns:
    - pd.DataFrame: A DataFrame with each row representing a node. Columns include the node's attribute,
                    degree, and either the count or probability of each attribute in its neighborhood.
    """
    # Collect unique attributes
    unique_attributes = set(nx.get_node_attributes(graph, attribute_name).values())

    # Prepare data for DataFrame
    data = []
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        neighbor_attrs = [graph.nodes[n].get(attribute_name, None) for n in neighbors]

        attr_counts = {}
        attr_counts[f"node_{attribute_name}"] = graph.nodes[node].get(attribute_name, None)
        attr_counts["node_index"] = node
        attr_counts["degree"] = len(neighbors)

        for attr in unique_attributes:
            if return_probs and attr_counts["degree"] > 0:  # Calculate probabilities
                attr_counts[f"p_{attr}"] = neighbor_attrs.count(attr) / attr_counts["degree"]
            else:  # Count occurrences
                attr_counts[f"n_{attr}"] = neighbor_attrs.count(attr)

        data.append(attr_counts)

    # Create DataFrame
    cols = ["node_index", f"node_{attribute_name}", "degree"] + \
           [f"{'p' if return_probs else 'n'}_{attr}" for attr in unique_attributes]
    df = pd.DataFrame(data, columns=cols)

    return df

def degree_distributions(G, outpath):
    # Calculate the degree of each node
    degrees = [degree for node, degree in G.degree()]
    hist, bin_edges = np.histogram(degrees, bins = range(0, max(degrees)))
    # Plot the degree distribution
    plt.scatter(bin_edges[:-1], hist, alpha=0.75, edgecolor='black')
    plt.title('Degree Distribution')
    plt.xlim((1,None))
    plt.ylim((1,None))
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.savefig(outpath)
    print(f"Degree distribution saved in {outpath}")


def betweenness_distributions(G, outpath):
    # Calculate the degree of each node
    betweenness_centrality = nx.betweenness_centrality(G)
    betweenness_data = [betweenness_centrality[node] for node in G.nodes()]
    hist, bin_edges = np.histogram(betweenness_data)
    # Plot the degree distribution
    fig, ax = plt.subplots()
    ax.scatter(bin_edges[:-1], hist, alpha=0.75, edgecolor='black')
    ax.set_title('Betweenness Distribution')
    ax.set_xlabel('Betweenness')
    ax.set_ylabel('Number of Nodes')
    fig.savefig(outpath)
    print(f"Betweenness distribution saved in {outpath}")

def eigenvector_distributions(G, outpath):
    # Calculate the degree of each node
    eigenvector_centrality = nx.eigenvector_centrality(G)
    eigenvector_centrality_data = [eigenvector_centrality[node] for node in G.nodes()]
    hist, bin_edges = np.histogram(eigenvector_centrality_data, bins = 10)
    fig, ax = plt.subplots(figsize = (15,15))

    # Plot the degree distribution
    ax.scatter(bin_edges[:-1], hist, alpha=0.75, edgecolor='black')
    ax.set_title('Eigenvector centrality distribution')
    ax.set_xlabel('Eigenvector')
    ax.set_ylabel('Number of Nodes')
    fig.savefig(outpath)
    print(f"Eigenvector distribution saved in {outpath}")

def plot_community_composition(G, attribute_name, outpath):
    # Detect communities
    communities_generator = nx.algorithms.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    communities = [list(c) for c in sorted(top_level_communities, key=len, reverse=True)]

    # Infer labels from the graph
    labels_per_node = [G.nodes[node][attribute_name] for node in G.nodes()]
    unique_labels = np.unique(labels_per_node)

    # Prepare data for stacked bar plot
    community_compositions = {}
    for comm_id, community in enumerate(communities):
      labels_community = [G.nodes[node][attribute_name] for node in community]
      label_count = Counter(labels_community)
      community_compositions[comm_id] = {label: label_count.get(label, 0) for label in unique_labels}

    # Prepare data for plotting
    indices = list(community_compositions.keys())
    bar_width = 0.35  # Width of the bars
    print("Community composition:")
    print(community_compositions)
    print()
    # Initialize a figure and axis for the plot
    fig, ax = plt.subplots(figsize = (15,15))

    # Loop through each label to stack the bars
    bottoms = [0] * len(indices)  # Keeps track of where the next bar starts
    for label in unique_labels:
        values = [community_compositions[idx].get(label, 0) for idx in indices]
        ax.bar(indices, values, bar_width, label=label, bottom=bottoms)
        # Update the bottom positions for the next label
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    # Set the position of the bars on the X-axis
    ax.set_xticks(indices)
    ax.set_xticklabels(indices)

    # Adding labels and title
    ax.set_xlabel('Community ID')
    ax.set_ylabel('Counts')
    ax.set_title('Counts of outcomes by community ID')
    ax.legend()

    # Show the plot
    fig.savefig(outpath)
    print(f"Community composition saved in {outpath}")


def print_neighbors_prob(df_neigh, label_col, out_path):
# Calculate the probabilities
    probabilities = {}
    for label_i in df_neigh[f'node_{label_col}'].unique():
        nodes_with_label_i = df_neigh[df_neigh[f'node_{label_col}'] == label_i]
        total_degree_i = nodes_with_label_i['degree'].sum()
        for label_j in df_neigh[f'node_{label_col}'].unique():
            col_name = f'n_{label_j}'
            total_neighbors_with_label_j = nodes_with_label_i[col_name].sum()
            probabilities[(label_i, label_j)] = total_neighbors_with_label_j / total_degree_i if total_degree_i else 0

    return probabilities

def heat_map_prob(probabilities, df_neigh, label_col, prob_heatmap_path):
    labels = sorted(df_neigh[f'node_{label_col}'].unique())
    prob_matrix = pd.DataFrame(index=labels, columns=labels, data=0.0)

    for (i, j), prob in probabilities.items():
        prob_matrix.loc[i, j] = prob

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(prob_matrix, cmap='viridis', interpolation='nearest')
    fig.colorbar(cax)

    # Adding annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{prob_matrix.iloc[i, j]:.2f}",
                           ha="center", va="center", color="w")

    ax.set_title('Probability Distribution Heatmap')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Label j')
    plt.ylabel('Label i')

    plt.savefig(prob_heatmap_path)
    print(f"Neighbour probability data saved in {prob_heatmap_path}")

