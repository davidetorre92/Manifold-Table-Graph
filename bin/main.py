import numpy as np
import pandas as pd

from utils import *
from config import *

import pickle

# Check if the dataframe exists
if os.path.exists(df_path):
    pass
else:
    raise FileNotFoundError(f"File containing data not found in {df_path}")

# Check if the output directories are available
# List of all file paths from config
output_paths = [preprocess_df_path, graph_path, graph_visualization_path, neigh_prob_path, 
              prob_heatmap_path, degree_distribution_outpath, betweenness_distribution_outpath, 
              community_composition_outpath]

for file_path in output_paths:
    # Extract directory path
    directory = os.path.dirname(file_path)
    
    # Check if the directory exists
    if not os.path.exists(directory):
        # If not, create it
        os.makedirs(directory)
        print(f"Directory '{directory}' has been created.")
    else:
        pass

if save_date_experiment:
  preprocess_df_path = append_date_to_filenames(preprocess_df_path)
  graph_path = append_date_to_filenames(graph_path)
  graph_visualization_path = append_date_to_filenames(graph_visualization_path)
  neigh_prob_path = append_date_to_filenames(neigh_prob_path)
  prob_heatmap_path = append_date_to_filenames(prob_heatmap_path)
  degree_distribution_outpath = append_date_to_filenames(degree_distribution_outpath)
  betweenness_distribution_outpath = append_date_to_filenames(betweenness_distribution_outpath)
  community_composition_outpath = append_date_to_filenames(community_composition_outpath)


try:
    if df_path.endswith('.csv'):
        df = pd.read_csv(df_path, index_col=0)
    elif df_path.endswith('.xlsx'):
        df = pd.read_excel(df_path, index_col=None)  # Changed index_col to None for consistency
    elif df_path.endswith('.pickle'):
        df = pd.read_pickle(df_path)
    elif df_path.endswith('.json'):
        df = pd.read_json(df_path)
    elif df_path.endswith('.parquet'):
        df = pd.read_parquet(df_path)
    elif df_path.endswith('.hdf') or df_path.endswith('.h5'):
        df = pd.read_hdf(df_path)
    else:
        # Suggesting action to the user
        supported_formats = ", ".join(["CSV", "Excel (.xlsx)", "Pickle", "JSON", "Parquet", "HDF5 (.hdf, .h5)"])
        raise ValueError(f"The file format is not supported. Please convert your file to one of the following supported formats: {supported_formats}.")
except Exception as e:
    print(f"Error reading the data file: {e}")

if drop_cols is not None:
    df = df.drop(drop_cols, axis = 1)

# Step 1: preprocess

drop_cols_pp = []

if ids_col is not None: drop_cols_pp.append(ids_col)
if label_col is not None: drop_cols_pp.append(label_col)
if pass_cols is not None: 
    if type(pass_cols) == list:
      drop_cols_pp.append(*pass_cols)
    elif type(pass_cols) == str:
      drop_cols_pp.append(pass_cols)
    else:
      raise NotImplemented

if drop_cols_pp != []:
  X = df.drop(drop_cols_pp, axis = 1)
else:
  X = df.values
if label_col is not None:
  y = df[label_col]
  y.reset_index(drop = True, inplace = True)
else: y = None
if ids_col is not None:
  ids = df[ids_col]
else:
  ids = None
if pass_cols is not None:
  pas = df[pass_cols]
else:
  pas = None

if preprocess_mode == 'infer':
    processed_df = infer_and_preprocess_data(X, scaler=scaler, unique_threshold=unique_threshold)
elif preprocess_mode == 'preprocess':
    processed_df = preprocess_data(df, categorical_cols=categorical_cols, scaler='standard')
else:
   raise NotImplemented

processed_df.to_pickle(preprocess_df_path)
print(f"Preprocessed dataframe saved in {preprocess_df_path}")

manifold_df = apply_manifold_learning(processed_df, technique, manifold_params)
manifold_df.to_pickle(manifold_df_path)
print(f"Dataframe with manifold techniques saved in {manifold_df_path}")

if relationship_mode == 'original':
   relationship_df = df.copy()
elif relationship_mode == 'preprocessed':
   relationship_df = processed_df.copy()
elif relationship_mode == 'manifold':
   relationship_df = manifold_df_path.copy()
else:
  raise NotImplemented


if graph_mode == 'neighbors':
    if graph_params is None:
        G = create_similarity_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = label_col, n_neighbors=5, distance_threshold = None)
    else:
        G = create_similarity_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = label_col, **graph_params)
elif graph_mode =='distance':
    if graph_params is None:
        G = create_distance_threshold_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = label_col, distance_function = 2, distance_threshold = 1.0)
    else:
        G = create_distance_threshold_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = label_col, **graph_params)
elif graph_mode == 'similarity':
    if graph_params is None:
        G = create_similarity_threshold_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = label_col, similarity_function = cos_sim, similarity_threshold =0.995)
    else:
        G = create_similarity_threshold_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = label_col, **graph_params)
else:
   raise NotImplemented

with open(graph_path, 'wb') as fp:
   pickle.dump(G, fp)

print(G)
print(f"This graph has been saved in: {graph_path}")

if label_col is not None:
    df_neigh = analyze_neighborhood_attributes(G, attribute_name = label_col)
    probabilities = print_neighbors_prob(df_neigh, label_col, neigh_prob_path)
    # Display the results
    for (i, j), prob in probabilities.items():
        print(f"P({j}|{i}) = {prob}")
    with open(neigh_prob_path, 'w') as fp:
        for (i, j), prob in probabilities.items():
            fp.write(f"P({j}|{i}) = {prob}")   

    heat_map_prob(probabilities, df_neigh, label_col, prob_heatmap_path)

# degree_distributions(G, degree_distribution_outpath)
betweenness_distributions(G, betweenness_distribution_outpath)
plot_community_composition(G, label_col, community_composition_outpath)
if plot_graph:
   pos = manifold_df.values
   matplotlib_graph_visualization(G, graph_visualization_path, palette = 'viridis', pos = pos)