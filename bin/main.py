import numpy as np
import pandas as pd

from utils import *
from config import *

import pickle

if df_path.endswith('.csv'):
    df = pd.read_csv(df_path, index_col = 0)
elif df_path.endswith('.xlsx'):
    df = pd.read_excel(df_path, index_col = False)
elif df_path.endswith('.pickle'):
    df = pd.read_pickle(df_path)
else: raise NotImplemented

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
# elif preprocess_mode = 'preprocess':
#     processed_df = preprocess_data(df, numerical_cols, scaler='standard', categorical_cols=):
else:
   raise NotImplemented

processed_df.to_pickle(preprocess_df_path)
print(f"Preprocessed dataframe saved in {preprocess_df_path}")

manifold_df = apply_manifold_learning(processed_df, technique, manifold_params)
manifold_df.to_pickle(manifold_df_path)
print(f"Dataframe with manifold techniques saved in {manifold_df_path}")

if graph_mode == 'neighbors':
    if graph_params is None:
        G = create_similarity_graph(df, processed_df, n_neighbors=5, distance_threshold = None, classification_attribute = y, classification_attribute_name = label_col)
    else:
        G = create_similarity_graph(df, processed_df, classification_attribute = y, classification_attribute_name = label_col, **graph_params)
elif graph_mode =='distance':
    if graph_params is None:
        G = create_distance_threshold_graph(df, processed_df, distance_function = 2, distance_threshold = 1.0, classification_attribute = y, classification_attribute_name = None)
    else:
        G = create_distance_threshold_graph(df, processed_df, classification_attribute = y, classification_attribute_name = None, **graph_params)
elif graph_mode == 'similarity':
    if graph_params is None:
        G = create_similarity_threshold_graph(df, processed_df, classification_attribute = y, classification_attribute_name = None, similarity_function = cos_sim, similarity_threshold =0.995)
    else:
        G = create_similarity_threshold_graph(df, processed_df, classification_attribute = y, classification_attribute_name = None, **graph_params)
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