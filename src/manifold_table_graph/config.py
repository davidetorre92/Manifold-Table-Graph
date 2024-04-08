from utils import *
# All path are relative to the main folder refer to README.m for the purpose of each variable
# ManifoldTableGraph

# [preprocessing options]
df_path = 'data/demo/make_blobs.csv'
preprocess_mode = 'infer' # ['infer', 'preprocess']
unique_threshold = 0.0
categorical_cols = None
preprocess_df_path = 'results/demo/preprocesed_df.pickle'
scaler = 'standard' # one among ['standard', 'minmax']
label_col = 'cl' # label to be predicted, leave None for no prediction
ids_col = None # column id
pass_cols = None # columns that should not be preprocessed
drop_cols = None # columns that won't take part to the model

# [manifold learning options]
technique = 'TSNE' # Options are 'TSNE', 'Isomap', 'MDS', 'LLE'
manifold_params = None # None will use pre-set parameters.
manifold_df_path = 'results/demo/manifold_df.pickle'

# [graph learning options]
relationship_mode = 'preprocessed'
graph_mode = 'similarity' # Pick one among ['neighbors', 'distance', 'similarity']
graph_params = {'similarity_function': cos_sim, 'similarity_threshold': 0.95} # dict, parameters to pass to the manifold learning algorithm. None will use pre-set parameters.

# [graph visualization options]
plot_graph = True
save_date_experiment = True
graph_path = 'results/demo/neighbors_graph.pickle'
graph_visualization_path = 'results/demo/graph.pdf'
neigh_prob_path = 'results/demo/neigh_prob.dat'
prob_heatmap_path = 'results/demo/neigh_prob.pdf'
degree_distribution_outpath = 'results/demo/degree.pdf'
betweenness_distribution_outpath = 'results/demo/betweenness.pdf'
community_composition_outpath = 'results/demo/community_composition.pdf'