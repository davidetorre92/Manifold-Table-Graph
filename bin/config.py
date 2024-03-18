# All path are relative to the main folder
# ManifoldTableGraph

# [preprocess]
df_path = 'data/demo/make_blobs.csv' # relative/path/to/df
preprocess_mode = 'infer' # mode: 'infer' if the categorical is unknow, 'preprocess' otherwise
preprocess_df_path = 'results/demo/preprocesed_df.pickle'
scaler = 'standard' # one among ['standard', 'minmax']
label_col = 'cl' # label to be predicted, leave None for no prediction
ids_col = None # column id
pass_cols = None # columns that should not be preprocessed
drop_cols = None # columns that won't take part to the model
unique_threshold = 0.0

# [manifold learning]
technique = 'TSNE' # name of the manifold learning technique to apply. Options are 'TSNE', 'Isomap', 'MDS', 'LLE
manifold_params = None # dict, parameters to pass to the manifold learning algorithm. None will use pre-set parameters.
manifold_df_path = 'results/demo/manifold_df.pickle'

# [graph options]
graph_mode = 'neighbors' # option for the graph. Pick one among ['neighbors', 'distance', 'similarity']
graph_params = None # dict, parameters to pass to the manifold learning algorithm. None will use pre-set parameters.
plot_graph = True
graph_path = 'results/demo/neighbors_graph.pickle'
graph_visualization_path = 'results/demo/graph.pdf'
neigh_prob_path = 'results/demo/neigh_prob.dat'
prob_heatmap_path = 'results/demo/neigh_prob.pdf'
degree_distribution_outpath = 'results/demo/degree.pdf'
betweenness_distribution_outpath = 'results/demo/betweenness.pdf'
community_composition_outpath = 'results/demo/community_composition.pdf'