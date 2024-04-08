import numpy as np
import pandas as pd
import os
from utils import *
import pickle
from datetime import datetime

class ManifoldGraph:
    """
    This class for graph creation from data.
    """
    
    def __init__(self, df, **config):
        """
        The constructor for the ManifoldGraph class.
        
        Parameters:
        - df: the dataframe. Can be a path/to/dataframe or a pandas dataframe.
        """
        # Default and placeholders
        _PP_INFER_STR = 'infer'
        _PP_PP_STR = 'preprocess'
        _preprocess_modes = [_PP_INFER_STR, _PP_PP_STR]
        _preprocess_default = _PP_INFER_STR
        _unique_threshold_default = 0.0
        _scalers = ['standard', 'minmax']
        _scaler_default = 'standard'

        _RS_OG_STR = 'original'        
        _RS_PP_STR = 'preprocessed'
        _relationship_modes = [_RS_OG_STR, _RS_PP_STR]
        _relationship_default = _RS_PP_STR

        _GP_NG_STR = 'neighbors'        
        _GP_DS_STR = 'distance'
        _GP_SM_STR = 'similarity'
        _graph_modes = [_GP_NG_STR, _GP_DS_STR, _GP_SM_STR]
        _graph_default = _GP_NG_STR

        _MLT_TSNE_STR = 'TSNE'        
        _MLT_ISO_STR = 'Isomap'
        _MLT_MDS_STR = 'MDS'
        _MLT_LLE_STR = 'LLE'
        _manifold_learning_techniques = [_MLT_TSNE_STR, _MLT_ISO_STR, _MLT_MDS_STR, _MLT_LLE_STR]
        _manifold_learning_technique_default = _MLT_TSNE_STR

        _OUTPUT_PATH_PICKLE_KEYS = ['manifold_df_path', 'preprocess_df_path', 'graph_path']
        _OUTPUT_PATH_PDF_KEYS = ['graph_visualization_path', 'neigh_prob_path', 
              'prob_heatmap_path', 'degree_distribution_outpath', 'betweenness_distribution_outpath', 
              'community_composition_outpath']

        if 'preprocess_mode' in config:
            preprocess_mode = config['preprocess_mode']
            if preprocess_mode in _preprocess_modes:
                self.preprocess_mode = preprocess_mode
            else:
                raise NotImplemented
        else:
            # set default
            self.preprocess_mode = _preprocess_default
        if self.preprocess_mode == _PP_INFER_STR:
            if 'unique_threshold' in config:
                self.unique_threshold = config['unique_threshold']
            else:
                self.unique_threshold = _unique_threshold_default
            if 'scaler' in config:
                scaler = config['scaler']
                if scaler in _scalers:
                    self.scaler = scaler
                else:
                    raise NotImplemented
            else:
                self.scaler = _scaler_default
        else:
            self.unique_threshold = None
            self.scaler = None

        if 'manifold_learning_technique' in config:
            manifold_learning_technique = config['manifold_learning_technique']
            if manifold_learning_technique in _manifold_learning_techniques:
                self.manifold_learning_technique = manifold_learning_technique
            else:
                self.manifold_learning_technique = _manifold_learning_technique_default

        if 'manifold_learning_params' in config:
            if type(config['manifold_learning_params']) == dict:
                self.manifold_learning_params = config['manifold_learning_params']
            else:
                raise ValueError


        if 'relationship_mode' in config:
            relationship_mode = config['relationship_mode']
            if relationship_mode in _relationship_modes:
                self.relationship_mode = relationship_mode
            else:
                raise NotImplemented
        else:
            # set default
            self.relationship_mode = _relationship_default

        if 'graph_mode' in config:
            graph_mode = config['graph_mode']
            if graph_mode in _graph_modes:
                self.graph_mode = graph_mode
            else:
                raise NotImplemented
        else:
            # set default
            self.graph_mode = _graph_default

        if 'graph_params' in config:
            if type(config['graph_params']) == dict:
                self.graph_params = config['graph_params']
            else:
                raise ValueError
        else:
            # set default
            self.graph_params = None

        if 'save_date_experiment' in config:
            save_date_experiment = bool(config['save_date_experiment'])
        else:
            # set default
            save_date_experiment = True

        if 'plot_graph' in config:
            self.plot_graph = bool(config['plot_graph'])
        else:
            # set default
            self.plot_graph = True


        for key in _OUTPUT_PATH_PICKLE_KEYS:
            if key in config:
                if config[key].endswith('.pickle'):
                    path = config[key]
                    if save_date_experiment: path = self._append_date_to_filenames(path)
                    setattr(self, key, path)
                else:
                    raise ValueError
            else:
                setattr(self, key, key + '.pickle')
        
        
        for key in _OUTPUT_PATH_PDF_KEYS:
            if key in config:
                if config[key].endswith('.pdf'):
                    path = config[key]
                    setattr(self, key, path)
                else:
                    raise ValueError
            else:
                setattr(self, key, key + '.pdf')

        if 'drop_cols' in config:
            self.drop_cols = config['drop_cols']
        else:
            self.drop_cols = None

        if 'ids_col' in config:
            self.ids_col = config['ids_col']
        else:
            self.ids_col = None

        if 'label_col' in config:
            self.label_col = config['label_col']
        else:
            self.label_col = None

        if 'pass_col' in config:
            self.pass_col = config['pass_col']
        else:
            self.pass_col = None

        self.df = self.__read_df__(df)

    def _append_date_to_filenames(self, path):
        current_datetime = datetime.now().strftime("_%Y%m%d%H%M")
        dirname, basename = os.path.split(path)
        name, ext = os.path.splitext(basename)
        updated_name = f"{name}{current_datetime}{ext}"
        updated_path = os.path.join(dirname, updated_name)
        return updated_path


    def __read_df__(self, df):
        if type(df) == pd.DataFrame: return df
        elif type(df) == str:
            df_path = df
            # Check if the dataframe exists
            if os.path.exists(df_path):
                pass
            else:
                raise FileNotFoundError(f"File containing data not found in {df_path}")
            try:
                if df_path.endswith('.csv'):
                    # read the first row of the CSV to determine if the first column is an index
                    peek_df = pd.read_csv(df_path, nrows=1)
                    # check if the first column looks like an index (e.g., unnamed or follows a specific pattern)
                    if peek_df.columns[0].startswith('Unnamed') or peek_df.columns[0].isdigit():
                        df = pd.read_csv(df_path, index_col=0)
                    else:
                        df = pd.read_csv(df_path)
                elif df_path.endswith('.xlsx'):
                    df = pd.read_excel(df_path, index_col=None)
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
            return df
        else:
            raise ValueError
        
    def run(self):
        if self.drop_cols is not None:
            df = df.drop(self.drop_cols, axis = 1)

        # Step 1: preprocess
        # Clean from nans
        n_nan_rows_before = self.df.shape[0]
        self.df.dropna(inplace=True)
        n_nan_rows_after = df.shape[0]
        n_rows_dropped = n_nan_rows_before - n_nan_rows_after

        print(f"Deleted {n_rows_dropped} rows containing NaNs.")
        # Now, preprocess.
        drop_cols_pp = []

        if self.ids_col is not None: drop_cols_pp.append(self.ids_col)
        if self.label_col is not None: drop_cols_pp.append(self.label_col)
        if self.pass_cols is not None: 
            if type(self.pass_cols) == list:
                drop_cols_pp.append(*self.pass_cols)
            elif type(self.pass_cols) == str:
                drop_cols_pp.append(self.pass_cols)
            else:
                raise NotImplemented

        if drop_cols_pp != []:
            X = df.drop(drop_cols_pp, axis = 1)
        else:
            X = df.values
        if self.label_col is not None:
            y = df[self.label_col]
            y.reset_index(drop = True, inplace = True)
        else: y = None
        
        if self.ids_col is not None:
            ids = df[self.ids_col]
        else:
            ids = None
        if self.pass_cols is not None:
            pas = df[self.pass_cols]
        else:
            pas = None

        if self.preprocess_mode == 'infer':
            processed_df = infer_and_preprocess_data(X, scaler=self.scaler, unique_threshold=self.unique_threshold)
        elif self.preprocess_mode == 'preprocess':
            processed_df = preprocess_data(df, categorical_cols=self.categorical_cols, scaler='standard')
        else:
            raise NotImplemented

        processed_df.to_pickle(self.preprocess_df_path)
        print(f"Preprocessed dataframe saved in {self.preprocess_df_path}")

        manifold_df = apply_manifold_learning(processed_df, self.manifold_learning_technique, self.manifold_learning_params)
        manifold_df.to_pickle(self.manifold_df_path)
        print(f"Dataframe with manifold techniques saved in {self.manifold_df_path}")

        if self.relationship_mode == 'original':
            relationship_df = df.copy()
        elif self.relationship_mode == 'preprocessed':
            relationship_df = processed_df.copy()
        else:
            raise NotImplemented


        if self.graph_mode == 'neighbors':
            if self.graph_params is None:
                G = create_similarity_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = self.label_col, n_neighbors=5, distance_threshold = None)
            else:
                G = create_similarity_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = self.label_col, **self.graph_params)
        elif self.graph_mode =='distance':
            if self.graph_params is None:
                G = create_distance_threshold_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = self.label_col, distance_function = 2, distance_threshold = 1.0)
            else:
                G = create_distance_threshold_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = self.label_col, **self.graph_params)
        elif self.graph_mode == 'similarity':
            if self.graph_params is None:
                G = create_similarity_threshold_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = self.label_col, similarity_function = cos_sim, similarity_threshold =0.995)
            else:
                G = create_similarity_threshold_graph(df, relationship_df, classification_attribute = y, classification_attribute_name = self.label_col, **self.graph_params)
        else:
            raise NotImplemented

        with open(self.graph_path, 'wb') as fp:
            pickle.dump(G, fp)

        print(G)
        print(f"This graph has been saved in: {self.graph_path}")

        if self.label_col is not None:
            df_neigh = analyze_neighborhood_attributes(G, attribute_name = self.label_col)
            probabilities = print_neighbors_prob(df_neigh, self.label_col, self.neigh_prob_path)
            # Display the results
            for (i, j), prob in probabilities.items():
                print(f"P({j}|{i}) = {prob}")
            with open(self.neigh_prob_path, 'w') as fp:
                for (i, j), prob in probabilities.items():
                    fp.write(f"P({j}|{i}) = {prob}")   

            heat_map_prob(probabilities, df_neigh, self.label_col, self.prob_heatmap_path)

        degree_distributions(G, self.degree_distribution_outpath)
        betweenness_distributions(G, self.betweenness_distribution_outpath)
        plot_community_composition(G, self.label_col, self.community_composition_outpath)
        if self.plot_graph:
            pos = manifold_df.values
            matplotlib_graph_visualization(G, self.graph_visualization_path, palette = 'viridis', pos = pos)
