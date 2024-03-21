# Manifold Table Graph

A Python library for creating a graph representation of tabular data.

## Features

- **Distance Metrics:** Evaluate distances between data points using user-defined metrics.
- **Graph Creation:** Generate graphs using the igraph library with edges based on distance metrics.
- **Visualization:** Plot the graph to understand how different data is distributed

**UPCOMING**
- **Centrality Measurement:** Assess centrality to classify the typicality of entries.
- **Clustering via community detection:** Search for communities in the graph to cluster the data points.

## Quickstart and Demo

### Downloading the Repository

To get started, clone the repository using the following command:

```bash
git clone https://https://github.com/davidetorre92/Manifold-Table-Graph
```
### Installing Requirements
Navigate to the repository directory and install the required dependencies using:
```bash
pip install -r requirements.txt
```
### Demo
The creation of a graph from tabular data happens in two steps:
1. Evaluate the distance between all pairs of nodes
2. Create a graph according to some threshold.

#### 1. Distance Evaluation

```python
python3 bin/graph_definition/evaluate_similarity.py -c config.ini
```

#### 2. Graph Creation
```python
python3 bin/graph_definition/graph_creation.py -c config.ini
```
### Tasks
#### Graph Visualization
Run the script for graph visualization using:
```python
python3 bin/tasks/visualization.py -c config.ini
```
This script will generate:
1. The graph itself
2. The degree distribution
3. The correlation between groups

The path to the corresponding data will be displayed on screen.

## Setup
1. Open the script graph_definition/evaluate_similarity.py.
2. Change the path of the desired file and the output similarity dataframe.
3. Run the script with the following command:

1. Open the script graph_definition/graph_creation.py.
2. Change the name of the similarity dataframe and set the threshold.
3. Run the script with the following command:

# Contacts
For any inquiries or futher discussion regarding this project, we invite to reach out to us. Our contacts are

- [Davide Torre](https://www.linkedin.com/in/davidetorre92/): d[dot]torre[at]iac[dot]cnr[dot]it
- [Davide Chicco](https://davidechicco.it): davidechicco[at]davidechicco[dot]it


