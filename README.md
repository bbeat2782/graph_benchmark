# Replication of Benchmark Metrics on Graph Neural Networks

In fields like biology, social networks, and transportation systems, it's hard to represent all relevant data in a tabular format. To represent these, we need to use a Graph-based approach. However, early methods struggled to fully grasp the complicated structures of graph data. This project explores various Graph Neural Networks (GNNs) on graph benchmark datasets and evaluates the performance of different approaches. Specifically, by assessing the capability of the transformer-based graph model, it aims to provide insights into its potential and possibility of exceeding traditional GNN architectures' effectiveness.

## Retrieving the data locally:

This repository uses datasets from the PyGeometric package (https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html), and it is specified in a .yaml file in the configs folder. The format is shown below, where `format` indicates one of the classes from the PyGeometric package and `name` indicates a specific dataset you want to retrieve.

```
dataset:
  format: PyG-LRGBDataset
  name: Peptides-func
```

## Running the project

After correctly including paths to the .yaml files, you will be able to run the project by

```
python run.py
```

### Building the project stages using `run.py`

## Reference