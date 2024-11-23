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

The following versions are used for this project.

```
python3 -m venv graph_benchmark
source graph_benchmark/bin/activate
pip install ipython ipykernel
ipython kernel install —user —name=graph_benchmark
pip install torch==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.6.1
pip install networkx==3.3
pip install numpy==1.26.4
pip install scikit-learn==1.4.2
pip install scipy==1.13.0
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install ogb==1.3.6
pip install tensorboardX==2.6.2.2
pip install performer-pytorch==1.1.4
pip install pytorch-lightning==2.4.0
pip install yacs==0.1.8
pip install torchmetrics==1.5.2
pip install matplotlib==3.9.2
pip install torch_scatter==2.1.2
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install transformers TODO which version
```

After correctly installing the packages and including paths to the .yaml files in `run.py`, you will be able to run the project by

```
python run.py
```

It took about 21 hours to run the current configurations with 2080ti GPU.

## Reference

This project uses GraphGPS code from the [`GraphGPS`](https://github.com/rampasek/GraphGPS) repository. The original code has been modified to fit the specific requirements and functionality of this implementation.

For more details about the original implementation, please visit the official repository: [https://github.com/rampasek/GraphGPS](https://github.com/rampasek/GraphGPS).