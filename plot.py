import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

file_path = 'results/result.json'
with open(file_path, "r") as f:
    results_data = json.load(f)

early_stopping_patience = 30

def extract_median_std(data, patience=early_stopping_patience):
    combine = []
    for i, row in data.items():
        combine.append(row[-(patience + 1)])
    return np.median(combine), np.std(combine)

def combine_change_num_layers(data, patience=early_stopping_patience):
    medians, stds = [], []
    for num_layer, history in histories.items():
        median, std = extract_median_std(history, patience=patience)
        medians.append(median)
        stds.append(std)
    return medians, stds

def pluralize_metric(metric_type):
    if metric_type.endswith("ys"):
        return metric_type[:-2] + "ies"
    return metric_type

results = {}

for dataset, models in results_data.items():
    results[dataset] = {}
    for model, metrics in models.items():
        results[dataset][model] = {}
        for metric in metrics.keys():
            if metric == 'elapsed_time':
                results[dataset][model][metric] = {
                    'median': metrics[metric],
                    'std': 0
                }
            elif "losses" not in metric and 'change' not in metric:
                median, std = extract_median_std(metrics[metric])
                results[dataset][model][metric] = {
                    'median': median,
                    'std': std
                }
            elif 'change' in metric:
                results[dataset][model]['change_num_layers'] = {}
                for change_metric, histories in metrics[metric].items():
                    if change_metric == 'elapsed_time':
                        results[dataset][model]['change_num_layers'][change_metric] = {
                            'medians': [histories],
                            'stds': [0]
                        }
                    elif 'losses' not in change_metric:
                        medians, stds = combine_change_num_layers(histories)
                        results[dataset][model]['change_num_layers'][change_metric] = {
                            'medians': medians,
                            'stds': stds
                        }

with open('results/result_extracted.json', "w") as f:
    json.dump(results, f, indent=4)


for dataset, model_results in results.items():
    print(f"Dataset: {dataset}")
    for model, metrics_results in model_results.items():
        print(f"  Model: {model}")
        for metric, stats in metrics_results.items():
            if metric == "change_num_layers":  # Special case for change_num_layers
                print(f"    Metric: {metric}")
                for change_metric, change_stats in stats.items():
                    # print(change_metric)
                    formatted_medians = [f"{median:.4f}" for median in change_stats['medians']]
                    formatted_stds = [f"{std:.4f}" for std in change_stats['stds']]
                    print(f"      Change Metric: {change_metric}")
                    print(f"        Medians by Layer: {formatted_medians}")
                    print(f"        Stds by Layer: {formatted_stds}")
            else:
                print(f"    Metric: {metric} - Median: {stats['median']:.4f}, Std: {stats['std']:.4f}")

global_model_colors = {
    "GCN": "tab:blue",
    "GIN": "tab:orange",
    "GAT": "tab:green",
}

for dataset, model_results in results.items():
    metric_types = ["train_accuracys", "val_accuracys", "test_accuracys"]
    if dataset == 'Peptides-func':
        metric_types = ["train_aps", "val_aps", "test_aps"]
    
    line_styles = {"train_accuracys": ':', "val_accuracys": '-', "test_accuracys": '--'}
    if dataset == 'Peptides-func':
        line_styles = {"train_aps": ':', "val_aps": '-', "test_aps": '--'}
    
    plt.figure(figsize=(12, 8))
    handles = []
    labels = []
    
    for metric_type in metric_types:
        plt.figure(figsize=(10, 6))
        legend_added = False
        for model, metrics_results in model_results.items():
            if "change_num_layers" in metrics_results:
                if metric_type in metrics_results["change_num_layers"]:
                    change_stats = metrics_results["change_num_layers"][metric_type]
                    medians = change_stats["medians"]
                    stds = change_stats["stds"]
                    elapsed_time = metrics_results['elapsed_time']['median']
                    
                    # Use the predefined global color and line style
                    line, = plt.plot(
                        medians,
                        label=f"{model.upper()} {metric_type.replace('_', ' ').capitalize()}",
                        color=global_model_colors.get(model.upper(), "tab:gray"),  # Default to gray if model not found
                        linestyle=line_styles[metric_type]
                    )
                    
                    # Store handles and labels for custom legend ordering
                    handles.append(line)
                    labels.append(f"{model.upper()} {metric_type.replace('_', ' ').capitalize()}")
                    
        if dataset == 'Peptides-func':
            x_ticks = range(4, 3 + len(medians) + 1)
        else:
            x_ticks = range(1, len(medians) + 1)
        plt.xticks(ticks=range(len(medians)), labels=x_ticks)

    # Dynamically construct ordered_labels for the current dataset
    metric_suffix = "aps" if dataset == 'Peptides-func' else "accuracys"
    ordered_models = ['GCN', 'GIN', 'GAT']
    ordered_metrics = [f"train {metric_suffix}", f"val {metric_suffix}", f"test {metric_suffix}"]
    ordered_labels = [f"{model} {metric.capitalize()}" for model in ordered_models for metric in ordered_metrics]
    
    # Sort handles and labels to match the desired order
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: ordered_labels.index(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    
    plt.title(f"Change in Number of Layers ({dataset})")
    plt.xlabel("Number of layers")
    plt.ylabel("Metric Value")
    
    # Add custom legend
    plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize='small')
    
    plt.grid(True, color='gray', alpha=0.2)
    
    # Save the figure for this dataset
    plt.savefig(f"figures/{dataset}_change_num_layers_combined.png", dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    
    # Close the figure to prevent memory issues
    plt.close()