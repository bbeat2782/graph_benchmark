import json
import matplotlib.pyplot as plt
import numpy as np

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
            if "losses" not in metric and 'change' not in metric:
                median, std = extract_median_std(metrics[metric])
                results[dataset][model][metric] = {
                    'median': median,
                    'std': std
                }
            elif 'change' in metric:
                results[dataset][model]['change_num_layers'] = {}
                for change_metric, histories in metrics[metric].items():
                    if 'losses' not in change_metric:
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


for dataset, model_results in results.items():
    metric_types = ["train_accuracys", "val_accuracys", "test_accuracys"]
    if dataset == 'Peptides-func':
        metric_types = ["train_aps", "val_aps", "test_aps"]

    for metric_type in metric_types:
        plt.figure(figsize=(10, 6))
        legend_added = False  # Track if any labels are added to the plot
        for model, metrics_results in model_results.items():
            if "change_num_layers" in metrics_results:
                if metric_type in metrics_results["change_num_layers"]:
                    change_stats = metrics_results["change_num_layers"][metric_type]
                    medians = change_stats["medians"]
                    stds = change_stats["stds"]
                    
                    # Plot medians
                    plt.plot(
                        medians, label=f"{model} {metric_type.replace('_', ' ').capitalize()} Median"
                    )
                    legend_added = True  # Label added
                    
                    # Plot standard deviation as shaded area
                    plt.fill_between(
                        range(len(medians)), 
                        [m - s for m, s in zip(medians, stds)], 
                        [m + s for m, s in zip(medians, stds)], 
                        alpha=0.2
                    )
                    x_ticks = range(1, len(medians) + 1)
                    plt.xticks(ticks=range(len(medians)), labels=x_ticks)

        plt.title(f"{pluralize_metric(metric_type).replace('_', ' ').capitalize()} - Change in Number of Layers ({dataset})")
        plt.xlabel("Number of layers")
        plt.ylabel(pluralize_metric(metric_type).replace('_', ' ').capitalize())
        if legend_added:
            plt.legend()

        plt.grid(True, color='gray', alpha=0.2)
        
        # Save the figure for this metric type
        plt.savefig(f"figures/{dataset}_change_num_layers_{metric_type}.png", dpi=300)
        
        # Display the plot
        plt.show()
        
        # Close the figure to prevent memory issues
        plt.close()