import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime


def get_col_labels(is_regularisation): 
    if is_regularisation : 
        return ["BNN Accuracy", "BNN Fair Training Accuracy",  
                "BNN Max Difference", "BNN Fair Training Max Difference", 
                "BNN Min Difference", "BNN Fair Training Min Difference", 
                "BNN Fairness Score", "BNN Fair Training Fairness Score",
                "BNN Avg Diff", "BNN Fair Training Avg Diff",
                "BNN Recall", "BNN Fair Training Recall", 
                "BNN Precision", "BNN Fair Training Precision", 
                "BNN Mean Entropy", "BNN Fair Training Mean Entropy"]
    else : 
        return ["BNN Accuracy", "BNN Fair Training Accuracy", "DNN Accuracy", "DNN Fair Training Accuracy", 
                "BNN Basic Score", "BNN Fair Training Basic Score", "DNN Basic Score", "DNN Fair Training Basic Score", 
                "BNN Max Difference", "BNN Fair Training Max Difference", "DNN Max Difference", "DNN Fair Training Max Difference", 
                "BNN Min Difference", "BNN Fair Training Min Difference", "DNN Min Difference", "DNN Fair Training Min Difference", 
                "BNN Fairness Score", "BNN Fair Training Fairness Score", "DNN Fairness Score", "DNN Fair Training Fairness Score"]

def get_trial_means(df_list):
    """
    Compiles list of trials down to means for each metric. 
    """
    df_means = pd.concat(df_list)
    df_means = df_means.groupby(df_means.index, sort=False).mean()

    return df_means

def get_labels(label): 
    cmap = 'RdBu'
    bar_label = '\u03B4\u002A'
    if(label == "BNN Fairness Score" or label == "BNN Fair Training Fairness Score"):
        cmap = 'Greens'
        bar_label = 'Threshold score'
    
    return (cmap, bar_label)
    

def generate_heatmaps(df, epsilon, delta, path, is_regularisation=True):
    """
    Generates the heat maps for a given metric vs architecture sizes 
    """
        
    labels = get_col_labels(is_regularisation)

    # Number of hidden layers in the model
    layers = [1, 2, 3, 4, 5]
    # Number of neurons per hidden layer in the model
    neurons = [64, 32, 16, 8, 4, 2]

    for label in labels:
        # The column label for the measurements in the dataframe has no spaces
        tmp = label
        measurement = tmp.replace(" ", "")

        # We want to extract just one column from the dataframe (i.e. one measurement, e.g. DNNMaxDifference)
        results = df[measurement].to_numpy()
        print("Measurement:", measurement)
        print("Results:", results)

        # Then, we want to convert the 1D array to a 2D array of shape (6, 5) (i.e. 6 rows (neurons), 5 columns (layers))
        # So it can be converted to a heatmap easily
        heatmap_data = np.reshape(results, (len(neurons), len(layers)))
        print("Heatmap Formatted Data:", heatmap_data)

        heatmap_df = pd.DataFrame(
            heatmap_data, columns=layers, index=neurons, dtype=float)
        plt.figure(figsize=(10, 6))
        sns.set(font_scale=1.5)

        cmap_label, bar_label = get_labels(label)

        sns.heatmap(heatmap_df,
                    cmap= cmap_label,
                    # annot=True,
                    cbar_kws={'label': bar_label},
                    fmt='.5g',
                    vmin=0,
                    vmax=1)

        plt.title(label + " - " + '\u03B4' + ": " + str(delta), fontsize=22)
        plt.xlabel('Number of Hidden Layers', fontsize=22)
        plt.ylabel('Number of Neurons (width)', fontsize=22)
        plt.savefig(
            f"{path}heatmaps/{measurement}_{epsilon}.png")
        plt.close()


def generate_epsilon_plots(df_list, path, is_regularisation=False):
    """
    Generates the epsilon values vs metrics for a given BNN architecture (such as L4N64 as shown in paper)
    """
    labels = get_col_labels(is_regularisation)

    eps = [0.00, 0.05, 0.10, 0.15, 0.20]
    layers = [1, 2, 3, 4, 5]
    neurons = [2, 4, 8, 16, 32, 64]

    for label in labels:
        for layer_num in layers:
            for neuron_num in neurons:
                tmp = label
                measurement = tmp.replace(" ", "")
                plt.figure(figsize=(10, 6))

                points = []
                print(type(df_list))
                for df in df_list:
                    print(type(df))
                    points.append(
                        df.loc[f"L{layer_num}N{neuron_num}", measurement])
                    print(f"L{layer_num}N{neuron_num}", measurement)
                    print(df.loc[f"L{layer_num}N{neuron_num}", measurement])

                plt.plot(eps, points, 'o-')
                plt.title(f"Epsilon vs {label}", fontsize=22)
                plt.xlabel("Epsilon", fontsize=22)
                plt.ylabel(label, fontsize=22)
                axis = plt.gca()
                axis.set_xlim([0, 0.20])
                axis.set_ylim([0, 1])
                if not os.path.exists(f"{path}EpsilonPlots/L{layer_num}N{neuron_num}/") :
                    os.makedirs(f"{path}EpsilonPlots/L{layer_num}N{neuron_num}/")
                plt.savefig(
                    f"{path}EpsilonPlots/L{layer_num}N{neuron_num}/{measurement}_L{layer_num}N{neuron_num}.png")


def main():
    """
    Generates the mean results across 10 trials for each epsilon value, as well as the heat maps and epsilon plots. 
    Graph outputs will appear in /graphs/Regularisation_plots_[epsilons]_[time of execution]. Epsilon plots will appear in epsilon 0.2.
    """
    eps = ["0.00", "0.05", "0.10", "0.15", "0.20"]
    delta = 1
    is_regularisation = True 
    trial_type = "Regularisation" if is_regularisation else "Training"
    # Position 0 will hold mean results across 10 trials at epsilon 0.00
    # Position 1 will hold mean results at epsilon 0.05, etc.
    mean_results_by_eps = []

    for epsilon in eps:
        file_names = []
        # Read in all files in the results directory
        for item in Path(f"./results/temp/{epsilon}").iterdir():
            if item.is_file():
                file_names.append(str(item))

        df_list = []
        for file in file_names:
            df = pd.read_csv(file, sep="\s+")
            df_list.append(df)

        df_means = get_trial_means(df_list)
        path = f"./graphs/{trial_type}_plots_{epsilon}_{datetime.now()}/"
        if not os.path.exists(path):
                os.makedirs(path)
                os.makedirs(f"{path}heatmaps")

        generate_heatmaps(df_means, epsilon, delta, path, is_regularisation=is_regularisation)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        f = open(
            f"{path}mean_eps_{epsilon}.csv", 'w')
        print(df_means, file=f)

        mean_results_by_eps.append(df_means)
    
    generate_epsilon_plots(mean_results_by_eps, path, is_regularisation=is_regularisation)

if __name__ == "__main__":
    main()
