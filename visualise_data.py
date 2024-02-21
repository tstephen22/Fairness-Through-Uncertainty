import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def get_trial_means(df_list):
    #  Read in the all 10 dfs from trials and return the average of each measurement across all trials
    df_means = pd.concat(df_list)
    # sort=False to maintain index/row order
    df_means = df_means.groupby(df_means.index, sort=False).mean()

    return df_means

def get_labels(label): 
    cmap = 'coolwarm'
    bar_label = '\u03B4\u002A'
    if(label == "BNN Fairness Score" or label == "BNN Adversary Fairness Score"):
        cmap = 'Greens'
        bar_label = 'Threshold score'
    
    return (cmap, bar_label)
    

def generate_heatmaps(df, epsilon, delta, is_regularisation=False):

    if not os.path.exists(f"./heatmaps/{"Regularisation" if is_regularisation else "Training"}/heatmaps_{epsilon}/"):
        os.makedirs(
            f"./heatmaps/{"Regularisation" if is_regularisation else "Training"}/heatmaps_{epsilon}/")
                    
    labels = ["BNN Accuracy", "BNN Adversary Accuracy", "DNN Accuracy", "DNN Adversary Accuracy", 
              "BNN Basic Score", "BNN Adversary Basic Score", "DNN Basic Score", "DNN Adversary Basic Score", 
              "BNN Max Difference", "BNN Adversary Max Difference", "DNN Max Difference", "DNN Adversary Max Difference", 
              "BNN Min Difference", "BNN Adversary Min Difference", "DNN Min Difference", "DNN Adversary Min Difference", 
              "BNN Fairness Score", "BNN Adversary Fairness Score", "DNN Fairness Score", "DNN Adversary Fairness Score"]
    
    if is_regularisation : 
        labels = ["BNN Accuracy", "BNN Adversary Accuracy",  
              "BNN Max Difference", "BNN Adversary Max Difference", 
              "BNN Min Difference", "BNN Adversary Min Difference", 
              "BNN Fairness Score", "BNN Adversary Fairness Score",
              "BNN Avg Diff", "BNN Adversary Avg Diff",]

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
            f"./heatmaps/{"Regularisation" if is_regularisation else "Training"}/heatmaps_{epsilon}/{measurement}_{epsilon}.png")


def generate_accuracy_fairness_plots(df, epsilon, is_regularisation=False):

    if not os.path.exists(f"./fairness_acc_plots/{"Regularisation" if is_regularisation else "Training"}/plots_{epsilon}/"):
                    os.makedirs(
                        f"./fairness_acc_plots/{"Regularisation" if is_regularisation else "Training"}/plots_{epsilon}/")
                    
    BNN__adv_labels = ["BNN Adversary Fairness Score", "BNN Adversary Max Difference",
                  "BNN Adversary Min Difference", "BNN Adversary Avg Diff"]
    BNN_labels = ["BNN Fairness Score", "BNN Max Difference",
                  "BNN Min Difference", "BNN Avg Diff"]

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    for label in BNN__adv_labels:
        tmp = label
        measurement = tmp.replace(" ", "")

        plt.figure(figsize=(10, 6))
        plt.plot(df["BNNAdversaryAccuracy"],
                 df[measurement], 'o')
        plt.title(f"{label} vs BNN Adversary Accuracy", fontsize=22)
        plt.xlabel("BNN Adversary Accuracy", fontsize=22)
        plt.ylabel(label, fontsize=22)
        axis = plt.gca()
        axis.set_xlim([0, 1])
        axis.set_ylim([0, 1])
        plt.savefig(
            f"./fairness_acc_plots/{"Regularisation" if is_regularisation else "Training"}/plots_{epsilon}/{measurement}_{epsilon}.png")

    for label in BNN_labels:
        tmp = label
        measurement = tmp.replace(" ", "")

        plt.figure(figsize=(10, 6))
        plt.plot(df["BNNAccuracy"],
                 df[measurement], 'o')
        plt.title(f"{label} vs BNN Accuracy", fontsize=22)
        plt.xlabel("BNN Accuracy", fontsize=22)
        plt.ylabel(label, fontsize=22)
        axis = plt.gca()
        axis.set_xlim([0, 1])
        axis.set_ylim([0, 1])
        plt.savefig(
            f"./fairness_acc_plots/{"Regularisation" if is_regularisation else "Training"}/plots_{epsilon}/{measurement}_{epsilon}.png")


def generate_epsilon_plots(df_list, is_regularisation=False):
    labels = ["BNN Fairness Score", "BNN Max Difference",
              "BNN Min Difference", "BNN Avg Diff", 
              "BNN Adversary Fairness Score", "BNN Adversary Max Difference",
              "BNN Adversary Min Difference", "BNN Adversary Avg Diff"]
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

                for df in df_list:
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

                if not os.path.exists(f"./epsilon_plots/{"Regularisation" if is_regularisation else "Training"}/L{layer_num}N{neuron_num}/"):
                    os.makedirs(
                        f"./epsilon_plots/{"Regularisation" if is_regularisation else "Training"}/L{layer_num}N{neuron_num}/")
                plt.savefig(
                    f"./epsilon_plots/{"Regularisation" if is_regularisation else "Training"}/L{layer_num}N{neuron_num}/{measurement}_L{layer_num}N{neuron_num}.png")


def main():
    eps = ["0.00", "0.05", "0.10", "0.15", "0.20"]
    delta = 1
    is_regularisation = True 
    # Position 0 will hold mean results across 10 trials at epsilon 0.00
    # Position 1 will hold mean results at epsilon 0.05, etc.
    mean_results_by_eps = []

    for epsilon in eps:
        file_names = []
        # Read in all files in the results directory
                           
        for item in Path(f"./results/{"Regularisation" if is_regularisation else "Training"}/epsilon_{epsilon}/").iterdir():
            if item.is_file():
                file_names.append(str(item))

        #  print(file_names)

        df_list = []
        for file in file_names:
            df = pd.read_csv(file, sep="\s+")
            df_list.append(df)

        #  print(df_list)

        df_means = get_trial_means(df_list)

        # print(df_means)

        generate_heatmaps(df_means, epsilon, delta, True, is_regularisation=is_regularisation)

        generate_accuracy_fairness_plots(df_means, epsilon, is_regularisation=is_regularisation)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        f = open(
            f"./results/mean_results/eps_{epsilon}.csv", 'w')
        print(df_means, file=f)

        mean_results_by_eps.append(df_means)

    # print(mean_results_by_eps)
    generate_epsilon_plots(mean_results_by_eps, is_regularisation=is_regularisation)


if __name__ == "__main__":
    main()
