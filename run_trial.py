# For installation of DeepBayes see https://stackoverflow.com/questions/23075397/python-how-to-edit-an-installed-package
import deepbayes.optimizers as optimizers
from tensorflow import keras
import tensorflow as tf
import os, contextlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from DNN_FGSM import DNN_FGSM
from BNN_FGSM import BNN_FGSM
from datetime import datetime
from tqdm import tqdm
from tensorflow.python.ops.numpy_ops import np_config
import random

np_config.enable_numpy_behavior()

def preprocess_data():
    # 1. Load UCI adult dataset
    features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]

    data_path = "./data/adult.data"
    input_data = pd.read_csv(data_path, names=features,
                             sep=r'\s*,\s*', engine='python', na_values="?")
   # input_data = input_data[:len(input_data)//2] #Half list of values for now
    print("\n>>Using ", len(input_data), " data points")
    # 2. Clean up dataset
    # Binary classification: 1 = >50K, 0 = <=50K
    y = (input_data['salary'] == '>50K').astype(int)

    # Features x contain categorical data, so use pandas.get_dummies to convert to one-hot encoding
    x = (input_data
         .drop(columns=['salary'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))

    # Normalise data
    x = x/np.max(x)

    # 3. Split data into training and test sets
    # 80% training, 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y)

    # 4. Preprocess training and test data
    # Flattening data to 1D vector
    x_train = x_train.values.reshape(x_train.shape[0], -1)
    x_test = x_test.values.reshape(x_test.shape[0], -1)

    # Convert class vectors to binary class matrices using one-hot encoding
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test


def train_DNN_model(x_train, y_train, model):
    # Printing the model to standard output
    model.summary()

    # Initialising some training parameters
    loss = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam()
    batch_size = 128
    epochs = 15     # See EarlyStopping callback below
    validation_split = 0.1
    # callbacks = [keras.callbacks.EarlyStopping(patience=2)]

    model.compile(loss=loss,
                  optimizer=optimizer, metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_split=validation_split)

    return model


def train_BNN_model(x_train, y_train, x_test, y_test, model, fair_reg=False, fair_epsilons=[]):
    # Printing the model to standard output
    model.summary()

    # Initialising some training parameters
    loss = keras.losses.CategoricalCrossentropy()
    # Deepbayes Adam optimizer doesn't seem to work properly
    # opt = optimizers.Adam()
    optimizer = optimizers.VariationalOnlineGuassNewton()
    batch_size = 128
    epochs = 15
    # validation_split = 0.1

    bayes_model = optimizer.compile(
        model, loss_fn=loss, batch_size=batch_size, epochs=epochs, 
        robust_train= 5 if fair_reg else 0,
        fair_epsilons=fair_epsilons) # train w/ fairness regularisation if true 

    # Why does it need x_test and y_test for training?
    bayes_model.train(x_train, y_train, x_test, y_test)

    return bayes_model


def get_adversarial_examples(model, test_data, epsilon, type):
    print(">>Creating adversarial examples")
    epsilons = np.full(100, epsilon)

    # Index 58 is the feature for gender (0 for Female, 1 for Male)
    epsilons[58] = 1.0
    adversarial_examples = np.ndarray(shape=(test_data.shape))

    for i in range(len(test_data)):
        if type == "DNN":
            adversarial = DNN_FGSM(
                model, test_data[i], keras.losses.categorical_crossentropy, epsilons)
        elif type == "BNN":
            adversarial = BNN_FGSM(
                model, test_data[i], keras.losses.categorical_crossentropy, epsilons)
        adversarial_examples[i] = adversarial
    print("Done.")
    return adversarial_examples


def get_fairness_score_basic(x_test_predictions, x_test_adversarial_predictions, type):
    classes = ["<=50K", ">50K"]
    count = 0

    for i in range(len(x_test_predictions)):
        if classes[np.argmax(x_test_predictions[i])] != classes[np.argmax(
                x_test_adversarial_predictions[i])]:
            count += 1

    basic_score = (1 - (count / len(x_test_predictions)))
    print(count, "/", len(x_test_predictions),
          "individuals classified differently after adversarial attack")
    print("Basic Fairness Score:", basic_score, "\n")

    return basic_score


def get_fairness_score(x_test_predictions, x_test_adversarial_predictions, type):
    differences = []

    if type == "DNN":
        for i in range(len(x_test_predictions)):
            difference = abs(x_test_predictions[i][0] -
                             x_test_adversarial_predictions[i][0])
            differences.append(difference)
    elif type == "BNN":
        for i in range(len(x_test_predictions)):
            difference = abs(x_test_predictions[i][0] -
                             x_test_adversarial_predictions[i][0]).numpy()
            differences.append(difference)

    max_diff = max(differences)
    min_diff = min(differences)
    avrg_diff = np.mean(differences)

    print("Maximum Difference:", max_diff)
    print("Minimum Difference:", min_diff)
    print("Average (Mean) Difference:", avrg_diff, "\n")

    return max_diff, min_diff, avrg_diff

def threshold_fairness_score(x_test_predictions, x_test_adversarial_predictions, delta, type):
    differences = []

    if type == "DNN":
        for i in range(len(x_test_predictions)):
            difference = abs(x_test_predictions[i][0] -
                             x_test_adversarial_predictions[i][0])
            differences.append(difference)
    elif type == "BNN":
        for i in range(len(x_test_predictions)):
            difference = abs(x_test_predictions[i][0] -
                             x_test_adversarial_predictions[i][0]).numpy()
            differences.append(difference)

    max_diff = max(differences)
    thres_score = (delta - max_diff)/delta
    print("Threshold adjusted fairness score : ", thres_score)
    return thres_score

def get_results(model, x_test, y_test, epsilon, delta, type, display_text):
    # Get predictions for x_test data (without attack)
    x_test_predictions = model.predict(x_test)

    # Get numpy array of x_test data converted to adversarial examples
    x_test_adversarial = get_adversarial_examples(model, x_test, epsilon, type)

    # Get predictions for x_test_adversarial data (with attack)
    x_test_adversarial_predictions = model.predict(x_test_adversarial)

    # Get accuracy of model
    if type == "DNN":
        score = model.evaluate(x_test, y_test)
        accuracy = score[1]
    elif type == "BNN":
        test_acc = np.mean(np.argmax(x_test_predictions, axis=1)
                           == np.argmax(y_test, axis=1))
        accuracy = test_acc

    print(f"\n ❗️{display_text} RESULTS❗️")
    print("Accuracy:", accuracy, "\n")

    basic_score = get_fairness_score_basic(
        x_test_predictions, x_test_adversarial_predictions, type)
    
    thres_score = threshold_fairness_score(x_test_predictions, x_test_adversarial_predictions, delta, type)

    max_diff, min_diff, avrg_diff = get_fairness_score(x_test_predictions,
                                                       x_test_adversarial_predictions, type)

    return basic_score, (max_diff, min_diff, avrg_diff) , thres_score, accuracy

def trainModel(neurons, layers, x_train, y_train, x_test, y_test, epsilons=[], train_dnn=True, train_bnn=True, fairness_reg=False):
        input_shape = x_train.shape[1]
        num_classes = y_train.shape[1]
        # Reason why model_DNN and model_BNN are defined separately (even though they're the same)
        # is to do with how Python passes values/objects through functions
        # http://scipy-lectures.org/intro/language/functions.html#passing-by-value

        model_DNN = keras.Sequential()
        model_DNN.add(keras.Input(shape=input_shape))

        model_BNN = keras.Sequential()
        model_BNN.add(keras.Input(shape=input_shape))

        for x in range(layers):
            model_DNN.add(keras.layers.Dense(
                neurons, activation="relu"))
            model_BNN.add(keras.layers.Dense(
                neurons, activation="relu"))

        model_DNN.add(keras.layers.Dense(
            num_classes, activation="softmax"))

        model_BNN.add(keras.layers.Dense(
            num_classes, activation="softmax"))
        
        if train_dnn : 
            trained_model_DNN = train_DNN_model(
            x_train, y_train, model_DNN)
        else : trained_model_DNN = None    
    
        if train_bnn : 
            trained_model_BNN = train_BNN_model(
            x_train, y_train, x_test, y_test, model_BNN, fair_reg=fairness_reg, fair_epsilons=epsilons)
        else : trained_model_BNN = None
        
        return (trained_model_BNN, trained_model_DNN)


def main():
    tf.keras.utils.disable_interactive_logging()
    x_train, x_test, y_train, y_test = preprocess_data()
    adversary_regularisation = True
    delta = 1 
    # Try 0.00, 0.05, 0.10, 0.15, 0.20
    eps = [0.0, 0.05, 0.10, 0.15]

    # Number of hidden layers in the model
    # layers = [1, 2, 3, 4, 5]
    layers = [1, 2, 3, 4, 5]
    neurons = [64, 32, 16, 8, 4, 2]
    # Number of neurons per hidden layer in the model
    # neurons = [64, 32, 16, 8, 4, 2]

    # Measurements we're recording during the trials
    measurements_training = ["BNNAccuracy", "BNNAdversaryAccuracy", "DNNAccuracy", "DNNAdversaryAccuracy",
                    "BNNMaxDifference", "BNNAdversaryMaxDifference", "DNNMaxDifference", "DNNAdversaryMaxDifference", 
                    "BNNMinDifference", "BNNAdversaryMinDifference", "DNNMinDifference", "DNNAdversaryMinDifference",
                    "BNNFairnessScore", "BNNAdversaryFairnessScore", "DNNFairnessScore", "DNNAdversaryFairnessScore"]

    measurements_adv = ["BNNAccuracy", "BNNAdversaryAccuracy",
                    "BNNMaxDifference", "BNNAdversaryMaxDifference",
                    "BNNMinDifference", "BNNAdversaryMinDifference", 
                    "BNNFairnessScore", "BNNAdversaryFairnessScore", 
                    "BNNAvgDiff", "BNNAdversaryAvgDiff"]
    # Order of models tested: L1N64, L2N64,... , L5N64, L1N32, ..., L5N32, ..., L1N2, ..., L5N2
    # Where, L = number of hidden layers (1, 2, 3, 4, 5)
    # and, N = number of neurons per layer, i.e width (64, 32, 16, 8, 4, 2)
    # Order is because of how we want the data to be displayed in heatmap (i.e. first five datapoints for a measurement correspond to the top/first row of heatmap)
    # Heatmap Layout:
    # N64
    # N32
    # N16
    # N8
    # N4
    # N2
    #    L1 L2 L3 L4 L5
    print("Running ", "Adversary Regularisation" if adversary_regularisation else "Adversary Training", "------------------------------------------" )
    # Number of neurons per layer (64, 32, 16, 8, 4, 2)
    for epsilon in eps: 
        print("> Testing epsilon=",epsilon)
        df = pd.DataFrame(columns=measurements_adv if adversary_regularisation else measurements_training)
        for neuron_num in neurons:
            # Number of layers (1, 2, 3, 4, 5)
            for layer_num in tqdm(layers):
            #Training--------------------------------------------------------------------------------------------------------------------------------------
                print("Network : L", layer_num, "N", neuron_num)
                if adversary_regularisation : 
                    #Epsilons 
                    epsilons = np.full(100, epsilon)

                    # Index 58 is the feature for gender (0 for Female, 1 for Male)
                    epsilons[58] = 1.0
                    print(">> Training Ordinary BNN model ...")
                    #Train without regularisation 
                    trained_model_BNN, _ = trainModel(neuron_num, layer_num, x_train, y_train, x_test, y_test, train_dnn=False)
                    print("> Done.\n>>Generating results.")
                    basic_score_BNN, (max_diff_BNN, min_diff_BNN, avg_diff_BNN), delta_BNN_res, accuracy_BNN = get_results(
                        trained_model_BNN, x_test, y_test, epsilon, delta, "BNN", "BNN - Normal, No regularisation")
                    print(">> Training Regularized BNN model ...")
                    #Train with regularisation
                    trained_model_BNN_adv, _ = trainModel(neuron_num, layer_num, x_train, y_train, x_test, y_test, train_dnn=False,
                                                        fairness_reg=True, epsilons=epsilons)
                    print("> Done.\n>>Generating results.")
                    basic_score_BNN_adv, (max_diff_BNN_adv, min_diff_BNN_adv, avg_diff_BNN_adv), delta_BNN_adv_res, accuracy_BNN_adv = get_results(
                        trained_model_BNN_adv, x_test, y_test, epsilon, delta, "BNN", "BNN - Adversarial Regularisation")
                    print(">> Writing out to dataframe ...")
                    new_row = pd.DataFrame([accuracy_BNN, accuracy_BNN_adv, 
                                            max_diff_BNN, max_diff_BNN_adv,  
                                            min_diff_BNN, min_diff_BNN_adv,
                                            delta_BNN_res, delta_BNN_adv_res,
                                            avg_diff_BNN, avg_diff_BNN_adv], index=measurements_adv, columns=[f"L{layer_num}N{neuron_num}"]).T
                    df = pd.concat((df, new_row))
                    print("> Done.\nTrial complete!\n")

                else: 
                    #Train normal BNN, DNN network with no adversarial training 
                    trained_model_BNN, trained_model_DNN = trainModel(neuron_num, layer_num, x_train, y_train, x_test, y_test)
                    # BNN 
                    basic_score_BNN, max_diff_BNN, min_diff_BNN, delta_BNN_res, accuracy_BNN = get_results(
                        trained_model_BNN, x_test, y_test, epsilon, delta, "BNN", "BNN - Normal")
                    
                    # DNN 
                    basic_score_DNN, max_diff_DNN, min_diff_DNN, delta_DNN_res, accuracy_DNN = get_results(
                        trained_model_DNN, x_test, y_test, epsilon, delta, "DNN", "DNN - Normal")
                    
                    #Create adversarials
                    adversarials_BNN = get_adversarial_examples(trained_model_BNN, x_train, epsilon, "BNN")
                    adversarials_DNN = get_adversarial_examples(trained_model_DNN, x_train, epsilon, "DNN")

                    #Concatenate onto training data 
                    # BNN
                    x_train_adv_BNN = np.concatenate([x_train, adversarials_BNN])
                    y_train_adv_BNN = np.concatenate([y_train, y_train]) 
                    # DNN
                    x_train_adv_DNN = np.concatenate([x_train, adversarials_DNN])
                    y_train_adv_DNN = np.concatenate([y_train, y_train]) 

                    #Train models using adversarials 
                    # BNN only
                    trained_model_BNN_adv, _ = trainModel(neuron_num, layer_num, x_train_adv_BNN, y_train_adv_BNN, x_test, y_test, train_dnn=False)
                    # DNN only
                    _, trained_model_DNN_adv = trainModel(neuron_num, layer_num, x_train_adv_DNN, y_train_adv_DNN, x_test, y_test, train_bnn=False)

                    #Get results 
                    # BNN
                    basic_score_BNN_adv, max_diff_BNN_adv, min_diff_BNN_adv, delta_BNN_adv_res, accuracy_BNN_adv = get_results(
                        trained_model_BNN_adv, x_test, y_test, epsilon, delta, "BNN", "BNN - With adversarial")
                    # DNN
                    basic_score_DNN_adv, max_diff_DNN_adv, min_diff_DNN_adv, delta_DNN_adv_res, accuracy_DNN_adv = get_results(
                        trained_model_DNN_adv, x_test, y_test, epsilon, delta, "DNN", "DNN - With adversarial")
                    # Dummy data for debugging
                    # accuracy_DNN = random.random()
                    # accuracy_BNN = random.random()
                    # basic_score_DNN = random.random()
                    # basic_score_BNN = random.random()
                    # max_diff_DNN = random.random()
                    # max_diff_BNN = random.random()
                    # min_diff_DNN = random.random()
                    # min_diff_BNN = random.random()
                    # avrg_diff_DNN = random.random()
                    # avrg_diff_BNN = random.random()

                    new_row = pd.DataFrame([accuracy_BNN, accuracy_BNN_adv, accuracy_DNN, accuracy_DNN_adv,
                                            basic_score_BNN, basic_score_BNN_adv, basic_score_DNN, basic_score_DNN_adv, 
                                            max_diff_BNN, max_diff_BNN_adv, max_diff_DNN, max_diff_DNN_adv, 
                                            min_diff_BNN, min_diff_BNN_adv, min_diff_DNN, min_diff_DNN_adv, 
                                            delta_BNN_res, delta_BNN_adv_res, delta_DNN_res, delta_DNN_adv_res], index=measurements_training, columns=[f"L{layer_num}N{neuron_num}"]).T
                    df = pd.concat((df, new_row))
                    #--------------------------------------------------------------------------------------------------------------------------------------
        print(">> Models tested. Writing out to file ...")
        # Pandas options to display all columns and rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        # pd.set_option('display.max_rows', None)
        # np.set_printoptions(linewidth=100000)
    
        f = open(f"./results/trial_{datetime.now()}_eps_{epsilon}_{'regularisation' if adversary_regularisation else 'fair_training'}.csv", 'a')
        print(df, file=f)
        print("Epsilon=", epsilon, " testing complete.")
    print("Full suite complete.")


if __name__ == "__main__":
    main()
