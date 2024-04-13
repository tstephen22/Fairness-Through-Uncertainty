# Originally created by Alice Doherty https://github.com/alicedoherty/bayesian-individual-fairness
# Configured and extended on by Theo Stephens to create an individual fairness regulariser, as specified in https://github.com/tstephen22/Fairness-Through-Awareness-BNN
# For installation of DeepBayes see https://stackoverflow.com/questions/23075397/python-how-to-edit-an-installed-package
import deepbayes.optimizers as optimizers
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from scoring import get_adversarial_examples, get_results, timed 
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def preprocess_data():
    """
    Preprocess the Adult dataset https://archive.ics.uci.edu/dataset/2/adult  
    """
    # Load adult dataset
    features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]

    data_path = "./data/adult.data"
    input_data = pd.read_csv(data_path, names=features,
                             sep=r'\s*,\s*', engine='python', na_values="?")

    print("\n>>Using ", len(input_data), " data points")
    # Clean up dataset
    # Binary classification: 1 = >50K, 0 = <=50K
    y = (input_data['salary'] == '>50K').astype(int)

    # Features x contain categorical data, so use pandas.get_dummies to convert to one-hot encoding
    x = (input_data
         .drop(columns=['salary'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))

    # Normalise data
    x = x/np.max(x)

    # Split data into training and test sets
    # 80% training, 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y)

    # Preprocess training and test data
    # Flattening data to 1D vector
    x_train = x_train.values.reshape(x_train.shape[0], -1)
    x_test = x_test.values.reshape(x_test.shape[0], -1)

    # Convert class vectors to binary class matrices using one-hot encoding
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test

# TRAINING --------------------------------------------------------------------------------------------------------------------

def train_DNN_model(x_train, y_train, model):
    """
    This is no longer used. Artifact from https://github.com/alicedoherty/bayesian-individual-fairness. 
    """
    # Printing the model to standard output
    model.summary()

    # Initialising training parameters
    loss = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam()
    batch_size = 128
    epochs = 15  
    validation_split = 0.1

    model.compile(loss=loss,
                  optimizer=optimizer, metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_split=validation_split)

    return model


def train_BNN_model(x_train, y_train, x_test, y_test, model, fair_reg=False, fair_epsilons=[]):
    """
    Trains our BNN model using VOGN and with fairness regularisation if specified.  
    To see the code for our regularisation method, see /deepbayes/deepbayes/optimizers/blrvi.py line 139 
    """
    # Printing the model to standard output
    model.summary()

    # Initialising some training parameters
    loss = keras.losses.CategoricalCrossentropy()
    # Our inference method 
    optimizer = optimizers.VariationalOnlineGuassNewton()
    batch_size = 128
    epochs = 15

    bayes_model = optimizer.compile(
        model, loss_fn=loss, batch_size=batch_size, epochs=epochs, 
        robust_train= 5 if fair_reg else 0,
        rob_lam=0.5,
        fair_epsilons=fair_epsilons) # train w/ fairness regularisation if true 

    bayes_model.train(x_train, y_train, x_test, y_test)

    return bayes_model

def trainModel(neurons, layers, x_train, y_train, x_test, y_test, epsilons=[], train_dnn=True, train_bnn=True, fairness_reg=False):
        """
        Constructs the architecture of our BNN before training it. 
        - This code was originally used by Alice Doherty https://github.com/alicedoherty/bayesian-individual-fairness
         """
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

trainModel = timed(trainModel) # Used to time the models training 

# --------------------------------------------------------------------------------------------------------------------

def main():
    """
    Executes all trials for the experiment. 
    - The outputs go in increasing layers with decreasing neurons (So L1N64, L1N32, L1N16, ... L2N64, L2N32, ... etc)
    - Outputs can be found in /results/temp under their corresponding epsilon value 
    - The calculation of metrics can be found in scoring.py 
    - To get the mean of the metrics for the trials, please run visualise.py 
    - Quite a big for loop, so we have set the number of trials and architecture sizes down if an example run is needed. Uncomment the block below to run the full experiment again.
    """
    tf.keras.utils.disable_interactive_logging()
    x_train, x_test, y_train, y_test = preprocess_data()

    fairness_regularisation = True
    delta = 1
    # Original experiment parameters  
    # trials = 5
    # eps = [0.00, 0.05, 0.10, 0.15, 0.20] #non-protected attribute epsilon values 0.0, 0.05, 0.10, 0.15, 0.20 
    # layers = [1, 2, 3, 4, 5]  #Number of hidden layers in the model
    # neurons = [64, 32, 16, 8, 4, 2] #Number of neurons per hidden layer in the model
    trials = 1
    eps = [0.05, 0.10, 0.15] #non-protected attribute epsilon values 
    layers = [1, 2, 3]  #Number of hidden layers in the model
    neurons = [8, 4, 2] #Number of neurons per hidden layer in the model

    # Measurements we're recording during the trials
    measurements_training = ["BNNAccuracy", "BNNFairTrainingAccuracy", "DNNAccuracy", "DNNFairTrainingAccuracy",
                    "BNNMaxDifference", "BNNFairTrainingMaxDifference", "DNNMaxDifference", "DNNFairTrainingMaxDifference", 
                    "BNNMinDifference", "BNNFairTrainingMinDifference", "DNNMinDifference", "DNNFairTrainingMinDifference",
                    "BNNFairnessScore", "BNNFairTrainingFairnessScore", "DNNFairnessScore", "DNNFairTrainingFairnessScore"]

     # Measurements used in preliminary experiments, not used here 
    measurements_fairness = ["Layer", "Neurons",
                    "BNNAccuracy", "BNNFairTrainingAccuracy",
                    "BNNMaxDifference", "BNNFairTrainingMaxDifference",
                    "BNNMinDifference", "BNNFairTrainingMinDifference", 
                    "BNNFairnessScore", "BNNFairTrainingFairnessScore", 
                    "BNNAvgDiff", "BNNFairTrainingAvgDiff",
                    "BNNRecall", "BNNFairTrainingRecall",
                    "BNNPrecision", "BNNFairTrainingPrecision",
                    "BNNMeanEntropy", "BNNFairTrainingMeanEntropy",
                    "BNNTime", "BNNFairTrainingTime"]
   
    print("Running ", "Fairness Regularisation" if fairness_regularisation else "Fairness Training", " for ", str(trials), " trials ------------------------------------------" )
    # Number of neurons per layer (64, 32, 16, 8, 4, 2)
    for trial in range(trials): 
        for epsilon in eps: 
            print("> Testing epsilon=",epsilon)
            df = pd.DataFrame(columns=measurements_fairness if fairness_regularisation else measurements_training)
            for neuron_num in tqdm(neurons):
                # Number of layers (1, 2, 3, 4, 5)
                for layer_num in layers:
                #Training--------------------------------------------------------------------------------------------------------------------------------------
                    print("Network : L", layer_num, "N", neuron_num)
                    if fairness_regularisation : 
                        #Epsilons 
                        epsilons = np.full(100, epsilon)

                        # Index 58 is the feature for gender (0 for Female, 1 for Male)
                        epsilons[58] = 1.0
                        print(">> Training Ordinary BNN model ...")
                        #Train without regularisation 
                        (trained_model_BNN, _), time_BNN = trainModel(neuron_num, layer_num, x_train, y_train, x_test, y_test, train_dnn=False)
                        print(f"> Done in {time_BNN} s.\n>>Generating results.")
                        (max_diff_BNN, min_diff_BNN, avg_diff_BNN), delta_BNN_res, (accuracy_BNN, recall_BNN, precision_BNN, entropy_BNN) = get_results(
                            trained_model_BNN, x_test, y_test, epsilon, delta, "BNN", "BNN - Normal, No regularisation")
                        print(">> Training Regularized BNN model ...")
                        #Train with regularisation
                        (trained_model_BNN_adv, _), time_BNN_adv = trainModel(neuron_num, layer_num, x_train, y_train, x_test, y_test, train_dnn=False,
                                                            fairness_reg=True, epsilons=epsilons)
                        print(f"> Done in {time_BNN_adv}.\n>>Generating results.")
                        (max_diff_BNN_adv, min_diff_BNN_adv, avg_diff_BNN_adv), delta_BNN_adv_res, (accuracy_BNN_adv, recall_BNN_adv, precision_BNN_adv, entropy_BNN_adv) = get_results(
                            trained_model_BNN_adv, x_test, y_test, epsilon, delta, "BNN", "BNN - Adversarial Regularisation")
                        print(">> Writing out to dataframe ...")
                        new_row = pd.DataFrame([layer_num, neuron_num,
                                                accuracy_BNN, accuracy_BNN_adv, 
                                                max_diff_BNN, max_diff_BNN_adv,  
                                                min_diff_BNN, min_diff_BNN_adv,
                                                delta_BNN_res, delta_BNN_adv_res,
                                                avg_diff_BNN, avg_diff_BNN_adv,
                                                recall_BNN, recall_BNN_adv,
                                                precision_BNN, precision_BNN_adv, 
                                                entropy_BNN, entropy_BNN_adv,
                                                time_BNN, time_BNN_adv], 
                                                index=measurements_fairness if fairness_regularisation else measurements_training, 
                                                columns=[f"L{layer_num}N{neuron_num}"]).T
                        df = pd.concat((df, new_row))
                        print("> Done.\nTrial complete!\n")

                    else: 
                        #
                        # This section was used in preliminary studies to see the effect of training BNNs on a dataset of fair examples 
                        #
                        #Train normal BNN, DNN network with no adversarial training 
                        trained_model_BNN, trained_model_DNN = trainModel(neuron_num, layer_num, x_train, y_train, x_test, y_test)
                        # BNN 
                        max_diff_BNN, min_diff_BNN, delta_BNN_res, accuracy_BNN = get_results(
                            trained_model_BNN, x_test, y_test, epsilon, delta, "BNN", "BNN - Normal")
                        
                        # DNN 
                        max_diff_DNN, min_diff_DNN, delta_DNN_res, accuracy_DNN = get_results(
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
                        max_diff_BNN_adv, min_diff_BNN_adv, delta_BNN_adv_res, accuracy_BNN_adv = get_results(
                            trained_model_BNN_adv, x_test, y_test, epsilon, delta, "BNN", "BNN - With adversarial")
                        # DNN
                        max_diff_DNN_adv, min_diff_DNN_adv, delta_DNN_adv_res, accuracy_DNN_adv = get_results(
                            trained_model_DNN_adv, x_test, y_test, epsilon, delta, "DNN", "DNN - With adversarial")

                        new_row = pd.DataFrame([accuracy_BNN, accuracy_BNN_adv, accuracy_DNN, accuracy_DNN_adv,
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
        
            f = open(f"./results/trial_{datetime.now()}_eps_{epsilon}_{'regularisation' if fairness_regularisation else 'fair_training'}.csv", 'a')
            print(df, file=f)
            print("Epsilon=", epsilon, " testing complete.")
        print("Full suite complete.")
    print("All ", str(trials), " trials complete.")


if __name__ == "__main__":
    main()
