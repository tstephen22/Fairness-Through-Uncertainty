from DNN_FGSM import DNN_FGSM
from BNN_FGSM import BNN_FAIR_FGSM
import pandas as pd
import numpy as np
from time import time
from tensorflow import keras
from deepbayes.analyzers import predictive_entropy


def timed(func):

    def timed_wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        end = time()-t1
        return result, end
    return timed_wrapper

def get_fairness_examples(model, test_data, epsilon, type):
    print(">>Creating fairness examples")
    epsilons = np.full(100, epsilon)

    # Index 58 is the feature for gender (0 for Female, 1 for Male)
    epsilons[58] = 1.0
    adversarial_examples = np.ndarray(shape=(test_data.shape))

    for i in range(len(test_data)):
        if type == "DNN":
            adversarial = DNN_FGSM(
                model, test_data[i], keras.losses.categorical_crossentropy, epsilons)
        elif type == "BNN":
            adversarial = BNN_FAIR_FGSM(
                model, test_data[i], keras.losses.categorical_crossentropy, epsilons)
        adversarial_examples[i] = adversarial
    print("Done.")
    return adversarial_examples

def get_fairness_score(x_test_predictions, x_test_fair_examples_predictions, type):
    differences = []

    if type == "DNN":
        for i in range(len(x_test_predictions)):
            difference = abs(x_test_predictions[i][0] -
                             x_test_fair_examples_predictions[i][0])
            differences.append(difference)
    elif type == "BNN":
        for i in range(len(x_test_predictions)):
            difference = abs(x_test_predictions[i][0] -
                             x_test_fair_examples_predictions[i][0]).numpy()
            differences.append(difference)

    max_diff = max(differences)
    min_diff = min(differences)
    avrg_diff = np.mean(differences)

    print("Maximum Difference:", max_diff)
    print("Minimum Difference:", min_diff)
    print("Average (Mean) Difference:", avrg_diff, "\n")

    return max_diff, min_diff, avrg_diff

def threshold_fairness_score(x_test_predictions, x_test_fair_examples_predictions, delta, type):
    differences = []

    if type == "DNN":
        for i in range(len(x_test_predictions)):
            difference = abs(x_test_predictions[i][0] -
                             x_test_fair_examples_predictions[i][0])
            differences.append(difference)
    elif type == "BNN":
        for i in range(len(x_test_predictions)):
            difference = abs(x_test_predictions[i][0] -
                             x_test_fair_examples_predictions[i][0]).numpy()
            differences.append(difference)

    max_diff = max(differences)
    thres_score = (delta - max_diff)/delta
    print("Threshold adjusted fairness score : ", thres_score)
    return thres_score

def get_accuracy(preds, true): 
    return np.mean(np.argmax(preds, axis=1) == np.argmax(true, axis=1))

def get_precision(preds, true, class_index):
    preds_class = np.argmax(preds, axis=1)
    true_class = np.argmax(true, axis=1)

    tp = np.sum((true_class == class_index) & (preds_class == class_index))
    fp = np.sum((true_class != class_index) & (preds_class == class_index)) 
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def get_macro_average_precision(preds, true):
    num_classes = preds.shape[1]
    precision_sum = 0
    
    # Calculate precision for each class and sum them
    for class_index in range(num_classes):
        precision_sum += get_precision(preds, true, class_index)
    
    macro_average_precision = precision_sum / num_classes
    return macro_average_precision

def get_recall(preds, true, class_index): 
    preds_class = np.argmax(preds, axis=1)
    true_class = np.argmax(true, axis=1)

    tp = np.sum((true_class == class_index) & (preds_class == class_index))
    fn = np.sum((true_class == class_index) & (preds_class != class_index))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def get_macro_average_recall(preds, true):
    num_classes = preds.shape[1]
    precision_sum = 0
    
    # Calculate precision for each class and sum them
    for class_index in range(num_classes):
        precision_sum += get_recall(preds, true, class_index)
    
    macro_average_precision = precision_sum / num_classes
    return macro_average_precision

def get_mean_predictive_entropy(model, x_test): 
    entropy = predictive_entropy(model, x_test)
    return np.mean(entropy)


def get_results(model, x_test, y_test, epsilon, delta, type, display_text):
    # Get predictions for x_test data 
    x_test_predictions = model.predict(x_test)

    # Get numpy array of x_test data fairness examples
    x_test_fairness_examples = get_fairness_examples(model, x_test, epsilon, type)

    # Get predictions for x_test_fairness_examples data 
    x_test_fairness_examples_predictions = model.predict(x_test_fairness_examples)

    # Get accuracy of model
    if type == "DNN":
        score = model.evaluate(x_test, y_test)
        accuracy = score[1]
    elif type == "BNN":
        print(x_test_predictions)
        accuracy = get_accuracy(x_test_predictions, y_test)
        precision = get_macro_average_precision(x_test_predictions, y_test)
        recall = get_macro_average_recall(x_test_predictions, y_test)
        mean_entropy = get_mean_predictive_entropy(model, x_test)



    print(f"\n ❗️{display_text} RESULTS❗️")
    print("Accuracy:", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("Mean predictive entropy: ", mean_entropy)
    
    thres_score = threshold_fairness_score(x_test_predictions, x_test_fairness_examples_predictions, delta, type)

    max_diff, min_diff, avrg_diff = get_fairness_score(x_test_predictions,
                                                       x_test_fairness_examples_predictions, type)

    return (max_diff, min_diff, avrg_diff) , thres_score, (accuracy, recall, precision, mean_entropy)