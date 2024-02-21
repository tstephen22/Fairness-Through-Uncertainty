# Fairness Through Awareness; fair training methods for Bayesian Neural Networks 

## Contents 

- `run_trial.py` - Script for running a suite of trials. This script will measure IF for BNNs and test experimental methods for improving IF; either `Adversarial training` or `Adversarial regularisation`
  depending on which is specified. 
- `visualise_data.py` - Script for visualising the results of the experiments.
- `requirements.txt` - Requirements required to run the provided code. Use `pip install -r requirements.txt`.
- `BNN_FGSM.py` and `DNN_FGSM.py` -  _Fair-FGSM_ for measuring individual fairness by [Alice Doherty](https://github.com/alicedoherty/bayesian-individual-fairness).
- `\deepbayes` - Directory containing the modified Deepbayes Python library to include Fair-FGSM. The original Deepbayes source code can be found [here](https://github.com/matthewwicker/deepbayes).
- `\data` - Directory containing the Adult dataset. The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/adult).
- `\results` - Directory containing the results of the experimental trials.
- `\heatmaps`, `\fairness_acc_plots`, `\epsilon_plots` - Directories containing the graphs visualising the experimental results. Separated by 
which form of improvement was used, `Training` or `Regularisation`
