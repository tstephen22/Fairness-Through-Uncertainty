# Fairness Through Awareness; fair training methods for Bayesian Neural Networks 

## Contents 

- `experiment.py` - Script for running all trials for our experiment. This script will measure and test our devised regulariser for improving IF for BNNs. 
- `scoring.py` - Python file for calculating all metrics for the trials of our experiment. It is imported into experiment.py. Kept separate for readability.  
- `visualise.py` - Script for creating the heat maps and epsilon plots of our results from the experiments. Creates the mean results across all experiment trials. 
- `requirements.txt` - Requirements required to run the provided code. Use `pip install -r requirements.txt`.
- `BNN_FGSM.py` and `DNN_FGSM.py` -  _Fair-FGSM_ for measuring individual fairness by [Alice Doherty](https://github.com/alicedoherty/bayesian-individual-fairness). DNN_FGSM is not used in our experiments. 
- `\deepbayes` - Directory containing the modified Deepbayes Python library to include Fair-FGSM as well as our fair training regulariser, which can be viewed at \deepbayes\deepbayes\optimizers\blrvi.py line 139. The original Deepbayes source code can be found [here](https://github.com/matthewwicker/deepbayes).
- `\data` - Directory containing the Adult dataset. The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/adult).
- `\results` - Directory containing the results of the experimental trials. Results for each epsilon will appear in their respective folder in \results\temp. Results outlined in our paper can be found in \Paper Results. 
- `\graphs` - Directory containing the visualisations (heat maps, epsilon plots and various metric graphs) of the results. Visualisations and mean results found in the paper can be found in \Paper Results.\Main

Trendmous credit to [Alice Doherty](https://github.com/alicedoherty/bayesian-individual-fairness), who's code and research this project is built-upon. 
