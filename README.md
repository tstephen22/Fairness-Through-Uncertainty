# Fairness Through Uncertainty; fair training methods for Bayesian Neural Networks 
Dissertation paper for the completement of my MSc in Computer Science :) Very happy to have achieved a first class honours for it!

## Abstract 
As machine learning models become more prevalent in our lives and take on
critical decision-making roles across various sectors of society, ensuring their decisions are
not only accurate but also fair has become ever more crucial.

This paper focuses on individual fairness in Bayesian Neural Networks (BNNs). Individual fairness 
ensures that similar individuals receive similar outcomes by a model.
Specifically, we undertake the definition of ε − δ individual fairness. BNNs offer advantages 
over deterministic neural networks due to their ability to quantify uncertainty and
effectively handle smaller datasets; a capability that has led to their adoption in critical
fields such as medicine.

In this paper, we construct a fairness regularisation method for BNNs by transferring
techniques from adversarial robustness training and employing the Fair-FGSM algorithm
[17] for generating similar inputs. This approach draws on existing research that highlights 
the similarity between adversarial robustness and individual fairness definitions.
The regulariser is designed to be simple, facilitating its integration into existing training 
procedures without extensive modifications. We also introduce a simple metric for
measuring and comparing ε − δ individual fairness models in an intuitive manner, the
Threshold-Fairness metric.

Through our experimentation on various model architecture sizes and similarity metric
parameters, encompassing a total of three-thousand models, we can attest that our devised
regulariser is effective at improving the individual fairness of a BNN. However, due the
to fairness-accuracy trade-off, there is a small degradation of model accuracy imposed by
the regulariser. Additionally, we empirically evaluate that there is a relationship between
the set of non-protected ε parameter values and the effectiveness of the regulariser at
improving fairness. We hope that our devised regulariser acts a starting point for more
sophisticated and adaptable individual fairness mechanisms, and that these findings will
serve as a foundational piece for future research into individual fairness in BNNs.


## Contents 
- `Fairness_Through_Uncertainty_TheoStephensKehoe.pdf` - Full dissertation paper. 
- `experiment.py` - Script for running all trials for our experiment. This script will measure and test our devised regulariser for improving IF for BNNs. 
- `scoring.py` - Python file for calculating all metrics for the trials of our experiment. It is imported into experiment.py. Kept separate for readability.  
- `visualise.py` - Script for creating the heat maps and epsilon plots of our results from the experiments. Creates the mean results across all experiment trials. 
- `requirements.txt` - Requirements required to run the provided code. Use `pip install -r requirements.txt`.
- `BNN_FGSM.py` and `DNN_FGSM.py` -  _Fair-FGSM_ for measuring individual fairness by [Alice Doherty](https://github.com/alicedoherty/bayesian-individual-fairness). DNN_FGSM is not used in our experiments. 
- `\deepbayes` - Directory containing the modified Deepbayes Python library to include Fair-FGSM as well as our fair training regulariser, which can be viewed at \deepbayes\deepbayes\optimizers\blrvi.py line 139. The original Deepbayes source code can be found [here](https://github.com/matthewwicker/deepbayes).
- `\data` - Directory containing the Adult dataset. The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/adult).
- `\results` - Directory containing the results of the experimental trials. Results for each epsilon will appear in their respective folder in \results\temp. Results outlined in our paper can be found in \Paper Results. 
- `\graphs` - Directory containing the visualisations (heat maps, epsilon plots and various metric graphs) of the results. Visualisations and mean results found in the paper can be found in \Paper Results\Main

Trendmous credit to [Alice Doherty](https://github.com/alicedoherty/bayesian-individual-fairness), who's code and research this project is built-upon. 
