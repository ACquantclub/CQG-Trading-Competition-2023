# Note For Case 3 Strategy:

For case 3, we attempted 3 seperate strategies and proceeded with the most succesful one.
Firstly, we attempted to proceed algorithmically and trade purely off certain indicators which we calculated based off of the initial data. The results from this were poor, so we switched to a model based approach utilizing scikit learn.
We attempted to use decision trees with varying results, but due to their inconsistency, we ultimately switched to random forests, which we tuned using hyperparameter grid searches and varying train-test splits of our data. We saved our model in a model.pkl file, then tested it on random data samples and observed our best results so far. We are ultimately submitting this model as our mtehod of choice for case 3.
