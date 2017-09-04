# Tennis Prediction Project

This project develops probabilistic models which can predict the outcomes of proffessional tennis matches. The intial work on the project was carried out as part of an MSc thesis and focuses on predicting a probability of either player winning prior to the match starting. However, the ultimate goal is also to develop models which make in-play predictions and predict the actual scores of the matches rather than just the overall outcome.

## Further Info
The TennisModellingModules contains modules which implement various probabilstic models. A description of most of these models can be found in the thesis report. The ExperimentScripts folder contains scripts for running experiments on the models. The Data folder contains the required input data files which covers historical results on mens singles tennis from 2004-2017. To run the experiment scripts the TennisModellingModules package should be added to the python path. Additionally the classes which load the data expect an enivroment variable 'ML_DATA_DIR' giving the path to the files in the data folder. The version of python used to run the orginal experiments was 2.7, however the code is written to also be compatible with python 3.



