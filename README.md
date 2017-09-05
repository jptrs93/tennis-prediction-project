# Tennis Prediction Project

This project is about developing probabilistic models which can predict the outcomes of proffessional tennis matches. Intial work on the project was part of an MSc thesis and focused on predicting a probability of either player winning prior to the match starting. However, the ultimate goal is also to develop models which make in-play predictions and predict the actual scores of the matches rather than just the overall outcome.

## Further Info
The TennisModellingModules contains modules which implement various probabilstic models. A description of most of these models can be found in the thesis report. The ExperimentScripts folder contains scripts for running experiments on the models. The Data folder contains the required input data files covering historical results on mens singles tennis from 2004-2017. To run the experiment scripts the TennisModellingModules package should be added to the python path. Additionally the classes which load the data files expect an enivroment variable 'ML_DATA_DIR' pointing to a location containing the files in the Data folder. The version of python used to run the orginal experiments was 2.7.

The historical data used in the project is obtained from GitHub and credited to Jeff Sackman (<a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/JeffSackmann" rel="dct:source">https://github.com/JeffSackmann</a>). The data is avaliable under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

