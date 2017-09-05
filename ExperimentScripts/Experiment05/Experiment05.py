"""

L2 regularisation constant grid search for combined point and game Free Parameter model. Standard recency weighting 
function (half life 240 days). Evaluates each model in its own process. 

"""

from tennismodelling import models
from tennismodelling import optimisers
import multiprocessing
from tennismodelling import initialisers

# Output files
output_files = ["../../Outputs/Experiment05/FreeParamPointGame_reg0_000005.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00001.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00002.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00005.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00010.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00015.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00020.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00025.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00030.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00035.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00040.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00045.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00050.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00055.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00060.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00065.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00070.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00075.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00100.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00125.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00150.csv",
                "../../Outputs/Experiment05/FreeParamPointGame_reg0_00200.csv"]

# Models
models = [models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.000005),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00001),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00002),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00005),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00010),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00015),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00020),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00025),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00030),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00035),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00040),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00045),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00050),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00055),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00060),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00065),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00070),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00075),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00100),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00125),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00150),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00200),half_life = 240,initialiser= initialisers.FreeParameterInitialiser())]


# Run models in parallel processes
if __name__ == '__main__':
    jobs = []
    for model, file in zip(models,output_files):
        p = multiprocessing.Process(target=model.run, args=(file,))
        jobs.append(p)
        p.start()
