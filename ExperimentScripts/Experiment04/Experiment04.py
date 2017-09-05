"""

Free parameter point model and combined point and game model. Fitted with and without bias terms. No regularisation.
Standard recency weighting function (half life 240 days). Evaluates each model in its own process.

"""
from tennismodelling import models
from tennismodelling import optimisers
import multiprocessing
from tennismodelling import initialisers

# Output files
output_files = ["../../Outputs/Experiment04/FreeParamPointGame_reg0_noBias.csv",
                "../../Outputs/Experiment04/FreeParamPointGame_reg0_Bias.csv",
                "../../Outputs/Experiment04/FreeParamPoint_reg0_noBias.csv",
                "../../Outputs/Experiment04/FreeParamPoint_reg0_Bias.csv"]

# Models
models = [models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0, use_bias =  False),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0, use_bias= True),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointModel(optimiser=optimisers.FreeParameterPointGradient(reg=0, use_bias= False),half_life = 240,initialiser= initialisers.FreeParameterInitialiser()),
          models.FreeParameterPointModel(optimiser=optimisers.FreeParameterPointGradient(reg=0,use_bias=True),half_life = 240,initialiser= initialisers.FreeParameterInitialiser())]

# Run models in parallel processes
if __name__ == '__main__':
    jobs = []
    for model, file in zip(models,output_files):
        p = multiprocessing.Process(target=model.run, args=(file,))
        jobs.append(p)
        p.start()
