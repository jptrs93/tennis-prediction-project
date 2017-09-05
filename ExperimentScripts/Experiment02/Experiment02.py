"""

Alternative Decay function hyper parameter grid search based on combined point and game Bradley-Terry model.
No regularisation. Evaluates each model in its own process. 

"""
from TennisModellingModules import models
from TennisModellingModules import data_providers
from TennisModellingModules import optimisers
import multiprocessing

# Output files
output_files = ["../../Outputs/Experiment02/PointGame_DecayFactor090.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor080.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor070.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor060.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor050.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor045.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor040.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor035.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor030.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor025.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor020.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor015.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor010.csv",
                "../../Outputs/Experiment02/PointGame_DecayFactor005.csv"]

# Models
models = [models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.9,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.8,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.7,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.6,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.5,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.45,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.4,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.35,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.3,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.25,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.2,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.15,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.1,decay_type=2),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 0.05,decay_type=2)]


# Run models in parallel processes
if __name__ == '__main__':
    jobs = []
    for model, file in zip(models,output_files):
        p = multiprocessing.Process(target=model.run, args=(file,))
        jobs.append(p)
        p.start()
