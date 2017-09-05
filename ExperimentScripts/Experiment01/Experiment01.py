"""

Half Life grid search for standard exponential decay function based on combined point and game Bradley-Terry model.
No regularisation. Evaluates each model in its own process.

"""
from tennismodelling import models
from tennismodelling import optimisers
import multiprocessing

# Output files
output_files = ["../../Outputs/Experiment01/PointGame_30DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_60DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_90DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_120DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_150DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_180DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_210DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_240DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_270DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_300DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_330DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_360DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_390DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_420DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_450DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_480DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_510DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_540DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_570DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_600DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_630DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_660DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_700DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_750DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_800DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_1000DayHalfLife.csv",
                "../../Outputs/Experiment01/PointGame_infDayHalfLife.csv"]

# Models
models = [models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 30),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 60),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 90),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 120),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 150),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 180),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 210),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 240),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 270),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 300),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 330),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 360),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 390),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 420),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 450),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 480),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 510),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 540),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 570),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 600),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 630),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 660),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 700),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 750),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 800),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 1000),
          models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0),half_life = 'inf')]


# Run models in parallel processes
if __name__ == '__main__':
    jobs = []
    for model, file in zip(models,output_files):
        p = multiprocessing.Process(target=model.run, args=(file,))
        jobs.append(p)
        p.start()
