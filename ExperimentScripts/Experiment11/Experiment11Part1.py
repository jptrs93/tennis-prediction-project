"""

Prior variance grid search for combined point and game Bradley-Terry model fitted using a Bayesian approach with
diagonal Covariance. Standard recency weighting function (half life 240 days). Evaluates each model in its own process.

"""
from tennismodelling import models
from tennismodelling import initialisers
from tennismodelling import optimisers
import multiprocessing

# Output files
output_files = ["../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior00_50.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior01_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior01_50.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior02_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior02_50.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior03_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior04_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior05_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior10_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior15_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior20_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior50_00.csv"]

# Models
models = [models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=0.5),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=0.5,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=1.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=1.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=1.5),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=1.5,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=2.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=2.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=2.5),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=2.5,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=3.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=3.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=4.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=4.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=5.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=5.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=10.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=10.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=15.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=15.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=20.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=20.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=50.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=50.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False))]


# Run models in parallel processes
if __name__ == '__main__':
    jobs = []
    for model, file in zip(models,output_files):
        p = multiprocessing.Process(target=model.run, args=(file,))
        jobs.append(p)
        p.start()
