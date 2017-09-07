"""

Extra search of prior variance values for combined point and game Bradley-Terry model fitted using a Bayesian approach
with diagonal Covariance. Standard recency weighting function (half life 240 days). Evaluates each model in its own process.

"""
from tennismodelling import models
from tennismodelling import initialisers
from tennismodelling import optimisers
import multiprocessing

# Output files
output_files = ["../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior06_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior07_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior08_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior09_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior11_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior12_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior13_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior14_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior16_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior17_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior18_00.csv",
                "../../Outputs/Experiment11/DiagonalCovariance/PointGame_Prior19_00.csv"]

# Models
models = [models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=6.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=6.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=7.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=7.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=8.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=8.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=9.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=9.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=11.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=11.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=12.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=12.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=13.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=13.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=14.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=14.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=16.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=16.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=17.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=17.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=18.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=18.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=19.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=19.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False))]

# Run models in parallel processes
if __name__ == '__main__':
    jobs = []
    for model, file in zip(models,output_files):
        p = multiprocessing.Process(target=model.run, args=(file,))
        jobs.append(p)
        p.start()
