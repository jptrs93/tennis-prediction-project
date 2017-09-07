"""

Prior variance grid search for combined point and game Bradley-Terry model fitted using a Bayesian approach
with full Covariance. Standard recency weighting function (half life 240 days). Evaluates each model in its own process.

"""
from tennismodelling import models
from tennismodelling import initialisers
from tennismodelling import optimisers
import multiprocessing

# Output files
output_files = ["../../Outputs/Experiment11/FullCovariance/PointGame_Prior01_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior02_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior04_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior05_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior06_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior07_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior08_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior09_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior10_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior11_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior12_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior13_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior14_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior15_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior20_00.csv",
                "../../Outputs/Experiment11/FullCovariance/PointGame_Prior50_00.csv"]

# Models
models = [models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=1.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=1.,
                                                                                                  use_correlations=False,
                                                                                                  use_samples=False)),
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=2.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=2.,
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
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=6.),
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
          models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=10.),
                                            optimiser=optimisers.BradleyTerryVariationalInference(prior_var=10.,
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
