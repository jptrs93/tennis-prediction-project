"""

Prior variance grid search for Filtered time series Bradley Terry model based on combined point and game level
information. Fitted using a full covariance matrix and drift parameters of 0.9. Evaluates each model in its own
process.

"""

from tennismodelling import models
from tennismodelling import optimisers
import multiprocessing
from tennismodelling import data_providers

# Output files
output_files = ["../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_975_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_975_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_975_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_950_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_950_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_950_Prior5_05.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_925_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_925_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_925_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_900_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_900_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_900_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_875_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_875_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_875_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_850_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_850_Prior5_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_850_Prior5_00.csv"]

# Models
models = [models.BayesianRatingModel(drift = 0.975, prior_var= 0.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 0.5, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.975, prior_var= 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.975, prior_var= 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianRatingModel(drift = 0.95, prior_var= 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.95, prior_var= 0.75,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.95, prior_var= 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianRatingModel(drift = 0.925, prior_var= 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.925, prior_var= 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.925, prior_var= 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 1.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianRatingModel(drift = 0.875, prior_var = 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.875, prior_var = 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.875, prior_var = 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianRatingModel(drift = 0.85, prior_var = 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.85, prior_var = 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.85, prior_var = 5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 5, use_correlations= True, use_samples= False),
                 surface = 'Grass')]



# Run models in parallel processes
if __name__ == '__main__':
    jobs = []
    for model, file in zip(models,output_files):
        p = multiprocessing.Process(target=model.run, args=(file,))
        jobs.append(p)
        p.start()
