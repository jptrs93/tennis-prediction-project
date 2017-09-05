"""

Prior variance grid search for Filtered time series Bradley Terry model based on combined point and game level
information. Fitted using a full covariance matrix and drift parameters of 0.9. Evaluates each model in its own
process.

"""

from TennisModellingModules import models
from TennisModellingModules import optimisers
import multiprocessing
from TennisModellingModules import data_providers

# Output files
output_files = ["../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior0_50.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior0_50.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior0_50.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior0_75.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior0_75.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior0_75.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior1_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior1_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior1_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior1_50.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior1_50.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior1_50.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior2_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior2_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior2_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior3_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior3_00.csv",
                "../../Outputs/Experiment07/PointGame/BradleyPointGameCorrelated_Drift_0_9_Prior3_00.csv"]

# Models
models = [models.BayesianRatingModel(drift = 0.9, prior_var= 0.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 0.5, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 0.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 0.5, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 0.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 0.5, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 0.75,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 0.75, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 0.75,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 0.75, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 0.75,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 0.75, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 1,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 1, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 1,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 1, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 1,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 1, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 1.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 1.5, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 1.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 1.5, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.9, prior_var= 1.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 1.5, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianRatingModel(drift = 0.9, prior_var = 2,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 2, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.9, prior_var = 2,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 2, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.9, prior_var = 2,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 2, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianRatingModel(drift = 0.9, prior_var = 3,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 3, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianRatingModel(drift = 0.9, prior_var = 3,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 3, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianRatingModel(drift = 0.9, prior_var = 3,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var= 3, use_correlations= True, use_samples= False),
                 surface = 'Grass')]


# Run models in parallel processes
if __name__ == '__main__':
    jobs = []
    for model, file in zip(models,output_files):
        p = multiprocessing.Process(target=model.run, args=(file,))
        jobs.append(p)
        p.start()
