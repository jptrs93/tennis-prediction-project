"""

Prior variance grid search for Filtered time series Free Parameter model based on combined point and game level
information. Fitted using a full covariance matrix and drift parameters of 0.9. Evaluates each model in its own
process.

"""

from tennismodelling import models
from tennismodelling import optimisers
import multiprocessing
from tennismodelling import data_providers

# Output files
output_files = ["../../Outputs/Experiment08/PointGame/FreeParameterPointGameCorrelated_Drift_0_9_Prior0_75.csv",
                "../../Outputs/Experiment08/PointGame/FreeParameterPointGameCorrelated_Drift_0_9_Prior0_75.csv",
                "../../Outputs/Experiment08/PointGame/FreeParameterPointGameCorrelated_Drift_0_9_Prior0_75.csv",
                "../../Outputs/Experiment08/PointGame/FreeParameterPointGameCorrelated_Drift_0_9_Prior1_00.csv",
                "../../Outputs/Experiment08/PointGame/FreeParameterPointGameCorrelated_Drift_0_9_Prior1_00.csv",
                "../../Outputs/Experiment08/PointGame/FreeParameterPointGameCorrelated_Drift_0_9_Prior1_00.csv",
                "../../Outputs/Experiment08/PointGame/FreeParameterPointGameCorrelated_Drift_0_9_Prior1_50.csv",
                "../../Outputs/Experiment08/PointGame/FreeParameterPointGameCorrelated_Drift_0_9_Prior1_50.csv",
                "../../Outputs/Experiment08/PointGame/FreeParameterPointGameCorrelated_Drift_0_9_Prior1_50.csv"]

# Models
models = [models.BayesianFreeParameterRatingModel(drift = 0.9, prior_var= 0.75,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.FreeParameterPointVariationalInference(prior_var= 0.75, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianFreeParameterRatingModel(drift = 0.9, prior_var= 0.75,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.FreeParameterPointVariationalInference(prior_var= 0.75, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianFreeParameterRatingModel(drift = 0.9, prior_var= 0.75,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.FreeParameterPointVariationalInference(prior_var= 0.75, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianFreeParameterRatingModel(drift = 0.9, prior_var= 1,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.FreeParameterPointVariationalInference(prior_var= 1, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianFreeParameterRatingModel(drift = 0.9, prior_var= 1,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.FreeParameterPointVariationalInference(prior_var= 1, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianFreeParameterRatingModel(drift = 0.9, prior_var= 1,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.FreeParameterPointVariationalInference(prior_var= 1, use_correlations= True, use_samples= False),
                 surface = 'Grass'),
          models.BayesianFreeParameterRatingModel(drift = 0.9, prior_var= 1.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.FreeParameterPointVariationalInference(prior_var= 1.5, use_correlations= True, use_samples= False),
                 surface = 'Hard'),
          models.BayesianFreeParameterRatingModel(drift = 0.9, prior_var= 1.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.FreeParameterPointVariationalInference(prior_var= 1.5, use_correlations= True, use_samples= False),
                 surface = 'Clay'),
          models.BayesianFreeParameterRatingModel(drift = 0.9, prior_var= 1.5,
                 data_provider= data_providers.VFineDataProvider(),level = 'PointGame',
                 optimiser = optimisers.FreeParameterPointVariationalInference(prior_var= 1.5, use_correlations= True, use_samples= False),
                 surface = 'Grass')]


# Run models in parallel processes
if __name__ == '__main__':
    jobs = []
    for model, file in zip(models,output_files):
        p = multiprocessing.Process(target=model.run, args=(file,))
        jobs.append(p)
        p.start()
