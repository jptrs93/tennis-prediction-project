from TennisModellingModules import models
from TennisModellingModules import optimisers
from TennisModellingModules import data_providers
from TennisModellingModules import initialisers
from TennisModellingModules import output as out
import numpy as np
import time
from TennisModellingModules import markov_chain as mcm
import os
import csv


# test models
model1 = models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyTerryGradient(reg=0.00001))
model2 = models.PointGameBradleyTerryModel(initialiser = initialisers.DefaultInitialiser() , optimiser=optimisers.BradleyTerryVariationalInference(prior_var=1, use_correlations=False, use_samples=False))
model3 = models.FreeParameterPointModel(initialiser=  initialisers.DefaultInitialiser(), optimiser = optimisers.FreeParameterPointGradient(use_bias=True))
model4 = models.FreeParameterPointModel(initialiser = initialisers.DefaultInitialiser(), optimiser = optimisers.FreeParameterPointVariationalInference(prior_var=1, use_correlations=False, use_samples=False, display=True))
model5 = models.JointOptTimeSeriesModel(steps = 4,optimiser=optimisers.JointOptTimeSeriesBradleyTerryVariationalInference(steps=4,drift=0.9,tol = 1e-8, use_correlations=False,display=True))
model6 = models.JointOptTimeSeriesModelRefined(steps = 4,optimiser=optimisers.JointOptTimeSeriesRefinedBradleyTerryVariationalInference(steps=4,drift=0.9,tol = 1e-8, use_correlations=False, display=True))
model7 = models.BayesianRatingModel(drift = 0.9, prior_var =  1, level = 'Game',
                 data_provider= data_providers.VFineDataProvider(),
                 optimiser = optimisers.BradleyTerryVariationalInference(prior_var = 1, use_correlations= False, use_samples= False, display=True),
                 surface = 'Hard')
model = models.BayesianFreeParameterRatingModel(drift = 0.9, prior_var =  1,
                 data_provider= data_providers.VFineDataProvider(), level = 'Game',
                 optimiser = optimisers.FreeParameterPointVariationalInference(prior_var= 1, use_correlations= True, use_samples= False, display=True),
                 surface = 'Hard')
model10 = models.FreeParameterPointGameModel(optimiser=optimisers.FreeParameterPointGradient(reg=0.00001),
                                          initialiser= initialisers.FreeParameterInitialiser(),half_life = 240)

model11 = models.PointGameSurfaceFactorModel(half_life = 240,optimiser=optimisers.SurfaceFactorModelGradient(num_factors=3,reg = 0.000001,hand_pick_init=False, display=True),repeats=2)
# for i in model:
#     pass

for i in range(5):
    start = time.time()
    rows = model.next()
    print('iteration {0}, time {1}'.format(i,time.time() -start))
    for row in rows:
        print(row)




# # for tt in t:
# #     print(tt)
# for i in range(10):
#     s = time.time()
#     S, M, E = data_provider.next()
#
#     rows = model.do_iteration(S,M,E)
#     print('Iteration time {0}'.format(time.time() - s))
#     for row in rows:
#         print(row)


# model = models.Bayesian_Rating_Model(drift = 0.9, prior_var =  1,
#                  optimiser = optimisers.TimeSeriesVariationalInference5(prior_var= 1, use_correlations= False, use_samples= False),
#                  data_provider = data_providers.CoarseDataProvider(data_file='atp_no_odds.csv'),
#                  surfaces = ['Hard'])
#
#
# for i in range(100):
#     # print(i)
#     s =time.time()
#     out =  model.next()
#     print('opt time {0}'.format(time.time()-s))
#     # for item in out:
#     #     print(item)
# # opt time 0.784749984741
# # opt time 1.43157696724
# # opt time 2.53977584839
# # opt time 0.00170302391052
# # opt time 0.00760507583618
