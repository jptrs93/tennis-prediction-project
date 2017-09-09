"""

Bradley-Terry model with infinite half life (no recency weighting), fitted using variational inference with a
diagonal covariance and prior variance of 15. This model is for comparison to joint optimisation model as its
results should be identical to joint optimisation model with 1 step. Runs the model across 10 processes.

"""

import multiprocessing as mp
from tennismodelling import output as out
from tennismodelling import models
from tennismodelling import optimisers
import Queue    # Change to 'queue' for python 3
from tennismodelling import initialisers

# Output file
output_file = "../../Outputs/Experiment06/PointGameBradleyTerry_InfHalfLife_DiagonalCovariance_Prior15.csv"
provider_model = models.Model()


def worker(q_in, q_out):
    """Worker function to compute predictions of each iteration.
    """
    model = models.PointGameBradleyTerryModel(initialiser=initialisers.BradleyTerryVariationalInferenceInitialiser(prior_var=15.),
                                              half_life='inf',
                                              optimiser=optimisers.BradleyTerryVariationalInference(prior_var=15.,
                                                                                                    use_correlations=False,
                                                                                                    use_samples=False))
    while 1:
        try:
            item = q_in.get(block=True, timeout=300)  # If idle for 5 minutes assume job finished
        except Queue.Empty:
            break
        output_lines = model.do_iteration(item[0],item[1],item[2])
        q_out.put(output_lines)

if __name__ == "__main__":

    # Create queues and process pool
    q_out = mp.Queue()
    q_in = mp.Queue()
    pool = mp.Pool(10, worker, (q_in,q_out, ))

    # Fill queues of work, with max 50 items in input queue
    for i, item in enumerate(provider_model.data_provider):
        q_in.put(item)
        if i > 50:
            output_lines = q_out.get()
            out.append_csv_rows(output_lines, output_file)

    # Handle final 50 items of output que
    for i in range(50):
        try:
            output_lines = q_out.get(block=True, timeout=300)  # If idle for 5 minutes assume finished
            out.append_csv_rows(output_lines, output_file)
        except Queue.Empty:
            break

    # Close ques and pool
    pool.terminate()
    q_in.close()
    q_out.close()
    pool.close()
