"""

Joint optimisation Bradley-Terry time series model with drift of 0.9 and prior of 1. Runs the 2nd quarter of the
evaluation across 10 cores.

"""

import multiprocessing as mp
from TennisModellingModules import output as out
from TennisModellingModules import models
from TennisModellingModules import optimisers
import Queue    # Change to 'queue' for python 3
from TennisModellingModules import initialisers

# Output file
output_file = "../../Outputs/Experiment06/Drift0_9Prior1.csv"
provider_model = models.Model()


def worker(q_in, q_out):
    """Worker function to compute predictions of each iteration.
    """
    model = models.JointOptTimeSeriesModelRefined(steps = 4, initialiser=initialisers.JointOptTimeSeriesInitialiser(),optimiser=optimisers.JointOptTimeSeriesRefinedBradleyTerryVariationalInference(steps=4,drift=0.9,tol = 1e-8, use_correlations=False, prior_var=1))
    while 1:
        try:
            item = q_in.get(block=True, timeout=300)  # If idle for 5 minutes assume job finished
        except Queue.Empty:
            break
        output_lines = model.do_iteration(item[0],item[1],item[2])
        q_out.put(output_lines)

if __name__ == "__main__":

    # Skip to one quarter of the way through model evaluation
    provider_model.skip_to(450)

    # Create queues and process pool
    q_out = mp.Queue()
    q_in = mp.Queue()
    pool = mp.Pool(10, worker, (q_in,q_out, ))

    # Fill queues of work, with max 50 items in input queue
    for i, item in enumerate(provider_model.data_provider):
        if i > 450:
            break
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
