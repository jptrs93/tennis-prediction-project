"""

Point Game multiple factor model with 5 factors per player. 5 repeats for each optimisation

"""

import multiprocessing as mp
from tennismodelling import output as out
from tennismodelling import models
from tennismodelling import optimisers
import Queue    # Change to 'queue' for python 3
from tennismodelling import initialisers

# Output file
output_file = "../../Outputs/Experiment12/PointGameBradleyTerry_5Factors_reg0_000001.csv"
provider_model = models.Model()


def worker(q_in, q_out):
    """Worker function to compute predictions of each iteration.
    """
    model = models.PointGameBradleyTerryModel(optimiser=optimisers.BradleyMultipleFactorGradient(num_factors=2, reg = 0.000001))

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
