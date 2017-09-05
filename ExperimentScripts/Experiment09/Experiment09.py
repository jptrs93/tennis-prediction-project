"""

Surface factor model with 2 and 3 factors for handpicked initialisation and random initialisation. Based on combined
point and game level information. For standard recency weighting function (half life 240 days). No regularisation.
Runs 4 models across a pool of 25 processes.

"""

import multiprocessing as mp
from TennisModellingModules import output as out
from TennisModellingModules import models
from TennisModellingModules import optimisers
import Queue

print('starting')
# Output files
output_file1 = "../../Outputs/Experiment09/PointGameSurfaceFactor_2Factors_HandPickedInit.csv"
output_file2 = "../../Outputs/Experiment09/PointGameSurfaceFactor_3Factors_HandPickedInit.csv"
output_file3 = "../../Outputs/Experiment09/PointGameSurfaceFactor_2Factors_RandomInit_5Repeats.csv"
output_file4 = "../../Outputs/Experiment09/PointGameSurfaceFactor_3Factors_RandomInit_5Repeats.csv"

provider_model = models.Model()


def worker(q_in, q_out1, q_out2, q_out3, q_out4):
    """Worker function to compute predictions on each iteration.
    """
    model1 = models.PointGameSurfaceFactorModel(optimiser=optimisers.SurfaceFactorModelGradient(num_factors=2,reg = 0,hand_pick_init=True),repeats=0)
    model2 = models.PointGameSurfaceFactorModel(optimiser=optimisers.SurfaceFactorModelGradient(num_factors=3,reg = 0,hand_pick_init=True),repeats=0)
    model3 = models.PointGameSurfaceFactorModel(optimiser=optimisers.SurfaceFactorModelGradient(num_factors=2,reg = 0,hand_pick_init=False),repeats=5)
    model4 = models.PointGameSurfaceFactorModel(optimiser=optimisers.SurfaceFactorModelGradient(num_factors=3,reg = 0, hand_pick_init=False),repeats=5)
    while 1:
        try:
            item = q_in.get(block=True, timeout=300)  # If idle for 5 minutes assume job finished
        except Queue.Empty:
            print('process ended due to time out')
            break
        output_lines = model1.do_iteration(item[0],item[1],item[2])
        q_out1.put(output_lines)
        output_lines = model2.do_iteration(item[0],item[1],item[2])
        q_out2.put(output_lines)
        output_lines = model3.do_iteration(item[0],item[1],item[2])
        q_out3.put(output_lines)
        output_lines = model4.do_iteration(item[0],item[1],item[2])
        q_out4.put(output_lines)


if __name__ == "__main__":
    # Create queues and process pool
    q_out1 = mp.Queue()
    q_out2 = mp.Queue()
    q_out3 = mp.Queue()
    q_out4 = mp.Queue()

    q_in = mp.Queue()
    pool = mp.Pool(25, worker, (q_in,q_out1,q_out2,q_out3,q_out4, ))

    # Fill queues of work, with max 50 items in input queue
    for i, item in enumerate(provider_model.data_provider):
        q_in.put(item)
        if i > 50:
            output_lines = q_out1.get()
            out.append_csv_rows(output_lines, output_file1)
            output_lines = q_out2.get()
            out.append_csv_rows(output_lines, output_file2)
            output_lines = q_out3.get()
            out.append_csv_rows(output_lines, output_file3)
            output_lines = q_out4.get()
            out.append_csv_rows(output_lines, output_file4)
            if i % 50 == 0:
                print('Iteration {0}'.format(i))

    # Handle final 150 items of output queue
    for i in range(50):
        try:
            output_lines = q_out1.get(block=True, timeout=300)  # If idle for 5 minutes assume job finished
            out.append_csv_rows(output_lines, output_file1)
            output_lines = q_out2.get(block=True, timeout=300)  # If idle for 5 minutes assume job finished
            out.append_csv_rows(output_lines, output_file2)
            output_lines = q_out3.get(block=True, timeout=300)  # If idle for 5 minutes assume job finished
            out.append_csv_rows(output_lines, output_file3)
            output_lines = q_out4.get(block=True, timeout=300)  # If idle for 5 minutes assume job finished
            out.append_csv_rows(output_lines, output_file4)
        except Queue.Empty:
            break

    # Close ques and pool
    pool.terminate()
    q_in.close()
    q_out1.close()
    q_out2.close()
    q_out3.close()
    q_out4.close()
    pool.close()
