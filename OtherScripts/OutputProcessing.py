"""

Gets a performance summary of models based on their output files.

"""

from TennisModellingModules import output as out

folders = ["../Outputs/Experiment01",
           "../Outputs/Experiment02",
           "../Outputs/Experiment03",
           "../Outputs/Experiment04",
           "../Outputs/Experiment05",
           "../Outputs/Experiment06",
           "../Outputs/Experiment07",
           "../Outputs/Experiment08",
           "../Outputs/Experiment09",
           "../Outputs/Experiment10"]

# Creates an output processor object for calculating the performance scores.
processor = out.OutputProcessor(year_start=2013,year_end=2017, min_matches= 0)
# Print baseline performances.
processor.print_baselines()
for folder in folders:
        print('---'*20)
        print(folder)
        processor.get_summary_stats(folder)
