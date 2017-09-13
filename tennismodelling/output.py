"""Output.

This module provides classes for writing and processing model output files.

"""
import csv
import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FormatStrFormatter


def open_csv(filename, mode='a'):
    """Open a csv file in proper mode depending on Python version.
    
    Args:
        filename (string) : Path to file to open
        mode (string) : Mode to open file in 
    """
    return(open(filename, mode=mode+'b') if sys.version_info[0] == 2 else
           open(filename, mode=mode, newline=''))


def append_csv_row(row, file_name):
    """Appends a row to a csv file on Python version.

    Args:
        row (list) : Row of data to append
        filename (string) : Path to file
    """
    results = open_csv(file_name, 'a')
    writer = csv.writer(results)
    writer.writerow(row)


def append_csv_rows(rows, file_name):
    """Appends multiple rows to a csv file

    Args:
        row (list) : List of rows to append
        filename (string) : Path to file
    """
    results = open_csv(file_name, 'a')
    writer = csv.writer(results)
    for row in rows:
        writer.writerow(row)


class OutputProcessor(object):
    """Class for processing model output files.
    TODO: tidy - structure is messy, parts are not concise, repeated chunks of code etc...
    """

    def __init__(self, min_matches=0, year_start=2013, year_end=2020, odds=1):
        """
        Args:
            min_matches (integer) : Minimum number of matches
            year_start (integer) : Start year of range of predictions to consider (inclusive)
            year_end (integer) : End year of range of predictions to consider (exclusive)
            odds (integer) : Odds to compare the model against. 1 for averaged bookmaker odds, 2 for pinnacle only odds,
                             3 for bet365 only odds, 4 for betfair exchange odds.
        """
        self.odds = odds
        self.pred_keys = self.get_keys(min_matches,year_start,year_end)
        self.get_odds(odds)

    def get_summary_stats(self,folder_location, files=False):
        """Process's output files in a folder and print a performance summary of predictions for each output file.

        Args:
            folder_location (string) : Path to folder location
            files (bool) : Set true if instead of a folder path a list of files is provided as the first argument.
        """
        if files:
            csv_files = folder_location
        else:
            all_files = os.listdir(folder_location)
            csv_files = [name for name in all_files if '.csv' in name]

        # Print headings of output
        print('---'*60)
        print_format = "{0:<65}|{1:<15}|{2:<15}|{3:<15}|{4:<15}|{5:<15}|{6:<15}|{7:<15}|{8:<15}"
        header = ['Model','Acc(m1)','Av Prob (m2)', 'Av ln(p) (m3)', 'ROI (m4)','No. Pred','No. Bets','% Bets Won','P-score']
        print(print_format.format(*header))
        print_format = "{0:<65}|{1:<15.4f}|{2:<15.4f}|{3:<15.4f}|{4:<15.4f}|{5:<15}|{6:<15}|{7:<15.3f}|{8:<15.3f}"
        # Loop for each output file and aggregate statistics

        for file in csv_files:
            if files:
                path = file
            else:
                path  = folder_location+'/'+file
            accuracy, av_prob, log_prob, profit, p_score, _, y = self.get_stats(path)
            # Print performance metrics for output file
            if len(file) > 50:
                file = file[-50:]
            try:
                output = [file, float(accuracy[0])/accuracy[1], av_prob[0]/av_prob[1], log_prob[0]/log_prob[1], profit[0]/profit[1], accuracy[1], profit[1], float(profit[2])/profit[1],float(p_score[0])/p_score[1]]
                print(print_format.format(*output))
            except ZeroDivisionError:
                pass

    def get_keys(self,min_matches, year_start, year_end):
        """Returns the subset of matches which the model predictions will be scored against.

        Args:
            min_matches (int) : minimum matches for each player
            year_start  Year start range (inclusive)
            year_end : Year end range (exclusive)
        Returns:
            Dictionary of matches
        """
        pred_keys = {}
        keys_file = os.path.join(os.environ['ML_DATA_DIR'], 'ATP_only_2013to2017.csv')
        rows = [row for row in csv.reader(open(keys_file))]
        rows.sort(key=lambda row: (row[0]))
        for row in rows:
            yr = int(row[0][0:4])
            if int(row[5]) > min_matches-1 and int(row[6]) > min_matches-1 and yr >= year_start and yr < year_end:
                key = row[0][:4] + row[1] + row[3] + row[4]
                pred_keys[key] = -1
        return pred_keys

    def get_odds(self, type = 1):
        """Fetches the betting odds for the subset of matches which are being considered.

        Args:
            type (integer) : Type of odds to compare the model against. 1 for averaged bookmaker odds, 2 for pinnacle
                             only odds, 3 for bet365 only odds, 4 for betfair exchange odds.
        Returns:

        """
        if type < 4:
            data_path = os.path.join(os.environ['ML_DATA_DIR'], 'BookieOdds_2004to2017.csv')
            rows = [row for row in csv.reader(open(data_path))]
            for row in rows:
                key = row[5][:4] + row[1] + row[10]+ row[20]
                if key in self.pred_keys:
                    try:
                        if type == 1:
                            self.pred_keys[key] = [float(row[53]),float(row[54])]
                        elif type ==2:
                            self.pred_keys[key] = [float(row[51]),float(row[52])]
                        elif type ==3:
                            self.pred_keys[key] = [float(row[49]),float(row[50])]
                    except:
                        self.pred_keys[key] = -1
        else:
            data_path = os.path.join(os.environ['ML_DATA_DIR'], 'Betfair_Odds.csv')
            rows = [row for row in csv.reader(open(data_path))]
            for row in rows[1:]:
                key = row[0][:4] + row[1] + row[2] + row[3]
                if key in self.pred_keys:
                    try:
                        if type == 4:
                            self.pred_keys[key] = [float(row[5]), float(row[6])]
                        else:
                            self.pred_keys[key] = [float(row[8]), float(row[9])]
                    except:
                        self.pred_keys[key] = -1

    def get_stats(self, path):
        """Evaluates and returns the performance statistics for the predictions in an output file.

        Args:
            path (string) : Path to output file
        Returns:
            statistics (tuple) : Prediction performance statistics
        """
        accuracy = [0, 0]
        av_prob = [0, 0]
        log_prob = [0, 0]
        profit = [0, 0, 0]
        p_score = [0, 0]
        probs = []
        years = []
        rows = [row for row in csv.reader(open(path))]
        rows.sort(key=lambda row: (row[0]))
        for row in rows:
            key = row[0][:4] + row[1] + row[3] + row[4]
            if key in self.pred_keys:
                # Accuracy aggregation

                non_decimal = re.compile(r'[^\d.]+')
                prob = non_decimal.sub('', row[11])
                probs += [float(prob)]
                years += [int(row[0][:6])]
                logprob = -np.log(float(prob))
                # Accuracy aggregation
                if float(prob) > 0.5:
                    accuracy[0] += 1
                    accuracy[1] += 1
                else:
                    accuracy[1] += 1
                # Average probability aggregation
                av_prob[0] += float(prob)
                av_prob[1] += 1
                # Average log probability aggregation
                log_prob[0] += float(logprob)
                log_prob[1] += 1
                # p-score
                if(float(prob)>0.5):
                    p_score[0] += float(prob)
                else:
                    p_score[0] -= (1-float(prob))
                p_score[1] += 1

                # Profit aggregation
                if self.pred_keys[key] == -1:
                    pass
                else:
                    w_prob = float(prob)
                    w_odds = self.pred_keys[key][0]
                    l_odds = self.pred_keys[key][1]
                    if self.odds > 3:
                        w_odds = 0.95*w_odds + 0.05
                        l_odds = 0.95*l_odds + 0.05

                    diffL = (1. - w_prob) - (1. / l_odds)
                    diffW = w_prob - (1. / w_odds)
                    marginL = 0
                    marginU = 1


                    if diffL > marginL and diffL < marginU and diffL > diffW:
                        profit[0] -= 1
                        profit[1] += 1
                    elif diffW > marginL and diffW < marginU and diffW > diffL:
                        if w_odds < 1:
                            print('odds error')

                        profit[0] += (w_odds - 1)
                        profit[1] += 1
                        profit[2] += 1
                    else:
                        pass

        return accuracy, av_prob, log_prob, profit, p_score, probs, years

    def get_bookmaker_baseline(self):
        """Evaluates and returns the performance statistics of the bookmaker implied probabilities.

        Returns:
            statistics (tuple) : Prediction performance statistics
        """
        accuracy = [0, 0]
        av_prob = [0, 0]
        log_prob = [0, 0]
        profit = [0, 0, 0]
        p_score = [0, 0]
        probs_list = []
        for key in self.pred_keys.keys():
            if self.pred_keys[key] == -1:
                pass
            else:
                oddsw = float(self.pred_keys[key][0])
                oddsl = float(self.pred_keys[key][1])
                wprob = 1. /oddsw
                lprob = 1. /oddsl
                prob = wprob/(wprob+lprob)
                probs_list += [prob]
                logprob = np.log(prob)
                # Accuracy aggregation
                if float(prob) > 0.5:
                    accuracy[0] += 1
                    accuracy[1] += 1
                else:
                    accuracy[1] += 1
                # Average probability aggregation
                av_prob[0] += float(prob)
                av_prob[1] += 1
                # Average log probability aggregation
                log_prob[0] += float(logprob)
                log_prob[1] += 1
                # p-score
                if(float(prob)>0.5):
                    p_score[0] += float(prob)
                else:
                    p_score[0] -= (1-float(prob))
                p_score[1] += 1

        return accuracy, av_prob, log_prob, profit, p_score, probs_list

    def get_atp_ranking_baseline(self):
        """Evaluates and returns the performance statistics for predictions based upon atp rankings.

        Returns:
            statistics (tuple) : Prediction performance statistics
        """

        keys_file = os.path.join(os.environ['ML_DATA_DIR'], 'ATP_only_2013to2017.csv')
        rows = [row for row in csv.reader(open(keys_file))]
        rows.sort(key=lambda row: (row[0]))

        accuracy = [0, 0]
        av_prob = [0, 0]
        log_prob = [0, 0]
        profit = [0, 0, 0]
        p_score = [0, 0]
        probs_list = []
        for row in rows:
            yr = int(row[0][0:4])
            key = row[0][:4] + row[1]  + row[3] + row[4]

            if key in self.pred_keys:
                try:
                    wrp = float(row[7])
                    lrp = float(row[8])
                    logit = 0.62*np.log(wrp/lrp)
                    prob = 1./(1.+np.exp(-logit))
                    logprob = np.log(prob)
                    probs_list += [prob]
                    if float(prob) > 0.5:
                        accuracy[0] += 1
                        accuracy[1] += 1
                    else:
                        accuracy[1] += 1
                    # Average probability aggregation
                    av_prob[0] += float(prob)
                    av_prob[1] += 1
                    # Average log probability aggregation
                    log_prob[0] += float(logprob)
                    log_prob[1] += 1
                    # p-score
                    if(float(prob)>0.5):
                        p_score[0] += float(prob)
                    else:
                        p_score[0] -= (1-float(prob))
                    p_score[1] += 1

                    # Profit aggregation
                    if self.pred_keys[key] == -1:
                        pass
                    else:  # Profit is recalculated (due to mistake in early output files)

                        w_prob = float(prob)
                        w_odds = self.pred_keys[key][0]
                        l_odds = self.pred_keys[key][1]
                        diffL = (1. - w_prob) - (1. / l_odds)
                        diffW = w_prob - (1. / w_odds)
                        margin = 0.00
                        if diffL > margin and diffL > diffW:
                            profit[0] -= 1
                            profit[1] += 1
                        elif diffW > margin and diffW > diffL:
                            if w_odds < 1:
                                print('odds error')
                            profit[0] += (w_odds -1)*0.95
                            profit[1] += 1
                            profit[2] += 1
                        else:
                            pass
                except ValueError:
                    pass

        return accuracy, av_prob, log_prob, profit, p_score, probs_list

    def print_baselines(self):
        """Prints the performance statisitics of the model baselines: bookmakers and atp rankings.
        """
        print('---'*60)
        print_format = "{0:<65}|{1:<15}|{2:<15}|{3:<15}|{4:<15}|{5:<15}|{6:<15}|{7:<15}|{8:<15}"
        header = ['Model','Acc(m1)','Av Prob (m2)', 'Av ln(p) (m3)', 'ROI (m4)','No. Pred','No. Bets','% Bets Won','P-score']
        print(print_format.format(*header))
        print_format = "{0:<65}|{1:<15.4f}|{2:<15.4f}|{3:<15.4f}|{4:<15.4f}|{5:<15}|{6:<15}|{7:<15.3f}|{8:<15.3f}"
        # Loop for each output file and aggregate statistics
        accuracy, av_prob, log_prob, profit, p_score, _ = self.get_bookmaker_baseline()
        output = ['Bookmaker Model', float(accuracy[0])/accuracy[1], av_prob[0]/av_prob[1], log_prob[0]/log_prob[1], 0, accuracy[1], 0, 0,float(p_score[0])/p_score[1]]
        print(print_format.format(*output))
        accuracy, av_prob, log_prob, profit, p_score, _ = self.get_atp_ranking_baseline()
        output = ['ATP Rankings', float(accuracy[0])/accuracy[1], av_prob[0]/av_prob[1], log_prob[0]/log_prob[1], profit[0]/profit[1], accuracy[1], profit[1], float(profit[2])/profit[1],float(p_score[0])/p_score[1]]
        print(print_format.format(*output))

    def make_averaged_file(self,files, outfile):
        """
        Creates a new output file with averaged predictions from other models.

        Args:
            files (list) : List of output files to average predictions from
            outfile (string) : Path to save averaged output file to
        """
        dict = {}
        var = {}
        for i, file in enumerate(files):
            rows = [row for row in csv.reader(open(file))]
            rows.sort(key=lambda row: (row[0]))
            for row in rows:
                key = row[0][:4] + row[1] + row[3] + row[4]
                if key in self.pred_keys:
                    non_decimal = re.compile(r'[^\d.]+')
                    prob = float(non_decimal.sub('', row[11]))
                    varpred = float(row[9]) + float(row[10])
                    if key not in dict:
                        dict[key] = [prob]
                        var[key] = [varpred]
                    else:
                        dict[key] += [prob]
                        var[key] += [varpred]

        keys_file = os.path.join(os.environ['ML_DATA_DIR'], 'ATP_only_2013to2017.csv')
        rows = [row for row in csv.reader(open(keys_file))]
        rows.sort(key=lambda row: (row[0]))
        out = []
        for row in rows:
            key = row[0][:4] + row[1]  + row[3] + row[4]
            if key in dict:
                out += [[row[0], row[1], row[2], row[3], row[4],'','','','','',np.mean(var[key]),np.mean(dict[key])]]
        append_csv_rows(out,outfile)

    @staticmethod
    def get_profit(prob,w_odds,l_odds, edge_margin = 0):
        """Calculates the profit based the odds and models predictions.

        Args:
            prob (float) : Models predicted probability for winning player
            w_odds (float) : Odds for winning player
            l_odds (float) : Odds for losing player
            edge_margin (float) : Probabilty edge required to place a bet

        Returns:
            profit (float) : Profit made (returns zero if no bet would have been placed)
        """

        diffL = (1. - prob) - (1. / l_odds)
        diffW = prob - (1. / w_odds)
        if diffL > edge_margin and diffL > diffW:
            return - 1
        elif diffW > edge_margin and diffW > diffL:
            return (w_odds - 1)*0.95
        else:
            return  0