"""Data providers.

This module provides classes for loading and iterating over the data set. Assumes there is an environment variable
'ML_DATA_DIR' which is set to the location of where the data files are stored.

"""

import os
from datetime import datetime
import csv


class CoarseDataProvider(object):
    """Basic data provider which provides data tournament by tournament at a time. This provider is designed for
    a model where filtering updates are made to the player parameters."""

    def __init__(self, data_file = 'ATP_and_Challenger_2005to2017.csv'):
        """
        Args:
            data_file (string) : Name of file containing data
        """

        # Get path to data file
        data_path = os.path.join(os.environ['ML_DATA_DIR'], data_file)
        assert os.path.isfile(data_path), ('Data file does not exist at expected path: ' + data_path)

        # Load and sort data. Primary sort by date, secondary sort by round and tertiary sort by match number
        self.round_sort_order = {'Q1': 0,'Q2': 1, 'Q3': 2, 'RR': 0,'R128': 3,'R64': 4,'R32': 5,'R16': 6,'QF': 7, 'SF': 8,'F': 9, 'BR': 9}
        self.data = [row for row in csv.reader(open(data_path))]
        self.headers = self.data[0]
        self.data = self.data[1:]
        self.data.sort(key=lambda val: (int(val[5]),self.round_sort_order[val[29]], int(val[6])))

        # Convert all of the dates to epoch time with an origin at the date of the first match in data
        self.epoch = datetime.strptime(self.data[0][5], '%Y%m%d')
        for row in self.data:
            row.append(row[5])  # Keep the original time format around at the end of the row
            match_date = datetime.strptime(row[5], '%Y%m%d')
            epoch_time = (match_date - self.epoch).days
            row[5] = epoch_time

        # Initialise some index's which will used to stream through data
        self.iteration = 0
        self.s = 0  # Index which is the starting point of the current batch
        self.m = 0  # Index which is the ending point of the current batch and starting point of the next batch
        self.e = 0  # Index which is the ending point of the next batch
        self.last_index = len(self.data) - 1  # last index
        self.initialise_indexs()

    def initialise_indexs(self):
        """Initialise indexes in preparation for first iteration.
        """
        self.iteration = 0
        self.s = 0
        self.e = 0
        self.updateE() # Required to make first iteration work
        self.m = self.e
        self.updateE()

    def updateS(self):
        """Update the index S.
        """
        self.s = self.m

    def updateM(self):
        """Update the index M.
        """
        self.m = self.e

    def updateE(self):
        """Update index E.
        """
        for i, row in enumerate(self.data[self.m:]):
            if row[5] > self.data[self.m][5]:
                self.e = self.m + i
                return


    def __iter__(self):
        return self

    def next(self):
        """Provides the indexes for the current and next batch of matches.

        Returns:
            S (int) - Staring index of current batch of matches
            M (int) - Ending index of current batch and starting index of the next batch of matches
            E (int) - Ending index of next batch of matches
        """
        self.iteration += 1
        if self.iteration > 1:
            self.updateS()
            self.updateM()
            self.updateE()
            if self.data[self.e][5] > (datetime(2020,1,1) - self.epoch).days or self.e == self.m: # Use only up to 2016
                self.initialise_indexs()
                raise StopIteration
        return self.s, self.m, self.e

    # Python 3.x compatibility
    def __next__(self):
        return self.next()


class MediumDataProvider(CoarseDataProvider):
    """Provides data in batches twice per tournament."""

    def updateE(self):
        """Update index E
        """
        medium_order = {'Q1': 0,'Q2': 0, 'Q3': 0, 'RR': 0,'R128': 1,'R64': 1,'R32': 1,'R16': 1,'QF': 2, 'SF': 2,'F': 2, 'BR': 2}
        roundf = medium_order[self.data[self.m][29]]
        for i, row in enumerate(self.data[self.m:]):
            if row[5] > self.data[self.m][5] or (medium_order[row[29]] != roundf and row[4] != 'C'):
                self.e = self.m + i
                break


class FineDataProvider(CoarseDataProvider):
    """Provides data in batches for every second round of each tournament."""

    def updateE(self):
        """Update index E.
        """
        fine_order = {'Q1': 0,'Q2': 0, 'Q3': 0, 'RR': 0,'R128': 1,'R64': 1,'R32': 1,'R16': 2,'QF': 2, 'SF': 3,'F': 3, 'BR': 3}
        roundf = fine_order[self.data[self.m][29]]
        for i, row in enumerate(self.data[self.m:]):
            if row[5] > self.data[self.m][5] or (fine_order[row[29]] != roundf and row[4] != 'C'):
                self.e = self.m + i
                break


class VFineDataProvider(CoarseDataProvider):
    """Steps per every round of every tournament. So no player ever plays twice within a step."""

    def updateE(self):
        """Update index E.
        """
        round = self.round_sort_order[self.data[self.m][29]]
        for i, row in enumerate(self.data[self.m:]):
            if row[5] > self.data[self.m][5] or self.round_sort_order[row[29]] != round:
                self.e = self.m + i
                break


class ThreeYearWindowProvider(VFineDataProvider):
    """Provides data with a 3 year history window required for particular models. This provider is designed for models
    where the parameters are 3 re-optimised for each batch of predictions based on 3 years of previous data."""

    def __init__(self, data_file='ATP_and_Challenger_2010to2017.csv'):
        super(ThreeYearWindowProvider,self).__init__(data_file=data_file)

    def initialise_indexs(self):
        """Initialise index's in preparation for first iteration"""
        self.iteration = 0
        self.s = 0
        self.e = 0
        first_date = self.data[0][5]
        for i, row in enumerate(self.data):
            if row[5] - first_date > 1094:
                self.m = i
                break
        self.updateE()

    def updateS(self):
        """
        Update index S.
        """
        current_date = self.data[self.e][5]
        for i, row in enumerate(self.data[self.s:]):
            if current_date - row[5] < 1096:
                self.s += i
                break