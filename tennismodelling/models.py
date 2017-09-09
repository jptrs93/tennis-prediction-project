"""Models.

This module provides classes which implement various probabilistic models for predicting the outcomes of tennis matches.

"""

from tennismodelling import data_providers
from tennismodelling import output as out
from tennismodelling import markovchain as mcm
from tennismodelling import optimisers
from tennismodelling import initialisers
import numpy as np
import csv

'----------------------------------------------------------------------------------------------------------------------'

'Base Model Classes'

'----------------------------------------------------------------------------------------------------------------------'


class Model(object):
    """Base model class."""

    def __init__(self, half_life=240, data_provider=data_providers.ThreeYearWindowProvider(),
                 surface_weights=None, decay_type=1):
        """
        Args:
            decay_type (int) : The type of recency weighting decay function to be used (1 or 2)
            half_life (float) : Half life in days of recency weighting function
            data_provider (date_providers object): An object for providing the data
            surface_weights (dict) : Dictionary of court surface weightings
        """
        self.decay_type = decay_type
        self.half_life = half_life
        self.data_provider = data_provider
        if surface_weights is None:   # Default weightings
            self.surface_weights = {'Grass' : {'Grass': 1,    'Clay': 0.01,'Carpet': 0.5,'Hard': 0.5},
                                    'Clay'  : {'Grass': 0.01, 'Clay': 1,   'Carpet': 0.1,'Hard': 0.1},
                                    'Carpet': {'Grass': 0.5,  'Clay': 0.1, 'Carpet': 1,  'Hard': 1},
                                    'Hard'  : {'Grass': 0.5,  'Clay': 0.1, 'Carpet': 1,  'Hard': 1}}
        else:
            self.surface_weights = surface_weights

    def decay_function(self, diff):
        """Recency weighting function.

        Args:
            diff (integer) : Difference between two dates in days
        Returns:
            weight (scalar) : Recency weighting
        """
        if self.decay_type == 2:
            return min(self.half_life,self.half_life**(diff / 365.))/self.half_life
        elif self.half_life == 'inf':
            return 1.
        else:
            return 0.5 ** (diff / float(self.half_life))

    def __iter__(self):
        return self

    def next(self):
        """Perform the next iteration of evaluating the model on the historical data (to be implemented in subclass)."""
        raise NotImplementedError

    # Python 3.x compatibility
    def __next__(self):
        return self.next()

    def skip_to(self, iteration):
        """Skip forward iterations of the model evaluation.

        Args:
            iteration (int) : Number of iterations to skip
        """
        for i in range(iteration):
            _ = self.data_provider.next()

    def run(self,output_file):
        """Run the model, writing predictions to an output file.

        Args:
            output_file (string) : Path to write output to
        """
        for i, output_rows in enumerate(self):
            out.append_csv_rows(output_rows, output_file)
            if i % 100 == 0: print('{0}, Iteration: {1}'.format(output_file,i))

    def check_written_rows(self,output_file):
        """Checks the output file of an unfinished model run and puts the data provider to the correct place in order
        to continue the model.

        Args:
            output_file(string) : Output file path
            """
        written_rows = [[row[0], row[1], row[2], row[3], row[4]] for row in csv.reader(open(output_file))]
        last_row = written_rows[-1]
        start_ind =0

        # set up index's of data provider to correct place
        for i, row in enumerate(self.data_provider.data):
            info = [row[-1], row[1], row[2], row[10], row[20]]
            if info == last_row:
                start_ind = i+1
        for S,M,E in self.data_provider:
            if M < start_ind:
                pass
            else:
                break
        self.data_provider.E = start_ind


class OptimisationModel(Model):
    """Another base class which contains some additional functions used in bradley terry and free parameter models."""

    def __init__(self, test_mode = False, *args, **kwargs):
        """
        Args:
            test_mode (bool) : True for testing only
        """
        self.surface = 'Hard'       # The current surface
        self.players = []           # List of players
        self.tournaments = []       # List of tournaments
        self.Z = np.array([])       # List of model output probabilities
        self.prediction_date = 0    # Epoch date of matches
        self.test_mode = test_mode
        super(OptimisationModel, self).__init__(*args, **kwargs)

    def update_players_list(self, S, M, min_matches = 2):
        """Creates a list of players who's skills are to be modelled.

        Args:
            S (integer) : Starting index in data of relevant matches
            M (integer) : Ending index in data of relevant matches
            min_matches (integer) : The minimum number of matches a player must have played to be included
        """
        players_list = {}
        for row in self.data_provider.data[S:M]:
            players_list[row[10]] = players_list .get(row[10],0) + 1
            players_list[row[20]] = players_list .get(row[20],0) + 1
        # pop players that have played less than the minimum number of matches
        self.players = [k for k, v in players_list.items() if v >= min_matches]

    def next(self):
        """Performs the next iteration of evaluating the model on the historical data.

        Returns:
            output (list) : A list of predictions for the next batch in the evaluation
        """
        S, M, E = self.data_provider.next()
        return self.do_iteration(S,M,E)

    def get_predictions(self, M, E):
        """Makes predictions for the next batch of matches in the historical data.

        Args:
            M (integer) : Starting index in data for matches to be predicted
            E (integer) : Ending index in data for matches to be predicted
        Returns:
            output (list) : List of output rows with models predictions
        """
        output = []
        for row in self.data_provider.data[M:E]:
            if row[4] not in ['Q','C','D'] and row[2] == self.surface:
                w_prob, l_prob = self.get_win_prob(row)
                if w_prob != -1:
                    output += [[row[-1], row[1], row[2], row[10], row[20], '', '', '','', '', '', w_prob, np.log(w_prob), '']]
        return output

    def get_win_prob(self, row):
        """Returns the models prediction for a match (to be implemented in subclass)."""
        raise NotImplementedError

    def do_iteration(self, S, M, E):
        """Fit the model based on the current range of data (to be implemented in subclass)."""
        raise NotImplementedError

'----------------------------------------------------------------------------------------------------------------------'

'Bradley Terry Models'

'----------------------------------------------------------------------------------------------------------------------'


class MatchBradleyTerryModel(OptimisationModel):
    """Class for implementing match level Bradley-Terry Model. In this model each player is represented bu a single
    free parameter which is optimised to minimise an error based on previous results."""

    def __init__(self, optimiser=optimisers.BradleyTerryGradient(), initialiser=initialisers.DefaultInitialiser(),
                 *args, **kwargs):
        """
        Args:
            initialiser : An initialiser object for initialising the model parameters of each optimisation
            optimiser : An optimiser object for optimising parameters of the model
        """
        self.optimiser = optimiser
        self.initialiser = initialiser
        super(MatchBradleyTerryModel, self).__init__(*args, **kwargs)

    def do_iteration(self, S, M, E):
        """Fits the model and makes predictions for the next batch of matches in the model evaluation.

        Args:
            S (integer) : Starting index in data of matches for fitting the model parameters
            M (integer) : Ending index in data for fitting the model parameters and starting index for predictions
            E (integer) : Ending index in data for matches to be predicted
        Returns:
            predictions : A list of predictions for matches in the range of index M to E
        """
        prediction_surfaces = set([row[2] for row in self.data_provider.data[M:E] if row[4] not in ['Q','C','D']])
        self.prediction_date = self.data_provider.data[M][5]
        self.update_players_list(S, M)

        predictions = []
        for surface in prediction_surfaces: # Model is re-fitted specific to each surface
            self.surface = surface
            R, W = self.extract_results(S, M)
            if self.test_mode: return R, W, self.players, self.tournaments
            self.optimiser.set_RW(R,W)
            # Optimise model parameters
            w_init = self.initialiser.get_parameters(self.players, surface)
            w = self.optimiser.optimise(w_init)
            self.initialiser.update_parameters(w, self.players, surface)
            self.Z = self.optimiser.get_probabilities(w)
            predictions += self.get_predictions(M,E)
        return predictions

    def get_win_prob(self,row):
        """Gets the models prediction for a match.

        Args:
            row (list) : Row of data corresponding to match to be predicted
        Returns:
            prediction (tuple) : Probabilities for the winning and losing player
        """
        if row[10] in self.players and row[20] in self.players:
            w_ind = self.players.index(row[10])
            l_ind= self.players.index(row[20])
            w_prob = self.Z[w_ind,l_ind]
            l_prob = self.Z[l_ind,w_ind]
            return w_prob, l_prob
        # If not enough information then return -1, -1
        return -1, -1

    def result_function(self, row):
        """Get the match level outcome for a historical match.

        Args:
            row (list) : Row of data corresponding to the historical match
        Returns:
            outcome (tuple) : Match outcome for the winning and losing player respectively
        """
        return 0.999, 0.001

    def extract_results(self, S, M):
        """Extracts results into matrix format to be used by optimiser for fitting the model parameters.

        Args:
            S (integer) : Starting index in data of relevant matches
            M (integer) : Ending index in data of relevant matches
        Returns:
            R (NxN Matrix) : A matrix containing the results between players
            W (NxN Matrix) : A matrix containing the weights associated with the results in R
        """

        R = np.zeros((len(self.players), len(self.players)), dtype=float)  # create a matrix to store results
        W = np.zeros_like(R, dtype=float)  # create matrix to store match weights

        for row in self.data_provider.data[S:M]:
            # Get the outcomes for winning and losing player
            winner_outcome, loser_outcome = self.result_function(row)
            if winner_outcome != -1 and row[10] in self.players and row[20] in self.players:
                # weightings
                recency_weight = self.decay_function(self.prediction_date - row[5])
                surface_weight = self.surface_weights[self.surface].get(row[2], 0.1)
                weight = recency_weight *surface_weight
                # player index's
                winner_index = self.players.index(row[10])  # index for winning player
                loser_index = self.players.index(row[20])  # index for losing player
                # update result matrix
                R[winner_index, loser_index] = (R[winner_index, loser_index] * W[winner_index, loser_index]
                                                + winner_outcome * weight) / (W[winner_index, loser_index] + weight)
                W[winner_index, loser_index] += weight
                R[loser_index, winner_index] = (R[loser_index, winner_index] * W[loser_index, winner_index]
                                                + loser_outcome * weight) / (W[loser_index, winner_index] + weight)
                W[loser_index, winner_index] += weight
        return R, W


class GameBradleyTerryModel(MatchBradleyTerryModel):
    """Class for implementing game level Bradley-Terry model."""

    def result_function(self, row):
        """Gets the game level outcome for a historical match.

        Args:
            row (list) : Row of data corresponding to the historical match
        Returns:
            outcome (tuple) : Outcome for the winning and losing player respectively
        """
        return float(row[49]), float(row[50])

    def get_win_prob(self, row):
        """Gets the models prediction for a match.

        Args:
            row (list) : Row of data corresponding to match to be predicted
        Returns:
            prediction (tuple) : Probabilities for the winning and losing player
        """
        best_of = int(row[28])
        winner_prob, loser_prob = super(GameBradleyTerryModel,self).get_win_prob(row)
        if winner_prob != -1:
            w_prob = mcm.match_chance_games(winner_prob, loser_prob , best_of)
            return w_prob, 1 - w_prob
        else:
            return -1, -1


class PointBradleyTerryModel(MatchBradleyTerryModel):
    """Class for implementing point level Bradley-Terry model."""

    def result_function(self, row):
        """Gets the point level outcome for a historical match.

        Args:
            row (list) : Row of data corresponding to the historical match
        Returns:
            outcome (tuple) : Outcome for the winning and losing player respectively
        """
        return float(row[53]), float(row[54])

    def get_win_prob(self, row):
        """Gets the models prediction for a match.

        Args:
            row (list) : Row of data corresponding to match to be predicted
        Returns:
            prediction (tuple) : Probabilities for the winning and losing player
        """
        best_of = int(row[28])
        winner_prob, loser_prob = super(PointBradleyTerryModel,self).get_win_prob(row)
        if winner_prob != -1:
            w_prob = mcm.match_chance(winner_prob, loser_prob, best_of)
            return w_prob, 1- w_prob
        else:
            return -1, -1


class PointGameBradleyTerryModel(GameBradleyTerryModel):
    """Class for implementing Bradley-Terry model based on game level but with integrated point information."""

    def result_function(self, row):
        """Gets the combined game and point level outcome for a historical match.

        Args:
            row (list) : Row of data corresponding to the historical match
        Returns:
            outcome (tuple) : Outcome for the winning and losing player respectively
        """
        return float(row[57]), float(row[58])

'----------------------------------------------------------------------------------------------------------------------'

'Free Parameter Models'

'----------------------------------------------------------------------------------------------------------------------'


class FreeParameterPointModel(PointBradleyTerryModel):
    """Class for implementing free parameter point model. In this model each player is modelled using two free
    parameters and the alternating service structure treated explicitly."""

    def result_function(self, row):
        """Gets the point level service outcomes for a historical match.

        Args:
            row (list) : Row of data corresponding to the historical match
        Returns:
            outcome (tuple) : Outcome for the winning and losing player respectively
        """
        return float(row[55]), float(row[56])


class FreeParameterGameModel(GameBradleyTerryModel):
    """Class for implementing free parameter game model."""

    def result_function(self, row):
        """Gets the game level service outcome for a historical match.

        Args:
            row (list) : Row of data corresponding to the historical match
        Returns:
            outcome (tuple) : Outcome for the winning and losing player respectively
        """
        return float(row[51]), float(row[52])


class FreeParameterPointGameModel(GameBradleyTerryModel):
    """Class for implementing free parameter game model with integrated point information."""

    def result_function(self, row):
        """Gets the game level outcome for a historical match.

        Args:
            row (list) : Row of data corresponding to the historical match
        Returns:
            outcome (tuple) : Outcome for the winning and losing player respectively
        """
        return float(row[59]), float(row[60])

'----------------------------------------------------------------------------------------------------------------------'

'Joint Optimisation Time Series Model'

'----------------------------------------------------------------------------------------------------------------------'


class JointOptTimeSeriesModel(PointGameBradleyTerryModel):
    """Class for implementing time series Bradley-Terry model. The model assumes a gaussian drift on player skills over
     time and each player is treated as having distinct skills at different times which are jointly optimised."""

    def __init__(self, steps=4, *args, **kwargs):
        """
        Args:
            steps (integer) : The number of time periods to break the history of matches into
        """
        self.steps = steps
        super(JointOptTimeSeriesModel,self).__init__(*args, **kwargs)
        self.half_life = 'inf'

    def extract_results(self, S, M):
        """Extracts results into matrix format to be used by optimiser for fitting the model parameters.

        Args:
            S (integer) : Starting index in data of relevant matches
            M (integer) : Ending index in data of relevant matches
        Returns:
            R (NxN Matrix) : A matrix containing the results between players
            W (NxN Matrix) : A matrix containing the weights associated with the results in R
        """
        steps = self.steps
        days_per_step = int((self.data_provider.data[M][5] - self.data_provider.data[S][5])/steps)
        n = len(self.players)
        R = np.zeros((n*steps, n*steps), dtype=float)
        W = np.zeros_like(R)
        s = S
        for a, i in enumerate(reversed(range(steps))):
            date_end = self.data_provider.data[M][5] - days_per_step*i
            for ind, ii in enumerate(self.data_provider.data[S:M]):
                if ii[5] > date_end:
                    m = ind + S
                    break
                m = ind + S + 1
            R1, W1 = super(JointOptTimeSeriesModel,self).extract_results(s, m)
            # Input the results into the result matrix
            if a < steps -1:
                R[a * n:(a + 1) * n, a * n:(a + 1) * n] = R1
                W[a * n:(a + 1) * n, a * n:(a + 1) * n] = W1
            else:
                R[a * n:, a * n:] = R1
                W[a * n:, a * n:] = W1
            s = m
        return R, W


class JointOptTimeSeriesModelRefined(JointOptTimeSeriesModel):
    """Extension of JointOptTimeSeriesModel. The 3 year history window is broken up into unequal rather than equal time
     periods which start coarse and get finer (right now is hard coded for only 4 steps)."""

    def extract_results(self, S, M):
        """Extracts results into matrix format to be used by optimiser for fitting the model parameters.

        Args:
            S (integer) : Starting index in data of relevant matches
            M (integer) : Ending index in data of relevant matches
        Returns:
            R (NxN Matrix) : A matrix containing the results between players
            W (NxN Matrix) : A matrix containing the weights associated with the results in R
        """
        steps = 4   # self.steps
        days_per_step = int((self.data_provider.data[M][5] - self.data_provider.data[S][5])/steps)
        n = len(self.players)
        R = np.zeros((n*steps, n*steps), dtype=float)
        W = np.zeros_like(R)
        s = S
        # 122, 243, 365
        for a, i in enumerate([122 + 243 + 365, 122 + 243, 122, 0]):
            date_end = self.data_provider.data[M][5] - i
            for ind, ii in enumerate(self.data_provider.data[S:M]):
                if ii[5] > date_end:
                    m = ind + S
                    break
                m = ind + S + 1

            R1, W1 = super(JointOptTimeSeriesModel, self).extract_results(s, m)
            # Input the results into the result matrix
            if a < steps -1:
                R[a * n:(a + 1) * n, a * n:(a + 1) * n] = R1
                W[a * n:(a + 1) * n, a * n:(a + 1) * n] = W1
            else:
                R[a * n:, a * n:] = R1
                W[a * n:, a * n:] = W1
            s = m
        return R, W


'----------------------------------------------------------------------------------------------------------------------'

'Surface Factor Models'

'----------------------------------------------------------------------------------------------------------------------'


class PointGameSurfaceFactorModel(PointGameBradleyTerryModel):
    """Class for implementing surface factor model. This model includes free parameters for each surface and
    multiple parameters for each player."""

    def __init__(self, repeats = 0, *args, **kwargs):
        """
        Args:
            repeats (integer) : The number times to repeat the optimisation at each iteration
        """
        self.repeats = repeats
        super(PointGameSurfaceFactorModel, self).__init__(*args, **kwargs)

    def extract_results(self, S, M):
        """Extracts results into matrix format to be used by optimiser for fitting the model parameters.

        Args:
            S (integer) : Starting index in data of relevant matches
            M (integer) : Ending index in data of relevant matches
        Returns:
            R (NxN Matrix) : A matrix containing the results between players
            W (NxN Matrix) : A matrix containing the weights associated with the results in R
        """
        R = np.zeros((3, len(self.players), len(self.players)), dtype=float)  # create a matrix to store results
        W = np.zeros_like(R, dtype=float)  # create matrix to store match weights
        for row in self.data_provider.data[S:M]:

            surf_ind = {'Grass' : 0, 'Clay': 1, 'Carpet' : 2, 'Hard' : 2}
            # recency weighting
            diff = self.prediction_date - row[5]
            weight = self.decay_function(diff)
            # Get the result for winning player
            result, loser_result  = self.result_function(row)

            if result != -1 and row[2] != '' and row[10] in self.players and row[20] in self.players:
                winner_index = self.players.index(row[10])  # index for winning player
                loser_index = self.players.index(row[20])  # index for losing player
                si = surf_ind.get(row[2],2)
                if weight > 0:
                    # Update R and W matrices
                    R[si,winner_index, loser_index] = (R[si,winner_index, loser_index] * W[si,
                        winner_index, loser_index] + result * weight) / (W[si,winner_index, loser_index] + weight)
                    W[si,winner_index, loser_index] += weight
                    W[si,loser_index, winner_index] += weight
                    R[si,loser_index, winner_index] = 1 - R[si,winner_index, loser_index]
        return R, W

    def do_iteration(self, S, M, E):
        """Fits the model and makes predictions for the next batch of matches in the model evaluation.

        Args:
            S (integer) : Starting index in data of matches for fitting the model parameters
            M (integer) : Ending index in data for fitting the model parameters and starting index for predictions
            E (integer) : Ending index in data for matches to be predicted
        Returns:
            predictions : A list of predictions for matches in the range of index M to E
        """
        if any([row[4] in ['A','F','G','M'] for row in self.data_provider.data[M:E]]):
            self.prediction_date = self.data_provider.data[M][5]
            self.update_players_list(S, M)
            R, W = self.extract_results(S, M)
            if self.test_mode: return R, W, self.players, self.tournaments

            # Optimisation model parameters
            self.optimiser.set_RW(R,W)
            w = self.optimiser.optimise()
            self.Z = self.optimiser.get_probabilities(w)
            for i in range(self.repeats):   # re-optimise and average result
                w = self.optimiser.optimise()
                self.Z += (self.optimiser.get_probabilities(w) - self.Z)/(i+2.)
        predictions = self.get_predictions(M,E)
        return predictions

    def get_win_prob(self, row):
        """Gets the models prediction for a match.

        Args:
            row (list) : Row of data corresponding to match to be predicted
        Returns:
            prediction (tuple) : Probabilities for the winning and losing player
        """
        if row[10] in self.players and row[20] in self.players:
            best_of = int(row[28])
            surf_ind = {'Grass': 0, 'Clay': 1, 'Carpet': 2, 'Hard': 2}
            si = surf_ind.get(row[2], 2)
            w_ind = self.players.index(row[10])
            l_ind= self.players.index(row[20])
            w_prob =self.Z[si,w_ind,l_ind]
            w_p = mcm.match_chance_games(w_prob, 1 - w_prob, best_of)
            return w_p, 1- w_p
        return -1, -1

    def get_predictions(self, M, E):
        """Makes predictions for the next batch of matches in the historical data.

        Args:
            M (integer) : Starting index in data for matches to be predicted
            E (integer) : Ending index in data for matches to be predicted
        Returns:
            output (list) : List of output rows with models predictions
        """
        output = []
        # Make the predictions
        for row in self.data_provider.data[M:E]:
            if row[4] not in ['Q','C','D']:
                w_prob, l_prob = self.get_win_prob(row)
                if w_prob != -1:
                    output += [[row[-1], row[1], row[2], row[10], row[20], '', '', '','', '', '', w_prob, np.log(w_prob), '']]
        return output

'----------------------------------------------------------------------------------------------------------------------'

'Rating Models'

'----------------------------------------------------------------------------------------------------------------------'


class BayesianRatingModel(object):
    """Implements a time series version of a Bradley-Terry model based on approximate filtering updates."""

    def __init__(self, drift=0.9, prior_var=1, level='Point', surface_weighting=True,
                 data_provider=data_providers.VFineDataProvider(),
                 optimiser=optimisers.BradleyTerryVariationalInference(prior_var=1, use_correlations=True, use_samples=False),
                 surface='Hard'):
        """
        Args:
            drift (scalar [0,1]) : Gaussian drift parameter
            prior_var (scalar) : The prior variance of player skills
            level (string) : Level of information to base the model on (match, game, point or combined point and game)
            surface_weighting (bool) : Indicates whether to use surface weightings
            data_provider (data_provider object) : Object for providing the data in correct manner
            optimiser (optimiser object) : Object for performing the optimisations (filtering updates)
            surface (string) :  which surface to model (only applicable if surface weightings are used)
        """
        self.drift = drift
        self.prior_var = prior_var
        self.level = level
        self.surface_weighting = surface_weighting
        self.data_provider = data_provider
        self.optimiser = optimiser
        self.surface = surface
        self.m = np.array([])       # Skill means of the players
        self.V = np.array([])       # Covariance matrix of the players skills
        self.Z = np.array([])       # Output probabilities
        self.name_indexes = {}      # Tracker of player names to indexes in m, V and Z
        self.last_appearance= {}    # Tracker of days since player last played a match
        self.archive_mean = {}      # Skill means of archived players
        self.archive_var = {}       # Skill variances of archived players

        if surface_weighting:
            self.surface_weights = {'Grass'  : {'Grass' : 1, 'Clay': 0.01, 'Carpet' : 0.5, 'Hard' : 0.5},
                                    'Clay'   : {'Grass' : 0.01, 'Clay': 1, 'Carpet' : 0.1, 'Hard' : 0.1},
                                    'Carpet' : {'Grass' : 0.5, 'Clay': 0.1, 'Carpet' : 1, 'Hard' : 1},
                                    'Hard'   : {'Grass' : 0.5, 'Clay': 0.1, 'Carpet' : 1, 'Hard' : 1}}
        else:
            self.surface_weights = {'Grass'  : {'Grass' : 1, 'Clay': 1, 'Carpet' : 1, 'Hard' : 1},
                                    'Clay'   : {'Grass' : 1, 'Clay': 1, 'Carpet' : 1, 'Hard' : 1},
                                    'Carpet' : {'Grass' : 1, 'Clay': 1, 'Carpet' : 1, 'Hard' : 1},
                                    'Hard'   : {'Grass' : 1, 'Clay': 1, 'Carpet' : 1, 'Hard' : 1}}

    def next(self):
        """Computes the next batch of predictions.

        Returns:
            output (list) : A list of predictions
        """
        S, M, E = self.data_provider.next()
        return self.do_iteration(S,M,E)

    def do_iteration(self, S, M, E):
        """Updates skills of players based on the current batch of matches and makes predictions for the next batch.

        Args:
            S (integer) : Starting index in data of matches for updating player skills
            M (integer) : Ending index in data for updating player skills and starting index for predictions
            E (integer) : Ending index in data for matches to be predicted
        Returns:
            predictions : A list of predictions for matches in the range of index M to E
        """
        # check for any new players and initialise the skill means and variances
        new_players = self.update_names(S,M)
        self.update_m(new_players)
        self.update_V(new_players)
        # get results for current batch
        R, W = self.extract_results(S,M)
        if np.any(W > 0):
            # update skills
            self.optimiser.set_RW(R,W)
            self.m, self.V = self.optimiser.optimise(prior = [self.m, self.V],display=False)
        if len(self.m) > 0:
            self.apply_drift(S,M)
            self.clean_old_players(M)
            self.Z = self.optimiser.get_probabilities([self.m,self.V])
        # get and return predictions for next batch
        return self.get_predictions(M,E)

    def get_predictions(self,M,E):
        """Makes predictions for the next batch of matches in the historical data.

        Args:
            M (integer) : Starting index in data for matches to be predicted
            E (integer) : Ending index in data for matches to be predicted
        Returns:
            output (list) : List of output rows with models predictions
        """
        output = []
        for row in self.data_provider.data[M:E]:
            if row[4] not in ['Q','C','D']:
                w_prob = self.get_win_prob(row)
                if w_prob != -1:
                    w_ind = self.name_indexes[row[10]]
                    l_ind = self.name_indexes[row[20]]
                    output += [[row[-1], row[1], row[2], row[10], row[20], '', '', self.m[w_ind],
                                self.m[l_ind], self.V[w_ind,w_ind], self.V[l_ind,l_ind], w_prob,
                                np.log(w_prob), '']]
        return output

    def get_win_prob(self, row):
        """Gets the models prediction for a match.

        Args:
            row (list) : Row of data corresponding to match to be predicted
        Returns:
            prediction (float) : Winner probability for predicted match
        """
        if (row[2] ==self.surface or(self.surface == 'Hard' and row[2] == 'Carpet') or not self.surface_weighting) and row[10] in self.name_indexes and row[20] in self.name_indexes:
            w_ind = self.name_indexes[row[10]]
            l_ind = self.name_indexes[row[20]]
            best_of = int(row[28])
            w_prob = self.Z[w_ind,l_ind]
            l_prob = self.Z[l_ind,w_ind]
            if self.level == 'Point':
                return mcm.match_chance(w_prob, l_prob, best_of)
            elif self.level == 'Match':
                return w_prob
            else:
                return mcm.match_chance_games(w_prob, l_prob, best_of)
        else:
            return -1

    def update_names(self,S,M):
        """Adds any new players to the player index tracker.

        Args:
            S (integer) : Starting index in data of matches for updating player skills
            M (integer) : Ending index in data for updating player skills
        Returns:
            new_players (list) : List of any new players
        """
        new_players = []
        for row in self.data_provider.data[S:M]:
            if row[10] not in self.name_indexes:
                self.name_indexes[row[10]] = len(self.name_indexes)
                new_players += [row[10]]
            if row[20] not in self.name_indexes:
                self.name_indexes[row[20]] = len(self.name_indexes)
                new_players += [row[20]]
            self.last_appearance[row[10]] = self.data_provider.data[S][5]
            self.last_appearance[row[20]] = self.data_provider.data[S][5]
        return new_players

    def clean_old_players(self, M):
        """Moves players who haven't played any recent matches into archive.

        Args:
            M (integer) : Ending index in data for updating player skills
        """
        current_date = self.data_provider.data[M][5]
        for key in self.last_appearance.keys():
            if current_date - self.last_appearance[key] > 100:    # Archive players with no matches in last 100 days
                if key in self.name_indexes:
                    # archive skill mean and variance
                    self.archive_var[key] = self.V[self.name_indexes[key],self.name_indexes[key]]
                    self.archive_mean[key] = self.m[self.name_indexes[key]]
                    # remove from index tracker
                    index = self.name_indexes[key]
                    self.name_indexes.pop(key)
                    # adjust other indexes in tracker
                    for key in self.name_indexes.keys():
                        if self.name_indexes[key] > index:
                            self.name_indexes[key] -=1
                    # remove from covariance and mean matrices
                    self.V = np.delete(self.V, index, 0)
                    self.V = np.delete(self.V, index, 1)
                    self.m = np.delete(self.m,index)

    def update_m(self, new_players):
        """Updates the vector of skill means for any new players. Initialises from archive if applicable otherwise to 0.

        Args:
            new_players (list) : list of new players
        """
        if len(new_players) > 0:
            new_means = np.array([self.archive_mean.get(player,0) for player in new_players])
            self.m = np.append(self.m, new_means)

    def update_V(self, new_players):
        """Updates the covariance matrix for any new players. Initialises new diagonal terms from archive if applicable
        otherwise to prior variance.

        Args:
            new_players (list) : list of new players
        """
        n_old = self.V.shape[0]
        if len(new_players) > 0:
            diag = np.append(np.zeros(n_old),np.array([self.archive_var.get(player,self.prior_var) for player in new_players]))
            temp = self.V
            self.V =np.diag(diag)
            self.V[:n_old,:n_old] = temp

    def apply_drift(self,S,M):
        """Applies gaussian drift to the skills and variances of players based on time in days between updates
        Args:

            S (integer) : Starting index in data of matches for updating player skills
            M (integer) : Ending index in data for updating player skills
        """
        time_diff = self.data_provider.data[M][5] - self.data_provider.data[S][5]
        drift = self.drift**(float(time_diff)/365)
        # drift mean
        self.m *= drift
        # drift variances
        diag_indices = np.diag_indices_from(self.V)
        self.V[diag_indices] = (drift**2)*self.V[diag_indices] + (1-(drift**2))*self.prior_var
        # apply drift to archived parameters
        for key in self.archive_mean.keys():
            self.archive_var[key] = (drift**2)*self.archive_var[key] + (1-(drift**2))*self.prior_var
            self.archive_mean[key] *= drift

    def extract_results(self, S, M):
        """extracts results from a set of data into matrix format for solving the Bradley
        Terry model.

        Args:
            S (integer) : Starting match index for training data
            M (integer) : Ending match index for training data
        Returns:
            R (NxN Matrix) : A matrix containing the results between players
            W (NxN Matrix) : A matrix containing the weights associated with the results in R
        """
        R = np.zeros((len(self.name_indexes), len(self.name_indexes)), dtype=float)
        W = np.zeros_like(R, dtype=float)
        for row in self.data_provider.data[S:M]:
            # down weight non atp matches
            if row[4] in ['D','Q','C']:
                weight = 0.95
            else:
                weight = 1.
            # Get the outcome of match
            w_outcome, l_outcome = self.result_function(row)
            if w_outcome != -1 and row[10] in self.name_indexes and row[20] in self.name_indexes:
                winner_index = self.name_indexes[row[10]]  # index for winning player
                loser_index = self.name_indexes[row[20]]  # index for losing player
                weight *= self.surface_weights[self.surface].get(row[2],0.1)
                if weight > 0:
                    # Update R and W matrices
                    R[winner_index, loser_index] = (R[winner_index, loser_index] * W[
                        winner_index, loser_index] + w_outcome * weight) / (W[winner_index, loser_index] + weight)
                    W[winner_index, loser_index] += weight
                    R[loser_index, winner_index] = (R[loser_index, winner_index] * W[
                        loser_index, winner_index] + l_outcome * weight) / (W[loser_index, winner_index] + weight)
                    W[loser_index, winner_index] += weight
        return R, W

    def result_function(self, row):
        """Returns the outcome of an historical match.

        Args:
            row (list) : row of match data in standard format
        Returns:
            R (float) : A value between 0 and 1 which is the result for the winning player
        """
        if self.level == 'Point':
            return float(row[53]), float(row[54])
        elif self.level == 'Game':
            return float(row[49]), float(row[50])
        elif self.level == 'Match':
            return 0.999, 0.001
        else:
            return float(row[57]), float(row[58])


class BayesianFreeParameterRatingModel(BayesianRatingModel):

    def clean_old_players(self,M):
        """Removes players who haven't played any recent matches.

        Args:
            M (integer) : Ending index in data for updating player skills
        """
        now = self.data_provider.data[M][5]
        for key in self.last_appearance.keys():
            if now - self.last_appearance[key] > 100:
                if key in self.name_indexes:
                    index = self.name_indexes[key]
                    self.name_indexes.pop(key)
                    # correct other indexes
                    for key in self.name_indexes.keys():
                        if self.name_indexes[key] > index:
                            self.name_indexes[key] -=1
                    # remove from covariances and means
                    n = int(self.V.shape[0]/2)
                    self.V = np.delete(self.V, index, 0)
                    self.V = np.delete(self.V, index, 1)
                    self.V = np.delete(self.V, index+n-1, 0)
                    self.V = np.delete(self.V, index+n-1, 1)
                    self.m = np.delete(self.m,index)
                    self.m = np.delete(self.m, index +n-1)


    def update_m(self,players):
        """If there is any new players in the index tracker then there means are added
        to the mean vectors initialised to 0.

        Args:
            new_players (list) : list of new players
        """
        n_old = int(self.m.shape[0]/2)
        n_new = len(self.name_indexes)
        if n_new > n_old:
            temp = self.m
            self.m = np.zeros(n_new*2)
            self.m[:n_old] = temp[:n_old]
            self.m[n_new:n_new+n_old] = temp[n_old:]

    def update_V(self,players):
        """Updates the covariance matrix for any new players. Initialises new diagonal terms to prior variance.

        Args:
            new_players (list) : list of new players
        """
        n_old = int(self.V.shape[0]/2)
        n_new = len(self.name_indexes)
        if self.V.size < 1:
            diag = np.ones(n_new * 2) * self.prior_var
            self.V = np.diag(diag)
        elif n_new >n_old:
            diag = np.ones(n_new*2)*self.prior_var
            temp = self.V
            self.V =np.diag(diag)
            self.V[:n_old,:n_old] = temp[:n_old,:n_old]
            self.V[:n_old,n_new:n_new+n_old] = temp[:n_old, n_old:]
            self.V[n_new:n_new + n_old, :n_old] = temp[ n_old:, :n_old]
            self.V[n_new:n_new + n_old, n_new:n_new + n_old] = temp[ n_old:, n_old:]

    def result_function(self, row):
        """Returns the outcome of a historical match

        Args:
            row (list) : row of match data in standard format
        Returns:
        """
        if self.level == 'Point':
            return float(row[55]), float(row[56])
        elif self.level == 'Game':
            return float(row[51]), float(row[52])
        else:
            return float(row[59]), float(row[60])



class Elo538(Model):
    """ELO rating model as described in [1].
    [1] Kovalchik, Stephanie Ann. "Searching for the GOAT of tennis win prediction."
    Journal of Quantitative Analysis in Sports 12.3 (2016): 127-138
    """

    def __init__(self, *args, **kwargs):

        self.ratings = {}           # stores player ratings
        self.life_time_matches = {} # stores total number of matches played by each player
        super(Elo538,self).__init__(*args,**kwargs)
        self.ii = 0

    def next(self):
        """Makes a prediction for the next match and then updates the player ratings.

        Returns:
            prediction (list) : Output row containing model prediction
        """
        if self.ii < len(self.data_provider.data):
            row = self.data_provider.data[self.ii]
            prediction = self.get_prediction(row)
            self.update_ratings(row)
            self.ii +=1
            return prediction
        else:
            self.ii = 0
            raise StopIteration
        
    def update_ratings(self,row):
        """Updates player rating based on a match.

        Args:
            row (list) : Row of data relating to the match
        """
        # Check for and initialise any new players
        if row[10] not in self.ratings:
            self.ratings[row[10]] = 1500
            self.life_time_matches[row[10]] = 0
        if row[20] not in self.ratings:
            self.ratings[row[20]] = 1500
            self.life_time_matches[row[20]] = 0
        # Add to player match counts
        self.life_time_matches[row[10]] += 1
        self.life_time_matches[row[20]] += 1

        # get ratings
        w_rating = self.ratings[row[10]]
        l_rating = self.ratings[row[20]]
        # update ratings
        prob = 1./(1.+10**(0.0025*l_rating - 0.0025*w_rating))
        w_k = (self.life_time_matches[row[10]] + 5.)**0.4
        l_k = (self.life_time_matches[row[20]] + 5.)**0.4
        # add 10% weight to grand slam matches
        if row[4] == 'G':
            w_k *= 1.1
            l_k *= 1.1
        self.ratings[row[10]] = w_rating + w_k*(1 - prob)
        self.ratings[row[20]] = l_rating + l_k*(0 -(1- prob))

    def get_prediction(self,row):
        """Returns the prediction for a match.

        Args:
            row (list) : Row of data relating to the match to be predicted
        Returns:
            prediction (list) : Output row containing model prediction
        """
        if row[10] in self.ratings and row[20] in self.ratings:
            w_rating = self.ratings[row[10]]
            l_rating = self.ratings[row[20]]
            prob = 1./(1.+10**(0.0025*l_rating - 0.0025*w_rating))
            output = [[row[-1], row[1], row[2], row[10], row[20], '', '',w_rating,l_rating, self.life_time_matches[row[10]],self.life_time_matches[row[20]] ,prob,np.log(prob), '']]
            return output
        else:
            return []




