"""Initialisers.

This module provides classes for initialising model parameters. These can be used so that the parameters of each fit
during the model evaluation are initialised based on the previous fit.

"""

import numpy as np


class DefaultInitialiser(object):
    """Default initialiser class which does nothing so initialisation will revert to default provided by optimiser."""

    def get_parameters(self, players, surface):
        """Returns parameters to initialise optimisation with, although for this class None is
        returned so that the optimisers default initialisation is used.

        Args:
            players (list) : List of players being modelled
            surface (string) : Surface being modelled
        Returns:
            None
        """
        return None

    def update_parameters(self, w, players, surface):
        """Updates the stored parameters for players to be used as future initialisations.

        Args:
            w (vector) : Vector of optimised parameters
            players (list) : List of players being modelled
            surface (string) : Surface being modelled
        """
        pass


class BradleyTerryInitialiser(object):
    """Class for initialising parameters in Bradley-Terry model based on previous fit."""

    def __init__(self):
        self.players = {'Hard': {}, 'Clay': {}, 'Carpet': {}, 'Grass': {}}  # For storing player parameters

    def get_parameters(self, players, surface):
        """Returns parameters to initialise optimisation with.

        Args:
            players (list) : List of players being modelled
            surface (string) : Surface being modelled
        Returns:
            w (vector) : Initialised model parameters
        """
        return np.array([self.players[surface].get(player, 0) for player in players])

    def update_parameters(self, w, players, surface):
        """Updates the stored parameters for players to be used as future initialisations.

        Args:
            w (vector) : Vector of optimised parameters
            players (list) : List of players being modelled
            surface (string) : Surface being modelled
        """
        pass
        for i, player in enumerate(players):
            self.players[surface][player] = w[i]


class FreeParameterInitialiser(BradleyTerryInitialiser):
    """Class for initialising parameters in Free Parameter model based on previous fit."""

    def get_parameters(self, players, surface):
        """Returns parameters to initialise optimisation with.

        Args:
            players (list) : List of players being modelled
            surface (string) : Surface being modelled
        Returns:
            w (vector) : Initialised model parameters
        """
        attacking = [self.players[surface].get(player,[0,0])[0] for player in players]
        defensive = [self.players[surface].get(player,[0,0])[1] for player in players]
        return np.array(attacking+defensive)

    def update_parameters(self, w, players, surface):
        """Updates the stored parameters for players to be used as future initialisations.

        Args:
            w (vector) : Vector of optimised parameters
            players (list) : List of players being modelled
            surface (string) : Surface being modelled
        """
        n= len(players)



class JointOptTimeSeriesInitialiser():
    """Class for initialising parameters in joint optimisation time series Bradley-Terry model based on previous fit.
    NOTE: unfinished -currently only initialises mean not covariance."""

    def __init__(self):
        self.players = {}       # Tracker of player name to indexes in m and V
        self.V = np.array([])   # Covariance matrix of previous fit
        self.m = np.array([])   # Mean vector of previous fit

    def get_parameters(self, players, surface):
        """Returns parameters to initialise optimisation with.

        Args:
            players (list) : List of players being modelled
            surface (string) : Surface being modelled
        Returns:
            tuple : Initialised mean and covariance
        """
        params = [self.players.get(player,0) for player in players]
        means = np.array(params + params + params +params)
        return [means, None]

    def update_parameters(self, w, players, surface):
        """Updates the stored parameters for players to be used as future initialisations.

        Args:
            w (tuple) : Tuple containing the mean and covariance of approximate posterior
            players (list) : List of players being modelled
            surface (string) : Surface being modelled
        """
        n= len(players)
        w = w[0]
        for i, player in enumerate(players):
            self.players[player] = w[i+n]



class BradleyTerryVariationalInferenceInitialiser(object):

    def __init__(self):

        self.player_means = {'Hard': {}, 'Clay': {}, 'Carpet': {}, 'Grass': {}}         # stores player skill means
        self.player_variances = {'Hard': {}, 'Clay': {}, 'Carpet': {}, 'Grass': {}}     # stores player skill variances


    def get_parameters(self, players, surface):
        """Returns parameters to initialise optimisation with.

        Args:
            players (list) : List of players being modelled
            surface (string) : Surface being modelled
        Returns:
            tuple : Initialised mean and covariance
        """
        means = np.array([self.player_means[surface].get(player,0) for player in players])
        vars = np.array([self.player_variances[surface].get(player,1) for player in players])
        V = np.diag(vars)
        return [means, V]

    def update_parameters(self, w,players, surface):
        """Updates the stored parameters for players to be used as future initialisations.

        Args:
            w (tuple) : Tuple containing the mean and covariance of approximate posterior
            players (list) : List of players being modelled
            surface (string) : Surface being modelled
        """
        means= w[0]
        V = w[1]
        for i, player in enumerate(players):
            self.player_means[surface][player] = means[i]
            self.player_variances[surface][player] = V[i,i]








