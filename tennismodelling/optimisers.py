"""Optimisers.

This module provides classes for performing parameter optimisation in the models

"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy import special


'----------------------------------------------------------------------------------------------------------------------'

'Base classes'

'----------------------------------------------------------------------------------------------------------------------'


class BaseOptimiser(object):
    """
    Base class for optimisers.
    """
    def __init__(self):
        """Base optimiser class.
        """
        self.R = np.array([])
        self.W = np.array([])
        self.n = 0
        self.num_params = 0

    def set_RW(self, R, W):
        """Update the matrices R and W.

        Args:
            R (NxN matrix) : Matrix of match results
            W (NxN matrix) : Matrix of weights corresponding to results in R
        """
        self.R = R
        self.W = W
        self.n = R.shape[1]
        self.set_num_params()

    def set_num_params(self):
        """Set the number of model parameters.
        """
        self.num_params = self.n


class GradientOptimiser(BaseOptimiser):
    """
    Base class for gradient based optimisers.
    """
    def __init__(self, error="ll", reg=0, reg_type='L2', init=np.random.randn, display=False):
        """
        Args:
            error (string) : The type of error function to use  - log likelihood ("ll") or square ("square")
            reg (scalar) : Regularisation constant for regularisation on parameters
            reg_type (string) : Type of regularisation to use "L2" or "L1"
            init (function) : A function for initialising numpy arrays
            display (bool) : Whether to display the optimisation details
        """
        self.display = display
        self.reg = reg
        self.error = error
        self.reg_type = reg_type
        self.init = init
        super(GradientOptimiser,self).__init__()

    def apply_regularisation(self, cost, wBar, w):
        """Applies regularisation.
        
        Args:
            w (vector) : vector of current parameters
            cost (scalar) : cost
            wBar (vector) : gradients of w
        """
        # L1 regularisation
        if self.reg_type == 'L1' and self.reg > 0:
            cost += self.reg * np.sum(np.abs(w))
            wBar += self.reg * (2 * (w > 0) - 1)
        # L2 regularisation
        elif self.reg > 0:  # L2
            cost += self.reg * np.dot(w,w)
            wBar += self.reg * 2 * w

    def cost_gradients(self,w):
        """Returns the cost and gradient of the model (to be implemented in sub class).
        """
        raise NotImplementedError

    def initialise_params(self):
        """Initialises the parameters of the model.
        """
        return 0.1*self.init(self.num_params)

    def pack_params(self,w):
        """Should pack the parameters into a single vector if not already in that form.
        """
        return w

    def optimise(self, w=None, max_iterations=250, display=False):
        """Optimises the model parameters (w).
        
        Args:
            w (vector) : Initial parameter settings to use
            display (bool) : Whether to print optimisation details
            max_iterations (int) : The maximum number of iterations to use in optimisation
        Returns:
            res.x (vector) : Vector of optimised parameters
        """
        # If no initial values given then use built in initialisation
        if w is None:
            w = self.initialise_params()
        w = self.pack_params(w)
        if self.display:
            display = True
        # Optimise using quasi-Newton method
        res = minimize(self.cost_gradients, x0=np.squeeze(w), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iterations, 'disp': display, 'gtol': 1e-8, 'ftol': 1e-8})
        return res.x

'----------------------------------------------------------------------------------------------------------------------'

'Bradley-Terry Model Optimisers for MAP fitting'

'----------------------------------------------------------------------------------------------------------------------'


class BradleyTerryGradient(GradientOptimiser):
    """
    Optimiser for solving parameters in a Bradley Terry model.
    """

    def cost_gradients(self, w):
        """Gets the cost and gradients for set of parameters w.
        
        Args:
            w (vector) : Vector of current parameters
        Returns:
           cost (scalar) : Current error of cost function
           wBar (vector) : Gradients with respect to w
        """

        Z1 = w[:,None] - w[None,:]
        Z2 = 1. / (1 + np.exp(Z1))
        ww = np.sum(self.W)  # total weight

        if self.error == "square":  # Square error cost function
            cost = np.sum(self.W*((Z2 - self.R)**2))/ ww
            Z1Bar = 2 * self.W * (Z2 - self.R) * Z2 * (1 - Z2) / ww
        else:  # Log likelihood cost function
            cost = - np.sum(self.W * self.R * np.log(Z2)) / ww
            Z1Bar = - self.R * self.W * (1 - Z2) / ww

        wBar = -np.sum(Z1Bar,axis = 1) + np.sum(Z1Bar,axis = 0)
        self.apply_regularisation(cost,wBar,w)
        return cost, wBar

    def get_probabilities(self, w):
        """Returns a matrix containing the model probabilities for all players.
        
        Args:
            w (vector) : vector of optimised parameters
        Returns:
            prob (matrix) : match up probabilities
        """
        Z1 = w[:, None] - w[None, :]
        Z2 = 1. / (1 + np.exp(Z1))
        return Z2


'----------------------------------------------------------------------------------------------------------------------'

'Free Parameter Point Model Optimisers for MAP fitting'

'----------------------------------------------------------------------------------------------------------------------'


class FreeParameterPointGradient(GradientOptimiser):
    """
    Optimiser for solving parameters in a Free Parameter model.
    """

    def __init__(self, use_bias=False, *args, **kwargs):
        """
        Args:
            use_bias (bool) : whether to use a bias term 
        """
        self.use_bias = use_bias
        super(FreeParameterPointGradient,self).__init__(*args,**kwargs)

    def cost_gradients(self, w):
        """
        Evaluates the cost and gradients for a set of parameters.
        
        Args:
            w (vector) : Vector of current parameters
        Returns:
            cost (scalar) : Value of cost function at w
            wBar (vector) : Vector of gradients
        """
        bias = 0
        if self.use_bias:
            A, D = np.split(w[:-1],2)
            bias = w[-1]
        else:
            A, D = np.split(w, 2)

        Z1 = A[:, None] - D[None, :]
        Z2 = 1. / (1 + np.exp(Z1 + bias))  # Model service probabilities
        ww = np.sum(self.W) # total weight

        if self.error == "square": # Square error cost function
            cost =  np.sum(self.W * ((Z2 - self.R) ** 2))/ ww
            Z1Bar = - 2 * self.W * (Z2 - self.R) * Z2 * (1 - Z2) / ww
        else: # Log likelihood cost function
            cost =  - np.sum(self.W * (self.R * np.log(Z2) + (1-self.R) *np.log(1-Z2))) / ww
            Z1Bar =  1 * self.W * (self.R - Z2)/ww

        wBar = np.append(np.sum(Z1Bar, axis=1),-np.sum(Z1Bar, axis=0))
        if self.use_bias:
            wBar = np.append(wBar,np.sum(Z1Bar))

        # Regularisation
        self.apply_regularisation(cost,wBar,w)
        return cost, wBar

    def initialise_params(self):
        """Initialise parameters.

        Returns:
            w (vector) : Initialised parameters
        """
        return 0.1*self.init(self.n*2)

    def get_probabilities(self, w):
        """Returns a matrix containing the model probabilities for all players.

        Args:
            w(vector) : vector of optimised parameters
        Returns:
            prob(matrix) : match up probabilities
        """
        bias = 0
        if self.use_bias:
            A, D = np.split(w[:-1],2)
            bias = w[-1]
        else:
            A, D = np.split(w, 2)

        Z1 = A[:, None] - D[None, :]
        Z2 = 1. / (1 + np.exp(Z1 + bias))
        return Z2

    def pack_params(self,w):
        """In this class this function is just being used to add the bias term (not ideal could maybe be
        made cleaner)

        Args:
            w (vector) : Vector of current parameters
        Returns:
            w (scalar) : Vector of current parameters
        """
        if self.use_bias:
            return np.append(w,np.array([0.5]))
        else:
            return w

'----------------------------------------------------------------------------------------------------------------------'

'Surface Factor Model Optimisers for maximum Likelihood fitting'

'----------------------------------------------------------------------------------------------------------------------'


class SurfaceFactorModelGradient(GradientOptimiser):
    """"Optimiser for solving surface factor model."""

    def __init__(self, surf_reg=0, num_factors=1, hand_pick_init=False, *args, **kwargs):
        """
        Args:
            surf_reg (scalar) : Regularisation to apply to surface parameters only
            num_factors (int) : Number of
            hand_pick_init (bool) : Whether to use a hand picked initialisation (only applies for 2 or 3 factors)
        """
        self.hand_pick = hand_pick_init
        self.surf_reg = surf_reg
        super(SurfaceFactorModelGradient,self).__init__(*args, **kwargs)
        self.num_factors = num_factors

    def set_num_params(self):
        """Sets the number of model parameters.
        """
        self.num_params = self.num_factors*self.n + self.R.shape[0]*max(1,self.num_factors)

    def cost_gradients(self, w):
        """Gets the cost and gradients for set of parameters w.

        Args:
            w (vector) : Current player attacking and defensive parameters
        Returns:
           E (scalar) : Current error of log likilood cost function
           wBar (N vector) : Gradients with respect to w
        """
        num_factors = max(1,self.num_factors)
        no_surf = self.R.shape[0]
        surf_params = np.exp(w[:no_surf*num_factors]).reshape(no_surf,-1) # surface parameters
        player_params = w[no_surf*num_factors:].reshape(num_factors,-1)

        Z0 = np.dot(surf_params,player_params)
        Z1 = Z0.reshape(no_surf,-1,1) - Z0.reshape(no_surf ,1,-1)
        Z2 = 1. / (1 + np.exp(-Z1))  # Model probabilities
        ww = np.sum(self.W)  # total weigh

        E = - np.sum(self.W * self.R * np.log(Z2)) / ww
        Z1Bar = self.R * self.W * (1 - Z2) / ww

        Z0Bar = - np.sum(Z1Bar,axis = 2) + np.sum(Z1Bar,axis = 1)
        surf_paramsBar = np.dot(Z0Bar,player_params.T)
        player_paramsBar = np.dot(surf_params.T,Z0Bar)
        if self.num_factors == 0:
            surf_paramsBar = np.zeros_like(surf_paramsBar)
        wBar = np.append(surf_paramsBar.reshape(-1)*np.exp(w[:no_surf*num_factors]),player_paramsBar.reshape(-1))

        self.apply_regularisation(E,wBar,w)
        return E, wBar

    def get_probabilities(self,w):
        """Returns a matrix containing the model probabilities for all players.

        Args:
            w(vector) : vector of optimised parameters
        Returns:
            prob(matrix) : match up probabilities
        """
        num_factors = max(1, self.num_factors)
        no_surf  = self.R.shape[0]
        surf_params = np.exp(w[:no_surf*num_factors]).reshape(no_surf,-1) # surface parameters
        player_params = w[no_surf*num_factors:].reshape(num_factors,-1)
        Z0 = np.dot(surf_params,player_params)
        Z1 = Z0.reshape(no_surf,-1,1) - Z0.reshape(no_surf,1,-1)
        Z2 = 1. / (1 + np.exp( - Z1))
        return Z2

    def initialise_params(self):
        """Initialise parameters.

        Returns:
            w (vector) : Initialised parameters
        """
        num_factors = max(1,self.num_factors)
        surf_params = 0.1*self.init(self.R.shape[0]*num_factors)

        if self.hand_pick and self.num_factors == 2:
            surf_params = np.array([2,0.5,0.25,2.5,1.7,0.8])
            surf_params = np.log(surf_params)
        if self.hand_pick and self.num_factors == 3:
            surf_params = np.array([3,0.5,2,1.2,3.5,0.4,2.5,1.3,1])
            surf_params = np.log(surf_params)
        # print(surf_params.reshape(self.R.shape[0],-1))
        player_params = np.zeros(num_factors*self.n)
        return np.append(surf_params,player_params)

    def apply_surf_reg(self, cost, wBar, w):
        """Applies surface parameter specific regularisation if applicable.

        Args:
            w (vector) : vector of current player parameters
            cost (scalar) : cost
            wBar (vector) : parameter gradients
        """
        num_factors = max(1,self.num_factors)
        surf_params = w[:self.R.shape[0]*num_factors]
        cost -= (self.reg - self.surf_reg) * np.dot(surf_params,surf_params)
        wBar[:self.R.shape[0]*num_factors] -= (self.reg - self.surf_reg) * 2 * w


'----------------------------------------------------------------------------------------------------------------------'

'Optimisers for stochastic variational inference'

'----------------------------------------------------------------------------------------------------------------------'


class BradleyTerryVariationalInference(BaseOptimiser):
    """Class for fitting approximate gaussian posteriors in a bradley terry model."""

    def __init__(self, use_correlations=True, use_samples=False, fast_approximation=False, prior_var=1, display=False, tol=1e-9):
        """
        Args:
            use_correlations (bool) : Whether to fit full or diagonal covariance
            use_samples (bool) : Whether to use sampled or closed form approximation of likelihood terms
            fast_approximation (bool) : Whether to use faster but less accurate version of closed form likelihood
                                        approximation. Only applies if use_samples = False
            prior_var (scalar) : Variance for default spherical prior covariance
            display (bool) : Whether to display the optimisation details
            tol (float) : Convergence tolerance to be used in optimisation
        """
        self.tol = tol
        self.display = display
        self.use_correlations = use_correlations
        self.prior_var = prior_var
        self.use_samples = use_samples
        self.fast_approximation = fast_approximation
        no_samples = 100
        self.samples = stats.norm.ppf(np.linspace(0,1,no_samples,endpoint=False) + 0.5 /no_samples).reshape(-1,1,1)
        super(BradleyTerryVariationalInference,self).__init__()

    def cost_gradientsL(self, m, L):
        """Cost and gradients with respect to the cholesky factor (L) of the covariance matrix (V). For fitting a
        full covariance matrix.

        Args:
            m(vector) : Current posterior means
            L(matrix) : Cholesky factor of current posterior covariance
        Returns:
            j(scalar) : Cost estimate
            mBar(vector) : Gradients of m
            LBar(matrix) : Gradients of L
        """

        # Unpack
        L = np.tril(L)
        diag = np.diag_indices_from(L)

        # j1 cross-entropy
        j1 = -np.sum(L[diag])
        L[diag] = np.exp(L[diag])
        V = np.dot(L,L.T)

        # J3 prior
        mdiff = self.Prior_mean - m
        mm = np.dot(self.Prior_Prec,mdiff)
        j3 = 0.5*(np.sum(self.Prior_Prec*np.dot(L,L.T).T) + np.dot(mdiff,mm))
        mBar = -mm
        LBar = np.dot(self.Prior_Prec,L)

        # j2 approximation
        j2, mubar, sigma2Bar = self.likelihood_approximation(m,V)
        mBar += np.sum(mubar,axis=1)
        mBar -= np.sum(mubar, axis=0)

        # L gradients (TODO: can probably be implemented without the loop)
        for r in range(self.n):
            for c in range(self.n):
                if sigma2Bar[r,c] ==0:
                    pass
                else:
                    LBar[r, :] += sigma2Bar[r,c]*2*(L[r,:] - L[c,:])
                    LBar[c, :] += sigma2Bar[r,c]*2*(-L[r,:] + L[c,:])

        LBar[diag] = LBar[diag]*L[diag] - 1
        j = j1 + j2 + j3
        return j, mBar, np.tril(LBar)

    def cost_gradientsV(self, m, Vdiag):
        """Cost and gradients with respect to V diagonal (for fitting a diagonal covariance only).

        Args:
            m (vector) : Current posterior means
            Vdiag (vector) : Current diagonal of posterior covariance
        Returns:
            j (scalar) : Cost estimate
            mBar (vector) : Gradients of m
            VdiagBar (matrix)  : Gradients of Vdiag
        """

        # j1 cross-entropy
        j1 = -0.5*np.sum(Vdiag)
        Vdiag = np.exp(Vdiag)

        # j3 prior
        VdiagBar = 0.5*np.diag(self.Prior_Prec)
        mdiff = self.Prior_mean - m
        mm = np.dot(self.Prior_Prec,mdiff)
        j3 = 0.5*(np.sum(Vdiag * np.diag(self.Prior_Prec)) + np.dot(mdiff,mm))
        mBar = -mm

        # j2 likelihood
        j2, muBar, sigma2Bar = self.likelihood_approximation(m,Vdiag)

        # add gradients of j2
        mBar += np.sum(muBar,axis=1)
        mBar -= np.sum(muBar, axis=0)
        VdiagBar += np.sum(sigma2Bar,axis = 1) + np.sum(sigma2Bar, axis = 0)
        VdiagBar = VdiagBar*Vdiag - 0.5     # (the 0.5 is the  gradient of j1)

        j = j1 + j2 + j3
        return j, mBar, VdiagBar

    def likelihood_approximation(self,m,V):
        """Approximates part of the cost and gradients of the likelihood terms.

        Args:
            m (vector) : Current posterior means
            V (matrix) : Current diagonal of posterior covariance
        Returns:
            j2 (scalar) : Part of the cost estimate
            mubar (vector) : Gradients with respect to dummy variable mu
            sigma2Bar(matrix) : Gradients with respect to dummy variable sigma2
        """

        if self.use_correlations:
            diag = np.diag_indices_from(V)
            sigma2 = (V[diag][:,None] + V[diag][None,:]) - 2*V
            mu = m[:,None] - m[None,:]
        else:
            sigma2 = (V[:,None] + V[None,:])
            mu = m[:,None] - m[None,:]
        sigma = np.sqrt(sigma2)

        # Estimate based on Samples
        if self.use_samples:
            diag = np.diag_indices_from(sigma)
            temp = self.samples* sigma.reshape(1,self.n,self.n)
            sigma[diag] = 1
            j2 = - np.sum(self.W * (self.R*np.mean(np.log(1 / (1 + np.exp(-temp-mu))),axis = 0)))
            mubar = -self.W * self.R * np.mean(1. / (1 + np.exp(temp + mu)), axis=0)
            sigma2Bar = 0.5 * self.W * self.R * np.mean(self.samples* (1 / (1. + np.exp(-temp - mu))),axis=0) / sigma

        # Estimate based on closed form approximation
        else:
            a = 0.692310472668
            b = 0.35811358159
            c = 0.443113462726
            if self.fast_approximation:
                a = 0.5/(np.pi/4)
                b = 2*(np.sqrt(np.pi)/4)**2
                c = np.sqrt(np.pi)/4

            temp1 = b*sigma2 + 1
            temp2 = c**2*sigma2+0.5

            # cost estimate
            j21 = -a/np.sqrt(temp1) * np.exp(-b*mu**2/(2*temp1))
            j221 = -0.5*mu * special.erf(c*mu/np.sqrt(2*temp2))
            j222 = -c*sigma2 *np.exp(-c**2*mu**2/(2*temp2))/(np.sqrt(2*np.pi)*np.sqrt(temp2))
            j33 = 0.5*mu
            j2 = -j21 - j221 - j222 - j33
            j2 = np.sum(self.R*self.W*j2)

            if self.fast_approximation:
                # mu gradients
                mubar = -self.W * self.R * (0.5 + 0.5*special.erf(-c*mu/np.sqrt(2*temp2)))
                # sigma2 gradients
                Z1 = np.pi*sigma2/16 + 0.5
                Z2 = np.pi*(mu**2)/32
                sigma2Bar = self.W*self.R*np.exp(-Z2/Z1)/(8*np.sqrt(2)*np.sqrt(Z1))
            else:
                # mu gradients
                mubar = j21*-2*b*mu/(2*b*sigma2 + 2)
                mubar += -0.5* special.erf(c*mu/np.sqrt(1+2*c**2*sigma2))
                mubar += - 0.5*c*mu/np.sqrt(1+2*c**2*sigma2)*2*np.exp(-(c*mu/np.sqrt(1+2*c**2*sigma2))**2)/np.sqrt(np.pi)
                mubar += j222*-2*c**2*mu/(2*c**2*sigma2+1)
                mubar += 0.5
                mubar = -self.R*self.W*mubar
                # sigma2 gradients
                sigma2Bar = j21*2*b*b*mu**2/(2*temp1)**2 - 0.5*b*j21/(temp1)
                sigma2Bar += -mu *np.exp(-(c*mu/np.sqrt(2*temp2))**2)/np.sqrt(np.pi) * -c**2*c*mu/(2*temp2)**(3./2.)
                sigma2Bar += -c *np.exp(-c**2*mu**2/(2*temp2))/(np.sqrt(2*np.pi)*np.sqrt(temp2))
                sigma2Bar += j222*-c**2*mu**2/(2*temp2)**2 *-1 *2*c**2 + j222/(temp2) *-0.5 * c**2
                sigma2Bar = -self.R*self.W*sigma2Bar

        return j2, mubar, sigma2Bar

    def optimise(self, w_init= None, prior = None, max_iter =200, display=False):
        """Optimise the parameters of the approximate posterior using quasi-Newton method.

        Args:
            w_init (list) : Containing [m, V] to initialise the posterior covariance and mean with. Or none if default
                            should be used
            prior (list) : List, [m, V] containing the mean and covariance of the prior. If None a default
                           spherical prior will be constructed
            display (bool) : Whether to display the optimisation information
            max_iter (int) : Maximum number of iterations to be used by optimiser
        Returns:
            m (vector) : Optimised mean of the approximate posterior
            V (matrix) : Optimised covariance of approximate posterior
        """

        # If no prior is given then use default prior
        if prior is None:
            self.initialise_prior()
        else:
            self.Prior_mean = prior[0]
            self.Prior_Prec = np.linalg.inv(prior[1])

        # Initialise posterior to prior
        if w_init is None:
            w = self.initialise_w()
        else:
            mean_init = w_init[0]
            V_init = w_init[1]
            w = self.initialise_w(mean_init,V_init)

        # Optimise
        if display:
            disp = True
        else:
            disp = self.display
        res = minimize(self.cost_gradients, x0=w.squeeze(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'disp': disp , 'gtol': self.tol, 'ftol': self.tol})

        # Extract optimised parameters m and V from flattened form
        w = res.x
        m, LV = self.unpack_params(w)
        if self.use_correlations:
            diag = np.diag_indices_from(LV)
            LV[diag] = np.exp(LV[diag])
            V = np.dot(LV,LV.T)
        else:
            LV = np.exp(LV)
            V = np.diag(LV)
        return m, V

    def get_probabilities(self, w):
        """Returns a matrix containing the model probabilities for all players.

        Args:
            w (tuple) : Containing mean vector m (w[0]) and covariance matrix V (w[1])
        Returns:
            prob(matrix) : match up probabilities
        """
        m = w[0]
        V= w[1]
        diag = np.diag_indices_from(V)
        sigma2 = (V[diag][:,None] + V[diag][None,:]) - 2*V
        mu = m[:,None] - m[None,:]
        Z = np.sqrt(1 + sigma2 * np.pi / 8.)
        prob = 1./(1+np.exp(-mu/Z))
        return prob

    def cost_gradients(self,w):
        """Flattened version of the cost and gradients functions for scipy minimise function

        Args:
            w(vector) : flatten vector containing parameters of m and L/V
        Returns:
            wBar(vector) : gradients of w
        """
        m, LV = self.unpack_params(w)
        if self.use_correlations:
            j, mBar, LVbar = self.cost_gradientsL(m, LV)
        else: # Diagonal Covariance
            j, mBar, LVbar = self.cost_gradientsV(m, LV)
        wBar = self.pack_params(mBar, LVbar)
        return j, wBar

    def initialise_prior(self):
        """Default initialisation for the prior covariance.
        """
        self.Prior_Prec = np.diag(np.ones(self.num_params)/self.prior_var)
        self.Prior_mean = np.zeros(self.num_params)

    def initialise_w(self, m = None, V = None):
        """Initialise the posterior mean and covariance. If none is given then they are initialised to the prior.

        Args:
            m (vector) : Mean to use as initialisation
            V (matrix) : Covariance to use as initialisation
        Returns:
            w (vector) : Packed parameters of initialised m and V in single vector ready for optimising
        """
        if m is None:
            m = self.Prior_mean
        diag = np.diag_indices_from(self.Prior_Prec)
        if V is None:
            Vdiag = 1. / self.Prior_Prec[diag]
            V = np.diag(Vdiag)
        if self.use_correlations:
            L = np.linalg.cholesky(V)
            L[diag] = np.log(L[diag])
            return self.pack_params(m,L)
        else:
            Vdiag = np.log(V[diag])
            return self.pack_params(m,Vdiag)

    def unpack_params(self,w):
        """Unpack the parameters from 1 dimensional form.

        Args:
            w (vector) : Parameters in 1 dimensional form
        Returns:
            m (vector) : Vector of means
            LV (vector/matrix) : Vector of covariance diagonal terms or Cholesky factor L depending on whether a full or
                                 diagonal covariance is being used
        """
        m = w[:self.num_params]
        if self.use_correlations:
            # extract flattened parameters and build covariance matrix
            tril_ind = np.tril_indices(self.num_params)
            LV = np.zeros((self.num_params, self.num_params))
            LV[tril_ind] = w[self.num_params:]
        else: # Diagonal Covariance
            LV = w[self.num_params:]
        return m, LV

    def pack_params(self,m,LV):
        """Pack parameters into 1 dimensional form.

        Args:
            m (vector) : Vector of means
            LV (vector/matrix) : Vector of covariance diagonal terms or Cholesky factor L depending on whether a full or
                                 diagonal covariance is being used
        Returns:
            w (vector) : Parameters in 1 dimensional form
        """
        if self.use_correlations:
            tril_ind = np.tril_indices(self.num_params)
            LV = LV[tril_ind]
            np.append(m, LV)
        else:
            np.append(m, LV)
        return np.append(m,LV)


class FreeParameterPointVariationalInference(BradleyTerryVariationalInference):
    """Optimiser for fitting approximate gaussian posteriors in a free parameter model."""

    def set_num_params(self):
        """Sets the numbers of model parameters.
        """
        self.num_params = self.n*2

    def cost_gradientsL(self, m, L):
        """Cost and gradients with respect to the cholesky factor (L) of the covariance matrix (V). For fitting a
        full covariance matrix.

        Args:
            m(vector) : Current posterior means
            L(matrix) : Cholesky factor of current posterior covariance
        Returns:
            j(scalar) : Cost estimate
            mBar(vector) : Gradients of m
            LBar(matrix) : Gradients of L
        """

        # Unpack
        L = np.tril(L)
        diag = np.diag_indices_from(L)

        # j1 cross-entropy
        j1 = -np.sum(L[diag])
        L[diag] = np.exp(L[diag])
        V = np.dot(L,L.T)

        # J3 prior
        mdiff = self.Prior_mean - m
        mm = np.dot(self.Prior_Prec,mdiff)
        j3 = 0.5*(np.sum(self.Prior_Prec*np.dot(L,L.T).T) + np.dot(mdiff,mm))
        mBar = -mm
        LBar = np.dot(self.Prior_Prec,L)

        # j2 likelihood
        j2, mubar, sigma2Bar = self.likelihood_approximation(m,V)    

        mBar[:self.n] += np.sum(mubar,axis=1)
        mBar[self.n:] -= np.sum(mubar, axis=0)
        # ********************************************************************************
        # L gradients
        for r in range(self.n):
            for c in range(self.n):
                if sigma2Bar[r,c] ==0:
                    pass
                else:
                    LBar[r, :] += sigma2Bar[r,c]*2*(L[r,:] - L[c+self.n,:])
                    LBar[c+self.n, :] += sigma2Bar[r,c]*2*(-L[r,:] + L[c+self.n,:])
        # ********************************************************************************
        LBar[diag] = LBar[diag]*L[diag] - 1
        j = j1 + j2 + j3

        L[diag] = np.log(L[diag])

        return j, mBar, np.tril(LBar)

    def cost_gradientsV(self, m, Vdiag):
        """Cost and gradients with respect to V diagonal (for fitting a diagonal covariance only).

         Args:
             m (vector) : Current posterior means
             Vdiag (vector) : Current diagonal of posterior covariance
         Returns:
             j (scalar) : Cost estimate
             mBar (vector) : Gradients of m
             VdiagBar (matrix)  : Gradients of Vdiag
         """

        # j1 cross-entropy
        j1 = -0.5*np.sum(Vdiag)
        Vdiag = np.exp(Vdiag)

        # j3 prior
        VdiagBar = 0.5*np.diag(self.Prior_Prec)
        mdiff = self.Prior_mean - m
        mm = np.dot(self.Prior_Prec,mdiff)
        j3 = 0.5*(np.sum(Vdiag * np.diag(self.Prior_Prec)) + np.dot(mdiff,mm))
        mBar = -mm

        # j2 likelihood
        j2, mubar, sigma2Bar = self.likelihood_approximation(m,Vdiag)

        # Add gradients of j3
        mBar[:self.n] += np.sum(mubar,axis=1)
        mBar[self.n:] -= np.sum(mubar, axis=0)
        VdiagBar += np.append(np.sum(sigma2Bar,axis = 1),np.sum(sigma2Bar, axis = 0))
        VdiagBar = VdiagBar*Vdiag - 0.5
        j = j1 + j2 + j3

        return j, mBar, VdiagBar

    def likelihood_approximation(self, m, V):
        """Approximates part of the cost and gradients of the likelihood terms.

        Args:
            m (vector) : Current posterior means
            V (matrix) : Current diagonal of posterior covariance
        Returns:
            j2 (scalar) : Part of the cost estimate
            mubar (vector) : Gradients with respect to dummy variable mu
            sigma2Bar(matrix) : Gradients with respect to dummy variable sigma2
        """
        if self.use_correlations:
            diag = np.diag_indices_from(V)
            sigma2 = (V[diag][:self.n][:,None] + V[diag][self.n:][None,:]) - V[:self.n,self.n:] - V[self.n:,:self.n].T
            mu = m[:self.n][:,None] - m[self.n:][None,:] + 0.5
        else:
            sigma2 = V[:self.n][:,None] + V[self.n:][None,:]
            mu = m[:self.n][:,None] - m[self.n:][None,:]+0.5
        sigma = np.sqrt(sigma2)

        # Estimate based on Samples
        if self.use_samples:
            temp = self.samples* sigma.reshape(1,self.n,self.n)
            j2 = - np.sum(self.W * (self.R*np.mean(np.log(1 / (1 + np.exp(-temp-mu))),axis = 0) + (1-self.R)*np.mean(np.log(1 / (1 + np.exp(temp+mu))),axis = 0)))
            mubar = -self.W * (self.R - np.mean(1. / (1 + np.exp(-temp - mu)), axis=0))
            sigma2Bar = 0.5 * self.W * np.mean(self.samples * (1. / (1 + np.exp(-temp - mu))),axis=0) / sigma

        # Estimate based on closed form approximation
        else:
            a = 0.692310472668
            b = 0.35811358159
            c = 0.443113462726
            if self.fast_approximation:
                a = 0.5/(np.pi/4)
                b = 2*(np.sqrt(np.pi)/4)**2
                c = np.sqrt(np.pi)/4

            temp1 = b*sigma2 + 1
            temp2 = c**2*sigma2+0.5

            # cost estimate
            j21 = -a/np.sqrt(temp1) * np.exp(-b*mu**2/(2*temp1))
            j221 = -0.5*mu * special.erf(c*mu/np.sqrt(2*temp2))
            j222 = -c*sigma2 *np.exp(-c**2*mu**2/(2*temp2))/(np.sqrt(2*np.pi)*np.sqrt(temp2))
            j33 = 0.5*mu
            j2t = j21 + j221 + j222 + j33
            j2 = - np.sum(self.W*self.R*j2t)
            j221 = 0.5*mu * special.erf(c*-mu/np.sqrt(2*temp2))
            j33 = -0.5*mu
            j2t = j21 + j221 + j222 + j33
            j2 += - np.sum(self.W*(1-self.R)*j2t)

            if self.fast_approximation:
                # mu gradients
                mubar = -self.W *(self.R + -0.5 + 0.5*special.erf(-c*mu/np.sqrt(2*temp2)))
                # sigma2 gradients
                Z1 = np.pi*sigma2/16 + 0.5
                Z2 = np.pi*(mu**2)/32
                sigma2Bar = self.W*self.R*np.exp(-Z2/Z1)/(8*np.sqrt(2)*np.sqrt(Z1))
            else:
                # mu gradients
                mubar = j21*-2*b*mu/(2*b*sigma2 + 2)
                mubar += -0.5* special.erf(c*mu/np.sqrt(1+2*c**2*sigma2))
                mubar += - 0.5*c*mu/np.sqrt(1+2*c**2*sigma2)*2*np.exp(-(c*mu/np.sqrt(1+2*c**2*sigma2))**2)/np.sqrt(np.pi)
                mubar += j222*-2*c**2*mu/(2*c**2*sigma2+1)
                mubar += 0.5
                mubar = -self.W*(self.R + mubar -1)
                # sigma2 gradients
                sigma2Bar = j21*2*b*b*mu**2/(2*temp1)**2 - 0.5*b*j21/(temp1)
                sigma2Bar += -mu *np.exp(-(c*mu/np.sqrt(2*temp2))**2)/np.sqrt(np.pi) * -c**2*c*mu/(2*temp2)**(3./2.)
                sigma2Bar += -c *np.exp(-c**2*mu**2/(2*temp2))/(np.sqrt(2*np.pi)*np.sqrt(temp2))
                sigma2Bar += j222*-c**2*mu**2/(2*temp2)**2 *-1 *2*c**2 + j222/(temp2) *-0.5 * c**2
                sigma2Bar = -self.W*sigma2Bar
        return j2, mubar, sigma2Bar

    def get_probabilities(self, w):
        """Returns a matrix containing the model probabilities for all players.

        Args:
            w (tuple) : Containing mean vector m (w[0]) and covariance matrix V (w[1])
        Returns:
            prob(matrix) : match up probabilities
        """
        m= w[0]
        V= w[1]
        n = int(V.shape[0]/2)
        diag = np.diag_indices_from(V)
        sigma2 = (V[diag][:n][:,None] + V[diag][n:][None,:]) - V[:n,n:] - V[n:,:n].T
        mu = m[:n][:,None] - m[n:][None,:] + 0.5
        Z = np.sqrt(1 + sigma2 * np.pi / 8)
        prob = 1/(1+np.exp(-mu/Z))
        return prob


class JointOptTimeSeriesBradleyTerryVariationalInference(BradleyTerryVariationalInference):
    """Optimiser for Bradley-Terry time series model based on joint optimisation approach."""

    def __init__(self, steps=4., drift=0.9, *args, **kwargs):
        """
        Args:
            steps (integer) : Number of blocks the history is broken into (should match that of model)
            drift (scalar [0,1]) : Gaussian drift parameter
        """
        self.steps = steps
        self.drift = drift
        super(JointOptTimeSeriesBradleyTerryVariationalInference,self).__init__(*args,**kwargs)

    def initialise_prior(self):
        """Constructs the prior precision matrix with appropriate correlations between the skills
        of players at different points in time.
        """
        drift = self.drift**(3./self.steps)
        n = int(self.n/self.steps)
        main_diagonal = np.ones(n*self.steps)
        main_diagonal[n:-n] = 1+drift**2
        off_diagonal = np.ones(n*(self.steps-1))*-drift
        Prior_Prec = np.diag(off_diagonal, k=-n)
        Prior_Prec +=Prior_Prec.T +np.diag(main_diagonal)
        Prior_Prec /= (self.prior_var*(1-drift**2))
        self.Prior_Prec = Prior_Prec
        self.Prior_mean = np.zeros(self.n)

    def get_probabilities(self, w):
        """Returns a matrix containing the model probabilities for all players.

        Args:
            w (tuple) : Containing mean vector m (w[0]) and covariance matrix V (w[1])
        Returns:
            prob(matrix) : match up probabilities
        """
        m = w[0]
        V = w[1]
        n = int(self.n / self.steps)
        diag = np.diag_indices_from(V)
        sigma2 = (V[diag][:,None] + V[diag][None,:]) - 2*V
        mu = m[:,None] - m[None,:]
        Z = np.sqrt(1 + sigma2 * np.pi / 8)
        prob = 1./(1+np.exp(-mu/Z))
        return prob[-n:,-n:]


class JointOptTimeSeriesRefinedBradleyTerryVariationalInference(JointOptTimeSeriesBradleyTerryVariationalInference):
    """Optimiser for Bradley-Terry time series model joint optimisation approach further refined with unequal
    steps in time."""

    def initialise_prior(self):
        """Constructs the prior precision matrix with appropriate correlations between the skills
        of players at different points in time.
        """
        self.steps = 4
        alpha1 = self.drift**(365./365.)  # 12 month gap
        alpha2 = self.drift**(243./365.)  # 8 month gap
        alpha3 = self.drift**(122./365.)  # 4 month gap
        n = int(self.n/4)

        main_diagonal = np.ones(n*self.steps)
        main_diagonal[:n] = 1./(self.prior_var*(1.-alpha1**2))
        main_diagonal[n:2*n] = 1./(self.prior_var*(1.-alpha1**2)) + (alpha2**2)/(self.prior_var*(1.-alpha2**2))
        main_diagonal[2*n:3*n] = 1./(self.prior_var*(1.-alpha2**2)) + (alpha3**2)/(self.prior_var*(1.-alpha3**2))
        main_diagonal[-n:] = 1./(self.prior_var*(1.-alpha3**2))

        off_diagonal = np.ones(n*(self.steps-1))
        off_diagonal[:n] = -alpha1/(self.prior_var*(1.-alpha1**2))
        off_diagonal[n:2*n] = -alpha2/(self.prior_var*(1.-alpha2**2))
        off_diagonal[-n:] = -alpha3/(self.prior_var*(1.-alpha3**2))

        Prior_Prec = np.diag(off_diagonal, k=-n)
        Prior_Prec += Prior_Prec.T +np.diag(main_diagonal)

        self.Prior_Prec = Prior_Prec
        self.Prior_mean = np.zeros(self.n)

'----------------------------------------------------------------------------------------------------------------------'

'Optimisers for models with multiple factors'

'----------------------------------------------------------------------------------------------------------------------'


class FreeParameterPoint_extrafactors(FreeParameterPointGradient):
    """An optimiser for a version of the free parameter point model with additional parameters with non-linear
    interactions."""

    def set_num_params(self):
        """Sets the number of model parameters.
        """
        self.num_params = self.n * 3 + self.use_bias

    def cost_gradients(self, w):
        """Evaluates the cost and gradients for a set of parameters.

        Args:
            w (vector) : Vector of current parameters
        Returns:
            cost (scalar) : Value of cost function at w
            wBar (vector) : Vector of gradients
        """
        bias = 0
        if self.use_bias:
            A, D, S1 = np.split(w[:-1],2)
            bias = w[-1]
        else:
            A, D, S1 = np.split(w, 3)

        Z1 = A[:, None] - D[None, :] + S1[:,None] - S1[:None]
        Z2 = 1. / (1 + np.exp(Z1 + bias))  # Model service probabilities
        ww = np.sum(self.W) # total weight

        if self.error == "square": # Square error cost function
            E =  np.sum(self.W * ((Z2 - self.R) ** 2))/ ww
            Z1Bar = - 2 * self.W * (Z2 - self.R) * Z2 * (1 - Z2) / ww
        elif self.error == "abs": # Absolute error cost function
            E = np.sum(self.W * np.abs(Z2 - self.R)) / ww
            Z1Bar = - self.W * Z2 * (1 - Z2) * (2 * ((Z2 - self.R) > 0) - 1)/ww
        else: # Log likelihood cost function
            E =  - np.sum(self.W * (self.R * np.log(Z2) + (1-self.R) *np.log(1-Z2))) / ww
            Z1Bar =  1 * self.W * (self.R  -  Z2)/ww

        wBar = np.append(np.sum(Z1Bar, axis=1),-np.sum(Z1Bar, axis=0))
        wBar = np.append(wBar,np.sum(Z1Bar, axis=1)-np.sum(Z1Bar, axis=0))
        if self.use_bias:
            wBar = np.append(wBar,np.sum(Z1Bar))

        # Regularisation
        if self.reg > 0:
            if self.reg_type == 'L1':
                E += self.reg * np.sum(np.abs(w))
                wBar[:] += self.reg*(2 * (w > 0) - 1)
            else: # L2
                E += self.reg*np.dot(w,w)
                wBar += self.reg*2*w
        return E, wBar

    def get_probabilities(self, w):
        """Returns a matrix containing the model probabilities for all players.

        Args:
            w (vector) : Vector of optimised parameters
        Returns:
            prob (matrix) : Match up probabilities
        """
        bias = 0
        if self.use_bias:
            A, D, S1 = np.split(w[:-1],3)
            bias = w[-1]
        else:
            A, D, S1 = np.split(w, 3)

        Z1 = A[:, None] - D[None, :] + S1[:, None] - S1[None, :]
        Z2 = 1. / (1 + np.exp(Z1 + bias))

        return Z2
