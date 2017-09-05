"""

Investigates approximation for ln(1/(1+exp(-x)). This is used to obtain closed form derivatives and cost estimates
of likelihood terms in KL divergence for variational inference.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.optimize import minimize
from scipy import stats

xx =10
x = np.linspace(-xx,xx,10000)
y_exact = np.log(1./(1.+np.exp(-x)))
a = 0.692310472668
b = 0.35811358159
c = 0.443113462726
# a = 2/np.pi
# b = np.pi/8
# c = np.sqrt(np.pi)/4
y_approx = -a*np.exp(-0.5*b*x**2) - 0.5*x*special.erf(c*x) + 0.5*x


# fit parameters of new approximation********************************************************
# n =100000
# a = np.sqrt(np.pi)/4.
# x = 10*stats.norm.ppf(np.linspace(0,1,n,endpoint=False) + 0.5/n)
# y_exact = np.log(1./(1.+np.exp(-x)))
# def res_f(xx,w):
#     c1 = w[0]
#     c2 = w[1]
#     c3 = w[2]
#     c4 = w[3]
#     f = -(c1 + 0.5)*np.exp(-c2 *a**2*xx**2) - c4*0.5*xx*special.erf(c3*a*xx) + 0.5*xx
#     res = np.abs(f-np.log(1./(1.+np.exp(-xx))))
#     return f, res
#
# def cost_grad(w):
#     c1 = w[0]
#     c2 = w[1]
#     c3 = w[2]
#     c4 = w[3]
#     f = -(c1 + 0.5)*np.exp(-c2 *a**2*x**2) - c4*0.5*x*special.erf(c3*a*x) + 0.5*x
#     cost = np.sum((f-y_exact)**2)
#     zbar = 2*(f - y_exact)
#     wbar = np.zeros_like(w)
#
#     wbar[0] = np.sum(-zbar*np.exp(-c2 *a**2*x**2))
#     wbar[1] = np.sum( zbar*(a**2*x**2 *c1*np.exp(-c2 *a**2*x**2) + a**2*x**2 *0.5*np.exp(-c2 *a**2*x**2)))
#     wbar[2] = np.sum(-zbar*a*x*np.exp(-(c3*a*x)**2)/np.sqrt(np.pi))
#     wbar[3] = np.sum(-zbar*0.5*x*special.erf(c3*a*x))
#     return cost, wbar
# w = np.ones(4)
# w[0] = 0
# res = minimize(cost_grad, x0=w.squeeze(), method='L-BFGS-B', jac=True,
#                        options={'maxiter': 50, 'disp': 1 , 'gtol': 1e-12, 'ftol': 1e-12})
# w =res.x
# y_approx, res = res_f(x,w)
# print('Maximum Error {0}'.format(np.max(res)))
# print('Location of max error {0}'.format(x[np.argmax(res)]))
# print('a = {0}'.format(w[0]+0.5))
# print('b = {0}'.format(2*w[1]*a**2))
# print('c = {0}'.format(w[2]*a))
# print('d = {0}'.format(w[3]*0.5))
# print('--'*20)
# ****************************************************************************************



# Plotting
fig1 = plt.figure(figsize=(8,6),facecolor='white')
ax1 = fig1.add_subplot(111)

# plot exact
ax1.plot(x,(y_exact),linewidth = 1,  label = 'Exact function')

# plot approximation
ax1.plot(x,(y_approx), '--', linewidth = 1, label = 'Approximation')


fsize = 18
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax1.legend(loc = 2, fontsize =fsize)
ax1.set_xlabel('x', fontsize = fsize)
ax1.set_ylabel('f(x)', fontsize = fsize)
ax1.set_xlim([-xx,xx])

plt.show()