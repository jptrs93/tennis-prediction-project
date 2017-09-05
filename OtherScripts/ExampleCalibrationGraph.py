"""

Plots an example of a calibration graph.

"""

import matplotlib.pyplot as plt
import numpy as np

fig1 = plt.figure(figsize=(12,8),facecolor='white')
ax1 = fig1.add_subplot(111)

x = np.linspace(0.5,1,500)
y_calibrated = x
y_under = y_calibrated + 2*(1-x)*(x-0.5)
y_over = y_calibrated +- 2*(1-x)*(x-0.5)

# Plotting
ax1.plot(x,y_calibrated,'--',color = 'k',linewidth = 2, label = 'Perfectly calibrated predictions')
ax1.plot(x,y_under,linewidth = 2, label = 'Example under confident predictions')
ax1.plot(x,y_over,linewidth = 2, label = 'Example over confident predictions')

# Formatting
fsize = 18
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax1.legend(loc = 'best', fontsize =fsize)
ax1.set_xlabel('Probability of prediction', fontsize = fsize)
ax1.set_ylabel('Percentage correct', fontsize = fsize)
ax1.set_xlim([0.5,1])
ax1.set_ylim([0.5,1])
plt.show()