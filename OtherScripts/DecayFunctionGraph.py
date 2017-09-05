"""

Plots an example recency weighting function graph.

"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,365*3,1000)
y = 0.5**(x/180)
y2 = 0.5**(x/240)
y3 = 0.5**(x/365)

# Plotting
fig1 = plt.figure(figsize=(8,6),facecolor='white')
ax1 = fig1.add_subplot(111)
ax1.plot(x,y,label= '180 Day half life')
ax1.plot(x,y2,label= '240 Day half life')
ax1.plot(x,y3,label= '365 Day half life')

# Formatting
fsize = 18
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax1.legend(loc = 2, fontsize =fsize)
ax1.set_xlabel('x', fontsize = fsize)
ax1.set_ylabel('f(x)', fontsize = fsize)

plt.show()