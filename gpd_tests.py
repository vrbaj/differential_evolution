from scipy.stats import genpareto
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

c = 0.1
r = genpareto.rvs(-1, loc=0, scale=1, size=1000)

ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()