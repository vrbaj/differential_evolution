from scipy.stats import genpareto
import matplotlib.pyplot as plt
import pickle

fig, ax = plt.subplots(1, 1)

c = 0.1
r = genpareto.rvs(-1, loc=0, scale=1, size=1000)

dbfile = open('gpd_sample', 'ab')
pickle.dump(r, dbfile)
dbfile.close()

ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()

print(genpareto.fit(r))
