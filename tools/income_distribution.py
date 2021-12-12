import numpy as np
import matplotlib.pyplot as plt

# incomes = np.random.zipf(1.5,size=1000)
# vals, bins = np.histogram(incomes,bins=np.logspace(np.log10(min(incomes)),np.log10(max(incomes)), 40),density=True)

incomes = np.random.lognormal(mean=np.log(30000),sigma=1.6,size=1000)
incomes = [i if i>1000 else 1000 for i in incomes]
incomes = [i if i<200000 else 200000 for i in incomes]
vals, bins = np.histogram(incomes,bins=np.logspace(np.log10(min(incomes)),np.log10(max(incomes)), 40),density=True)

#print(vals2)

#plt.plot(bins[1:],vals,'o-')
plt.plot(bins[1:],vals,'o-')
plt.xscale('log')
plt.yscale('log')
plt.show()