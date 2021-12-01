import matplotlib.pyplot as plt
import numpy as np

a,b = 1.25,3
func = lambda x: 1/(1+np.exp(-(x/a -b)))

x = np.linspace(0,10,50)
npf = np.vectorize(func)
y = npf(x)

plt.plot(x,y)
plt.show()