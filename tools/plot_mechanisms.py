import matplotlib.pyplot as plt
import numpy as np

# income
x = np.linspace(0,100000,100)
y = x**2
plt.plot(x,y)
plt.title('Score Income')
plt.xlabel('income')
plt.ylabel('score')
plt.savefig('score_income.png')
plt.show()

# score income match
x = np.linspace(10000,40000,100)
y = np.abs(30000-x)
plt.plot(x,y)
plt.title('Score Income Match')
plt.xlabel('location median income')
plt.ylabel('score')
plt.savefig('score_income_match.png')
plt.show()

# Houisng cost
x = np.linspace(0,1,100)
y = -0.5*x
plt.plot(x,y)
plt.title('Score Housing Cost')
plt.xlabel('normalized housing cost')
plt.ylabel('score')
plt.savefig('score_housing_cost.png')
plt.show()

# Score low income
x = np.linspace(1000,50000,100)
y = 10**(-5)*(1/(x+1))
plt.plot(x,y)
plt.title('Score Low Income Forced Move')
plt.xlabel('income')
plt.ylabel('score')
plt.savefig('score_low_income.png')
plt.show()

# mapping
x = np.linspace(0,10,100)
a = 1.25
b = 6
y = 1/(1+np.exp(-(x/a -b)))
plt.plot(x,y)
plt.title('Score Mapping')
plt.xlabel('score')
plt.ylabel('Probability of Moving')
plt.savefig('score_mapping.png')
plt.show()