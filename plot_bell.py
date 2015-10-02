import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

'''
#code for plotting bell curve
mean = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(-6,8,100)
plt.plot(x,mlab.normpdf(x,0,sigma),label='Mean=0, Var=1')
plt.plot(x,mlab.normpdf(x,2,sigma),label='Mean=2, Var=1')
plt.axis([-6, 8, 0, .50])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Sample to show overlap in distibutions')
plt.show()
'''

'''
#for plotting misclassification error for various means
y=[.499,.12,.013,.0004]
x=[0,1,2,3]
plt.plot(x,y,label='Misclassification Error')
plt.axis([0, 3.5, 0, .55])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
#plt.title('Misclassification Error for Vario')
plt.show()
'''
y=[.53, .42125 ,.27,.22,.13625,.08,.04875,.0175,.00875,.01,0]
x=[0  , .25 ,.5,.75,1,1.25,1.5,1.75,2,2.25,2.5]
plt.plot(x,y)#,label='PR Curve for Best Classifier')
plt.axis([0, 2.5, 0, .6])
plt.xlabel('Mean Of Class 2')
plt.ylabel('Misclassification Error')
plt.legend()
#plt.title('PR Curv')
plt.show()