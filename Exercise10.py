"""
Created on Mon Nov 12 18:01:33 2018

@author: marissaandersen
"""
#Question 1
import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

data=pandas.read_csv('data.txt', sep=",", header=0)
print (data)
 
ggplot(data,aes(x='x',y='y'))+geom_point()+theme_classic()

def line(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x 
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum() 
    return nll

initialGuess=numpy.array([1,1,1])
fit=minimize(line,initialGuess,method="Nelder-Mead",options={'disp': True},args=data)

    
def quad(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    expected=B0+B1*obs.x+B2*((obs.x)**2)  
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

initialGuess=numpy.array([1,1,1,1])
fit2=minimize(quad,initialGuess,method="Nelder-Mead",options={'disp': True},args=data)

from scipy import stats 

print (fit,fit2)

teststat=2*(fit.fun-fit2.fun)
df=len(fit2.x)-len(fit.x)
pval=1-stats.chi2.cdf(teststat,df)

#linear is more apropriate because adding in that extra variable dosen't make the model more accurate 
#p-value is too close to 1 

#Question 2 
import pandas
import scipy
import scipy.integrate as spint
from plotnine import *

def ddSim(y,t0,r1,r2,a11,a12,a22,a21): 
    N1=y[0] 
    N2=y[1]
    dN1dt=r1*(1-N1*a11-N2*a12)*N1
    dN2dt=r2*(1-N2*a22-N1*a21)*N2
    return [dN1dt,dN2dt]

N0=[0.01,0.01]
times=range(0,100)

params=(0.5,100,2,0.5,2,0.5)
modelSim=spint.odeint(func=ddSim,y0=N0,t=times,args=params)
modelOutput=pandas.DataFrame({"t":times,"N1":modelSim[:,0],"N2":modelSim[:,1]})
a=ggplot(modelOutput,aes(x="t",y="N1"))+geom_line(color="red")+theme_classic()
a+geom_line(modelOutput,aes(x="t", y="N2"),color="blue")


params2=(0.5,100,1,2,1,2)
modelSim=spint.odeint(func=ddSim,y0=N0,t=times,args=params2)
modelOutput=pandas.DataFrame({"t":times,"N1":modelSim[:,0],"N2":modelSim[:,1]})
b=ggplot(modelOutput,aes(x="t",y="N1"))+geom_line(color="red")+theme_classic()
b+geom_line(modelOutput,aes(x="t", y="N2"),color="blue")

params3=(0.5,100,1,.6,2,.3)
modelSim=spint.odeint(func=ddSim,y0=N0,t=times,args=params3)
modelOutput=pandas.DataFrame({"t":times,"N1":modelSim[:,0],"N2":modelSim[:,1]})
c=ggplot(modelOutput,aes(x="t",y="N1"))+geom_line(color="red")+theme_classic()
c+geom_line(modelOutput,aes(x="t", y="N2"),color="blue")
