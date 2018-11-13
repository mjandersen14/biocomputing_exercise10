"""
Created on Mon Nov 12 18:01:33 2018

@author: marissaandersen
"""

#QUESTION 1 
#this is the script for the walk through. Modify to answer the questions. 
import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

N=25
x=numpy.random.uniform(0,20,size=N)
y=3*x+5

#this generates our data set so need to make it the data.txt file 
y=y+numpy.random.randn(N)*3
df=pandas.DataFrame({'x':x,'y':y})

ggplot(df,aes(x='x',y='y'))+geom_point()+theme_classic()

def nllike(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

initialGuess=numpy.array([1,1,1])
fit=minimize(nllike,initialGuess,method="Nelder-Mead",options={'disp': True},args=df)

print(fit.x)

#how do you incorperate the two models together? 
#I don't get the liklihood function (ok i kinda get it but i don't get what 
#the diffrerent variables represent) 
#how do we incorperate the 


#################### QUESTION SCRIPT #########################
import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

data=pandas.read_csv('data.txt', sep=",", header=0)
print (data)
 
ggplot(data,aes(x='x',y='y'))+geom_point()+theme_classic()

def nllike(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x 
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum() 
    return nll

initialGuess=numpy.array([1,1,1])
fit=minimize(nllike,initialGuess,method="Nelder-Mead",options={'disp': True},args=data)

    
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

teststat=2*(fit.fun-fit2.fun)
df=len(fit2.x)-len(fit.x)
1-stats.chi2.cdf(teststat,df)

#linear is more apropriate because adding in that extra variable dosen't do anything 













#QUESTION 2
##walkthrough script 
import pandas
import scipy #WHAT? 
import scipy.integrate as spint #WHY 
from plotnine import *
#costom function made defining state variable, time, and two parameters
def ddSim(y,t0,r,K,Ant,Atn): #Ant and Atn were defined in the paper as well as K and r.
    Nn=y[0] #gives us the column we are using for Nn (state variables) aka the observations? 
    Nt=y[1]
    dNndt=r*(1-(Nn+Ant)/K)*Nn
    dNtdt=r*(1-(Nt+Atn)/K)*Nt
    return [dNtdt,dNndt]
params=(0.5,10,0.5,2)
N0=[0.01,0.01]
times=range(0,100)
modelSim=spint.odeint(func=ddSim,y0=N0,t=times,args=params)
modelOutput=pandas.DataFrame({"t":times,"N":modelSim[:,0]})
ggplot(modelOutput,aes(x="t",y="N"))+geom_line()+theme_classic()



################ QUESTION SCRIPT ####################
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

params2=(0.5,100,2,2,2,2)
modelSim=spint.odeint(func=ddSim,y0=N0,t=times,args=params2)
modelOutput=pandas.DataFrame({"t":times,"N1":modelSim[:,0],"N2":modelSim[:,1]})
b=ggplot(modelOutput,aes(x="t",y="N1"))+geom_line(color="red")+theme_classic()
b+geom_line(modelOutput,aes(x="t", y="N2"),color="blue")

params3=(0.5,100,1,.6,2,.3)
modelSim=spint.odeint(func=ddSim,y0=N0,t=times,args=params3)
modelOutput=pandas.DataFrame({"t":times,"N1":modelSim[:,0],"N2":modelSim[:,1]})
c=ggplot(modelOutput,aes(x="t",y="N1"))+geom_line(color="red")+theme_classic()
c+geom_line(modelOutput,aes(x="t", y="N2"),color="blue")












































