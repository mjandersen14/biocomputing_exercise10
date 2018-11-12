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
from scipy.optimize import minimize #what does this mean? 
from scipy.stats import norm #lol what 
from plotnine import *

data=pandas.read_csv('data.txt', sep=",", header=0)
print (data)
#is it going to hurt me later on that I'm using a dataframe and not a numpy array? 
ggplot(data,aes(x='x',y='y'))+geom_point()+theme_classic()

def nllike(p,xaxis):
#changed obs to xaxis becuase i feel like it makes more sense but does it? what is it? 
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x #hypothesis 
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum() #null model 
    #can i just add in the quadratic here? 
    return nll
# is this just the same for any linear model? what would change between models? 
# I don't get what any of the lines mean or represent here.. 
    
y=m*x+b


























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

def ddSim(y,t0,r1,r2,N1a11,N2a12,N2a22,N1a21): 
    N1=y[0] 
    N2=y[1]
    dN1dt=r1*(1-N1a11-N2a12)*N1
    dN2dt=r2*(1-N2a22-N1a21)*N2
    return [dN1dt,dN2dt]
params=(0.5,100,#how do I know the other parameters? do I just make them up till they look good?)
N0=[0.01,0.01]
times=range(0,100)
modelSim=spint.odent(func=ddSim,y0=N0,t=times,args=params)
modelOutput=pandas.dataframe
#I also don't get the last part of the queston? is that saying try out a few different alphas 
#and see what one works the best? 
















































