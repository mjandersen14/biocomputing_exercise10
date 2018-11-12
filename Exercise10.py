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


################ REAL SCRIPT #########################
import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

data=numpy.loadtxt('data.txt', delimiter=",", dtype="string")
print (data)
#i dont think i can do this as a string so how do I load it as a float? it 
#won't convert them to float and idk why 
























#QUESTION 2












