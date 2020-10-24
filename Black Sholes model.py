#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 18:06:12 2020

@author: ziranxu
"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

class CallOption:
    def __init__(self,S0,R,V,K,T):
        self.s0 = S0
        self.r = R
        self.v = V
        self.k = K
        self.t = T
    def d1(self):
        d1 = (np.log(self.s0/self.k)+(self.r+0.5*self.v**2)*self.t)/(self.v*np.sqrt(self.t))
        return d1
    def d2(self):
        d2 = (np.log(self.s0/self.k)+(self.r-0.5*self.v**2)*self.t)/(self.v*np.sqrt(self.t))
        return d2
    def call_value(self):
        c = stats.norm.cdf(self.d1(),0,1)*self.s0-self.k*np.exp(-self.r*self.t)*stats.norm.cdf(self.d2(),0,1)
        return c
    def delta(self):
        return stats.norm.cdf(self.d1(),0,1)
    def vega(self):
        return self.s0*stats.norm.cdf(self.d1(),0,1)*np.sqrt(self.t)
    def gamma(self):
        return stats.norm.pdf(self.d1(),0,1)/(self.s0*self.v*np.sqrt(self.t))

class PutOption(CallOption):
    def put_value(self):
        p = self.k*np.exp(-self.r*self.t)*stats.norm.cdf(-self.d2(),0,1)-self.s0*stats.norm.cdf(-self.d1(),0,1)
        return p
    def delta(self):
        return -stats.norm.cdf(-self.d1(),0,1)

#  Calculate the Call and Put through the true value
s0 = 2.47
k = 2.5
r = 0.04
sigma = 0.2
t1 = pd.datetime(2018,11,28)
t2 = pd.datetime(2019,1,23)
t = (t2-t1).days/365

C = CallOption(s0,r,sigma,k,t)
P = PutOption(s0,r,sigma,k,t)
c_value = C.call_value()
c_delta = C.delta()
c_vega = C.vega()
c_gamma = C.gamma()
p_value = P.put_value()
p_delta = P.delta()
p_vega = P.vega()
p_gamma = P.gamma()

print("The Call Option value is {:.4f} , delta value is {:.4f}, vega value is {:.4f}, gamma \
      value is {:.4f}".format(c_value,c_delta,c_vega,c_gamma))
print("The Put Option value is {:.4f} , delta value is {:.4f}, vega value is {:.4f}, gamma \
      value is {:.4f}".format(p_value,p_delta,p_vega,p_gamma))

# explore the Option values change with Strike price change
k_test = np.linspace(2.25,2.65,80)
C_test = CallOption(s0,r,sigma,k_test,t)
P_test = PutOption(s0,r,sigma,k_test,t)
c_test = C_test.call_value()
p_test = P_test.put_value()

plt.figure()
plt.subplot(2,1,1)
plt.plot(k_test,c_test)
plt.grid(True)
plt.xlabel("strike price")
plt.ylabel("call option value")
plt.subplot(2,1,2)
plt.plot(k_test,p_test)
plt.grid(True)
plt.xlabel("strike price")
plt.ylabel("put option value")

# explore the Greek values change with Strike price change
c_delta_test = C_test.delta()
p_delta_test = P_test.delta()
c_gamma_test = C_test.gamma()
p_gamma_test = P_test.gamma()
c_vega_test = C_test.vega()
p_vega_test = P_test.vega()

plt.figure()
plt.subplot(2,1,1)
plt.plot(k_test,c_delta_test)
plt.grid(True)
plt.xlabel("strike price")
plt.ylabel("call option delta")
plt.subplot(2,1,2)
plt.plot(k_test,p_delta_test)
plt.grid(True)
plt.xlabel("strike price")
plt.ylabel("put option delta")

plt.figure()
plt.subplot(2,1,1)
plt.plot(k_test,c_gamma_test)
plt.grid(True)
plt.xlabel("strike price")
plt.ylabel("call option gamma")
plt.subplot(2,1,2)
plt.plot(k_test,p_gamma_test)
plt.grid(True)
plt.xlabel("strike price")
plt.ylabel("put option gamma")

plt.figure()
plt.subplot(2,1,1)
plt.plot(k_test,c_vega_test)
plt.grid(True)
plt.xlabel("strike price")
plt.ylabel("call option vega")
plt.subplot(2,1,2)
plt.plot(k_test,p_vega_test)
plt.grid(True)
plt.xlabel("strike price")
plt.ylabel("put option vega")