#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,quad
from scipy.interpolate import interp1d
from numpy import genfromtxt
from scipy.optimize import curve_fit 
from scipy import optimize as opt 
from scipy.optimize import minimize, rosen, rosen_der
from scipy.optimize import fsolve
from scipy.optimize import fmin
from scipy import optimize
import scipy as sp
from matplotlib import rc, rcParams

plt.rcParams["text.usetex"] = False
fontsize=26
legendfontsize=22
font = {'size' : fontsize}
rc('font',**font)
rc('text', usetex=False)
rc('font', family='serif', serif='Computer Modern Roman')

golden_ratio=1.61803398875 #Standard Python ratio for plot width/height=8/6.
G = 4.302*10**(-6) #kpc/Msol*km²/s²


# In[10]:


# abstract base class for the DM halo
class DMHalo:
    def __init__(self, rho_0):
        self.rho_0 = rho_0
        
    def density(self, r):
        pass
    
    def __str__(self):
        return "DMHalo"
    
    def mass(self, r):
        dMdR = lambda m, r: r**2 * self.density(r)  # This is the integrand
        return 4*np.pi * odeint(dMdR, 0., np.append(1e-3*np.min(r), r))[1:, 0]   
        # This construction is used so that we don't start with M=0 at r[0], but at 1e-6 * r[0], 
        #    so that the value at r[0] is more realistic


# In[12]:


class IsoHalo(DMHalo): 
    num = 10000
    M_b = 0.
    
    def __init__(self,rho_0,sigma_0):
        super().__init__(rho_0)
        self.sigma_0 = sigma_0
        self.r_min = 1e3 * 4.8e-14 * 6 # r_isco in pc
        self.r_max = 1e4
        
    def solve_dgl(self,r):
        def rhs(x,logr):
            r = np.exp(logr)
            rho, M_dm = np.exp(x)
            drhodr = -(1./self.sigma_0**2)*(rho/r**2)*(M_dm+self.M_b)
            dMdr = 4.0*np.pi*r**2* rho 
            return [r*drhodr/rho,r*dMdr/M_dm]

        # initial conditions
        M_0 = (4.*np.pi/3.)*self.rho_0*self.r_min**3
        x_0 = [np.log(self.rho_0), np.log(M_0)]
        x = odeint(rhs,x_0,np.log(np.append(self.r_min,r)), rtol=1e-10, atol=1e-10)
        rho = np.exp(x[1:,0])
        M = np.exp(x[1:,1])
        return rho, M
        
    def density(self, r):
        return IsoHalo.solve_dgl(self,r)[0]
    
    def __str__(self):
        return "IsoHalo"

    def mass(self, r):
        return IsoHalo.solve_dgl(self,r)[1]


# In[4]:


class NFW(DMHalo):
    def __init__(self, rho_s, r_s):
        self.rho_s = rho_s
        self.r_s = r_s
    
    def density(self, r):
        return self.rho_s / (r/self.r_s) / (1. + r/self.r_s)**2
    
    def __str__(self):
        return "NFW"
    
    def mass(self, r):
        return  4*np.pi*self.rho_s * self.r_s**3  *                   (np.log((self.r_s+r)/(self.r_s)) + self.r_s/(self.r_s + r) - 1.)

    def FromVirial(M200,c):    # M_sol, c has no dimension
        H0 = 70e-3 * 1e-3 / 299792458   # km/s/Mpc to pc^-1
        M200 = M200 * 4.7963e-14  # M_sol to pc
        r200 = (M200/(100*H0**2))**1/3
        r_s = r200/c   # pc
        rho_s = M200/((4*np.pi*r_s**3*np.log(1+c)) - (c/(1+c)))   # pc^-2
        return NFW(rho_s,r_s)       


# In[5]:


class MatchedSIDM(DMHalo):   # iso + nfw
    def __init__(self, rho_0, sigma_0, r_m, rho_s, r_s):
        super().__init__(rho_0)
        self.sigma_0 = sigma_0
        self.inner = IsoHalo(rho_0, sigma_0)
        self.outer = NFW(rho_s, r_s)
        self.r_m = r_m
        
    def __str__(self):
        return "MatchedSIDM"
    
    def density(self,r):
        return np.where(r < self.r_m,
                           self.inner.density(r),
                           self.outer.density(r) )

    def mass(self,r):
        return np.where(r < self.r_m,
                        self.inner.mass(r),
                        self.outer.mass(r) - self.outer.mass(self.r_m) + self.inner.mass(self.r_m) ) 
                        # Second and third term are unnecessary if we assume them to be equal in the first place

    def matching(nfw, r_m):
        # boundary conditions
        rho_m = nfw.density(r_m)
        M_m = nfw.mass(r_m)

        def model(z, x):
            h, dh = z
            ddh = -(2./x)*dh - np.exp(h)
            return [dh, ddh]

        xa_val = np.geomspace(1e-5, 1e12, 1500)  
        x_min = min(xa_val)

        z2 = odeint(model,[0.,-x_min/3.], xa_val)
        [h_val, hp_val] = [z2[:,0], z2[:,1]]

        func_h = interp1d(xa_val,h_val)
        func_hp = interp1d(xa_val,hp_val)

        def ratio(x): 
            return -np.exp(-func_h(x))*func_hp(x)/x

        max_ratio_index = np.argmax(ratio(xa_val))
        max_xval = xa_val[max_ratio_index]
        xa_val_new = np.geomspace(1e-5,max_xval,1500)
        
        ### inverse ratio == x(ratio)
        inv_ratio = interp1d(ratio(xa_val_new), xa_val_new)
        
        ### matching
        ratio_NFW = M_m/(4.*np.pi*rho_m*r_m**3)

        print(ratio_NFW, ratio(xa_val), 
                  ratio_NFW > np.min(ratio(xa_val)) and ratio_NFW < np.max(ratio(xa_val)) )
        x_m = inv_ratio(ratio_NFW)

        r_star = r_m / x_m

        def func_h_r(r):
            return func_h(r/r_star)

        rho_0 = rho_m*np.exp(-func_h_r(r_m)) # rho_iso(r_1) = rho_1
        sigma_0 = np.sqrt(4*np.pi*rho_0) * r_star
        
        print("Matching result: rho_0= ", rho_0 / 4.7963e-23,'M_sol/kpc³', " - sigma_0 = ", sigma_0 * 299792.458,'km/s')
        return MatchedSIDM(rho_0, sigma_0, r_m, nfw.rho_s, nfw.r_s)
    
    def core_dynamic(self,r_m):
        def integrand(r):
            return ((self.inner.mass(r))**2 - (self.outer.mass(r))**2) / r**2
        D_U = quad(integrand,0,r_m)
        D_U = - 1/2 * D_U[0]
        if  D_U > 0:
            print('Core growing solution.')
        else:
            print('Core collapse solution.')
        return 'D_U = ',D_U

    def find_sigma_m(self,nfw,r_m,t_age):        #sigma_m in cm²/g
        years_to_sec = 365*24*60*60
        t_age = t_age * years_to_sec  # 10e9
        unit_conversion = (3.09e16)**3 * (1e5)**2 / (2e33)
    
        rho_m = nfw.density(r_m) / (4.7963e-23)    # M_sol/kpc^3
        self.sigma_0 = self.sigma_0 * 299792.458   # km/s
        print(self.sigma_0)
        print('rho_m',rho_m)
        sigma_m = np.sqrt(np.pi)/(rho_m*4*self.sigma_0*t_age) * unit_conversion
        return sigma_m


# In[6]:


class Spike(DMHalo):
    def __init__(self,rho_spike, r_spike, gamma):
        DMHalo.__init__(self, 0.)
        self.r_spike = r_spike
        self.gamma = gamma
        self.rho_spike = rho_spike
        
    def __str__(self):
        return "Spike"
    
    def density(self,r):
        return self.rho_spike * (self.r_spike/r)**self.gamma
    
    def mass(self,r):
        return 4*np.pi*self.rho_spike*self.r_spike**self.gamma * r**(3.-self.gamma) / (3.-self.gamma)


# In[7]:


class Spike_NFW(Spike, NFW):
    def __init__(self, rho_s, r_s, r_spike, gamma):
        Spike.__init__(self,rho_s * r_s/r_spike / (1+r_spike/r_s)**2,r_spike,gamma)
        NFW.__init__(self,rho_s, r_s)
        self.r_min = 0.
        print('rho_spike = ',self.rho_spike/4.8e-14)

    def density(self, r):
        return np.where(r < self.r_spike,                         np.where(r > self.r_min, self.rho_spike * (self.r_spike/r)**self.gamma, 0.),                         NFW.density(self,r))

    def mass(self, r):
        return np.where(r < self.r_spike,
                        np.where(r > self.r_min, Spike.mass(self,r) - Spike.mass(self,self.r_min), 0),
                        NFW.mass(self,r) - NFW.mass(self,self.r_spike) + Spike.mass(self,self.r_spike) - Spike.mass(self,self.r_min))


    def FromNFW(nfw, M_bh, gamma):
        r = np.geomspace(1e-3*nfw.r_s, 1e3*nfw.r_s)
        M_to_r = interp1d(nfw.mass(r), r, kind='cubic', bounds_error=True)
        r_h = M_to_r(2.* M_bh)
        r_spike = 0.2*r_h
        print('r_spike = ',r_spike,'r_h = ',r_h)
        return Spike_NFW(nfw.rho_s, nfw.r_s, r_spike, gamma)

    def __str__(self):
        return "SpikedNFW"


# In[8]:


class SpikedSIDM(Spike, MatchedSIDM):
    def __init__(self, r_spike, gamma, sidm):
        Spike.__init__(self, sidm.density(r_spike), r_spike, gamma)
        MatchedSIDM.__init__(self, sidm.inner.rho_0, sidm.inner.sigma_0, sidm.r_m, sidm.outer.rho_s, sidm.outer.r_s)
    
    def density(self, r):
        return np.where( r < self.r_spike, 
                        Spike.density(self, r),
                        MatchedSIDM.density(self, r))
    
    def mass(self, r):
        return np.where( r < self.r_spike,
                        Spike.mass(self, r),
                        MatchedSIDM.mass(self, r) - MatchedSIDM.mass(self, self.r_spike) + Spike.mass(self, self.r_spike) )
    
    def __str__(self):
        return "SpikedSIDM"
    
    def FromBH(sidm, m_bh, gamma):
        r_spike = 0.2 * m_bh / sidm.inner.sigma_0**2
        return SpikedSIDM(r_spike, gamma, sidm)


# In[ ]:





# In[ ]:




