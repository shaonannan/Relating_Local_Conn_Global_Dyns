#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:46:09 2022

@author: yuxiushao
"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib

from numpy import linalg as la
from scipy import linalg as scpla
import scipy
# import seaborn as sb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmath import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve,leastsq,minimize
import scipy.integrate
from math import tanh,cosh
import math
import time

from functools import partial
import random

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# from sympy import *
from scipy.linalg import schur, eigvals
from scipy.special import comb, perm
import scipy.stats as stats

extras_require = {'PLOT':['matplotlib>=1.1.1,<3.0']},

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
### SELF DEFINED
from UtilFuncs import *
from Sparse_util import *
from Meanfield_Dyn_util import *
from Connect_util import *



Nt=np.array([600,600])#([750,750]) # 750
NE,NI=Nt[0],Nt[1]
N=NE+NI
Nparams=np.array([NE,NI])
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
nrank,ntrial,neta,ngavg=1,30,2,11

''' ## Three \bar{J} cases '''
### connectivity setting -- original
JI,JE,a,b      = 0.5,2.0,0.0,0.0
JEE,JIE,JEI,JII=JE+a,JE-a,JI-b,JI+b
''' Am -- J(g0=0), l0(g0=0), r0(g0=0), S=R1(B1-lambda0In-1)^(-1)L1.T '''
Am,Jsv=generate_meanmat_eig(Nparams,JEE,JIE,JEI,JII)


### eigendecomposition
xAm,yAm =np.ones((N,1)),np.ones((N,1))
yAm[:NE,0],yAm[NE:,0] = yAm[:NE,0]*N/NE*JE,-yAm[NE:,0]*N/NI*JI
eigvAm   = np.zeros(N)
eigvAm[0]=JE-JI
### singular value 
sigvAm = np.sqrt(2*(JE**2+JI**2))
print('line 74: predicted singular value:',sigvAm)
uAm,sAm,vAmh = la.svd(Am)
print('line 76: numerical cal singular value:',sAm[0])
'''Network Setting f'''
xee,xei,xie,xii=1.0,1.0,1.0,1.0
coeffeta=np.array([1.0,1.0,1.0])
gaverage = 1.0
signeta  = np.ones(3)
''' Recording Variables '''
# M,N -- with lambda
eigvseries_eig = np.zeros((ntrial,N),dtype=complex)
eigvseries_svd = np.zeros((ntrial,N),dtype=complex)
''' All random matrices for each trials '''
''' Iterative Processing '''
for iktrial in range(ntrial):
    ##>>>>>>>>>>> g0!=0, >>>>>>>>
    Xsym  = iidGaussian([0,gaverage/np.sqrt(N)],[N,N])
    XsymT = Xsym.copy().T
    X0    = Xsym.copy()  
    J0    = Am.copy()+X0.copy()
    uJ,sJ,vJh = la.svd(J0)
    ### reconstruct using SVD
    Jsvd = np.reshape(uJ[:,0],(-1,1))@np.reshape(vJh[0,:],(1,-1))*sJ[0]
    ### calculate the eigenvalue 
    eigv_svd, _=la.eig(Jsvd)
    eigv_eig, _=la.eig(J0)
    
    ## l,r without lambda
    eigvseries_eig[iktrial,:]=eigv_eig.copy()
    eigvseries_svd[iktrial,:]=eigv_svd.copy()
    
figsv, axsv = plt.subplots(figsize=(4,3))
xxx = np.linspace(0,2,50)
predict_sv_hist = 2*np.sqrt(4-xxx**2)
predict_sv_hist = predict_sv_hist/np.sum(predict_sv_hist)/(xxx[1]-xxx[0])
axsv.hist(Bsigvseries[:,:].flatten(),bins=50,facecolor='gray',alpha=0.5,density=True)
axsv.plot(xxx,predict_sv_hist,c='tab:red',lw=1.5)
axsv.scatter(sigvAm,0,s=80,c='',marker='o',edgecolor='tab:red')
axsv.scatter(predict_svalue,0,s=80,c='',marker='o',edgecolor='black')


xticks = [.0,2.0]
xlims  = [.0,3.5]

yticks = [0,0.8]
ylims  = [0,0.8]

axsv.set_xlim(xlims)
axsv.set_ylim(ylims)
axsv.set_xticks(xticks)
axsv.set_yticks(yticks)
# axsv.set_aspect('equal')

figev, axev = plt.subplots(figsize=(4,3))
axev.scatter(np.real(Beigvseries[0,1:]),np.imag(Beigvseries[0,1:]),s=10,c='tab:blue',alpha=0.25)
axev.scatter(np.real(Beigvseries[:,0]),np.imag(Beigvseries[:,0]),s=20,c='tab:blue',alpha=0.5)
axev.scatter(eigvAm[0].real,eigvAm[0].imag,s=80,c='',marker='o',edgecolor='tab:red')

axev.spines['right'].set_color('none')
axev.spines['top'].set_color('none')
# X axis location
axev.xaxis.set_ticks_position('bottom')
axev.spines['bottom'].set_position(('data', 0))

xticks = [-1.0,1.0]
xlims  = [-1.5,2.0]
r_g = gaverage
axev.set_yticks(yticks)
axev.set_aspect('equal')

theta = np.linspace(0, 2 * np.pi, 200)
xr = r_g*(1+0)*np.cos(theta)
yr = r_g*(1-0)*np.sin(theta)

axev.plot(xr, yr, color="black", linewidth=1.5,linestyle='--') # >>>>>