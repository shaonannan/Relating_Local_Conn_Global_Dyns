#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 2022

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
'''
WITHOUT SYMMETRY, 
* increase JE to change from JE-JI<1 to JE-JI>1
* constant g0 # randomness
'''
from functools import partial
shiftx=1.5

# data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_iid_PositiveTF_0_4VS1.npz"
# # np.savez(data_name, **data)
# data      = np.load(data_name)
RERUN = 1

JE    = 0.6


Nt    = np.array([1200,300])#([750, 750])
NE,NI = Nt[0], Nt[1]
N     = NE+NI
Nparams  = np.array([NE, NI])
Npercent = Nparams/N
nrank, ntrial, neta, nvec, nmaxFP = 1,3, 1, 2, 3
cuttrials = int(ntrial/2)
nJI = 10#10
JIseries = np.linspace(1.5, 2.4, nJI) # 1.2 2.1
''' Network Setting  '''
#### compare to connectivity ~~~~~
gaverage = 0.8
xee,xei,xie,xii=1.0,0.5,0.2,0.8
# heterogeneous degree of symmetry: amplitudes and signs
coeffetaEsub  = np.array([1.0, 1.0, 1.0])
coeffetaTotal = np.array([1.0, 1.0, 1.0])
# coeffetaTotal = np.zeros(3)
signetaEsub   = np.ones(3)
signetaTotal  = np.ones(3)

ppercentEsub    = np.ones(2)
ppercentEsub[0] = 0.5
ppercentEsub[1] = 1.0-ppercentEsub[0]
# E->total ppercentEsub[0]/2.0,ppercentEsub[1]/2.0, (I) 1/2.0

'''  Recording Variables. '''
# perturbed + symmetric eigenvectors
Reigvecseries, Leigvecseries = np.zeros(
    (nJI, ntrial, N, nvec*2)), np.zeros((nJI, ntrial, N, nvec*2))
ReigvecTseries, LeigvecTseries = np.zeros(
    (nJI, ntrial, N, nvec*2)), np.zeros((nJI, ntrial, N, nvec*2))
if(RERUN==0):
    ReigvecTseries[:,:,:,0], LeigvecTseries[:,:,:,0] = data['ReigvecTseries'],data['LeigvecTseries']
Beigvseries = np.zeros((nJI, ntrial, N), dtype=complex)
# reference, eigenvectors of matrix with iid perturbations
x0series, y0series = np.zeros((nJI, ntrial, N), dtype=complex), np.zeros(
    (nJI, ntrial, N), dtype=complex)

''' Nullclines parameters  '''
kappaintersect_Full = np.zeros((nJI, ntrial, nmaxFP*2))
kappaintersect_R1   = np.zeros((nJI, ntrial, nmaxFP*2))

''' simulating parameters '''
tt  = np.linspace(0, 500, 500)
dt  = tt[2]-tt[1]
ntt = len(tt)
xfpseries_Full = np.zeros((nJI, ntrial, N, ntt))
xfpseries_R1   = np.zeros((nJI, ntrial, N, ntt))
# @YX 04DEC -- RECORD THE DYNAMICS OF KAPPA
kappaseries_R1 = np.zeros((nJI, ntrial, ntt))


if(RERUN==0):
    kappaintersect_Full = data['kappaintersect_Full']
    kappaintersect_R1   = data['kappaintersect_R1']

    xfpseries_Full[:,:,:,-1] = data['xfpseries_Full']
    xfpseries_R1[:,:,:,-1]   = data['xfpseries_R1']

if (RERUN):
    ''' Iterative Processing '''
    print('>>>>>>> simulating neuronal activity ... >>>>>>')
    for iktrial in range(ntrial):
        Xsym  = iidGaussian([0, gaverage/np.sqrt(N)], [N, N])
        XsymT = Xsym.copy().T
        X0    = Xsym.copy()
        for idxje, JIv in enumerate(JIseries):
            JI, JE, a, b = JIv, JE, 0, 0
            JEE, JIE, JEI, JII = JE+a, JE-a, JI-b, JI+b
            Am, Jsv = generate_meanmat_eig(Nparams, JEE, JIE, JEI, JII)
            meigvecAm, neigvecAm = np.ones((N, 1)), np.ones((N, 1))
            neigvecAm[:NE, 0], neigvecAm[NE:, 0] = N*JE/NE, -N*JI/NI
            eigvAm   = (JE-JI)*np.ones(N)
            xAm, yAm = np.reshape(meigvecAm[:, 0], (N, 1)), np.reshape(neigvecAm[:, 0].copy(), (N, 1))
            eta = 0
            etaset = eta*np.ones(6)
            for iloop in range(3):
                etaset[iloop]   = etaset[iloop]*coeffetaEsub[iloop]
                etaset[iloop+3] = etaset[iloop+3]*coeffetaTotal[iloop]
            Xinit = Xsym.copy()
            X     = Xinit.copy()
            X[:NE, :NE] *= (xee)
            X[:NE, NE:] *= (xei)
            X[NE:, :NE] *= (xie)
            X[NE:, NE:] *= (xii)

            # overall
            J  = X.copy()+Am.copy()
            JT = J.copy().T
            ''' iid Gaussian randomness '''
            eigvJ, leigvec, reigvec, xnorm0, ynorm0 = decompNormalization(
                J, xAm, yAm, xAm, yAm, nparams=Nparams, sort=0, nrank=1)
            Reigvecseries[idxje, iktrial, :, 0], Leigvecseries[idxje,iktrial, :, 0] = xnorm0[:, 0].copy(), ynorm0[:, 0].copy()
            Beigvseries[idxje, iktrial, :] = eigvJ.copy()

            # Rank One Approximation, Nullclines of \kappa
            xnorm0, ynorm0 = Reigvecseries[idxje, iktrial, :, 0].copy(), Leigvecseries[idxje, iktrial, :, 0]
            xnorm0, ynorm0 = np.reshape(xnorm0, (N, 1)), np.reshape(ynorm0, (N, 1))

            ''' Full Connectivity -- Dynamics '''
            Jpt = J.copy()
            if(iktrial<cuttrials):
                xinit = np.random.normal(5.0,1E-1, (1, N))
            else:
                xinit = np.random.normal(0,1e-2,(1,N))
            xinit = np.squeeze(np.abs(xinit))
            xtemporal = odesimulationP(tt, xinit, Jpt, 0)
            xfpseries_Full[idxje, iktrial, :, :] = xtemporal.T.copy()
            # 2 KAPPA_0
            kappanum = np.zeros(3)
            xact = np.squeeze(xfpseries_Full[idxje, iktrial, :, -1])
            # use yAm -- unperturbed centre.
            if(1):#(ymu[0])*(yAm[0, 0]) > 0):
                kappanum[0] = yAm[0, 0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0] + yAm[NE, 0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            else:
                kappanum[0] = -yAm[0, 0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0] -yAm[NE, 0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            kappaintersect_Full[idxje, iktrial, :3] = kappanum[:].copy()

            ''' use perturbation theorem to calculate the perturbed vectors '''
            xnormt, ynormt = X.copy()@xAm.copy()/eigvAm[0], X.copy().T@yAm.copy()/eigvAm[0]
            xnormt, ynormt = xnormt+xAm.copy(), ynormt+yAm.copy()
            
            ReigvecTseries[idxje, iktrial, :, 0], LeigvecTseries[idxje,iktrial, :, 0] = xnormt[:, 0].copy(), ynormt[:, 0].copy()

            # renew -- perturbation result
            if(1):#(ymu[0])*(yAm[0, 0]) > 0):
                kappanum[0] = np.squeeze(np.reshape(ynormt,(1,N))@np.reshape((1.0+np.tanh(xact-shiftx)),(N,1)))/N
            else:
                mvec_norm = np.reshape(xnormt,(N,1))
                kappanum[0] = np.squeeze(np.reshape(np.squeeze(xact),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            kappaintersect_Full[idxje, iktrial, :3] = kappanum[:].copy()

            r1Matt = np.real(xnormt@ynormt.T)
            r1Matt = r1Matt/N
            # START SIMULATING AND CALCULATE KAPPA AND xfpsereis_R1
            # use the same initial values
            xtemporal = odesimulationP(tt, xinit, r1Matt, 0)
            xfpseries_R1[idxje, iktrial, :, :] = xtemporal.T.copy()
            kappaseries_R1[idxje, iktrial, :] = np.squeeze(xtemporal.copy()@np.reshape(xnormt, (-1, 1)))/np.sum(xnormt**2)
            ''' kappa dynamics    '''
            # 2 populations
            kappanum = np.zeros(3)
            xact = np.squeeze(xfpseries_R1[idxje, iktrial, :, -1])
            # use yAm -- unperturbed centre.
            if(1):#(ymu[0])*(yAm[0, 0]) > 0):
                kappanum[0] = yAm[0, 0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0] + \
                    yAm[NE, 0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            else:
                kappanum[0] = -yAm[0, 0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0] - \
                    yAm[NE, 0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            kappaintersect_R1[idxje, iktrial, :3] = kappanum[:].copy()

def dyn_analysis():
    #### A. intersections of the equilibrium of kappa
    gmat = np.array([xee, xei, xie, xii])*gaverage
    gee, gei, gie, gii = gmat[0], gmat[1], gmat[2], gmat[3]
    if(len(Nparams) == 2):
        NE, NI = Nparams[0], Nparams[1]
    else:
        NE1, NE2, NI = Nparams[0], Nparams[1], Nparams[2]
        NE = NE1+NE2
    N = NE+NI
    xintersect = np.linspace(1.5, 2.4, 2)#np.linspace(1.2, 1.9, 2)
    kappa_x    = np.linspace(-1, 4, 30)
    Sx         = np.zeros((len(xintersect), len(kappa_x)))
    F0         = np.zeros_like(Sx)
    F1         = np.zeros_like(Sx)
    for idx, JIU in enumerate(xintersect):
        for idxx, x in enumerate(kappa_x):
            muphi, delta0phiE, delta0phiI = x, x**2 * \
                (gee**2*NE/N+gei**2*NI/N)/(JE-JIU)**2, x**2 * \
                (gie**2*NE/N+gii**2*NI/N)/(JE-JIU)**2
            Sx[idx, idxx] = -x+(JE*PhiP(muphi, delta0phiE) -
                                JIU*PhiP(muphi, delta0phiI))
            F0[idx,idxx]  = x 
            F1[idx,idxx]  = (JE*PhiP(muphi, delta0phiE) -JIU*PhiP(muphi, delta0phiI))
    fig, ax = plt.subplots(figsize=(4, 4))
    xticks  = [-1,0,4]#np.linspace(-1, 4, 3)
    xlims   = [-1, 4]
    yticks  = [-4,0,1]#np.linspace(-1.0, 1.0, 3)
    ylims   = [-4, 1.0]
    # ax.plot(kappa_x,np.zeros_like(kappa_x),c='k',lw=1.0)
    ax.plot(kappa_x, Sx[0, :], c='gray', lw=1)
    ax.plot(kappa_x, Sx[1, :], c='tab:purple', lw=1.0)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    
    
    fig, ax = plt.subplots(figsize=(4, 4))
    xticks  = [-1,0,4]#np.linspace(-1, 5, 2)
    xlims   = [-1, 4]
    yticks  = [-4,0,1]#np.linspace(-1., 5, 2)
    ylims   = [-4.0,1]
    # ax.plot(kappa_x,np.zeros_like(kappa_x),c='k',lw=1.0)
    ax.plot(kappa_x, F0[0, :], c='gray', lw=1)
    ax.plot(kappa_x, F1[0, :], c='tab:purple',linestyle='--',alpha=0.75)
    ax.plot(kappa_x, F1[1, :], c='tab:purple')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    


    variance_x_num, variance_kappa_num   = np.zeros((nJI, ntrial, 2)), np.zeros((nJI, ntrial, 2))
    variance_full_theo, variance_R1_theo = np.zeros((nJI, ntrial, 3,2)), np.zeros((nJI, ntrial, 3,2))
    mu_x_num, mu_kappa_num   = np.zeros((nJI, ntrial, 2)), np.zeros((nJI, ntrial, 2))
    mu_full_theo, mu_R1_theo = np.zeros((nJI, ntrial,3, 2)), np.zeros((nJI, ntrial, 3, 2))
    for idxji in range(nJI):
        for iktrial in range(ntrial):
            # numerical results for Full Mat
            variance_x_num[idxji, iktrial, 0], variance_x_num[idxji, iktrial, 1] = np.std(
                xfpseries_Full[idxji, iktrial, :NE, -1])**2, np.std(xfpseries_Full[idxji, iktrial, NE:, -1])**2
            mu_x_num[idxji, iktrial, 0], mu_x_num[idxji, iktrial, 1] = np.mean(
                xfpseries_Full[idxji, iktrial, :NE, -1]), np.mean(xfpseries_Full[idxji, iktrial, NE:, -1])
            # numerical results for Rank one Appriximation Mat
            variance_kappa_num[idxji, iktrial, 0], variance_kappa_num[idxji, iktrial, 1] = np.std(
                xfpseries_R1[idxji, iktrial, :NE, -1])**2, np.std(xfpseries_R1[idxji, iktrial, NE:, -1])**2
            mu_kappa_num[idxji, iktrial, 0], mu_kappa_num[idxji, iktrial, 1] = np.mean(
                xfpseries_R1[idxji, iktrial, :NE, -1]), np.mean(xfpseries_R1[idxji, iktrial, NE:, -1])

    #### B. theo. latent dynamical variable kappa
    kappa_theo_iid = np.zeros((nJI, 3))
    for idxji in range(nJI):
        gmat = np.array([xee,xei,xie,xii])#general case# np.array([gaverage, gaverage, gaverage, gaverage])
        gmat = gmat * gaverage
        init_k = -1.0#1.1*np.max(kappaintersect_R1[idxje, np.where(kappaintersect_Full[idxje, :,0] >= 0)[0], 0])
        # kappa_theo_iid[idxje, 0] 
        kappa_max= fsolve(iidperturbationP, init_k, args=(JE, JIseries[idxji], gmat, Nparams),xtol=1e-8,maxfev=1000)
        residual0=np.abs(iidperturbationP(kappa_max, JE, JIseries[idxji], gmat, Nparams))
        print('>>>>>>>>>residual:',residual0)
        init_k = 0.5*init_k
        # kappa_theo_iid[idxje, 1] 
        kappa_middle= fsolve(iidperturbationP, init_k, args=(JE, JIseries[idxji], gmat, Nparams),xtol=1e-8,maxfev=1000)
        residual1=np.abs(iidperturbationP(kappa_middle,JE, JIseries[idxji], gmat, Nparams))
        init_k = 0.0
        kappa_theo_iid[idxji, 2] = fsolve(iidperturbationP, init_k, args=(JE, JIseries[idxji], gmat, Nparams),xtol=1e-8,maxfev=1000)

        if(residual0>1e-2):
            kappa_theo_iid[idxji,0]=kappa_theo_iid[idxji,2]
        else:
            kappa_theo_iid[idxji,0]=kappa_max
        if(residual1>1e-2):
            kappa_theo_iid[idxji,1]=kappa_theo_iid[idxji,2]
        else:
            kappa_theo_iid[idxji,1]=kappa_middle

        ### variance
        # ## theoretical
        deltainitE, meaninitE = np.max(variance_x_num[idxji, :, 0]), kappa_theo_iid[idxji, 0]
        deltainitI, meaninitI = np.max(variance_x_num[idxji, :, 1]), kappa_theo_iid[idxji, 0]
        INIT = [meaninitE, meaninitI, deltainitE, deltainitI]
        statsfull = fsolve(iidfull_mudelta_consistencyP, INIT,args=(JE, JIseries[idxji], gmat, Nparams),xtol=1e-6)
        mu_full_theo[idxji, :,0, 0], variance_full_theo[idxji,:,0, 0] = statsfull[0], statsfull[2]
        mu_full_theo[idxji, :,0, 1], variance_full_theo[idxji,:,0, 1] = statsfull[1], statsfull[3]

        mu_R1_theo[idxji, :,0, 0], variance_R1_theo[idxji, :, 0,0] = kappa_theo_iid[idxji,0], kappa_theo_iid[idxji, 0]**2*(gee**2*NE/N+gei**2*NI/N)/(JE-JIseries[idxji])**2
        mu_R1_theo[idxji, :,0, 1], variance_R1_theo[idxji, :,0, 1] = kappa_theo_iid[idxji,0], kappa_theo_iid[idxji, 0]**2*(gie**2*NE/N+gii**2*NI/N)/(JE-JIseries[idxji])**2


    clrs = ['k','b','r']
    mean_pos_kappa_full, mean_neg_kappa_full = np.zeros(nJI), np.zeros(nJI)
    std_pos_kappa_full, std_neg_kappa_full = np.zeros(nJI), np.zeros(nJI)
    pos_t0=-0.7

    fig, ax = plt.subplots(figsize=(4, 2))

    xticks = np.linspace(JIseries[0],JIseries[-1],2)#np.linspace(1.2, 2.0, 2)
    xlims  = [JIseries[0],JIseries[-1]]#[1.1, 2.2]
    yticks = np.linspace(-0.15, -0.05, 2)
    ylims  = [-0.15, -0.05]
    
    ax.plot(JIseries, kappa_theo_iid[:, 0],c='tab:purple', linewidth=1.5)
    ax.plot(JIseries, kappa_theo_iid[:, 2],c='tab:purple', linewidth=1.5)
    ax.plot(JIseries, kappa_theo_iid[:, 1],c='tab:purple', linewidth=1.5)


    for idxji in range(nJI):
        pos_full = kappaintersect_Full[idxji, np.where(
            kappaintersect_Full[idxji, :, 0] >= pos_t0)[0], 0]
        mean_pos_kappa_full[idxji] = np.mean(pos_full)
        std_pos_kappa_full[idxji]  = np.std(pos_full)

    low_bound = np.zeros_like(mean_pos_kappa_full)
    for imax in range(len(mean_pos_kappa_full)):
        low_bound[imax]=min(0,mean_pos_kappa_full[imax]+std_pos_kappa_full[imax])
    ax.fill_between(JIseries, mean_pos_kappa_full-std_pos_kappa_full,low_bound, alpha=0.5, facecolor='black')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    fig, ax2   = plt.subplots(2, 1, figsize=(5, 5),sharex=True,tight_layout=True)
    ax2[0].plot(JIseries, np.mean(variance_R1_theo[:, :, 0,0], axis=1), color='tab:red',linewidth=1.5, label=r'$\Delta_{R1}^E$ theo.')
    ax2[1].plot(JIseries, np.mean(variance_R1_theo[:, :, 0,1], axis=1), color='tab:blue',linewidth=1.5, label=r'$\Delta_{R1}^I$ theo.')
    
    varE_mean, varE_std = np.mean(variance_x_num[:, :,0], axis=1),np.std(variance_x_num[:, :,0], axis=1)
    ax2[0].fill_between(JIseries, varE_mean-varE_std, varE_mean+varE_std, facecolor='tab:red',alpha=0.75)
    # ax2[0].plot(JIseries, np.mean(variance_full_theo[:, :, 0,0], axis=1), color='orange',linewidth=1.5, label=r'$\Delta_{R1}^E$ theo.')
    varI_mean, varI_std = np.mean(variance_x_num[:, :,1], axis=1),np.std(variance_x_num[:, :,1], axis=1)
    ax2[1].fill_between(JIseries, varI_mean-varI_std, varI_mean+varI_std, facecolor='tab:blue',alpha=0.75)
    ax2[0].set_xlim(xlims)
    ax2[0].set_ylim([0.0025,0.005])
    ax2[1].set_ylim([0.00045,0.0008])
    ax2[0].set_yticks([0.0025,0.005])
    ax2[1].set_yticks([0.00045,0.0008])

    ijesample=[1,5,9]
    yticks = np.linspace(-3.0, 3.0, 3)
    ylims = [-3.0, 3.0]#[1.1, 2.2]
    xticks = np.linspace(-1.5, 1.5, 3)
    xlims = [-1.5, 1.5]
    import matplotlib.cm as cm
    
    
    figE,axE=plt.subplots(figsize=(2,2))#,sharex=True,sharey=True,tight_layout=True)
    figI,axI=plt.subplots(figsize=(2,2))#,sharex=True,sharey=True,tight_layout=True)
    iktrial =-1
    for iii, idxje in enumerate(ijesample):
        idrandsE     = np.random.choice(np.arange(0,NE),size=100,replace=False)
        idrandsI     = np.random.choice(np.arange(NE,N),size=100,replace=False)
        xactfull_E   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsE, -1])
        xactfull_I   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsI, -1])
        deltaxfull_E = xactfull_E - np.mean(xactfull_E)
        deltaxfull_I = xactfull_I - np.mean(xactfull_I)
    
        # if (np.mean(xactfull_E)<0):
        #     deltaxfull_E=-deltaxfull_E
        # if(np.mean(xactfull_I)<0):
        #     deltaxfull_I=-deltaxfull_I
    
        deltam_E = ReigvecTseries[idxje,iktrial,idrandsE,0]-np.mean(ReigvecTseries[idxje,iktrial,idrandsE,0])#xAm[0,0]
        deltam_I = ReigvecTseries[idxje,iktrial,idrandsI,0]-np.mean(ReigvecTseries[idxje,iktrial,idrandsI,0])#xAm[-1,0]
        axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c=cm.Reds((JIseries[idxje]-JIseries[0])/(JIseries[-1]-JIseries[0])),alpha=0.25)
        axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c=cm.Blues((JIseries[idxje]-JIseries[0])/(JIseries[-1]-JIseries[0])),alpha=0.25)
        ### predicted
        deltaxpred_E = kappa_theo_iid[idxje,2]*np.array(xlims)
        axE.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')
        axI.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')
    
    yticks = np.linspace(-0.5, 0.5, 3)
    ylims = [-0.5, 0.5]#[1.1, 2.2]
    xticks = np.linspace(-1.5, 1.5, 3)
    xlims = [-1.5, 1.5]
    
    axE.set_xlim(xlims)
    axE.set_ylim(ylims)
    axE.set_xticks(xticks)
    axE.set_yticks(yticks)
    
    axE.spines['top'].set_color('none')
    axE.spines['right'].set_color('none')
    axE.xaxis.set_ticks_position('bottom')
    axE.spines['bottom'].set_position(('data',0))
    axE.yaxis.set_ticks_position('left')
    axE.spines['left'].set_position(('data',0))
    
    axI.set_xlim(xlims)
    axI.set_ylim(ylims)
    axI.set_xticks(xticks)
    axI.set_yticks(yticks)
    
    axI.spines['top'].set_color('none')
    axI.spines['right'].set_color('none')
    axI.xaxis.set_ticks_position('bottom')
    axI.spines['bottom'].set_position(('data',0))
    axI.yaxis.set_ticks_position('left')
    axI.spines['left'].set_position(('data',0))
    

# dyn_analysis()


shiftlen = 0
idxc = 0
### A. eigenvalue spectrum
idxgavg,idxtrial= 0,0# 0,3#9,3 # 6,3,0
idxgavgsample   = np.array([0,nJI-1])
figtspt,axtspt  = plt.subplots(figsize=(5,3))
cm=['b','g','c']

idrands=np.random.choice(np.arange(nrank,N),size=600,replace=False)
idxgavg = idxgavgsample[1] ### the first
axtspt.scatter(np.real(Beigvseries[idxgavg,idxtrial,nrank:]),np.imag(Beigvseries[idxgavg,idxtrial,nrank:]),s=10,c='b',alpha=0.25) # >>>>>>>>>>>>>>
axtspt.scatter(np.real(Beigvseries[idxgavg,idxtrial,idrands]),np.imag(Beigvseries[idxgavg,idxtrial,idrands]),s=10,c='b',alpha=0.25) 
axtspt.scatter(np.real(Beigvseries[idxgavg,idxtrial,0]),np.imag(Beigvseries[idxgavg,idxtrial,0]),s=50,c='b',alpha=0.75) 
axtspt.set_aspect('equal')

axtspt.scatter(JE-JIseries[idxgavg],0,s=50,c='',marker='o',edgecolor='red') # 
axtspt.spines['right'].set_color('none')
axtspt.spines['top'].set_color('none')
axtspt.xaxis.set_ticks_position('bottom')
axtspt.spines['bottom'].set_position(('data', 0))
        
aee,aei,aie,aii=xee,xei,xie,xii
eta=0
theta = np.linspace(0, 2 * np.pi, 200)
# first do not multiply at
ahomo = gaverage
xee_,xei_,xie_,xii_=ahomo*aee/np.sqrt(N),ahomo*aei/np.sqrt(N),ahomo*aie/np.sqrt(N),ahomo*aii/np.sqrt(N)
gmat = np.array([[NE*xee_**2,NI*xei_**2],[NE*xie_**2,NI*xii_**2]])

eigvgm,eigvecgm=la.eig(gmat) 
r_g2 = np.max(eigvgm)
r_g  = np.sqrt(r_g2)
print(r_g)
eta=0
longaxis,shortaxis=(1+eta)*r_g ,(1-eta)*r_g 
xr = longaxis*np.cos(theta)
yr = shortaxis*np.sin(theta)
axtspt.plot(xr, yr, color="gray", linewidth=0.5,linestyle='--',label=r'ellipse') # >>>>>

#### @YX 2709 original one ----
xticks = [-1.5,0,1.0]
xlims  = [-2.0,1.5]
yticks = [-1.0,1.0]
ylims  = [-1.0,1.0]

axtspt.set_xlim(xlims)
axtspt.set_ylim(ylims)
axtspt.set_xticks(xticks)
axtspt.set_yticks(yticks)
axtspt.set_aspect('equal')   


# gmat = np.array([xee, xei, xie, xii])*gaverage
# gee, gei, gie, gii = gmat[0], gmat[1], gmat[2], gmat[3]
# if(len(Nparams) == 2):
#     NE, NI = Nparams[0], Nparams[1]
# else:
#     NE1, NE2, NI = Nparams[0], Nparams[1], Nparams[2]
#     NE = NE1+NE2
# N = NE+NI


# #### B. theo. latent dynamical variable kappa
# kappa_theo_iid = np.zeros((nJI, 3))
# for idxji in range(nJI):
#     gmat = np.array([xee,xei,xie,xii])#general case# np.array([gaverage, gaverage, gaverage, gaverage])
#     gmat = gmat * gaverage
#     init_k = -1.0#1.1*np.max(kappaintersect_R1[idxje, np.where(kappaintersect_Full[idxje, :,0] >= 0)[0], 0])
#     # kappa_theo_iid[idxje, 0] 
#     kappa_max= fsolve(iidperturbationP, init_k, args=(JE, JIseries[idxji], gmat, Nparams),xtol=1e-8,maxfev=1000)
#     residual0=np.abs(iidperturbationP(kappa_max, JE, JIseries[idxji], gmat, Nparams))
#     print('>>>>>>>>>residual:',residual0)
#     init_k = 0.5*init_k
#     # kappa_theo_iid[idxje, 1] 
#     kappa_middle= fsolve(iidperturbationP, init_k, args=(JE, JIseries[idxji], gmat, Nparams),xtol=1e-8,maxfev=1000)
#     residual1=np.abs(iidperturbationP(kappa_middle,JE, JIseries[idxji], gmat, Nparams))
#     init_k = 0.0
#     kappa_theo_iid[idxji, 2] = fsolve(iidperturbationP, init_k, args=(JE, JIseries[idxji], gmat, Nparams),xtol=1e-8,maxfev=1000)

#     if(residual0>1e-2):
#         kappa_theo_iid[idxji,0]=kappa_theo_iid[idxji,2]
#     else:
#         kappa_theo_iid[idxji,0]=kappa_max
#     if(residual1>1e-2):
#         kappa_theo_iid[idxji,1]=kappa_theo_iid[idxji,2]
#     else:
#         kappa_theo_iid[idxji,1]=kappa_middle

# ijesample=[1,5,9]
# yticks = np.linspace(-3.0, 3.0, 3)
# ylims = [-3.0, 3.0]#[1.1, 2.2]
# xticks = np.linspace(-1.5, 1.5, 3)
# xlims = [-1.5, 1.5]
# import matplotlib.cm as cm


# figE,axE=plt.subplots(figsize=(2,2))#,sharex=True,sharey=True,tight_layout=True)
# figI,axI=plt.subplots(figsize=(2,2))#,sharex=True,sharey=True,tight_layout=True)
# iktrial =-1
# for iii, idxje in enumerate(ijesample):
#     idrandsE     = np.random.choice(np.arange(0,NE),size=100,replace=False)
#     idrandsI     = np.random.choice(np.arange(NE,N),size=100,replace=False)
#     xactfull_E   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsE, -1])
#     xactfull_I   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsI, -1])
#     deltaxfull_E = xactfull_E - np.mean(xactfull_E)
#     deltaxfull_I = xactfull_I - np.mean(xactfull_I)

#     if (np.mean(xactfull_E)<0):
#         deltaxfull_E=-deltaxfull_E
#     if(np.mean(xactfull_I)<0):
#         deltaxfull_I=-deltaxfull_I

#     deltam_E = ReigvecTseries[idxje,iktrial,idrandsE,0]-np.mean(ReigvecTseries[idxje,iktrial,idrandsE,0])#xAm[0,0]
#     deltam_I = ReigvecTseries[idxje,iktrial,idrandsI,0]-np.mean(ReigvecTseries[idxje,iktrial,idrandsI,0])#xAm[-1,0]
#     axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c=cm.Reds((JIseries[idxje]-JIseries[0])/(JIseries[-1]-JIseries[0])),alpha=0.25)
#     axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c=cm.Blues((JIseries[idxje]-JIseries[0])/(JIseries[-1]-JIseries[0])),alpha=0.25)
#     ### predicted
#     deltaxpred_E = kappa_theo_iid[idxje,2]*np.array(xlims)
#     axE.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')
#     axI.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')

# yticks = np.linspace(-0.5, 0.5, 3)
# ylims = [-0.5, 0.5]#[1.1, 2.2]
# xticks = np.linspace(-1.5, 1.5, 3)
# xlims = [-1.5, 1.5]

# axE.set_xlim(xlims)
# axE.set_ylim(ylims)
# axE.set_xticks(xticks)
# axE.set_yticks(yticks)

# axE.spines['top'].set_color('none')
# axE.spines['right'].set_color('none')
# axE.xaxis.set_ticks_position('bottom')
# axE.spines['bottom'].set_position(('data',0))
# axE.yaxis.set_ticks_position('left')
# axE.spines['left'].set_position(('data',0))

# axI.set_xlim(xlims)
# axI.set_ylim(ylims)
# axI.set_xticks(xticks)
# axI.set_yticks(yticks)

# axI.spines['top'].set_color('none')
# axI.spines['right'].set_color('none')
# axI.xaxis.set_ticks_position('bottom')
# axI.spines['bottom'].set_position(('data',0))
# axI.yaxis.set_ticks_position('left')
# axI.spines['left'].set_position(('data',0))
    
    
    

# def list_to_dict(lst, string):
#     """
#     Transform a list of variables into a dictionary.
#     Parameters
#     ----------
#     lst : list
#         list with all variables.
#     string : str
#         string containing the names, separated by commas.
#     Returns
#     -------
#     d : dict
#         dictionary with items in which the keys and the values are specified
#         in string and lst values respectively.
#     """
#     string = string[0]
#     string = string.replace(']', '')
#     string = string.replace('[', '')
#     string = string.replace('\\', '')
#     string = string.replace(' ', '')
#     string = string.replace('\t', '')
#     string = string.replace('\n', '')
#     string = string.split(',')
#     d = {s: v for s, v in zip(string, lst)}
#     return d

# lst = [kappaintersect_Full, kappaintersect_R1,
#         xfpseries_Full[:,:,:,-1], xfpseries_R1[:,:,:,-1],
#         ReigvecTseries[:,:,:,0], LeigvecTseries[:,:,:,0]]
# stg = ["kappaintersect_Full, kappaintersect_R1,"
#         "xfpseries_Full, xfpseries_R1, "
#         "ReigvecTseries, LeigvecTseries"]
# data = list_to_dict(lst=lst, string=stg)
# data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_inh_Response.npz"
# np.savez(data_name, **data)