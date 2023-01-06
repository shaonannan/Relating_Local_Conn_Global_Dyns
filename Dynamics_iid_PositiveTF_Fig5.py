# -*- coding: utf-8 -*-
"""
@author: Yuxiu Shao
Network with independent random components;
Full-rank Gaussian Network and Rank-one Mixture of Gaussian Approximation
Dynamics
"""

"""
Help Functions
"""
import numpy as np
import matplotlib.pylab as plt

# import seaborn as sb
from functools import partial
# from sympy import *
extras_require = {'PLOT':['matplotlib>=1.1.1,<3.0']},
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

### import SELF DEFINED functions
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
shiftx  =  1.5 ### param for positive transfer function

RERUN = 1 ### 0 rerun, 1 reloading data saved
if RERUN ==0:
    data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_iid_PositiveTF_0_4VS1.npz"
    # np.savez(data_name, **data)
    data      = np.load(data_name)


JI    = 0.6 ### redefined 
Nt    = np.array([1200,300])#([750, 750])
NE,NI = Nt[0], Nt[1]
N     = NE+NI
Nparams  = np.array([NE, NI])
Npercent = Nparams/N
nrank, ntrial, neta, nvec, nmaxFP = 1,30, 1, 2, 3
cuttrials = int(ntrial/2)
nJE = 20#10
JEseries = np.linspace(1.5, 2.4, nJE) # 1.2 2.1


### network settings 
# gaverage = 0.001 # no random component#
gaverage = 0.8   # effects of random variances
### original 0.001#0.6

# xee,xei,xie,xii=1.0,1.0,1.0,1.0 # homogeneous variance 
xee,xei,xie,xii=1.0,0.5,0.2,0.8 # heterogeneous variance


coeffetaEsub    = np.array([1.0, 1.0, 1.0])
coeffetaTotal   = np.array([1.0, 1.0, 1.0])
# coeffetaTotal = np.zeros(3)
signetaEsub     = np.ones(3)
signetaTotal    = np.ones(3)

ppercentEsub    = np.ones(2) # multiple excitatory subpopulations
ppercentEsub[0] = 0.5
ppercentEsub[1] = 1.0-ppercentEsub[0]
# E->total ppercentEsub[0]/2.0,ppercentEsub[1]/2.0, (I) 1/2.0

### recording variables 
Reigvecseries, Leigvecseries   = np.zeros((nJE, ntrial, N, nvec*2)), np.zeros((nJE, ntrial, N, nvec*2))
ReigvecTseries, LeigvecTseries = np.zeros((nJE, ntrial, N, nvec*2)), np.zeros((nJE, ntrial, N, nvec*2))
if(RERUN==0):
    ReigvecTseries[:,:,:,0], LeigvecTseries[:,:,:,0] = data['ReigvecTseries'],data['LeigvecTseries']
Beigvseries = np.zeros((nJE, ntrial, N), dtype=complex)
#### statistical properties of the elements on the rank one eigenvectors 
armu, sigrcov = np.zeros((nJE, ntrial, 2), dtype=complex), np.zeros((nJE, ntrial, 2), dtype=complex)  # 2 for E and I
almu, siglcov = np.zeros((nJE, ntrial, 2), dtype=complex), np.zeros((nJE, ntrial, 2), dtype=complex)
siglr = np.zeros((nJE, ntrial, 2), dtype=complex)
# reference, eigenvectors of matrix with iid perturbations
x0series, y0series = np.zeros((nJE, ntrial, N), dtype=complex), np.zeros((nJE, ntrial, N), dtype=complex)


#### dynamical parameters 
tt  = np.linspace(0, 500, 500)
dt  = tt[2]-tt[1]
ntt = len(tt)
xfpseries_Full = np.zeros((nJE, ntrial, N, ntt))
xfpseries_R1   = np.zeros((nJE, ntrial, N, ntt))
# @YX -- temporal evolution of kappa 
kappaseries_R1 = np.zeros((nJE, ntrial, ntt))
#### fixed point of kappa
kappaintersect_Full = np.zeros((nJE, ntrial, nmaxFP*2))
kappaintersect_R1   = np.zeros((nJE, ntrial, nmaxFP*2))

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
        for idxje, JEv in enumerate(JEseries):
            JI, JE, a, b = 0.6, JEv, 0, 0
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
            #### iid random component
            eigvJ, leigvec, reigvec, xnorm0, ynorm0 = decompNormalization(
                J, xAm, yAm, xAm, yAm, nparams=Nparams, sort=0, nrank=1)
            Reigvecseries[idxje, iktrial, :, 0], Leigvecseries[idxje,iktrial, :, 0] = xnorm0[:, 0].copy(), ynorm0[:, 0].copy()
            Beigvseries[idxje, iktrial, :] = eigvJ.copy()

            axrmu, aylmu, sigxr, sigyl, sigcov = numerical_stats(xnorm0, ynorm0, xAm, yAm, eigvJ, nrank, 2, ppercent=np.array([0.5, 0.5]))
            armu[idxje, iktrial, :], almu[idxje,iktrial, :] = axrmu[:, 0], aylmu[:, 0]
            sigrcov[idxje, iktrial, :], siglcov[idxje,iktrial, :] = sigxr[:, 0], sigyl[:, 0]
            siglr[idxje, iktrial, :] = sigcov[:, 0, 0]

            # Rank One Approximation, Nullclines of \kappa
            xnorm0, ynorm0 = Reigvecseries[idxje, iktrial, :, 0].copy(), Leigvecseries[idxje, iktrial, :, 0]
            xnorm0, ynorm0 = np.reshape(xnorm0, (N, 1)), np.reshape(ynorm0, (N, 1))
            xmu, ymu = armu[idxje, iktrial, :].copy(), almu[idxje,iktrial, :].copy()
            xsig, ysig = sigrcov[idxje, iktrial, :].copy(), siglcov[idxje, iktrial, :].copy()
            yxcov = siglr[idxje, iktrial, :].copy()

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
            # # ## normalize ### not necessary
            # projection = np.reshape(xnormt, (1, N))@np.reshape(ynormt, (N, 1))
            # ynormt     = ynormt/projection*Beigvseries[idxje, iktrial, 0]*N
            # # ---------------------------------------------------------------
            # _, _, xnormt, ynormt = Normalization(xnormt.copy(), ynormt.copy(), xAm.copy(), yAm.copy(), nparams=Nparams, sort=0, nrank=1)
            ReigvecTseries[idxje, iktrial, :, 0], LeigvecTseries[idxje,iktrial, :, 0] = xnormt[:, 0].copy(), ynormt[:, 0].copy()

            # perturbation result
            if(1):
                kappanum[0] = np.squeeze(np.reshape(ynormt,(1,N))@np.reshape((1.0+np.tanh(xact-shiftx)),(N,1)))/N
            else:
                mvec_norm   = np.reshape(xnormt,(N,1))
                kappanum[0] = np.squeeze(np.reshape(np.squeeze(xact),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            kappaintersect_Full[idxje, iktrial, :3] = kappanum[:].copy()

            r1Matt = np.real(xnormt@ynormt.T)
            r1Matt = r1Matt/N
            # use the same initial values
            xtemporal = odesimulationP(tt, xinit, r1Matt, 0)
            xfpseries_R1[idxje, iktrial, :, :] = xtemporal.T.copy()
            # @YX --- ADD "Evolution of \kappa(t)", kappa is the projection of x on m x.Tm = \kappa m.Tm=\kappa |m|2
            kappaseries_R1[idxje, iktrial, :] = np.squeeze(xtemporal.copy()@np.reshape(xnormt, (-1, 1)))/np.sum(xnormt**2)
            #### kappa's dynamcis
            # 2 populations
            kappanum = np.zeros(3)
            xact     = np.squeeze(xfpseries_R1[idxje, iktrial, :, -1])
            # use yAm -- unperturbed centre.
            if(1):
                # do not change
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
    for idx, JEU in enumerate(xintersect):
        for idxx, x in enumerate(kappa_x):
            muphi, delta0phiE, delta0phiI = x, x**2 * \
                (gee**2*NE/N+gei**2*NI/N)/(JEU-JI)**2, x**2 * \
                (gie**2*NE/N+gii**2*NI/N)/(JEU-JI)**2
            Sx[idx, idxx] = -x+(JEU*PhiP(muphi, delta0phiE) - JI*PhiP(muphi, delta0phiI))
            F0[idx,idxx]  = x 
            F1[idx,idxx]  = (JEU*PhiP(muphi, delta0phiE) -JI*PhiP(muphi, delta0phiI))
    fig, ax = plt.subplots(figsize=(4, 4))
    xticks  = [-1,0,4]#np.linspace(-1, 4, 3)
    xlims   = [-1, 4]
    yticks  = [-1,0,1]#np.linspace(-1.0, 1.0, 3)
    ylims   = [-1.0, 1.0]
    # ax.plot(kappa_x,np.zeros_like(kappa_x),c='k',lw=1.0)
    ax.plot(kappa_x, Sx[0, :], c='gray', lw=1)
    ax.plot(kappa_x, Sx[1, :], c='tab:purple', lw=1.0)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # # ax.set_aspect('equal')
    

    fig, ax = plt.subplots(figsize=(4, 4))
    xticks  = [-1,0,4]#np.linspace(-1, 5, 2)
    xlims   = [-1, 4]
    yticks  = [-1,0,4]#np.linspace(-1., 5, 2)
    ylims   = [-1.0,4]
    # ax.plot(kappa_x,np.zeros_like(kappa_x),c='k',lw=1.0)
    ax.plot(kappa_x, F0[0, :], c='gray', lw=1)
    ax.plot(kappa_x, F1[0, :], c='tab:purple',linestyle='--',alpha=0.75)
    ax.plot(kappa_x, F1[1, :], c='tab:purple')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # # ax.set_aspect('equal')
    

    #### B. theo. latent dynamical variable kappa
    kappa_theo_iid = np.zeros((nJE, 3))
    for idxje in range(nJE):
        gmat = np.array([xee,xei,xie,xii])#general case# np.array([gaverage, gaverage, gaverage, gaverage])
        gmat = gmat * gaverage
        init_k = 3
        kappa_max= fsolve(iidperturbationP, init_k, args=(JEseries[idxje], JI, gmat, Nparams),xtol=1e-8,maxfev=1000)
        residual0=np.abs(iidperturbationP(kappa_max,JEseries[idxje], JI, gmat, Nparams))
        init_k = 0.5*init_k
        kappa_middle= fsolve(iidperturbationP, init_k, args=(JEseries[idxje], JI, gmat, Nparams),xtol=1e-8,maxfev=1000)
        residual1=np.abs(iidperturbationP(kappa_middle,JEseries[idxje], JI, gmat, Nparams))
        init_k = 0.0
        kappa_theo_iid[idxje, 2] = fsolve(iidperturbationP, init_k, args=(JEseries[idxje], JI, gmat, Nparams),xtol=1e-8,maxfev=1000)

        if(residual0>1e-2):
            kappa_theo_iid[idxje,0]=kappa_theo_iid[idxje,2]
        else:
            kappa_theo_iid[idxje,0]=kappa_max
        if(residual1>1e-2):
            kappa_theo_iid[idxje,1]=kappa_theo_iid[idxje,2]
        else:
            kappa_theo_iid[idxje,1]=kappa_middle

    clrs = ['k','b','r']
    mean_pos_kappa_full, mean_neg_kappa_full = np.zeros(nJE), np.zeros(nJE)
    std_pos_kappa_full, std_neg_kappa_full = np.zeros(nJE), np.zeros(nJE)
    pos_t0=0.7

    fig, ax = plt.subplots(figsize=(5, 3))

    xticks = np.linspace(JEseries[0],JEseries[-1],2)#np.linspace(1.2, 2.0, 2)
    xlims  = [JEseries[0],JEseries[-1]]#[1.1, 2.2]
    yticks = np.linspace(0.0, 4.0, 2)
    ylims  = [0.0, 4.0]
    
    ax.plot(JEseries, kappa_theo_iid[:, 0],c='tab:purple', linewidth=1.5)
    ax.plot(JEseries, kappa_theo_iid[:, 2],c='tab:purple', linewidth=1.5)
    ax.plot(JEseries, kappa_theo_iid[:, 1],c='tab:purple', linewidth=1.5)



    nkappa,nJEs  = 100,100
    kappa_line = np.linspace(yticks[0],yticks[-1],nkappa)
    JE_lin     = np.linspace(JEseries[0],JEseries[-1],nJEs)
    kappas_show=np.zeros((nJEs,nkappa))

    gmat = np.array([xee,xei,xie,xii])
    gmat = gmat * gaverage

    for idxje in range(nJEs):
        for idxkappa,kappamesh in enumerate(kappa_line):
            kappas_show[idxje,idxkappa]=np.abs(iidperturbationP(kappamesh,JE_lin[idxje], JI, gmat, Nparams))
    ax.imshow(kappas_show.T,extent = [xlims[0] , xlims[-1], ylims[-1] , ylims[0]],cmap='summer',vmin=0,vmax=0.5)
    ax.set_aspect('auto')

    for idxje in range(nJE):
        pos_full = kappaintersect_Full[idxje, np.where(
            kappaintersect_Full[idxje, :, 0] >= pos_t0)[0], 0]
        mean_pos_kappa_full[idxje] = np.mean(pos_full)
        std_pos_kappa_full[idxje] = np.std(pos_full)

    low_bound = np.zeros_like(mean_pos_kappa_full)
    for imax in range(len(mean_pos_kappa_full)):
        low_bound[imax]=max(0,mean_pos_kappa_full[imax]-std_pos_kappa_full[imax])
    ax.fill_between(JEseries, mean_pos_kappa_full+std_pos_kappa_full,low_bound, alpha=0.5, facecolor='black')
    

    mean_pos_kappa_full, mean_neg_kappa_full = np.zeros(nJE), np.zeros(nJE)
    std_pos_kappa_full, std_neg_kappa_full = np.zeros(nJE), np.zeros(nJE)

    for idxje in range(nJE):
        pos_full = kappaintersect_Full[idxje, np.where(
            kappaintersect_Full[idxje, :, 0] < pos_t0)[0], 0]
        mean_pos_kappa_full[idxje] = np.mean(pos_full)
        std_pos_kappa_full[idxje]  = np.std(pos_full)

    low_bound = np.zeros_like(mean_pos_kappa_full)
    for imax in range(len(mean_pos_kappa_full)):
        low_bound[imax]=max(0,mean_pos_kappa_full[imax]-std_pos_kappa_full[imax])
    ax.fill_between(JEseries, mean_pos_kappa_full+std_pos_kappa_full,low_bound, alpha=0.5, facecolor='black')

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    #### numerical results
    numkappa= np.zeros((nJE,ntrial))
    pavgkappa,pstdkappa = np.zeros(nJE),np.zeros(nJE)
    navgkappa,nstdkappa = np.zeros(nJE),np.zeros(nJE)
    for iJE in range(nJE):
        signXE = np.mean(1.0+np.tanh(np.squeeze(xfpseries_Full[iJE,:,:NE,-1])-shiftx),axis=1)
        signXI = np.mean(1.0+np.tanh(np.squeeze(xfpseries_Full[iJE,:,NE:,-1])-shiftx),axis=1)
        ptrialXE,ntrialXE = np.where(signXE>=pos_t0)[0],np.where(signXE<pos_t0)[0]
        print('>>>P/N TRIAL',(ptrialXE),ntrialXE)
        ### CALCULATE [PHIXE/I]
        if len(ptrialXE)>0:
            for iktrial in ptrialXE:
                mvec_norm = np.reshape(ReigvecTseries[iJE,iktrial,:,0],(N,1))
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(np.squeeze(xfpseries_Full[iJE,iktrial,:,-1]),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            pavgkappa[iJE],pstdkappa[iJE] = np.mean(numkappa[iJE,ptrialXE]),np.std(numkappa[iJE,ptrialXE])
        if len(ntrialXE)>0:
            for iktrial in ntrialXE:
                mvec_norm = np.reshape(ReigvecTseries[iJE,iktrial,:,0],(N,1))
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(np.squeeze(xfpseries_Full[iJE,iktrial,:,-1]),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            navgkappa[iJE],nstdkappa[iJE] = np.mean(numkappa[iJE,ntrialXE]),np.std(numkappa[iJE,ntrialXE])

    ax.plot(JEseries, pavgkappa,c='black', linewidth=1.5, linestyle='--')
    ax.plot(JEseries, navgkappa,c='black', linewidth=1.5, linestyle='--')


    ### calculate mean and variance of the synaptic input x, for each realization
    import matplotlib.patches as mpatches
    variance_x_num, variance_kappa_num = np.zeros(
        (nJE, ntrial, 2)), np.zeros((nJE, ntrial, 2))
    mu_x_num, mu_kappa_num = np.zeros((nJE, ntrial, 2)), np.zeros((nJE, ntrial, 2))
    variance_full_theo, variance_R1_theo = np.zeros(
        (nJE, ntrial, 3,2)), np.zeros((nJE, ntrial, 3,2))
    mu_full_theo, mu_R1_theo = np.zeros(
        (nJE, ntrial,3, 2)), np.zeros((nJE, ntrial, 3, 2))
    for idxje in range(nJE):
        for iktrial in range(ntrial):
            # numerical results for Full Mat
            variance_x_num[idxje, iktrial, 0], variance_x_num[idxje, iktrial, 1] = np.std(
                xfpseries_Full[idxje, iktrial, :NE, -1])**2, np.std(xfpseries_Full[idxje, iktrial, NE:, -1])**2
            mu_x_num[idxje, iktrial, 0], mu_x_num[idxje, iktrial, 1] = np.mean(
                xfpseries_Full[idxje, iktrial, :NE, -1]), np.mean(xfpseries_Full[idxje, iktrial, NE:, -1])
            # numerical results for Rank one Appriximation Mat
            variance_kappa_num[idxje, iktrial, 0], variance_kappa_num[idxje, iktrial, 1] = np.std(
                xfpseries_R1[idxje, iktrial, :NE, -1])**2, np.std(xfpseries_R1[idxje, iktrial, NE:, -1])**2
            mu_kappa_num[idxje, iktrial, 0], mu_kappa_num[idxje, iktrial, 1] = np.mean(
                xfpseries_R1[idxje, iktrial, :NE, -1]), np.mean(xfpseries_R1[idxje, iktrial, NE:, -1])

    for idxje in range(nJE):
        # ## theoretical
        gmat = np.array([xee,xei,xie,xii])#general case# np.array([gaverage, gaverage, gaverage, gaverage])
        gmat = gmat *gaverage
        gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
        deltainitE, meaninitE = np.max(variance_x_num[idxje, :, 0]), kappa_theo_iid[idxje, 0]
        # np.mean(kappaintersect[idxg,:,0])
        deltainitI, meaninitI = np.max(variance_x_num[idxje, :, 1]), kappa_theo_iid[idxje, 0]
        INIT = [meaninitE, meaninitI, deltainitE, deltainitI]
        statsfull = fsolve(iidfull_mudelta_consistencyP, INIT,args=(JEseries[idxje], JI, gmat, Nparams),xtol=1e-6)
        mu_full_theo[idxje, :,0, 0], variance_full_theo[idxje,:,0, 0] = statsfull[0], statsfull[2]
        mu_full_theo[idxje, :,0, 1], variance_full_theo[idxje,:,0, 1] = statsfull[1], statsfull[3]

        deltainitE, meaninitE = np.min(variance_x_num[idxje, :, 0]), kappa_theo_iid[idxje, 2]
        # np.mean(kappaintersect[idxg,:,0])
        deltainitI, meaninitI = np.min(variance_x_num[idxje, :, 1]), kappa_theo_iid[idxje, 2]
        INIT = [meaninitE, meaninitI, deltainitE, deltainitI]

        statsfull = fsolve(iidfull_mudelta_consistencyP, INIT,args=(JEseries[idxje], JI, gmat, Nparams),xtol=1e-6)
        mu_full_theo[idxje, :,2, 0], variance_full_theo[idxje,:,2, 0] = statsfull[0], statsfull[2]
        mu_full_theo[idxje, :,2, 1], variance_full_theo[idxje,:,2, 1] = statsfull[1], statsfull[3]

        mu_R1_theo[idxje, :,0, 0], variance_R1_theo[idxje, :, 0,0] = kappa_theo_iid[idxje,0], kappa_theo_iid[idxje, 0]**2*(gee**2*NE/N+gei**2*NI/N)/(JEseries[idxje]-JI)**2
        mu_R1_theo[idxje, :,0, 1], variance_R1_theo[idxje, :,0, 1] = kappa_theo_iid[idxje,0], kappa_theo_iid[idxje, 0]**2*(gie**2*NE/N+gii**2*NI/N)/(JEseries[idxje]-JI)**2

        mu_R1_theo[idxje, :,2, 0], variance_R1_theo[idxje, :,2, 0] = kappa_theo_iid[idxje,2], kappa_theo_iid[idxje, 2]**2*(gee**2*NE/N+gei**2*NI/N)/(JEseries[idxje]-JI)**2
        mu_R1_theo[idxje, :, 2,1], variance_R1_theo[idxje, :,2, 1] = kappa_theo_iid[idxje,2], kappa_theo_iid[idxje, 2]**2*(gie**2*NE/N+gii**2*NI/N)/(JEseries[idxje]-JI)**2

    #### D. fill-between kappa positive and negative
    # Figures reflecting how mean and variance change with random gain of iid Gaussian matrix
    clrs = 'kbr'
    fig, ax2 = plt.subplots(2, 1, figsize=(5, 5), sharex=True,tight_layout=True)
    # -----------------  mean act------------------------------------------------------------
    pmean_x_numE, pmean_kappa_numE, pmean_x_numI, pmean_kappa_numI = np.zeros(nJE), np.zeros(nJE), np.zeros(nJE), np.zeros(nJE)
    pstd_x_numE, pstd_kappa_numE, pstd_x_numI, pstd_kappa_numI = np.zeros(nJE), np.zeros(nJE), np.zeros(nJE), np.zeros(nJE)

    nmean_x_numE, nmean_kappa_numE, nmean_x_numI, nmean_kappa_numI = np.zeros(nJE), np.zeros(nJE), np.zeros(nJE), np.zeros(nJE)
    nstd_x_numE, nstd_kappa_numE, nstd_x_numI, nstd_kappa_numI = np.zeros(nJE), np.zeros(nJE), np.zeros(nJE), np.zeros(nJE)

    pos_t0=0.7
    for i in range(nJE):
        pmean_x_numE[i], pmean_kappa_numE[i] = np.mean(mu_x_num[i, np.where(mu_x_num[i, :, 0] <pos_t0)[0], 0]), np.mean(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 0] <pos_t0)[0], 0])
        pstd_x_numE[i], pstd_kappa_numE[i] = np.std(mu_x_num[i, np.where(mu_x_num[i, :, 0] <pos_t0)[0], 0]), np.std(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 0]  <pos_t0)[0], 0])

        pmean_x_numI[i], pmean_kappa_numI[i] = np.mean(mu_x_num[i, np.where(mu_x_num[i, :, 1] <pos_t0)[0], 1]), np.mean(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 1] <pos_t0)[0], 1])
        pstd_x_numI[i], pstd_kappa_numI[i] = np.std(mu_x_num[i, np.where(mu_x_num[i, :, 1] <pos_t0)[0], 1]), np.std(mu_kappa_num[i, np.where(mu_kappa_num[i,:, 1] <pos_t0)[0], 1])
    ### keep the mean values ~~~~~~~~~~~~~~~~~
    low_bound = np.zeros_like(pmean_x_numE)
    for imax in range(len(pmean_x_numE)):
        low_bound[imax]=max(0,pmean_x_numE[imax]-pstd_x_numE[imax])
    ax2[0].fill_between(JEseries, pmean_x_numE+pstd_x_numE,low_bound, alpha=0.3, facecolor='tab:red')
    low_bound = np.zeros_like(pmean_x_numI)
    for imax in range(len(pmean_x_numI)):
        low_bound[imax]=max(0,pmean_x_numI[imax]-pstd_x_numI[imax])
    ax2[1].fill_between(JEseries, pmean_x_numI+pstd_x_numI,low_bound, alpha=0.3, facecolor='tab:blue')

    for i in range(nJE):
        pmean_x_numE[i], pmean_kappa_numE[i] = np.mean(mu_x_num[i, np.where(mu_x_num[i, :, 0]  >=pos_t0), 0]), np.mean(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 0] >=pos_t0)[0], 0])
        pstd_x_numE[i], pstd_kappa_numE[i] = np.std(mu_x_num[i, np.where(mu_x_num[i, :, 0] >=pos_t0), 0]), np.std(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 0] >=pos_t0)[0], 0])

        pmean_x_numI[i], pmean_kappa_numI[i] = np.mean(mu_x_num[i, np.where(mu_x_num[i, :, 1] >=pos_t0), 1]), np.mean(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 1] >=pos_t0)[0], 1])
        pstd_x_numI[i], pstd_kappa_numI[i] = np.std(mu_x_num[i, np.where(mu_x_num[i, :, 1] >=pos_t0), 1]), np.std(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 1] >=pos_t0)[0], 1])
    
    ### keep the mean values ~~~~~~~~~~~~~~~~~
    low_bound = np.zeros_like(pmean_x_numE)
    for imax in range(len(pmean_x_numE)):
        low_bound[imax]=max(0,pmean_x_numE[imax]-pstd_x_numE[imax])
    ax2[0].fill_between(JEseries, pmean_x_numE+pstd_x_numE,low_bound, alpha=0.3, facecolor='tab:red')
    low_bound = np.zeros_like(pmean_x_numI)
    for imax in range(len(pmean_x_numI)):
        low_bound[imax]=max(0,pmean_x_numI[imax]-pstd_x_numI[imax])
    ax2[1].fill_between(JEseries, pmean_x_numI+pstd_x_numI,low_bound, alpha=0.3, facecolor='tab:blue')

    #### theoretical solutions of kappa
    ax2[0].plot(JEseries, np.mean(mu_R1_theo[:, :, 0,0], axis=1), color='tab:red',linewidth=1.5, label=r'$\mu_{R1}^E$ theo.')
    ax2[1].plot(JEseries, np.mean(mu_R1_theo[:, :, 0,1], axis=1), color='tab:blue',linewidth=1.5, label=r'$\mu_{R1}^I$ theo.')
    ax2[0].plot(JEseries, np.mean(mu_R1_theo[:, :, 2,0], axis=1), color='tab:red',linewidth=1.5, label=r'$\mu_{R1}^E$ theo.')
    ax2[1].plot(JEseries, np.mean(mu_R1_theo[:, :, 2,1], axis=1), color='tab:blue',linewidth=1.5, label=r'$\mu_{R1}^I$ theo.')

    ax2[0].plot(JEseries, np.mean(mu_full_theo[:, :, 0,0], axis=1), color='tab:red',linewidth=1.5, alpha=0.25)
    ax2[1].plot(JEseries, np.mean(mu_full_theo[:, :, 0,1], axis=1), color='tab:blue',linewidth=1.5, alpha=0.25)
    ax2[0].plot(JEseries, np.mean(mu_full_theo[:, :, 2,0], axis=1), color='tab:red',linewidth=1.5, alpha=0.25)
    ax2[1].plot(JEseries, np.mean(mu_full_theo[:, :, 2,1], axis=1), color='tab:blue',linewidth=1.5, alpha=0.25)

    xticks = np.linspace(1.2, 2.0, 2)
    xlims = [1.2, 2.0]#[1.1, 2.2]
    yticks = [-1,0,1]#np.linspace(-1.0, 1.0, 2)
    ylims = [-1., 1.]

    # for i in range(2):
    #     ax2[i].set_xlim(xlims)
    #     ax2[i].set_ylim(ylims)
    #     ax2[i].set_xticks(xticks)
    #     ax2[i].set_yticks(yticks)
    #     # ax2[i].legend()

    #### statistical properties of synaptic input x
    fig, ax2   = plt.subplots(2, 1, figsize=(5, 5),sharex=True,tight_layout=True)
    fige, ax2e = plt.subplots(2, 1, figsize=(5, 5),sharex=True,tight_layout=True) ### ERROR BAR
    # -----------------  mean act------------------------------------------------------------
    pmean_x_numE, pmean_x_numI= np.zeros(nJE), np.zeros(nJE)
    pstd_x_numE, pstd_x_numI  = np.zeros(nJE), np.zeros(nJE)
    pos_t0 = 0.15
    pos_t1 = 0.05


    for i in range(nJE):
        pmean_x_numE[i] = np.mean(variance_x_num[i,np.where(variance_x_num[i, :, 0]>= pos_t0)[0], 0])
        pstd_x_numE[i]  = np.std(variance_x_num[i,np.where(variance_x_num[i, :, 0] >= pos_t0)[0], 0])

        pmean_x_numI[i] = np.mean(variance_x_num[i,np.where(variance_x_num[i, :, 1]>= pos_t1)[0], 1])
        pstd_x_numI[i]  = np.std(variance_x_num[i,np.where(variance_x_num[i, :, 1] >= pos_t1)[0], 1])
    ax2[0].plot(JEseries, pmean_x_numE, color='tab:red',linewidth=1.5, linestyle='--')
    ax2[1].plot(JEseries, pmean_x_numI, color='tab:blue',linewidth=1.5, linestyle='--')

    low_bound = np.zeros_like(pmean_x_numE)
    for imax in range(len(pmean_x_numE)):
        low_bound[imax]=max(0,pmean_x_numE[imax]-pstd_x_numE[imax])
    ax2[0].fill_between(JEseries, pmean_x_numE+pstd_x_numE,low_bound, alpha=0.3, facecolor='tab:red')
    low_bound = np.zeros_like(pmean_x_numI)
    for imax in range(len(pmean_x_numI)):
        low_bound[imax]=max(0,pmean_x_numI[imax]-pstd_x_numI[imax])
    ax2[1].fill_between(JEseries, pmean_x_numI+pstd_x_numI,low_bound, alpha=0.3, facecolor='tab:blue')
    ### error bar >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ax2e[0].errorbar(JEseries, pmean_x_numE,pstd_x_numE/np.sqrt(ntrial),c='tab:red')
    ax2e[1].errorbar(JEseries, pmean_x_numI,pstd_x_numI/np.sqrt(ntrial), c='tab:blue')

    ### >>>>>>>>>>>>>> second fixed point >>>>>>>>>>>
    for i in range(nJE):
        pmean_x_numE[i] = np.mean(variance_x_num[i,np.where(variance_x_num[i,:, 0]< pos_t0)[0], 0])
        pstd_x_numE[i]  = np.std(variance_x_num[i,np.where(variance_x_num[i,:, 0]<pos_t0)[0], 0])
        pmean_x_numI[i] = np.mean(variance_x_num[i,np.where(variance_x_num[i, :, 1]<pos_t1)[0], 1])
        pstd_x_numI[i]  = np.std(variance_x_num[i,np.where(variance_x_num[i, :, 1]<pos_t1)[0], 1])
    ax2[0].plot(JEseries, pmean_x_numE, color='tab:red',linewidth=1.5, linestyle='--')
    ax2[1].plot(JEseries, pmean_x_numI, color='tab:blue',linewidth=1.5, linestyle='--')
    low_bound = np.zeros_like(pmean_x_numE)
    for imax in range(len(pmean_x_numE)):
        low_bound[imax]=max(0,pmean_x_numE[imax]-pstd_x_numE[imax])
    ax2[0].fill_between(JEseries, pmean_x_numE+pstd_x_numE,low_bound, alpha=0.3, facecolor='tab:red')
    low_bound = np.zeros_like(pmean_x_numI)
    for imax in range(len(pmean_x_numI)):
        low_bound[imax]=max(0,pmean_x_numI[imax]-pstd_x_numI[imax])
    ax2[1].fill_between(JEseries, pmean_x_numI+pstd_x_numI,low_bound, alpha=0.3, facecolor='tab:blue')
    ### error bar >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ax2e[0].errorbar(JEseries, pmean_x_numE,pstd_x_numE/np.sqrt(ntrial),c='tab:red')
    ax2e[1].errorbar(JEseries, pmean_x_numI,pstd_x_numI/np.sqrt(ntrial), c='tab:blue')


    ax2[0].plot(JEseries, np.mean(variance_R1_theo[:, :, 0,0], axis=1), color='tab:red',linewidth=1.5, label=r'$\Delta_{R1}^E$ theo.')

    ax2[1].plot(JEseries, np.mean(variance_R1_theo[:, :, 0,1], axis=1), color='tab:blue',linewidth=1.5, label=r'$\Delta_{R1}^I$ theo.')
    ax2[0].plot(JEseries, np.mean(variance_R1_theo[:, :, 2,0], axis=1), color='tab:red',linewidth=1.5, label=r'$\Delta_{R1}^E$ theo.')

    ax2[1].plot(JEseries, np.mean(variance_R1_theo[:, :, 2,1], axis=1), color='tab:blue',linewidth=1.5, label=r'$\Delta_{R1}^I$ theo.')

    ax2e[0].plot(JEseries, np.mean(variance_R1_theo[:, :, 0,0], axis=1), color='tab:red',linewidth=1.5, label=r'$\Delta_{R1}^E$ theo.')

    ax2e[1].plot(JEseries, np.mean(variance_R1_theo[:, :, 0,1], axis=1), color='tab:blue',linewidth=1.5, label=r'$\Delta_{R1}^I$ theo.')
    ax2e[0].plot(JEseries, np.mean(variance_R1_theo[:, :, 2,0], axis=1), color='tab:red',linewidth=1.5, label=r'$\Delta_{R1}^E$ theo.')

    ax2e[1].plot(JEseries, np.mean(variance_R1_theo[:, :, 2,1], axis=1), color='tab:blue',linewidth=1.5, label=r'$\Delta_{R1}^I$ theo.')

    xticks = np.linspace(JEseries[0],JEseries[-1],2)#np.linspace(1.2, 2.0, 2)
    xlims  = [JEseries[0],JEseries[-1]]#[1.2, 2.0]#[1.1, 2.2]
    # yticks = np.linspace(.0, 1.5, 2)
    # ylims  = [0., 1.5]
    ### ---4VS1----
    yticks = np.linspace(.0, 2.0, 2)
    ylims  = [0., 2.0]

    # for i in range(2):
    #     ax2[i].set_xlim(xlims)
    #     ax2[i].set_ylim(ylims)
    #     ax2[i].set_xticks(xticks)
    #     ax2[i].set_yticks(yticks)

    # for i in range(2):
    #     ax2e[i].set_xlim(xlims)
    #     ax2e[i].set_ylim(ylims)
    #     ax2e[i].set_xticks(xticks)
    #     ax2e[i].set_yticks(yticks)

    #### ----- 4VS1 -----
    ax2[0].set_xlim(xlims)
    ax2[0].set_ylim(ylims)
    ax2[0].set_xticks(xticks)
    ax2[0].set_yticks(yticks)

    ax2[1].set_ylim([0,0.50])
    ax2[1].set_yticks([0,0.50])


    ### adding comparison between --> fluctuations in connectivity lead to fluctuations in dynamic
    ijesample=[19]
    yticks = np.linspace(-3.0, 3.0, 3)
    ylims = [-3.0, 3.0]#[1.1, 2.2]
    xticks = np.linspace(-1.5, 1.5, 3)
    xlims = [-1.5, 1.5]

    # figE,axE=plt.subplots(figsize=(2,4))#,sharex=True,sharey=True,tight_layout=True)
    # figI,axI=plt.subplots(figsize=(2,4))#,sharex=True,sharey=True,tight_layout=True)
    # # iktrial =0
    # for iii, idxje in enumerate(ijesample):
    #     iktrial=0
    #     idrandsE     = np.random.choice(np.arange(0,NE),size=100,replace=False)
    #     idrandsI     = np.random.choice(np.arange(NE,N),size=100,replace=False)
    #     xactfull_E   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsE, -1])
    #     xactfull_I   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsI, -1])
    #     deltaxfull_E = xactfull_E - np.mean(xactfull_E)
    #     deltaxfull_I = xactfull_I - np.mean(xactfull_I)

    #     # if (np.mean(xactfull_E)<0):
    #     #     deltaxfull_E=-deltaxfull_E
    #     # if(np.mean(xactfull_I)<0):
    #     #     deltaxfull_I=-deltaxfull_I

    #     deltam_E = ReigvecTseries[idxje,iktrial,idrandsE,0]#-xAm[0,0]
    #     deltam_E = deltam_E - np.mean(deltam_E)
    #     deltam_I = ReigvecTseries[idxje,iktrial,idrandsI,0]#-xAm[-1,0]
    #     deltam_I = deltam_I - np.mean(deltam_I)
    #     # axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c=cm.Reds((JEseries[idxje]-JEseries[0])/(JEseries[-1]-JEseries[0])),alpha=0.75)
    #     # axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c=cm.Blues((JEseries[idxje]-JEseries[0])/(JEseries[-1]-JEseries[0])),alpha=0.75)
    #     axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c='tab:red',alpha=0.75)
    #     axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c='tab:blue',alpha=0.75)
    #     ### predicted
    #     deltaxpred_E = kappa_theo_iid[idxje,0]*np.array(xlims)
    #     axE.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')
    #     axI.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')

    #     iktrial=-1
    #     xactfull_E   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsE, -1])
    #     xactfull_I   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsI, -1])
    #     deltaxfull_E = xactfull_E - np.mean(xactfull_E)
    #     deltaxfull_I = xactfull_I - np.mean(xactfull_I)

    #     if (np.mean(xactfull_E)<0):
    #         deltaxfull_E=-deltaxfull_E
    #     if(np.mean(xactfull_I)<0):
    #         deltaxfull_I=-deltaxfull_I

    #     deltam_E = ReigvecTseries[idxje,iktrial,idrandsE,0]#-xAm[0,0]
    #     deltam_E = deltam_E - np.mean(deltam_E)
    #     deltam_I = ReigvecTseries[idxje,iktrial,idrandsI,0]#-xAm[-1,0]
    #     deltam_I = deltam_I - np.mean(deltam_I)
    #     axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c='tab:red',alpha=0.25)
    #     axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c='tab:blue',alpha=0.25)

    #     ### predicted
    #     deltaxpred_E = kappa_theo_iid[idxje,2]*np.array(xlims)
    #     axE.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')
    #     axI.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')
    #     # axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c=cm.Reds((JEseries[idxje]-JEseries[0])/(JEseries[-1]-JEseries[0])),alpha=0.25)
    #     # axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c=cm.Blues((JEseries[idxje]-JEseries[0])/(JEseries[-1]-JEseries[0])),alpha=0.25)

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

    ### adding comparison between --> fluctuations in connectivity lead to fluctuations in dynamic
    ijesample=[1,13,19]
    yticks = np.linspace(-3.0, 3.0, 3)
    ylims = [-3.0, 3.0]#[1.1, 2.2]
    xticks = np.linspace(-1.5, 1.5, 3)
    xlims = [-1.5, 1.5]
    import matplotlib.cm as cm

    figE,axE=plt.subplots(figsize=(2,4))#,sharex=True,sharey=True,tight_layout=True)
    figI,axI=plt.subplots(figsize=(2,4))#,sharex=True,sharey=True,tight_layout=True)
    # iktrial =0
    for iii, idxje in enumerate(ijesample):
        iktrial=1
        idrandsE     = np.random.choice(np.arange(0,NE),size=100,replace=False)
        idrandsI     = np.random.choice(np.arange(NE,N),size=100,replace=False)
        xactfull_E   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsE, -1])
        xactfull_I   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsI, -1])
        deltaxfull_E = xactfull_E - np.mean(xactfull_E)
        deltaxfull_I = xactfull_I - np.mean(xactfull_I)

        if (np.mean(xactfull_E)<0):
            deltaxfull_E=-deltaxfull_E
        if(np.mean(xactfull_I)<0):
            deltaxfull_I=-deltaxfull_I

        deltam_E = ReigvecTseries[idxje,iktrial,idrandsE,0]-1.0
        deltam_I = ReigvecTseries[idxje,iktrial,idrandsI,0]-1.0#xAm
        axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c=cm.Reds((JEseries[idxje]-JEseries[0])/(JEseries[-1]-JEseries[0])),alpha=0.75)
        axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c=cm.Blues((JEseries[idxje]-JEseries[0])/(JEseries[-1]-JEseries[0])),alpha=0.75)
        # axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c='tab:red',alpha=0.75)
        # axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c='tab:blue',alpha=0.75)
        ### predicted
        deltaxpred_E = kappa_theo_iid[idxje,0]*np.array(xlims)
        axE.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')
        axI.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')

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

        if (np.mean(xactfull_E)<0):
            deltaxfull_E=-deltaxfull_E
        if(np.mean(xactfull_I)<0):
            deltaxfull_I=-deltaxfull_I

        deltam_E = ReigvecTseries[idxje,iktrial,idrandsE,0]-np.mean(ReigvecTseries[idxje,iktrial,idrandsE,0])#xAm[0,0]
        deltam_I = ReigvecTseries[idxje,iktrial,idrandsI,0]-np.mean(ReigvecTseries[idxje,iktrial,idrandsI,0])#xAm[-1,0]
        axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c=cm.Reds((JEseries[idxje]-JEseries[0])/(JEseries[-1]-JEseries[0])),alpha=0.25)
        axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c=cm.Blues((JEseries[idxje]-JEseries[0])/(JEseries[-1]-JEseries[0])),alpha=0.25)
        # axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c='tab:red',alpha=0.75)
        # axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c='tab:blue',alpha=0.75)
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
    '''
    '''

dyn_analysis()


def list_to_dict(lst, string):
    """
    Transform a list of variables into a dictionary.
    Parameters
    ----------
    lst : list
        list with all variables.
    string : str
        string containing the names, separated by commas.
    Returns
    -------
    d : dict
        dictionary with items in which the keys and the values are specified
        in string and lst values respectively.
    """
    string = string[0]
    string = string.replace(']', '')
    string = string.replace('[', '')
    string = string.replace('\\', '')
    string = string.replace(' ', '')
    string = string.replace('\t', '')
    string = string.replace('\n', '')
    string = string.split(',')
    d = {s: v for s, v in zip(string, lst)}
    return d

lst = [kappaintersect_Full, kappaintersect_R1,
        xfpseries_Full[:,:,:,-1], xfpseries_R1[:,:,:,-1],
        ReigvecTseries[:,:,:,0], LeigvecTseries[:,:,:,0]]
stg = ["kappaintersect_Full, kappaintersect_R1,"
        "xfpseries_Full, xfpseries_R1, "
        "ReigvecTseries, LeigvecTseries"]
data = list_to_dict(lst=lst, string=stg)
data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_iid_PositiveTF_0_4VS1.npz"
if RERUN ==1:
    np.savez(data_name, **data)


# lst = [kappaintersect_Sparse, kappaintersect_Full, kappaintersect_R1,
#        xfpseries_Full[:,:,:,-1], xfpseries_R1[:,:,:,-1], xfpseries_Sparse[:,:,:,-1]
#        RsprvecTseries[:,:,:,0], Rsprvecseries[:,:,:,0],
#        LsprvecTseries[:,:,:,0], Lsprvecseries[:,:,:,0],
#        Bsprvseries[:,:,0],
#        sigrcov, siglcov]
# stg = ["kappaintersect_Sparse, kappaintersect_Full, kappaintersect_R1,"
#        "xfpseries_Full, xfpseries_R1, xfpseries_Sparse,"
#        "ReigvecTseries, Rsprvecseries,"
#        "LeigvecTseries, Lsprvecseries,"
#        "Bsprvseries,"
#        "sigrcov, siglcov"]
# data = list_to_dict(lst=lst, string=stg)
# data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_spr_PositiveTF.npz"
# np.savez(data_name, **data)


#
