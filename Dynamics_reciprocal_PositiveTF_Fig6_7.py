# -*- coding: utf-8 -*-
"""
@author: Yuxiu Shao
Network with reciprocal random components;
Full-rank Gaussian Network and Gaussian-mixture Rank-one Approximation
Dynamics
"""


import numpy as np
import matplotlib.pylab as plt

from numpy import linalg as la
# import seaborn as sb
from functools import partial
# from sympy import *

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

"""
WITH SYMMETRY, 
* increase eta
* constant and homogeneous g0 # randomness
"""
shiftx = 1.5
RERUN = 0
if RERUN ==0:
    # data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_reciprocal_PositiveTF_negativeAlleigv_kappa_4VS1.npz"
    data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_reciprocal_PositiveTF_kappa_4VS1.npz"
    data      = np.load(data_name)



''' Network Setting  '''
Nt     = np.array([1200,300])#([750, 750])
NE, NI = Nt[0], Nt[1]
N      = NE+NI
Nparams  = np.array([NE, NI])
Npercent = Nparams/N
nrank, ntrial, neta, nvec, nmaxFP = 1,30,21,2,3# 30, 5, 2, 3
cuttrials=int(ntrial/2.0)
etaseries = np.linspace(0.0, 1.0, neta)

# heterogeneous random gain
gaverage = 0.8
# xee,xei,xie,xii=1.0,1.0,1.0,1.0 # homogeneous scaled standard deviation
xee,xei,xie,xii=1.0,0.5,0.2,0.8 # heterogeneous scaled standard deviation

coeffetaEsub  = np.array([1.0, 1.0, 1.0])
coeffetaTotal = np.array([1.0, 1.0, 1.0]) 
coeffeta      = np.array([1.0, 1.0, 1.0])

signetaEsub   = np.ones(3)
signetaTotal  = np.ones(3)
# signetaTotal[2] = -1  # @ YX ineg v.s. ipos
# sign of reciprocity

# #### negative feedback ~~~~~~~~~~~
# signetaEsub  = -np.ones(3)
# signetaTotal = -np.ones(3)

ppercentEsub = np.ones(2)
ppercentEsub[0] = 0.5
ppercentEsub[1] = 1.0-ppercentEsub[0]
# E->total ppercentEsub[0]/2.0,ppercentEsub[1]/2.0, (I) 1/2.0

### recording variables
Reigvecseries, Leigvecseries = np.zeros((neta, ntrial, N, nvec*2), dtype=complex), np.zeros((neta, ntrial, N, nvec*2), dtype=complex)
ReigvecTseries, LeigvecTseries = np.zeros((neta, ntrial, N, nvec*2), dtype=complex), np.zeros((neta, ntrial, N, nvec*2), dtype=complex)
Beigvseries = np.zeros((neta, ntrial, N), dtype=complex) # eigenvectors and eigenvalues obtained from eigendecomposition 

if(RERUN==0):
    ReigvecTseries[:,:,:,0], LeigvecTseries[:,:,:,0] = data['ReigvecTseries'],data['LeigvecTseries']
    # Beigvseries = data['Beigvseries']
Zrandommat=np.zeros((neta,ntrial,N,N))
#### statistical properties of the connectivity matrix 
armu, sigrcov = np.zeros((neta, ntrial, 2), dtype=complex), np.zeros((neta, ntrial, 2), dtype=complex)  # 2 for E and I
almu, siglcov = np.zeros((neta, ntrial, 2), dtype=complex), np.zeros((neta, ntrial, 2), dtype=complex)
siglr = np.zeros((neta, ntrial, 2), dtype=complex) # mean, variance and covariance of entries on eigenvectors
x0series, y0series = np.zeros((ntrial, N), dtype=complex), np.zeros((ntrial, N), dtype=complex)


#### params for dynamical variable kappa
tt = np.linspace(0, 500, 500)
dt = tt[2]-tt[1]
ntt = len(tt)
xfpseries_Full = np.zeros((neta, ntrial, N, ntt))
xfpseries_R1   = np.zeros((neta, ntrial, N, ntt))

if(RERUN==0):
    kappaintersect_Full = data['kappaintersect_Full']
    kappaintersect_R1   = data['kappaintersect_R1']

    xfpseries_Full[:,:,:,-1] = data['xfpseries_Full']
    xfpseries_R1[:,:,:,-1]   = data['xfpseries_R1']

kappaseries_R1 = np.zeros((neta, ntrial, ntt))
#### fixed point of kappa
kappaintersect_Full = np.zeros((neta, ntrial, nmaxFP*2))
kappaintersect_R1   = np.zeros((neta, ntrial, nmaxFP*2))

### network params -- mean connectivity matrix 
#### positive feedback
JI,JE,a,b    =  0.6,1.9,0.0,0.0 
# #### negative feedback
# JI,JE,a,b    =  0.6,2.2,0.0,0.0 
JEE, JIE, JEI, JII = JE+a, JE-a, JI-b, JI+b
Am, Jsv = generate_meanmat_eig(Nparams, JEE, JIE, JEI, JII)
meigvecAm, neigvecAm = np.ones((N, 1)), np.ones((N, 1))
neigvecAm[:NE, 0], neigvecAm[NE:, 0] = N*JE/NE, -N*JI/NI
eigvAm = (JE-JI)*np.ones(N)
xAm, yAm = np.reshape(meigvecAm[:, 0], (N, 1)), np.reshape(neigvecAm[:, 0].copy(), (N, 1))


if (RERUN):
    print('>>>>>>> simulating neuronal activity ... >>>>>>')
    for iktrial in range(ntrial):
        '''    ##>>>>>>>>>>> g0!=0, >>>>>>>>     '''
        Xsym  = iidGaussian([0, gaverage/np.sqrt(N)], [N, N])
        XsymT = Xsym.copy().T
        X0    = Xsym.copy() # generate the independent random components
        # BASE superimposed network with independent random components
        J0 = Am.copy()+X0.copy()
        eigvJ0, leig0, reig0, x0, y0 = decompNormalization(J0, xAm, yAm, xAm, yAm, nparams=Nparams, sort=1, nrank=1)

        # recording references (iid random component)
        x0series[iktrial, :], y0series[iktrial,:] = x0[:, 0].copy(), y0[:, 0].copy()
        for idxeta, eta in enumerate(etaseries):
            etaset = eta*np.ones(6)
            for iloop in range(3):
                etaset[iloop]   = etaset[iloop]*coeffetaEsub[iloop]
                etaset[iloop+3] = etaset[iloop+3]*coeffetaTotal[iloop]
            Xinit = Xsym.copy()
            # >>>>>>>>.Subcircuit Exc sym >>>>>>>>>.
            hNE = int(ppercentEsub[0]*NE)
            # when eta = 0, asqr = 0, aamp = 0, XT-0, X-1
            #### E1--E1 
            asqr = (1-np.sqrt(1-etaset[0]**2))/2.0
            aamp = np.sqrt(asqr)
            Xinit[:hNE, :hNE] = signetaEsub[0]*aamp*XsymT[:hNE,:hNE].copy()+np.sqrt(1-aamp**2)*Xsym[:hNE, :hNE].copy()
            ####  E1--E2
            asqr = (1-np.sqrt(1-etaset[1]**2))/2.0
            aamp = np.sqrt(asqr)
            Xinit[:hNE, hNE:NE] = signetaEsub[1]*aamp*XsymT[:hNE,hNE:NE].copy()+np.sqrt(1-aamp**2)*Xsym[:hNE, hNE:NE].copy()
            Xinit[hNE:NE, :hNE] = signetaEsub[1]*aamp*XsymT[hNE:NE,:hNE].copy()+np.sqrt(1-aamp**2)*Xsym[hNE:NE, :hNE].copy()
            #### E2--E2
            asqr = (1-np.sqrt(1-etaset[2]**2))/2.0
            aamp = np.sqrt(asqr)
            Xinit[hNE:NE, hNE:NE] = signetaEsub[2]*aamp*XsymT[hNE:NE,hNE:NE].copy()+np.sqrt(1-aamp**2)*Xsym[hNE:NE, hNE:NE].copy()
            # >>>>>>> Total >>>>>
            #### E1I-B
            hNE = int(ppercentEsub[0]*NE)
            asqr = (1-np.sqrt(1-(etaset[3])**2))/2.0
            aamp = np.sqrt(asqr)
            Xinit[NE:, :hNE] = signetaTotal[0]*aamp*XsymT[NE:, :hNE].copy()+np.sqrt(1-aamp**2)*Xsym[NE:, :hNE].copy()
            Xinit[:hNE, NE:] = signetaTotal[0]*aamp*XsymT[:hNE,NE:].copy()+np.sqrt(1-aamp**2)*Xsym[:hNE, NE:].copy()
            #### E2I-B
            asqr = (1-np.sqrt(1-(etaset[4])**2))/2.0
            aamp = np.sqrt(asqr)
            Xinit[NE:, hNE:NE] = signetaTotal[1]*aamp*XsymT[NE:,hNE:NE].copy()+np.sqrt(1-aamp**2)*Xsym[NE:, hNE:NE].copy()
            Xinit[hNE:NE, NE:] = signetaTotal[1]*aamp*XsymT[hNE:NE,NE:].copy()+np.sqrt(1-aamp**2)*Xsym[hNE:NE, NE:].copy()
            ## II ##
            asqr = (1-np.sqrt(1-etaset[5]**2))/2.0
            aamp = np.sqrt(asqr)
            Xinit[NE:, NE:] = signetaTotal[2]*aamp*XsymT[NE:, NE:].copy()+np.sqrt(1-aamp**2)*Xsym[NE:, NE:].copy()

            X = Xinit.copy()
            X[:NE, :NE] *= (xee)
            X[:NE, NE:] *= (xei)
            X[NE:, :NE] *= (xie)
            X[NE:, NE:] *= (xii)

            # superimposed network with low-rank + full-rank
            J  = X.copy()+Am.copy()
            JT = J.copy().T
            eigvJ, leigvec, reigvec, xnorm0, ynorm0 = decompNormalization(J, x0, y0, xAm, yAm, nparams=Nparams, sort=1, nrank=1)
            Reigvecseries[idxeta, iktrial, :, 0], Leigvecseries[idxeta,iktrial, :, 0] = xnorm0[:, 0].copy(), ynorm0[:, 0].copy()
            Beigvseries[idxeta, iktrial, :] = eigvJ.copy()

            axrmu, aylmu, sigxr, sigyl, sigcov = numerical_stats(xnorm0, ynorm0, x0, y0, eigvJ, nrank, 2, ppercent=np.array([NE/N,NI/N]))
            armu[idxeta, iktrial, :], almu[idxeta,iktrial, :] = axrmu[:, 0], aylmu[:, 0]
            sigrcov[idxeta, iktrial, :], siglcov[idxeta,iktrial, :] = sigxr[:, 0], sigyl[:, 0]
            siglr[idxeta, iktrial, :] = sigcov[:, 0, 0]

            # Rank One Approximation, Nullclines of \kappa
            xnorm0, ynorm0 = Reigvecseries[idxeta, iktrial, :, 0].copy(), Leigvecseries[idxeta, iktrial, :, 0]
            xnorm0, ynorm0 = np.reshape(xnorm0, (N, 1)), np.reshape(ynorm0, (N, 1))
            xmu, ymu = armu[idxeta, iktrial, :].copy(), almu[idxeta, iktrial, :].copy()
            xsig, ysig = sigrcov[idxeta, iktrial, :].copy(), siglcov[idxeta, iktrial, :].copy()
            yxcov = siglr[idxeta, iktrial, :].copy()

            #### dynamics of the superposition connectivity 
            Jpt   = J.copy()
            if iktrial<cuttrials:
                xinit = np.random.normal(5.0,1E-1, (1, N))
            else:
                xinit = np.random.normal(0,1e-2,(1,N))
            xinit = np.squeeze(np.abs(xinit))
            xtemporal = odesimulationP(tt, xinit, Jpt, 0)
            xfpseries_Full[idxeta, iktrial, :, :] = xtemporal.T.copy()
            kappanum = np.zeros(3)
            xact = np.squeeze(xfpseries_Full[idxeta, iktrial, :, -1])
            # use yAm -- unperturbed centre.
            if(1):
                kappanum[0] = yAm[0, 0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0] + yAm[NE, 0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            else:
                kappanum[0] = -yAm[0, 0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0] - yAm[NE, 0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            kappaintersect_Full[idxeta, iktrial, :3] = kappanum[:].copy()  

            # xnormt, ynormt = xnorm0, ynorm0
            eigvnorm = np.real(Beigvseries[idxeta,iktrial,0])#eigvAm[0]#eigvJ0[0]#
            xnormt, ynormt = (X.copy())@xAm.copy()/eigvnorm, (X.copy()).T@yAm.copy()/eigvnorm
            xnormt, ynormt = xnormt+xAm.copy(), ynormt+yAm.copy()

            # ### ~~~~~~~ not necessary ~~~~~~~~~
            # # ## normalize
            # projection = np.reshape(xnormt, (1, N))@np.reshape(ynormt, (N, 1))
            # ynormt = ynormt/projection*Beigvseries[idxeta, iktrial, 0]*N
            # # ---------------------------------------------------------------
            # _, _, xnormt, ynormt = Normalization(xnormt.copy(), ynormt.copy(
            # ), x0.copy(), y0.copy(), nparams=Nparams, sort=0, nrank=1)
            ReigvecTseries[idxeta, iktrial, :, 0], LeigvecTseries[idxeta,iktrial, :, 0] = xnormt[:, 0].copy(),ynormt[:, 0].copy()
            # renew -- perturbation result
            if(1):
                kappanum[0] = np.squeeze(ynormt.copy().T@(1.0+np.tanh(xact.copy()-shiftx)))/N#np.squeeze(np.reshape(ynormt,(1,N))@np.reshape((1.0+np.tanh(xact-shiftx)),(N,1)))/N
            else:
                mvec_norm = np.reshape(xnormt,(N,1))
                kappanum[0] = np.squeeze(np.reshape(np.squeeze(xact),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            kappaintersect_Full[idxeta, iktrial, :3] = kappanum[:].copy()
            
            #### dynamics of the rank one connectivity 
            r1Matt = np.real(xnormt@ynormt.T)
            r1Matt = r1Matt/N
            xtemporal = odesimulationP(tt, xinit, r1Matt, 0)
            xfpseries_R1[idxeta, iktrial, :, :] = xtemporal.T.copy()
            kappaseries_R1[idxeta, iktrial, :] = np.squeeze(
                xtemporal.copy()@np.reshape(xnormt, (-1, 1)))/np.sum(xnormt**2)
            # 2 populations
            kappanum = np.zeros(3)
            xact = np.reshape(np.squeeze(
                xfpseries_R1[idxeta, iktrial, :, -1]), (N, 1))
            kappanum[0] = np.squeeze(ynormt.copy().T@(1.0+np.tanh(xact.copy()-shiftx)))/N
            kappaintersect_R1[idxeta, iktrial, :3] = kappanum[:].copy()

            Zrandommat[idxeta,iktrial,:,:]=X.copy()


def mann_whitney_u_test(distribution_1, distribution_2):
    """
    Perform the Mann-Whitney U Test, comparing two different distributions.
    Args:
       distribution_1: List. 
       distribution_2: List.
    Outputs:
        u_statistic: Float. U statisitic for the test.
        p_value: Float.
    """
    u_statistic, p_value = stats.mannwhitneyu(distribution_1, distribution_2)
    return u_statistic, p_value

def analysis():
    #### A. the intersection pattern of kappa
    gmat = np.array([xee, xei, xie, xii])*gaverage
    gee, gei, gie, gii = gmat[0], gmat[1], gmat[2], gmat[3]
    if(len(Nparams) == 2):
        NE, NI = Nparams[0], Nparams[1]
    else:
        NE1, NE2, NI = Nparams[0], Nparams[1], Nparams[2]
        NE = NE1+NE2
    N = NE+NI
    npercent   = Nparams/N
    idxetas    = [0,7,neta-1]
    xintersect = etaseries[idxetas]#np.linspace(0.90, 1, 3)#np.linspace(0, 1, 2)
    kappa_x    = np.linspace(-5, 5, 30)
    Sx = np.zeros((len(xintersect), len(kappa_x)))
    F0 = np.zeros_like(Sx)
    F1 = np.zeros_like(Sx)
    F2 = np.zeros_like(Sx)

    #### B. calculate eigenvalue
    ''' rewrite as a function '''
    nAm = np.reshape(yAm.copy(), (N, 1))
    mAm = np.reshape(xAm.copy(), (N, 1))
    nAm, mAm     = np.real(nAm), np.real(mAm)
    lambda_theo2 = np.zeros((neta, 3), dtype=complex)
    lambda_num   = np.transpose(np.squeeze(Beigvseries[:, :, :2]), (2, 1, 0))
    cutoff = 2

    for idxeta, eta in enumerate(etaseries):
        etaE, etaB, etaI = eta*coeffetaEsub[0], eta*coeffetaTotal[0], eta*coeffetaTotal[2] ## Total[0]--etaset[3]
        etaE, etaB, etaI = etaE*signetaEsub[0], etaB*signetaTotal[0], etaI*signetaTotal[2]
        etaset = np.array([etaE, etaB, etaI])
        # gee,gei,gie,gii
        gmat = np.array([xee, xei, xie, xii])*gaverage

        Jparams = np.array([JE, JI])
        lambda_theo2[idxeta, :], _ = CubicLambda(Jparams, nAm, mAm, eigvAm, gmat, etaset, Nparams,)

    ## nonlinearity -- using lambda rather than lambda0 as normalization
    for idx, etaU in enumerate(xintersect):
        eigvuse = np.real(lambda_theo2[idxetas[idx], 0]) # nonlinearity -- using lambda rather than lambda0 as normalization
        etaset = etaU*np.ones(6)
        for iloop in range(3):
            etaset[iloop]   = etaset[iloop]*coeffetaEsub[iloop]*signetaEsub[iloop]
            etaset[iloop+3] = etaset[iloop+3]*coeffetaTotal[iloop]*signetaTotal[iloop]
        for idxx, x in enumerate(kappa_x):
            sigmam2E, sigmam2I = gee**2*npercent[0]/(eigvuse)**2+gei**2*npercent[1]/(eigvuse)**2, gie**2*npercent[0]/(eigvuse)**2+gii**2*npercent[1]/(eigvuse)**2
            delta0phiE, delta0phiI = x**2*sigmam2E, x**2*sigmam2I
            muphiE, muphiI = x, x
            delta_kappa    = -x+JE*PhiP(muphiE, delta0phiE)-JI*PhiP(muphiI, delta0phiI)
            F0[idx,idxx]   = x
            F1[idx,idxx]   = JE*PhiP(muphiE, delta0phiE)-JI*PhiP(muphiI, delta0phiI)

            # eigvuse = eigvuse#eigvAm[0]# np.mean(np.real(Beigvseries[idx,:,0]))#
            sigmaE_term   = (gee**2*JE*etaset[0]-gie*gei*JI*etaset[3]) / eigvuse**2*derPhiP(muphiE, delta0phiE)*x*npercent[0]
            sigmaI_term   = (gei*gie*JE*etaset[3]-gii**2*JI*etaset[5]) / eigvuse**2*derPhiP(muphiI, delta0phiI)*x*npercent[1]
            F2[idx,idxx]  = (sigmaE_term+sigmaI_term)
            Sx[idx, idxx] = delta_kappa+(sigmaE_term+sigmaI_term)
    '''
    ## simply using lambda0=JE-JI as normalization
    for idx, etaU in enumerate(xintersect):
        etaset = etaU*np.ones(6)
        for iloop in range(3):
            etaset[iloop]   = etaset[iloop]*coeffetaEsub[iloop]*signetaEsub[iloop]
            etaset[iloop+3] = etaset[iloop+3]*coeffetaTotal[iloop]*signetaTotal[iloop]
        for idxx, x in enumerate(kappa_x):
            sigmam2E, sigmam2I = gee**2*npercent[0]/(JE-JI)**2+gei**2*npercent[1]/(
                JE-JI)**2, gie**2*npercent[0]/(JE-JI)**2+gii**2*npercent[1]/(JE-JI)**2
            delta0phiE, delta0phiI = x**2*sigmam2E, x**2*sigmam2I
            muphiE, muphiI = x, x
            delta_kappa = -x+JE*PhiP(muphiE, delta0phiE)-JI*PhiP(muphiI, delta0phiI)
            F0[idx,idxx]= x
            F1[idx,idxx]= JE*PhiP(muphiE, delta0phiE)-JI*PhiP(muphiI, delta0phiI)

            eigvuse = eigvAm[0]# np.mean(np.real(Beigvseries[idx,:,0]))#
            # using lambda/lambda0
            sigmaE_term = (gee**2*JE*etaset[0]-gie*gei*JI*etaset[3]) / \
                eigvuse**2*derPhiP(muphiE, delta0phiE)*x*npercent[0]
            sigmaI_term = (gei*gie*JE*etaset[3]-gii**2*JI*etaset[5]) / \
                eigvuse**2*derPhiP(muphiI, delta0phiI)*x*npercent[1]
            F2[idx,idxx]= (sigmaE_term+sigmaI_term)
            Sx[idx, idxx] = delta_kappa+(sigmaE_term+sigmaI_term)
    '''

    fig, ax = plt.subplots(figsize=(4, 4))
    xticks = [-1,0,3]
    xlims  = [-1.,3.]
    yticks = [-2,0,2]
    ylims  = [-2,2]
    ax.plot(kappa_x, Sx[0, :], c='gray', lw=1)
    ax.plot(kappa_x, Sx[1, :], c='tab:purple', lw=1.0)
    ax.plot(kappa_x, Sx[2, :], c='tab:purple', lw=1.0)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    fig, ax = plt.subplots(figsize=(4, 4))
    # xticks  = [-1,0,3]
    # xlims   = [-1,3]
    # yticks  = [-1,0,3]
    # ylims   = [-1,3]

    ### negative
    xticks  = [-2,0,3]
    xlims   = [-2,3]
    yticks  = [-12,0,3]
    ylims   = [-12,3]
    ax.plot(kappa_x, F0[2, :], c='gray', lw=1)
    ax.plot(kappa_x, F1[2, :], c='tab:blue', lw=1.0)
    ax.plot(kappa_x, F2[2, :], c='tab:purple', lw=1.0)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    
    pos_t0=0.7
    ### C. calculate kappa using lamba theoy
    kappa_theo_iid = np.zeros((neta, 3))
    for idxeta in range(neta):
        alphaa = idxeta*1.0/neta
        etaset = etaseries[idxeta]*np.ones(6)
        for iloop in range(3):
            etaset[iloop]   = etaset[iloop]*coeffetaEsub[iloop]*signetaEsub[iloop]
            etaset[iloop+3] = etaset[iloop+3]*coeffetaTotal[iloop]*signetaTotal[iloop]
        # @YX ---- GMAT HETEROGENEOUS
        gmat = np.array([xee, xei, xie, xii])*gaverage

        eigvuse = np.real(lambda_theo2[idxeta,0]) # eigvAm[0]#
        init_k  = 1.0*np.max(kappaintersect_R1[idxeta,:, 0])
        kappa_max= fsolve(symperturbationP, init_k, args=(JE, JI, gmat, Nparams, etaset, eigvuse),xtol=1e-6,maxfev=1000)
        residual0=np.abs(symperturbationP(kappa_max,JE, JI, gmat, Nparams, etaset, eigvuse))
        init_k  = 0.6*(alphaa*np.min(kappaintersect_R1[idxeta,:, 0])+(1-alphaa)*np.max(kappaintersect_Full[idxeta,:, 0]))
        # init_k  = 0.5*init_k#(alphaa*np.max(kappaintersect_Full[idxeta,:, 0])+(1-alphaa)*np.min(kappaintersect_Full[idxeta,:, 0]))
        # kappa_theo_iid[idxeta, 1] 
        kappa_middle= fsolve(symperturbationP, init_k, args=(JE, JI, gmat, Nparams, etaset, eigvuse),xtol=1e-6,maxfev=1000)
        residual1=np.abs(symperturbationP(kappa_middle,JE, JI, gmat, Nparams, etaset, eigvuse))
        init_k  = 0.0
        kappa_theo_iid[idxeta, 2] = fsolve(symperturbationP, init_k, args=(JE, JI, gmat, Nparams, etaset, eigvuse),xtol=1e-6,maxfev=1000)
        if(residual0>1e-3):
            kappa_theo_iid[idxeta,0]=kappa_theo_iid[idxeta,2]
        else:
            kappa_theo_iid[idxeta,0]=kappa_max
        if(residual1>1e-3):
            kappa_theo_iid[idxeta,1]=kappa_theo_iid[idxeta,2]
        else:
            kappa_theo_iid[idxeta,1]=kappa_middle

    #### compute critical eta for dynamical state transition
    signeta = np.zeros(3)
    signeta[0],signeta[1],signeta[2]=signetaEsub[0],signetaTotal[0],signetaTotal[2]
    # gee,gei,gie,gii
    gmat    = np.array([xee, xei, xie, xii])*gaverage
    Jparams = np.array([JE, JI])
    int_eta = 0.1
    transitioneta = fsolve(TransitionEta, int_eta, args=(Jparams, yAm, xAm, eigvAm, gmat, signeta, Nparams,))
    targetvalue = (1.0-eigvAm[0])
    transitioneta_ = fsolve(TransitionOver, int_eta, args=(targetvalue, Jparams, yAm, xAm, eigvAm, gmat, signeta, Nparams,))
    print("transition eta LAMBDA:", transitioneta)
    print("transition eta OVERLAP:", transitioneta_)

    ### B1. PLOTTING
    clrs = ['k','b','r']
    mean_pos_kappa_full, mean_neg_kappa_full = np.zeros(neta), np.zeros(neta)
    std_pos_kappa_full, std_neg_kappa_full = np.zeros(neta), np.zeros(neta)
    

    fig, ax = plt.subplots(figsize=(5, 3))
    xticks  = np.linspace(.0, 1.0, 2)
    xlims   = [-0.0, 1.0]#[-0.1, 1.1]
    yticks  = [0,2.5]
    ylims   = [0,2.5]

    for ig in range(neta):
        pos_full = kappaintersect_Full[ig, np.where(kappaintersect_Full[ig, :, 0] >= pos_t0)[0], 0]
        mean_pos_kappa_full[ig] = np.mean(pos_full)
        std_pos_kappa_full[ig] = np.std(pos_full)
    low_bound = np.zeros_like(mean_pos_kappa_full)
    for imax in range(len(mean_pos_kappa_full)):
        low_bound[imax]=max(0,mean_pos_kappa_full[imax]-std_pos_kappa_full[imax])
    ax.fill_between(etaseries, mean_pos_kappa_full+std_pos_kappa_full,low_bound, alpha=0.5, facecolor='tab:purple')

    for ig in range(neta):
        pos_full = kappaintersect_Full[ig, np.where(kappaintersect_Full[ig, :, 0] < pos_t0)[0], 0]
        mean_pos_kappa_full[ig] = np.mean(pos_full)
        std_pos_kappa_full[ig] = np.std(pos_full)
    low_bound = np.zeros_like(mean_pos_kappa_full)
    for imax in range(len(mean_pos_kappa_full)):
        low_bound[imax]=max(0,mean_pos_kappa_full[imax]-std_pos_kappa_full[imax])
    ax.fill_between(etaseries, mean_pos_kappa_full+std_pos_kappa_full,low_bound, alpha=.5, facecolor='tab:purple')

    ax.plot(etaseries, kappa_theo_iid[:, 0],c='tab:purple', linewidth=1.5)
    ax.plot(etaseries, kappa_theo_iid[:, 2],c='tab:purple', linewidth=1.5)
    ax.plot(etaseries, kappa_theo_iid[:, 1],c='tab:purple', linewidth=1.5)

    # for iktrial in range(ntrial):
    #     ax.scatter(etaseries,kappaintersect_Full[:,iktrial,0],c=etaseries,cmap='Purples',alpha=0.75)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    nkappa  = 100
    netass  = 100
    kappa_line = np.linspace(yticks[0],yticks[-1],nkappa)
    eta_lines  = np.linspace(xticks[0],xticks[-1],netass)
    kappas_show=np.zeros((netass,nkappa))
    gmat = np.array([xee, xei, xie, xii])*gaverage
    for idxeta, etamesh in enumerate(eta_lines):
        for idxkappa, kappamesh in enumerate(kappa_line):
            etaE, etaB, etaI = etamesh*coeffetaEsub[0], etamesh*coeffetaTotal[0], etamesh*coeffetaTotal[2]
            etaE, etaB, etaI = etaE*signetaEsub[0], etaB*signetaTotal[0], etaI*signetaTotal[2]
            etaset = np.array([etaE, etaB, etaI])
            lambda_theo, _ = CubicLambda(Jparams, nAm, mAm, eigvAm, gmat, etaset, Nparams,)

            etaset = etamesh*np.ones(6)
            for iloop in range(3):
                etaset[iloop]   = etaset[iloop]*coeffetaEsub[iloop]*signetaEsub[iloop]
                etaset[iloop+3] = etaset[iloop+3]*coeffetaTotal[iloop]*signetaTotal[iloop]
            eigvuse = np.real(lambda_theo[0])
            kappas_show[idxeta,idxkappa]=np.abs(symperturbationP(kappamesh,JE, JI, gmat, Nparams, etaset, eigvuse))
    ax.imshow(kappas_show.T,extent = [xlims[0] , xlims[-1], ylims[-1] , ylims[0]],cmap='summer',vmin=0,vmax=0.3)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_aspect('auto')



    ### definition of kappa: 1) projection on the rank one vector m (normalized by |m|^2)
    ### 2) projecting firing rate phi(x) onto the left eigenvector n
    numkappa= np.zeros((neta,ntrial))
    pavgkappa,pstdkappa = np.zeros(neta),np.zeros(neta)
    navgkappa,nstdkappa = np.zeros(neta),np.zeros(neta)
    for iJE in range(neta):
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
    ax.plot(etaseries, pavgkappa,c='black', linewidth=1.5, linestyle='--')
    ax.fill_between(etaseries, pavgkappa+pstdkappa,pavgkappa-pstdkappa, alpha=0.5, facecolor='tab:purple')
    ax.plot(etaseries, navgkappa,c='black', linewidth=1.5, linestyle='--')
    ax.fill_between(etaseries, navgkappa+nstdkappa,navgkappa-nstdkappa, alpha=0.5, facecolor='tab:purple')


    numkappa= np.zeros((neta,ntrial))
    pavgkappa,pstdkappa = np.zeros(neta),np.zeros(neta)
    navgkappa,nstdkappa = np.zeros(neta),np.zeros(neta)
    for iJE in range(neta):
        signXE = np.mean(1.0+np.tanh(np.squeeze(xfpseries_Full[iJE,:,:NE,-1])-shiftx),axis=1)
        signXI = np.mean(1.0+np.tanh(np.squeeze(xfpseries_Full[iJE,:,NE:,-1])-shiftx),axis=1)
        ptrialXE,ntrialXE = np.where(signXE>=pos_t0)[0],np.where(signXE<pos_t0)[0]
        print('>>>P/N TRIAL',(ptrialXE),ntrialXE)
        ### CALCULATE [PHIXE/I]
        if len(ptrialXE)>0:
            for iktrial in ptrialXE:
                nvec_norm = np.reshape(LeigvecTseries[iJE,iktrial,:,0],(N,1))
                phix      = 1.0+np.tanh(np.squeeze(xfpseries_Full[iJE,iktrial,:,-1])-shiftx)
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(phix,(1,N))@nvec_norm)/N
            pavgkappa[iJE],pstdkappa[iJE] = np.mean(numkappa[iJE,ptrialXE]),np.std(numkappa[iJE,ptrialXE])
        if len(ntrialXE)>0:
            for iktrial in ntrialXE:
                nvec_norm = np.reshape(LeigvecTseries[iJE,iktrial,:,0],(N,1))
                phix      = 1.0+np.tanh(np.squeeze(xfpseries_Full[iJE,iktrial,:,-1])-shiftx)
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(phix,(1,N))@nvec_norm)/N
            navgkappa[iJE],nstdkappa[iJE] = np.mean(numkappa[iJE,ntrialXE]),np.std(numkappa[iJE,ntrialXE])

    ax.plot(etaseries, pavgkappa,c='red', linewidth=1.5, linestyle='--')
    ax.plot(etaseries, navgkappa,c='red', linewidth=1.5, linestyle='--')
    

    #### C. mean and variance of synaptic input x for each trial
    ## test two expressions of variance ##
    variance_x_num, variance_kappa_num = np.zeros((neta, ntrial, 2)), np.zeros((neta, ntrial, 2))
    mu_x_num, mu_kappa_num = np.zeros((neta, ntrial, 2)), np.zeros((neta, ntrial, 2))

    # analytically calculate the variance and mean, using the expression (self-consistency)
    variance_full_theo, mu_full_theo = np.zeros((neta, ntrial, 3,2)), np.zeros((neta, ntrial, 3,2))
    variance_R1_theo, mu_R1_theo = np.zeros((neta, ntrial,3, 2)), np.zeros((neta, ntrial,3, 2))

    gee, gei, gie, gii = gaverage*xee, gaverage*xei, gaverage*xie, gaverage*xii
    for idxje in range(neta):
        for iktrial in range(ntrial):
            # numerical results for Full Mat
            variance_x_num[idxje, iktrial, 0], variance_x_num[idxje, iktrial, 1] = np.std(
                xfpseries_Full[idxje, iktrial, :NE, -1])**2, np.std(xfpseries_Full[idxje, iktrial, NE:, -1])**2
            mu_x_num[idxje, iktrial, 0], mu_x_num[idxje, iktrial, 1] = np.mean(
                xfpseries_Full[idxje, iktrial, :NE, -1]), np.mean(xfpseries_Full[idxje, iktrial, NE:, -1])

            # numerical results for Rank one Appriximation Mat
            variance_kappa_num[idxje, iktrial, 0], variance_kappa_num[idxje, iktrial, 1] = np.std(xfpseries_R1[idxje, iktrial, :NE, -1])**2, np.std(xfpseries_R1[idxje, iktrial, NE:, -1])**2
            mu_kappa_num[idxje, iktrial, 0], mu_kappa_num[idxje, iktrial, 1] = np.mean(
                xfpseries_R1[idxje, iktrial, :NE, -1]), np.mean(xfpseries_R1[idxje, iktrial, NE:, -1])

        # @YX 2410 ADD --- COMPLETE -- R1 -- SELF-CONSISTENCY
        # consistency = [x[2]-(gee**2*NE/N+gei**2*NI/N)/(JE-JI)**2*(JE*PhimeanE-JI*PhimeanI)**2,x[0]-(JE*PhimeanE-JI*PhimeanI),x[3]-(gie**2*NE/N+gii**2*NI/N)/(JE-JI)**2*(JE*PhimeanE-JI*PhimeanI)**2,x[1]-(JE*PhimeanE-JI*PhimeanI)]


        # #### CHANGE EIGVUSE
        eigvnorm = np.real(lambda_theo2[idxje,0])# nonlinearity --- using lambda
        # eigvnorm = (JE-JI) # using lambda0
        #### INDIVIDUAL UNCHANGED
        mu_R1_theo[idxje, :, 0,0], variance_R1_theo[idxje, :,0, 0] = kappa_theo_iid[idxje,0], kappa_theo_iid[idxje, 0]**2*(gee**2*NE/N+gei**2*NI/N)/(eigvnorm)**2
        mu_R1_theo[idxje, :,0, 1], variance_R1_theo[idxje, :, 0,1] = kappa_theo_iid[idxje,0], kappa_theo_iid[idxje, 0]**2*(gie**2*NE/N+gii**2*NI/N)/(eigvnorm)**2
        mu_R1_theo[idxje, :,2,0], variance_R1_theo[idxje, :,2,0] = kappa_theo_iid[idxje,2], kappa_theo_iid[idxje,2]**2*(gee**2*NE/N+gei**2*NI/N)/(eigvnorm)**2
        mu_R1_theo[idxje, :,2,1], variance_R1_theo[idxje, :,2,1] = kappa_theo_iid[idxje,2], kappa_theo_iid[idxje,2]**2*(gie**2*NE/N+gii**2*NI/N)/(eigvnorm)**2
    # Figures reflecting how mean and variance change with random gain of iid Gaussian matrix
    fig, ax2   = plt.subplots(2, 1, figsize=(5, 5),sharex=True,tight_layout=True)
    fige, ax2e = plt.subplots(2, 1, figsize=(5, 5),sharex=True,tight_layout=True) ### ERROR BAR
    # -----------------  mean act------------------------------------------------------------
    pmean_x_numE, pmean_x_numI= np.zeros(neta), np.zeros(neta)
    pstd_x_numE, pstd_x_numI  = np.zeros(neta), np.zeros(neta)
    pos_t0 = 0.15
    pos_t1 = 0.05

    for i in range(neta):
        pmean_x_numE[i] = np.mean(variance_x_num[i,np.where(variance_x_num[i, :, 0]>= pos_t0)[0], 0])
        pstd_x_numE[i]  = np.std(variance_x_num[i,np.where(variance_x_num[i, :, 0] >= pos_t0)[0], 0])

        pmean_x_numI[i] = np.mean(variance_x_num[i,np.where(variance_x_num[i, :, 1]>= pos_t1)[0], 1])
        pstd_x_numI[i]  = np.std(variance_x_num[i,np.where(variance_x_num[i, :, 1] >= pos_t1)[0], 1])

    low_bound = np.zeros_like(pmean_x_numE)
    for imax in range(len(pmean_x_numE)):
        low_bound[imax]=max(0,pmean_x_numE[imax]-pstd_x_numE[imax])
    ax2[0].fill_between(etaseries, pmean_x_numE+pstd_x_numE,low_bound, alpha=0.3, facecolor='tab:red')
    low_bound = np.zeros_like(pmean_x_numI)
    for imax in range(len(pmean_x_numI)):
        low_bound[imax]=max(0,pmean_x_numI[imax]-pstd_x_numI[imax])
    ax2[1].fill_between(etaseries, pmean_x_numI+pstd_x_numI,low_bound, alpha=0.3, facecolor='tab:blue')
    ax2[0].plot(etaseries, pmean_x_numE, color='tab:red',linewidth=1.5, linestyle='--')
    ax2[1].plot(etaseries, pmean_x_numI, color='tab:blue',linewidth=1.5, linestyle='--')
    ### error bar >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ax2e[0].errorbar(etaseries, pmean_x_numE,pstd_x_numE/np.sqrt(ntrial),c='tab:red')
    ax2e[1].errorbar(etaseries, pmean_x_numI,pstd_x_numI/np.sqrt(ntrial), c='tab:blue')


    ### >>>>>>>>>>>>>> second fixed point >>>>>>>>>>>
    for i in range(neta):
        pmean_x_numE[i] = np.mean(variance_x_num[i,np.where(variance_x_num[i,:, 0]< pos_t0)[0], 0])
        pstd_x_numE[i]  = np.std(variance_x_num[i,np.where(variance_x_num[i,:, 0]<pos_t0)[0], 0])
        pmean_x_numI[i] = np.mean(variance_x_num[i,np.where(variance_x_num[i, :, 1]<pos_t1)[0], 1])
        pstd_x_numI[i]  = np.std(variance_x_num[i,np.where(variance_x_num[i, :, 1]<pos_t1)[0], 1])

    low_bound = np.zeros_like(pmean_x_numE)
    for imax in range(len(pmean_x_numE)):
        low_bound[imax]=max(0,pmean_x_numE[imax]-pstd_x_numE[imax])
    ax2[0].fill_between(etaseries, pmean_x_numE+pstd_x_numE,low_bound, alpha=0.3, facecolor='tab:red')
    low_bound = np.zeros_like(pmean_x_numI)
    for imax in range(len(pmean_x_numI)):
        low_bound[imax]=max(0,pmean_x_numI[imax]-pstd_x_numI[imax])
    ax2[1].fill_between(etaseries, pmean_x_numI+pstd_x_numI,low_bound, alpha=0.3, facecolor='tab:blue')  
    ax2[0].plot(etaseries, pmean_x_numE, color='tab:red',linewidth=1.5, linestyle='--')
    ax2[1].plot(etaseries, pmean_x_numI, color='tab:blue',linewidth=1.5, linestyle='--')
    ### error bar >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ax2e[0].errorbar(etaseries, pmean_x_numE,pstd_x_numE/np.sqrt(ntrial),c='tab:red')
    ax2e[1].errorbar(etaseries, pmean_x_numI,pstd_x_numI/np.sqrt(ntrial), c='tab:blue')

    # BUT PLOT THEORETICAL Value
    ax2[0].plot(etaseries, np.mean(variance_R1_theo[:, :,0, 0], axis=1),color='tab:red', linewidth=1.5)
    ax2[1].plot(etaseries, np.mean(variance_R1_theo[:, :, 0,1], axis=1),color='tab:blue', linewidth=1.5)
    ax2[0].plot(etaseries, np.mean(variance_R1_theo[:, :,2, 0], axis=1),color='tab:red', linewidth=1.5)
    ax2[1].plot(etaseries, np.mean(variance_R1_theo[:, :,2,1], axis=1),color='tab:blue', linewidth=1.5)

    ax2e[0].plot(etaseries, np.mean(variance_R1_theo[:, :,0, 0], axis=1),color='tab:red', linewidth=1.5)
    ax2e[1].plot(etaseries, np.mean(variance_R1_theo[:, :, 0,1], axis=1),color='tab:blue', linewidth=1.5)
    ax2e[0].plot(etaseries, np.mean(variance_R1_theo[:, :,2, 0], axis=1),color='tab:red', linewidth=1.5)
    ax2e[1].plot(etaseries, np.mean(variance_R1_theo[:, :,2,1], axis=1),color='tab:blue', linewidth=1.5)

    xticks = np.linspace(.0, 1.0, 2)
    xlims  = [-0.0, 1.0]#[-0.1, 1.1]
    yticks = [0,2.0]
    ylims  = [0,2.0]


    # for i in range(2):
    ax2[0].set_xlim(xlims)
    ax2[0].set_ylim(ylims)
    ax2[0].set_xticks(xticks)
    ax2[0].set_yticks(yticks)
    ax2[0].legend()

    ax2[1].set_ylim([0,0.25])
    ax2[1].set_yticks([0,0.25])
    ax2[1].legend()
    ax2[1].set_xlabel(r'reciprocal connectivity $\eta$')

    # for i in range(2):
    ax2e[0].set_xlim(xlims)
    ax2e[0].set_ylim(ylims)
    ax2e[0].set_xticks(xticks)
    ax2e[0].set_yticks(yticks)
    ax2e[0].legend()
    ax2e[1].set_ylim([0,0.25])
    ax2e[1].set_yticks([0,0.25])
    ax2e[1].legend()
    ax2e[1].set_xlabel(r'reciprocal connectivity $\eta$')

    #### C2. means -- positive and negative, if separated
    # Figures reflecting how mean and variance change with random gain of iid Gaussian matrix
    fig, ax2 = plt.subplots(2, 1, figsize=(5, 5),sharex=True,tight_layout=True)
    # -----------------  mean act------------------------------------------------------------
    pmean_x_numE, pmean_kappa_numE, pmean_x_numI, pmean_kappa_numI = np.zeros(neta), np.zeros(neta), np.zeros(neta), np.zeros(neta)
    pstd_x_numE, pstd_kappa_numE, pstd_x_numI, pstd_kappa_numI = np.zeros(neta), np.zeros(neta), np.zeros(neta), np.zeros(neta)
    nmean_x_numE, nmean_kappa_numE, nmean_x_numI, nmean_kappa_numI = np.zeros(neta), np.zeros(neta), np.zeros(neta), np.zeros(neta)
    nstd_x_numE, nstd_kappa_numE, nstd_x_numI, nstd_kappa_numI = np.zeros(neta), np.zeros(neta), np.zeros(neta), np.zeros(neta)

    # @YX 2908 add gavg -- main-text: IGNORE
    pmean_gavg_numE, pmean_gavg_numI = np.zeros(neta), np.zeros(neta)
    pstd_gavg_numE, pstd_gavg_numI   = np.zeros(neta), np.zeros(neta)
    nmean_gavg_numE, nmean_gavg_numI = np.zeros(neta), np.zeros(neta)
    nstd_gavg_numE, nstd_gavg_numI   = np.zeros(neta), np.zeros(neta)

    pos_t0 = 0.7

    for i in range(neta):
        pmean_x_numE[i], pmean_kappa_numE[i] = np.mean(mu_x_num[i, np.where(mu_x_num[i, :, 0] >= pos_t0)[0], 0]), np.mean(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 0] >= pos_t0)[0], 0])
        pstd_x_numE[i], pstd_kappa_numE[i]   = np.std(mu_x_num[i, np.where(mu_x_num[i, :, 0] >= pos_t0)[0], 0]), np.std(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 0] >= pos_t0)[0], 0])

        pmean_x_numI[i], pmean_kappa_numI[i] = np.mean(mu_x_num[i, np.where(mu_x_num[i, :, 1] >= pos_t0)[0], 1]), np.mean(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 1] >= pos_t0)[0], 1])
        pstd_x_numI[i], pstd_kappa_numI[i]   = np.std(mu_x_num[i, np.where(mu_x_num[i, :, 1] >= pos_t0)[0], 1]), np.std(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 1] >= pos_t0)[0], 1])

    low_bound = np.zeros_like(pmean_x_numE)
    for imax in range(len(pmean_x_numE)):
        low_bound[imax]=max(0,pmean_x_numE[imax]-pstd_x_numE[imax])
    ax2[0].fill_between(etaseries, pmean_x_numE+pstd_x_numE,low_bound, alpha=0.3, facecolor='red')
    low_bound = np.zeros_like(pmean_x_numI)
    for imax in range(len(pmean_x_numI)):
        low_bound[imax]=max(0,pmean_x_numI[imax]-pstd_x_numI[imax])
    ax2[1].fill_between(etaseries, pmean_x_numI+pstd_x_numI,low_bound, alpha=0.3, facecolor='blue')

    ### >>>>>>>>>>>>>> second fixed point >>>>>>>>>>>
    for i in range(neta):
        pmean_x_numE[i], pmean_kappa_numE[i] = np.mean(mu_x_num[i, np.where(mu_x_num[i, :, 0] < pos_t0)[0], 0]), np.mean(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 0] <pos_t0)[0], 0])
        pstd_x_numE[i], pstd_kappa_numE[i]   = np.std(mu_x_num[i, np.where(mu_x_num[i, :, 0] <pos_t0)[0], 0]), np.std(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 0] <pos_t0)[0], 0])

        pmean_x_numI[i], pmean_kappa_numI[i] = np.mean(mu_x_num[i, np.where(mu_x_num[i, :, 1] <pos_t0)[0], 1]), np.mean(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 1] <pos_t0)[0], 1])
        pstd_x_numI[i], pstd_kappa_numI[i]   = np.std(mu_x_num[i, np.where(mu_x_num[i, :, 1] <pos_t0)[0], 1]), np.std(mu_kappa_num[i, np.where(mu_kappa_num[i, :, 1] <pos_t0)[0], 1])


    low_bound = np.zeros_like(pmean_x_numE)
    for imax in range(len(pmean_x_numE)):
        low_bound[imax]=max(0,pmean_x_numE[imax]-pstd_x_numE[imax])
    ax2[0].fill_between(etaseries, pmean_x_numE+pstd_x_numE,low_bound, alpha=0.3, facecolor='red')
    low_bound = np.zeros_like(pmean_x_numI)
    for imax in range(len(pmean_x_numI)):
        low_bound[imax]=max(0,pmean_x_numI[imax]-pstd_x_numI[imax])
    ax2[1].fill_between(etaseries, pmean_x_numI+pstd_x_numI,low_bound, alpha=0.3, facecolor='blue')
    
    # ----------------------------- theoretical value obtained using rank-one approximation -----
    ax2[0].plot(etaseries, np.mean(mu_R1_theo[:, :,0,0], axis=1), color='tab:red',linewidth=1.5)
    ax2[1].plot(etaseries, np.mean(mu_R1_theo[:, :,0,1], axis=1), color='tab:blue',linewidth=1.5,)
    ax2[0].plot(etaseries, np.mean(mu_R1_theo[:, :,2,0], axis=1), color='tab:red',linewidth=1.5)
    ax2[1].plot(etaseries, np.mean(mu_R1_theo[:, :,2,1], axis=1), color='tab:blue',linewidth=1.5,)

    xticks = np.linspace(.0, 1.0, 2)
    xlims  = [-0.0, 1.0]#[-0.1, 1.1]
    yticks = [0,3.0]
    ylims  = [0,3.0]

    for i in range(2):
        ax2[i].set_xlim(xlims)
        ax2[i].set_ylim(ylims)
        ax2[i].set_xticks(xticks)
        ax2[i].set_yticks(yticks)
        # ax2[i].legend()
    ax2[1].set_xlabel(r'reciprocal connectivity $\eta$')

    ### adding comparison between --> fluctuations in connectivity lead to fluctuations in dynamic
    ijesample=[9,20]#[10,20]#upper and bottom #[1,20]
    import matplotlib.cm as cm
    for iktrial in range(0,cuttrials,5):
        figE,axE=plt.subplots(figsize=(2,4))#,sharex=True,sharey=True,tight_layout=True)
        figI,axI=plt.subplots(figsize=(2,4))#,sharex=True,sharey=True,tight_layout=True)
        # iktrial =0 
        # yticks = np.linspace(-4.5, 4.5, 3)
        # ylims = [-4.5, 4.5]#[1.1, 2.2]
        yticks = np.linspace(-3, 3, 3)
        ylims = [-3, 3]#same as iid ---
        xticks = np.linspace(-1.5, 1.5, 3)
        xlims = [-1.5, 1.5]
        for iii, idxje in enumerate(ijesample):
            idrandsE     = np.random.choice(np.arange(0,NE),size=100,replace=False)
            idrandsI     = np.random.choice(np.arange(NE,N),size=100,replace=False)
            xactfull_E   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsE, -1])
            xactfull_I   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsI, -1])
            deltaxfull_E = xactfull_E - np.mean(xactfull_E)
            deltaxfull_I = xactfull_I - np.mean(xactfull_I)


            deltam_E = ReigvecTseries[idxje,iktrial,idrandsE,0]-np.mean(ReigvecTseries[idxje,iktrial,idrandsE,0])#xAm[0,0]
            deltam_I = ReigvecTseries[idxje,iktrial,idrandsI,0]-np.mean(ReigvecTseries[idxje,iktrial,idrandsI,0])#xAm[-1,0]
            axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c=cm.Reds((etaseries[idxje]-etaseries[ijesample[0]-1])/(1-etaseries[ijesample[0]-1])),alpha=0.75)
            axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c=cm.Blues((etaseries[idxje]-etaseries[ijesample[0]-1])/(1-etaseries[ijesample[0]-1])),alpha=0.75)
            # axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c='tab:red',alpha=0.75)
            # axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c='tab:blue',alpha=0.75)
            print('>>>>>>>>> eta:',etaseries[idxje])
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

        iktrial=iktrial+cuttrials
        figE,axE=plt.subplots(figsize=(2,2))#,sharex=True,sharey=True,tight_layout=True)
        figI,axI=plt.subplots(figsize=(2,2))#,sharex=True,sharey=True,tight_layout=True)
        for iii, idxje in enumerate(ijesample):
            idrandsE     = np.random.choice(np.arange(0,NE),size=100,replace=False)
            idrandsI     = np.random.choice(np.arange(NE,N),size=100,replace=False)
            xactfull_E   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsE, -1])
            xactfull_I   = np.squeeze(xfpseries_Full[idxje, iktrial, idrandsI, -1])
            deltaxfull_E = xactfull_E - np.mean(xactfull_E)
            deltaxfull_I = xactfull_I - np.mean(xactfull_I)

            deltam_E = ReigvecTseries[idxje,iktrial,idrandsE,0]-np.mean(ReigvecTseries[idxje,iktrial,idrandsE,0])#xAm[0,0]
            deltam_I = ReigvecTseries[idxje,iktrial,idrandsI,0]-np.mean(ReigvecTseries[idxje,iktrial,idrandsI,0])#xAm[-1,0]
            axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c=cm.Reds((etaseries[idxje]-etaseries[ijesample[0]-1])/(1-etaseries[ijesample[0]-1])),alpha=0.25)
            axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c=cm.Blues((etaseries[idxje]-etaseries[ijesample[0]-1])/(1-etaseries[ijesample[0]-1])),alpha=0.25)
            ### predicted
            deltaxpred_E = kappa_theo_iid[idxje,2]*np.array(xlims)
            axE.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')
            axI.plot(xlims,deltaxpred_E,lw=1.5,c='gray',linestyle='--')

        yticks = np.linspace(-0.3, 0.3, 3)
        ylims = [-0.3, 0.3]#[1.1, 2.2]
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

    # #### A. the intersection pattern of kappa
    # gmat = np.array([xee, xei, xie, xii])*gaverage
    # gee, gei, gie, gii = gmat[0], gmat[1], gmat[2], gmat[3]
    # if(len(Nparams) == 2):
    #     NE, NI = Nparams[0], Nparams[1]
    # else:
    #     NE1, NE2, NI = Nparams[0], Nparams[1], Nparams[2]
    #     NE = NE1+NE2
    # N = NE+NI
    # npercent   = Nparams/N
    # xintersect = etaseries
    # kappa_x    = np.linspace(-5, 5, 100)
    # Sx = np.zeros((len(xintersect), len(kappa_x)))
    # F0 = np.zeros_like(Sx)
    # F1 = np.zeros_like(Sx)
    # F2 = np.zeros_like(Sx)

    # # reuse JEJI
    # for idx, etaU in enumerate(xintersect):
    #     etaset = etaU*np.ones(6)
    #     for iloop in range(3):
    #         etaset[iloop]   = etaset[iloop]*coeffetaEsub[iloop]*signetaEsub[iloop]
    #         etaset[iloop+3] = etaset[iloop+3]*coeffetaTotal[iloop]*signetaTotal[iloop]
    #     for idxx, x in enumerate(kappa_x):
    #         sigmam2E, sigmam2I = gee**2*npercent[0]/(lambda_theo2[idx,0])**2+gei**2*npercent[1]/(
    #             lambda_theo2[idx,0])**2, gie**2*npercent[0]/(lambda_theo2[idx,0])**2+gii**2*npercent[1]/(lambda_theo2[idx,0])**2
    #         delta0phiE, delta0phiI = x**2*sigmam2E, x**2*sigmam2I
    #         muphiE, muphiI = x, x
    #         delta_kappa = -x+JE*PhiP(muphiE, delta0phiE)-JI*PhiP(muphiI, delta0phiI)
    #         F0[idx,idxx]= x
    #         F1[idx,idxx]= JE*PhiP(muphiE, delta0phiE)-JI*PhiP(muphiI, delta0phiI)

    #         eigvuse = lambda_theo2[idx,0]#eigvAm[0]# np.mean(np.real(Beigvseries[idx,:,0]))#
    #         # using lambda/lambda0
    #         sigmaE_term = (gee**2*JE*etaset[0]-gie*gei*JI*etaset[3]) / \
    #             eigvuse**2*derPhiP(muphiE, delta0phiE)*x*npercent[0]
    #         sigmaI_term = (gei*gie*JE*etaset[3]-gii**2*JI*etaset[5]) / \
    #             eigvuse**2*derPhiP(muphiI, delta0phiI)*x*npercent[1]
    #         F2[idx,idxx]= (sigmaE_term+sigmaI_term)
    #         Sx[idx, idxx] = delta_kappa+(sigmaE_term+sigmaI_term)

    #     fig, ax = plt.subplots(figsize=(4, 4))
    #     xticks = [-1,0,3]
    #     xlims  = [-1.,3.]
    #     yticks = [-2,0,2]
    #     ylims  = [-2,2]
    #     ax.plot(kappa_x, Sx[idx, :], c='tab:purple', lw=1.0)
    #     ax.set_xlim(xlims)
    #     ax.set_ylim(ylims)
    #     ax.set_xticks(xticks)
    #     ax.set_yticks(yticks)


def conn_analysis():

    ### A. eigenvalue spectrum
    yticks = np.linspace(-2.50,2.5,2)
    ylims  = [-2.5,2.5]
    xticks = np.linspace(-0.,1.,2)
    xlims  = [-0.0,1.0]
    cm='bgc'
    envelopeRe = np.zeros(neta)
    envelopeIm = np.zeros(neta)
    coeff=coeffeta.copy()
    nAm=np.reshape(yAm.copy(),(N,1))
    mAm=np.reshape(xAm.copy(),(N,1))
    nAm,mAm=np.real(nAm),np.real(mAm)
    lambda_theo2 = np.zeros((neta,3),dtype=complex)
    lambda_theo2_delta = np.zeros(neta)
    lambda_num = np.transpose(np.squeeze(Beigvseries[:,:,:2]),(2,1,0))
    cutoff = 2

    signeta=np.ones(3)
    signeta[0],signeta[1],signeta[2]=signetaEsub[0],signetaTotal[0],signetaTotal[2]

    for idxeta,eta in enumerate(etaseries):
        #### @YX 0409 notice -- might be cell-type specific  reciprocal conn.
        etaE,etaB,etaI=eta*coeff[0],eta*coeff[1],eta*coeff[2]
        etaE,etaB,etaI = etaE*signeta[0],etaB*signeta[1],etaI*signeta[2]
        etaset = np.array([etaE,etaB,etaI])
        # gee,gei,gie,gii 
        gmat = np.array([xee,xei,xie,xii])*gaverage
        
        Jparams = np.array([JE,JI])
        lambda_theo2[idxeta,:],lambda_theo2_delta[idxeta]=CubicLambda(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams,)
        #### SORT THE ROOTS @YX 07DEC
        realroots=np.real(np.squeeze(lambda_theo2[idxeta,:]))
        sidx = np.argsort(realroots)
        roots_uno = lambda_theo2[idxeta,:].copy()
        for i in range(len(sidx)):
            lambda_theo2[idxeta,i]=roots_uno[sidx[2-i]]
        realsort = np.sort(np.real(Beigvseries[idxeta,:,:]),axis=1)
        envelopeRe[idxeta]=np.mean(realsort[:,-2]) 
        # amax_value = np.amax(arr, axis=0)
        imagsort = np.sort(np.imag(Beigvseries[idxeta,:,:]),axis=1)
        envelopeIm[idxeta]=np.mean(imagsort[:,-3])


    idxgavg,idxtrial=0,4#9,3 # 6,3,0
    idxgavgsample=np.array([15,neta-1])#np.array([0,4,8])#np.array([9])#np.array([8])#
    figtspt,axtspt=plt.subplots(figsize=(5,3))
    cm='bgc'
    shiftlen=0.0#0.2#
    for idxc,idxgavg in enumerate(idxgavgsample):
        print('index:',idxc,idxgavg)
        idrands     = np.random.choice(np.arange(nrank,N),size=600,replace=False)
        axtspt.scatter(np.real(Beigvseries[idxgavg,:,idrands]),np.imag(Beigvseries[idxgavg,:,idrands]),s=5,c=cm[idxc],alpha=0.25) # >>>>>>>>>>>>>>
        axtspt.scatter(np.real(Beigvseries[idxgavg,:,0]),np.imag(Beigvseries[idxgavg,:,0])-shiftlen*idxc,s=20,c=cm[idxc],alpha=0.25)#marker='^',alpha=0.5) # >>>>>>>>>>>>
        axtspt.set_aspect('equal')

        axtspt.scatter(np.real(lambda_theo2[idxgavg,0]),np.imag(lambda_theo2[idxgavg,0]),s=80,c='',marker='o',edgecolor=cm[idxc]) # 

        axtspt.scatter(np.real(lambda_theo2[idxgavg,1]),np.imag(lambda_theo2[idxgavg,1]),s=80,c='',marker='o',edgecolor=cm[idxc]) # 
        axtspt.scatter(np.real(lambda_theo2[idxgavg,2]),np.imag(lambda_theo2[idxgavg,2]),s=80,c='',marker='o',edgecolor=cm[idxc]) # 

        axtspt.spines['right'].set_color('none')
        axtspt.spines['top'].set_color('none')
        # X axis location
        axtspt.xaxis.set_ticks_position('bottom')
        axtspt.spines['bottom'].set_position(('data', 0))

        aee,aei,aie,aii=xee,xei,xie,xii
        # first do not multiply at
        ahomo=gaverage
        xee_,xei_,xie_,xii_=ahomo*aee/np.sqrt(2),ahomo*aei/np.sqrt(2),ahomo*aie/np.sqrt(2),ahomo*aii/np.sqrt(2)
        gmat     = np.array([[xee_**2,xei_**2],[xie_**2,xii_**2]])
        gaverage_=0
        for i in range(2):
            for j in range(2):
                gaverage_+=gmat[i,j]/2 # ntype=2
        gaverage_=np.sqrt(gaverage_)
        eigvgm,eigvecgm=la.eig(gmat) 
        r_g2=np.max(eigvgm)
        r_g = np.sqrt(r_g2)

    axtspt.set_title(r'$\eta=$'+np.str(eta))

    # # #### @YX 2709 original one ----
    # xticks = np.linspace(-1,1,3)
    # xlims = [-2.0,2.0]
    # yticks = np.linspace(-1.0,1.0,2)
    # ylims = [-1.5,1.5]

    # #### ~~~~~ negative eta ~~~~
    xticks = np.linspace(-0.5,1.0,4)
    xlims = [-0.5,1.3]
    yticks = np.linspace(-0.5,0.5,3)
    ylims = [-1.0,1.0]


    # xticks = np.linspace(-1,1,3)
    # xlims = [-1.0,1.0]
    # yticks = np.linspace(-0.5,0.5,2)
    # ylims = [-0.5,0.5]
    
    # xticks = np.linspace(-0.5,1,4)
    # xlims = [-0.5,1.0]
    # yticks = np.linspace(-0.5,0.5,2)
    # ylims = [-0.5,0.5]
    # axtspt.set_xlim(xlims)
    # axtspt.set_ylim(ylims)
    # axtspt.set_xticks(xticks)
    # axtspt.set_yticks(yticks)
    # axtspt.set_aspect('equal')

    ''' rewrite as a function '''
    yticks = np.linspace(-2.50,2.5,2)
    ylims = [-2.5,2.5]
    xticks = np.linspace(-0.,1.,2)
    xlims = [-0.0,1.0]
    cm='bgc'
    lambda_num = np.transpose(np.squeeze(Beigvseries[:,:,:2]),(2,1,0))
    fig,ax=plt.subplots(figsize=(4,4))
    ax.plot(etaseries,lambda_theo2[:,0],linestyle='--',c='black',label='2 cut off')
    ax.plot(etaseries,np.mean(lambda_num[0,:,:],axis=0),'b',label='2 cut off')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ############################################################################
    ''' THREE SOLUTIONS '''
    # ## -------- SMALL ONE ----------
    # yticks = np.linspace(-1,1,2)
    # ylims = [-1.0,1.0]
    ## ------- ORIGINAL ONE ---------
    yticks = np.linspace(-2.50,2.5,2)
    ylims = [-2.5,2.5]
    xticks = np.linspace(-0.,1.,2)
    xlims = [-0.0,1.0]
    fig,ax=plt.subplots(figsize=(4,4))
    for i in range(3):
        ax.plot(etaseries,lambda_theo2[:,i].real,linestyle='--',color=cm[i],linewidth=1.5,label='theo')
    mean_lambda_num = np.mean(lambda_num[:2,:,:].real,axis=1)
    std_lambda_num  = np.std(lambda_num[:2,:,:].real,axis=1)
    ax.fill_between(etaseries,mean_lambda_num[0,:]+std_lambda_num[0,:],mean_lambda_num[0,:]-std_lambda_num[0,:],alpha=0.3,facecolor=cm[0])
    ax.fill_between(etaseries,-(envelopeRe),envelopeRe, color='gray',alpha=0.2)


    ## ORIGINAL ONE WITH LARGER
    yticks = np.linspace(-1.50,1.5,2)
    ylims = [-1.5,1.5]
    # ### @YX 02DEC -- IMAGINARY
    # yticks = np.linspace(-0.5,0.5,2)
    # ylims = [-0.5,0.5]
    fig,ax=plt.subplots(figsize=(4,2)) 
    for i in range(3):
        ax.plot(etaseries,np.imag(lambda_theo2[:,i]),color=cm[i],linestyle='--',linewidth=1.5,label='theo')
    ax.fill_between(etaseries,-(envelopeIm),envelopeIm, color='gray',alpha=0.2)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    mean_lambda_num = np.mean(lambda_num[:2,:,:].imag,axis=1)
    std_lambda_num  = np.std(lambda_num[:2,:,:].imag,axis=1)
    # ax.plot(etaseries,mean_lambda_num[0,:],c=cm[0],linewidth=1.5,) ### NO NEED MEAN NUM
    ax.fill_between(etaseries,mean_lambda_num[0,:]+std_lambda_num[0,:],mean_lambda_num[0,:]-std_lambda_num[0,:],alpha=0.3,facecolor=cm[0])


    #### B. ENTRIES ON RANK-ONE VECTORS
    idxtrial=9#16#
    ### @YX 2609 ORIGINAL
    idxeta=9# 
    alphaval=0.10
    edgv='black'
    cm='br'
    '''loading vector changing'''
    meanmE,meanmI,meannE,meannI=np.zeros(nrank),np.zeros(nrank),np.zeros(nrank),np.zeros(nrank)
    ### --------------------------------------------------------
    mEvec,mIvec,nEvec,nIvec=np.squeeze(Reigvecseries[idxeta,:,:NE,0]),np.squeeze(Reigvecseries[idxeta,:,NE:,0]),np.squeeze(Leigvecseries[idxeta,:,:NE,0]),np.squeeze(Leigvecseries[idxeta,:,NE:,0])
    # mEvec,mIvec,nEvec,nIvec=np.squeeze(ReigvecTseries[idxeta,:,:NE,0]),np.squeeze(ReigvecTseries[idxeta,:,NE:,0]),np.squeeze(LeigvecTseries[idxeta,:,:NE,0]),np.squeeze(LeigvecTseries[idxeta,:,NE:,0])
    mEvec,mIvec,nEvec,nIvec=mEvec.flatten(),mIvec.flatten(),nEvec.flatten(),nIvec.flatten()
    scale_std=3.0
    for irank in range(nrank):
        meanmEtotal,stdmEtotal = np.mean(mEvec),np.std(mEvec)
        varmE = mEvec - meanmEtotal
        idxwhere = np.where(np.abs(varmE)>scale_std*stdmEtotal)
        mEvec[idxwhere]=meanmEtotal
        meanmE[irank]=np.mean(mEvec)
        # puring I
        meanmItotal,stdmItotal = np.mean(mIvec),np.std(mIvec)
        varmI = mIvec - meanmItotal
        idxwhere = np.where(np.abs(varmI)>scale_std*stdmItotal)
        mIvec[idxwhere]=meanmItotal
        meanmI[irank]=np.mean(mIvec)
        
        # n vector
        # puring E
        meannEtotal,stdnEtotal = np.mean(nEvec),np.std(nEvec)
        varnE = nEvec - meannEtotal
        idxwhere = np.where(np.abs(varnE)>scale_std*stdnEtotal)
        nEvec[idxwhere]=meannEtotal
        meannE[irank]=np.mean(nEvec)

        # puring I
        meannItotal,stdnItotal = np.mean(nIvec),np.std(nIvec)
        varnI = nIvec - meannItotal
        idxwhere = np.where(np.abs(varnI)>scale_std*stdnItotal)
        nIvec[idxwhere]=meannItotal
        meannI[irank]=np.mean(nIvec)

    ''' calculate directions '''
    noiseE,noiseI = np.zeros((NE*ntrial,2,nrank)),np.zeros((NI*ntrial,2,nrank))
    dirvecE,dirvecI=np.zeros((2,2)),np.zeros((2,2))
    for irank in range(nrank):
        # E 0 M 1 N
        noiseE[:,0,irank],noiseE[:,1,irank]= nEvec-meannE[irank],mEvec-meanmE[irank]
        noiseI[:,0,irank],noiseI[:,1,irank]= nIvec-meannI[irank],mIvec-meanmI[irank]
    # m1n1
    covdirE,covdirI=np.squeeze(noiseE[:,:,0]).T@np.squeeze(noiseE[:,:,0]),np.squeeze(noiseI[:,:,0]).T@np.squeeze(noiseI[:,:,0])
    _,dirvecE=la.eig(covdirE)
    _,dirvecI=la.eig(covdirI) 
    for i in range(2):
        for j in range(2):
            dirvecE[i,j]*=(1*np.sqrt(N))
            dirvecI[i,j]*=(1*np.sqrt(N))

    ### @YX 2709 ORIGINAL
    fig,ax=plt.subplots(figsize=(5,3))
    # ax.scatter(nIvec,mIvec,s=5.0,c='blue',alpha=alphaval)#cm[1],alpha=alphaval)
    # ax.scatter(nEvec,mEvec,s=5.0,c='red',alpha=alphaval)#cm[0],alpha=alphaval)

    ax.plot([meannE[0],meannE[0]+dirvecE[0,0]],[meanmE[0],meanmE[0]+dirvecE[1,0]],color=edgv,linestyle='--',linewidth=1.5)
    ax.plot([meannE[0],meannE[0]+dirvecE[0,1]],[meanmE[0],meanmE[0]+dirvecE[1,1]],color=edgv,linestyle='--',linewidth=1.5)
    ax.plot([meannI[0],meannI[0]+dirvecI[0,0]],[meanmI[0],meanmI[0]+dirvecI[1,0]],color=edgv,linestyle='--',linewidth=1.5)
    ax.plot([meannI[0],meannI[0]+dirvecI[0,1]],[meanmI[0],meanmI[0]+dirvecI[1,1]],color=edgv,linestyle='--',linewidth=1.5)

    confidence_ellipse(np.real(nEvec),np.real(mEvec),ax,edgecolor=edgv)
    confidence_ellipse(np.real(nIvec),np.real(mIvec),ax,edgecolor=edgv)

    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)

    ### @YX 2709 ORIGINAL
    ax.set_aspect('equal')
    # ### ~~~~~~~~ original ~~~~~~~~~~
    # xticks = np.linspace(-10,10,2)
    # xlims = [-10,13]
    # yticks = np.linspace(-2,2,2)
    # ylims = [-3,4]

    ### ~~~~~~~~ negative eta ~~~~~~~~~~
    xticks = np.linspace(-5,5,2)
    xlims  = [-5,8]
    yticks = np.linspace(0,2,2)
    ylims  = [-0.5,2.5]


    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    ax.set_title(r'$\eta=$'+str(etaseries[idxeta]))


    ax.scatter(nIvec[:NI],mIvec[:NI],s=5.0,c='tab:blue',alpha=alphaval)#cm[1],alpha=alphaval)
    ax.scatter(nEvec[:NE],mEvec[:NE],s=5.0,c='tab:red',alpha=alphaval)#cm[0],alpha=alphaval)


    ### C. COMPARE INDIVIDUAL ENTRY ON RANK-ONE VECTORS
    xtickms = np.linspace(-2.5,5.5,2)
    xlimms = [-2.5,5.5]
    ytickms = np.linspace(-2.5,5.5,2)
    ylimms = [-2.5,5.5]

    xticks = np.linspace(-8,14,2)
    xlims = [-8,14]
    yticks = np.linspace(-8,14,2)
    ylims = [-8,14]
    '''# CHOOSE ONE TRIAL'''
    axisshift= 1.0
    x0,y0=xAm.copy(),yAm.copy()
    # idxtrial,idxtrial_=4,0 #  ### @YX DALE'S LAW
    idxtrial,idxtrial_ = 8,3 # ### @YX original
    fig,ax=plt.subplots(2,2,figsize=(4,4))
    idxtrial=0
    idxetasample=np.array([4,8])

    for i in range(len(idxetasample)):
        ax[0,i].plot(xticks,yticks,color='darkred',linestyle='--')
        ax[1,i].plot(xticks,yticks,color='darkred',linestyle='--')
        #### @YX modify 2508 -- redundancy
        #### @YX modify 2508 -- from Reigvecseries[i,...] to Reigvecseries[idxgsamples[i]...]
        ax[0,i].scatter(np.real(Reigvecseries[idxetasample[i],idxtrial,:NE,0]),np.real(ReigvecTseries[idxetasample[i],idxtrial,:NE,0]),s=2,c='tab:red',alpha=0.5)
        ax[1,i].scatter(np.real(Leigvecseries[idxetasample[i],idxtrial,:NE,0]),np.real(LeigvecTseries[idxetasample[i],idxtrial,:NE,0]),s=2,c='tab:red',alpha=0.5)
        
        ax[0,i].scatter(np.real(Reigvecseries[idxetasample[i],idxtrial,NE:,0]),np.real(ReigvecTseries[idxetasample[i],idxtrial,NE:,0]),s=2,c='tab:blue',alpha=0.5)
        ax[1,i].scatter(np.real(Leigvecseries[idxetasample[i],idxtrial,NE:,0]),np.real(LeigvecTseries[idxetasample[i],idxtrial,NE:,0]),s=2,c='tab:blue',alpha=0.5)
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])

    for i in range(2):
        ax[1,i].set_xlim(xlims)
        ax[1,i].set_ylim(ylims)
    ax[1,0].set_yticks(yticks)

    for i in range(2):
        ax[0,i].set_xlim(xlimms)
        ax[0,i].set_ylim(ylimms)
    ax[0,0].set_yticks(ytickms)
    for i in range(2):
        ax[1,i].set_xticks(xticks)
    # print(np.shape(rprime_the),np.shape(rprime_num))
    # ax.scatter(np.real(rprime_the),rprime_num,c='gray',alpha=0.5)
    # ax.set_xlim([-0.05,0.05])
    # ax.set_ylim([-0.05,0.05])
    '''
    # CHOOSE TWO TRIALS
    '''
    fig,ax=plt.subplots(2,2,figsize=(4,4))
    for i in range(len(idxetasample)):
        ax[0,i].plot(xticks,yticks,color='darkred',linestyle='--')
        ax[1,i].plot(xticks,yticks,color='darkred',linestyle='--')

        ax[0,i].scatter(np.real(Reigvecseries[idxetasample[i],idxtrial_,:NE,0]),np.real(ReigvecTseries[idxetasample[i],idxtrial,:NE,0]),s=2,c='tab:red',alpha=0.5)
        ax[1,i].scatter(np.real(Leigvecseries[idxetasample[i],idxtrial_,:NE,0]),np.real(LeigvecTseries[idxetasample[i],idxtrial,:NE,0]),s=2,c='tab:red',alpha=0.5)
        
        ax[0,i].scatter(np.real(Reigvecseries[idxetasample[i],idxtrial_,NE:,0]),np.real(ReigvecTseries[idxetasample[i],idxtrial,NE:,0]),s=2,c='tab:blue',alpha=0.5)
        ax[1,i].scatter(np.real(Leigvecseries[idxetasample[i],idxtrial_,NE:,0]),np.real(LeigvecTseries[idxetasample[i],idxtrial,NE:,0]),s=2,c='tab:blue',alpha=0.5)
    
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])



    for i in range(2):
        ax[1,i].set_xlim(xlims)
        ax[1,i].set_ylim(ylims)
    ax[1,0].set_yticks(yticks)

    for i in range(2):
        ax[0,i].set_xlim(xlimms)
        ax[0,i].set_ylim(ylimms)
    ax[0,0].set_yticks(ytickms)
    for i in range(2):
        ax[1,i].set_xticks(xticks)
    # print(np.shape(rprime_the),np.shape(rprime_num))
    # ax.scatter(np.real(rprime_the),rprime_num,c='gray',alpha=0.5)
    # ax.set_xlim([-0.05,0.05])
    # ax.set_ylim([-0.05,0.05])

    ### C. Gaussian mixture model
    from scipy import stats
    np.random.seed(41)
    nvbins = np.linspace(-10,10,100)
    mvbins = np.linspace(-3,4,)

    nEkde = stats.gaussian_kde(np.real(nEvec))
    nIkde = stats.gaussian_kde(np.real(nIvec))
    mEkde = stats.gaussian_kde(np.real(mEvec))
    mIkde = stats.gaussian_kde(np.real(mIvec))
    xx = np.linspace(-10, 10, 1000)
    fign, axn = plt.subplots(figsize=(5,1))
    axn.hist(np.real(nEvec), density=True, bins=nvbins, facecolor='tab:red', alpha=0.3)
    axn.plot(xx, nEkde(xx),c='tab:red')
    axn.hist(np.real(nIvec), density=True, bins=nvbins, facecolor='tab:blue',alpha=0.3)
    axn.plot(xx, nIkde(xx),c='tab:blue')
    axn.set_xlim(xlims)

    yy  = np.linspace(-3,4,70)
    figm, axm = plt.subplots(figsize=(3,1))
    axm.hist(np.real(mEvec), density=True, bins=mvbins,facecolor='red', alpha=0.3)
    axm.plot(yy, mEkde(yy),c='tab:red')

    axm.hist(np.real(mIvec), density=True, bins=mvbins,facecolor='blue' ,alpha=0.3)
    axm.plot(yy, mIkde(yy),c='tab:blue')
    axm.set_xlim(ylims)


    ### >>> D theoretical lambda and sigmanm
    DeltaLR,DeltaLR_num = np.zeros((neta,3),dtype=complex),np.zeros((neta,ntrial,3),dtype=complex)
    DeltaLR_eigv = np.zeros((neta,ntrial),dtype=complex)
    DeltaLR_      = np.zeros((neta,3),dtype=complex)

    Jinit = np.zeros((ntrial,N,N),dtype=complex)
    ## Jinit same for identical trial # realization
    Leigvecinit,Reigvecinit = np.zeros((ntrial,NE*2),dtype=complex),np.zeros((ntrial,NE*2),dtype=complex)

    for idxeta,eta in enumerate (etaseries):
        etaE,etaB,etaI=eta*coeff[0],eta*coeff[1],eta*coeff[2]
        etaE,etaB,etaI = etaE*signeta[0],etaB*signeta[1],etaI*signeta[2]
        etaset = np.array([etaE,etaB,etaI])
        # gee,gei,gie,gii 
        gmat = np.array([xee,xei,xie,xii])*gaverage
        Jparams = np.array([JE,JI])
        eigtemp = np.real(lambda_theo2[idxeta,:]) ####nonliearity using lambda
        # eigtemp = np.mean(np.real(Beigvseries[idxeta,:,0]))*np.ones(2) #### using lambda0
        #### lambda + nonlinearity
        DeltaLR_[idxeta,:] = sigmaOverlap(Jparams,nAm,mAm,eigtemp,gmat,etaset,Nparams,)   
        ### lambda0
        DeltaLR[idxeta,:] = sigmaOverlap(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams,)  
        # # #### @YX 03DEC  USING DEFINITION
        # DeltaLR[idxeta,2]=np.squeeze(yAm.T@Zrandommat[idxeta,0,:,:]@Zrandommat[idxeta,0,:,:]@xAm/eigvAm[0]**2)/N
        # DeltaLR[idxeta,0]=np.squeeze(yAm.T@Zrandommat[idxeta,0,:,:NE]@Zrandommat[idxeta,0,:NE,:]@xAm/eigvAm[0]**2)/NE
        # DeltaLR[idxeta,1]=np.squeeze(yAm.T@Zrandommat[idxeta,0,:,NE:]@Zrandommat[idxeta,0,NE:,:]@xAm/eigvAm[0]**2)/NI

        # DeltaLR[idxeta,:] = sigmaOverlap(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams,)
        for itrial in range(ntrial):
            eigvaluem = Beigvseries[idxeta,itrial,0]

            # numerical value
            changeL_num = np.reshape(np.squeeze(Leigvecseries[idxeta,itrial,:,0].copy()),(N,1))
            changeL_num[:NE,0]-=np.mean(changeL_num[:NE,0])
            changeL_num[NE:,0]-=np.mean(changeL_num[NE:,0])
            changeR_num = np.reshape(np.squeeze(Reigvecseries[idxeta,itrial,:,0].copy()),(N,1))
            changeR_num[:NE,0]-=np.mean(changeR_num[:NE,0])
            changeR_num[NE:,0]-=np.mean(changeR_num[NE:,0])

            DeltaLR_num[idxeta,itrial,2]=np.squeeze(changeL_num.T@changeR_num)/N
            DeltaLR_num[idxeta,itrial,0]=np.sum(changeL_num[:NE,0]*changeR_num[:NE,0])/NE
            DeltaLR_num[idxeta,itrial,1]=np.sum(changeL_num[NE:,0]*changeR_num[NE:,0])/NI
            
            # DeltaLR_num[idxeta,itrial,2]=siglr[idxeta,itrial,0]*NE/N+siglr[idxeta,itrial,1]*NI/N
            # DeltaLR_num[idxeta,itrial,0]=siglr[idxeta,itrial,0]
            # DeltaLR_num[idxeta,itrial,1]=siglr[idxeta,itrial,1]

    ### >>>> D1. as function of eta
    fig,ax=plt.subplots(figsize=(5,5))
    cm='rbk'
    ijump=2
    for i in range(3):
        meanvec, stdvec = np.mean(np.squeeze(DeltaLR_num[:,:,i]),axis=1),np.std(np.squeeze(DeltaLR_num[:,:,i]),axis=1)
        ax.plot(etaseries,DeltaLR[:,i],c=cm[i], linewidth=0.75,linestyle='--')
        ax.plot(etaseries,DeltaLR_[:,i],c=cm[i],linewidth=0.75)
        ax.fill_between(etaseries,meanvec+stdvec,meanvec-stdvec,alpha=0.3,facecolor=cm[i])

    # yticks = np.linspace(0.0,1.0,2)
    # ylims = [-0.1,1.2]

    # yticks = np.linspace(0.0,0.4,2)
    # ylims = [0.0,0.4]

    yticks = np.linspace(0.0,0.5,2)
    ylims = [0.0,0.5]

    # yticks = np.linspace(0.0,0.2,2)
    # ylims = [0.0,0.25]
    xticks = np.linspace(0.0,1.0,2)
    xlims = [-0.01,1.01]
    ax.set_xlim(xlims)
    ax.set_xticks(xticks)
    ax.set_ylim(ylims)
    ax.set_yticks(yticks)
    # ax.set_aspect('equal')

    fig,ax=plt.subplots(figsize=(5,5))
    import matplotlib.cm as cm
    mean_DeltaLR_num,std_DeltaLR_num = np.mean(DeltaLR_num[:,:,2],axis=1),np.std(DeltaLR_num[:,:,2],axis=1) 
    x  = np.real(lambda_theo2[:,0])-eigvAm[0]
    y1 = mean_DeltaLR_num+std_DeltaLR_num
    y2 = mean_DeltaLR_num-std_DeltaLR_num

    polygon  = ax.fill_between(x, y1, y2, lw=0, color='none')
    xlim     = plt.xlim()
    ylim     = plt.ylim()
    verts    = plt.vstack([p.vertices for p in polygon.get_paths()])
    gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='Blues', aspect='auto',
                          extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
    
    ax.plot(lambda_theo2[:,0]-eigvAm[0],DeltaLR[:,2],c='black',linewidth=0.75,linestyle='--')
    ax.plot(lambda_theo2[:,0]-eigvAm[0],DeltaLR_[:,2],c='black',linewidth=0.75)
    # ax.plot(lambda_theo2_delta[:],DeltaLR[:,2],c='black',linewidth=2.0,label='2 cut off')

    # yticks = np.linspace(0.0,0.6,2)
    # ylims = [-0.1,0.7]
    # xticks = np.linspace(0.0,0.6,2)
    # xlims = [-0.1,0.7]

    yticks = np.linspace(0.0,0.4,2)
    ylims = [-0.05,0.4]
    xticks = np.linspace(0.0,0.4,2)
    xlims = [-0.05,0.4]
    ### ~~~~~~~~~~~~ Suppl. nonlinearity
    yticks = np.linspace(0.0,0.25,2)
    ylims = [-0.0,0.25]
    xticks = np.linspace(0.0,0.25,2)
    xlims = [-0.0,0.25]

    # yticks = np.linspace(0.0,0.25,2)
    # ylims = [0.0,0.25]
    # xticks = np.linspace(0.0,0.25,2)
    # xlims = [0.0,0.25]

    # yticks = np.linspace(0.0,0.2,2)
    # ylims = [-0.05,0.25]
    # xticks = np.linspace(0.0,0.2,2)
    # xlims = [-0.05,0.25]

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta \lambda$')
    ax.set_ylabel(r'$\sigma_{nm}$')

# conn_analysis()
analysis()

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
        ReigvecTseries[:,:,:,0], LeigvecTseries[:,:,:,0], Beigvseries]
stg = ["kappaintersect_Full, kappaintersect_R1,"
        "xfpseries_Full, xfpseries_R1, "
        "ReigvecTseries, LeigvecTseries, Beigvseries"]
data = list_to_dict(lst=lst, string=stg)
if RERUN==1:
    data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_reciprocal_PositiveTF_negativeAlleigv_kappa_4VS1.npz"
    np.savez(data_name, **data)

# #### A. the intersection pattern of kappa
# gmat = np.array([xee, xei, xie, xii])*gaverage
# gee, gei, gie, gii = gmat[0], gmat[1], gmat[2], gmat[3]
# if(len(Nparams) == 2):
#     NE, NI = Nparams[0], Nparams[1]
# else:
#     NE1, NE2, NI = Nparams[0], Nparams[1], Nparams[2]
#     NE = NE1+NE2
# N = NE+NI
# npercent   = Nparams/N
# xintersect = [0.15,0.25,0.35,0.45,0.55]#etaseries
# kappa_x    = np.linspace(-5, 5, 100)
# Sx = np.zeros((len(xintersect), len(kappa_x)))
# F0 = np.zeros_like(Sx)
# F1 = np.zeros_like(Sx)
# F2 = np.zeros_like(Sx)

# # reuse JEJI
# for idx, etaU in enumerate(xintersect):
#     etaset = etaU*np.ones(6)
#     for iloop in range(3):
#         etaset[iloop]   = etaset[iloop]*coeffetaEsub[iloop]*signetaEsub[iloop]
#         etaset[iloop+3] = etaset[iloop+3]*coeffetaTotal[iloop]*signetaTotal[iloop]
#     for idxx, x in enumerate(kappa_x):
#         sigmam2E, sigmam2I = gee**2*npercent[0]/(JE-JI)**2+gei**2*npercent[1]/(
#             JE-JI)**2, gie**2*npercent[0]/(JE-JI)**2+gii**2*npercent[1]/(JE-JI)**2
#         delta0phiE, delta0phiI = x**2*sigmam2E, x**2*sigmam2I
#         muphiE, muphiI = x, x
#         delta_kappa = -x+JE*PhiP(muphiE, delta0phiE)-JI*PhiP(muphiI, delta0phiI)
#         F0[idx,idxx]= x
#         F1[idx,idxx]= JE*PhiP(muphiE, delta0phiE)-JI*PhiP(muphiI, delta0phiI)

#         eigvuse = eigvAm[0]# np.mean(np.real(Beigvseries[idx,:,0]))#
#         # using lambda/lambda0
#         sigmaE_term = (gee**2*JE*etaset[0]-gie*gei*JI*etaset[3]) / \
#             eigvuse**2*derPhiP(muphiE, delta0phiE)*x*npercent[0]
#         sigmaI_term = (gei*gie*JE*etaset[3]-gii**2*JI*etaset[5]) / \
#             eigvuse**2*derPhiP(muphiI, delta0phiI)*x*npercent[1]
#         F2[idx,idxx]= (sigmaE_term+sigmaI_term)
#         Sx[idx, idxx] = delta_kappa+(sigmaE_term+sigmaI_term)

#     fig, ax = plt.subplots(figsize=(4, 4))
#     xticks = [-1,0,3]
#     xlims  = [-1.,3.]
#     yticks = [-2,0,2]
#     ylims  = [-2,2]
#     ax.plot(kappa_x, Sx[idx, :], c='tab:purple', lw=1.0)
#     ax.set_xlim(xlims)
#     ax.set_ylim(ylims)
#     ax.set_xticks(xticks)
#     ax.set_yticks(yticks)

# fig, ax = plt.subplots(figsize=(4, 4))
# xticks  = [-1,0,3]
# xlims   = [-1,3]
# yticks  = [-1,0,3]
# ylims   = [-1,3]
# ax.plot(kappa_x, F0[1, :], c='gray', lw=1)
# ax.plot(kappa_x, F1[1, :], c='tab:blue', lw=1.0)
# ax.plot(kappa_x, F2[1, :], c='tab:purple', lw=1.0)
# ax.set_xlim(xlims)
# ax.set_ylim(ylims)
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)
