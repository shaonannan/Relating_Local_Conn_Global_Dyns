# -*- coding: utf-8 -*-
"""
@author: Yuxiu Shao
Sparse EI Network, Full-rank Gaussian Network and Rank-one Mixture of Gaussian Approximation
Connectivity, Dynamics
"""


import numpy as np
import matplotlib.pylab as plt
import matplotlib

from numpy import linalg as la
# import seaborn as sb

# from sympy import *

extras_require = {'PLOT':['matplotlib>=1.1.1,<3.0']},

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)


### import self-defined functions
from UtilFuncs import *
from Sparse_util import *
from Meanfield_Dyn_util import *
from Connect_util import *

def cal_outliers_general_complex(x, tau_set, coeffs, Jparams, Nparams, gmat, eigvAm):
    """
    Parameters
    ----------
    x : 
        complex eigenvalue solutions.
    tau_set : array
        motif statistics.
    coeffs : 
        local statistics.
    Jparams : 
        mean connectivity.
    Nparams : 
        network scale.
    gmat : 
        random variance.
    eigvAm : 
        unperturbed eigenvaules.

    Returns
    -------
    results : 
        solutions.

    """
    NE, NI = int(Nparams[0]), int(Nparams[1])
    N      = NE+NI
    alphaE,alphaI = NE/N, NI/N 
    JE, JI = Jparams[0],Jparams[1]

    ### mean connectivity 
    Mbar = np.ones((N,1))
    Nbar = np.zeros((N,1))
    Nbar[:NE,0],Nbar[NE:,0] = JE*N/NE, -JI*N/NI

    gee, gei, gie, gii = gmat[0],gmat[1],gmat[2],gmat[3]
    zchn = np.zeros((2,2))
    zchn[0,0],zchn[0,1] = alphaE*tau_set['tchn'][0,0]*gee**2+alphaI*tau_set['tchn'][1,0]*gei*gie, alphaE*tau_set['tchn'][0,1]*gee*gei+alphaI*tau_set['tchn'][1,1]*gei*gii 
    zchn[1,0], zchn[1,1] = alphaE*tau_set['tchn'][0,0]*gie*gee+alphaI*tau_set['tchn'][1,0]*gii*gie, alphaE*tau_set['tchn'][0,1]*gie*gei+alphaI*tau_set['tchn'][1,1]*gii**2

    Emean, Ediff = (zchn[0,0]+zchn[1,0])/2*N, (zchn[0,0]-zchn[1,0])/2*N  
    Imean, Idiff = (zchn[0,1]+zchn[1,1])/2*N, (zchn[0,1]-zchn[1,1])/2*N   
    # Imean *=(-1)

    U,V = np.zeros((N,2)), np.zeros((N,2))
    U[:,0], U[:NE,1], U[NE:,1] = 1, 1, -1 
    V[:NE,0],V[NE:,0] = Emean, Imean 
    V[:NE,1],V[NE:,1] = Ediff, Idiff 
    ### negative
    U *=(-1)
    V /=(N)

    ### diagonal matrix A 
    #### upper 
    xall = complex(x[0],x[1])
    # upper  =xall**2-(alphaE*tau_set['trec'][0,0]*gee**2+alphaI*tau_set['trec'][1,0]*gei*gie)+(alphaE*tau_set['tchn'][0,0]*gee**2+alphaI*tau_set['tchn'][1,0]*gei*gie)
    # bottom = xall**2-(alphaE*tau_set['trec'][0,1]*gie*gei+alphaI*tau_set['trec'][1,1]*gii**2)+(alphaE*tau_set['tchn'][0,1]*gie*gei+alphaI*tau_set['tchn'][1,1]*gii**2)
    upper_r  = (x[0]**2-x[1]**2)-(alphaE*tau_set['trec'][0,0]*gee**2+alphaI*tau_set['trec'][1,0]*gei*gie)+(alphaE*tau_set['tchn'][0,0]*gee**2+alphaI*tau_set['tchn'][1,0]*gei*gie)
    upper_i  = 2*x[0]*x[1]
    bottom_r = (x[0]**2-x[1]**2)-(alphaE*tau_set['trec'][0,1]*gie*gei+alphaI*tau_set['trec'][1,1]*gii**2)+(alphaE*tau_set['tchn'][0,1]*gie*gei+alphaI*tau_set['tchn'][1,1]*gii**2)
    bottom_i = 2*x[0]*x[1]
    I2 = np.eye(2)

    A_1 = np.zeros((N,N),dtype=complex)
    for i in range(NE):
        A_1[i,i] = complex(upper_r/(upper_r**2+upper_i**2),-upper_i/(upper_r**2+upper_i**2))
    for i in range(NE,N):
        A_1[i,i] = complex(bottom_r/(bottom_r**2+bottom_i**2),-bottom_i/(bottom_r**2+bottom_i**2))
    # for i in range(NE):
    #     A_1[i,i] =1/upper
    # for i in range(NE,N):
    #     A_1[i,i] = 1/bottom

    ### step-by-step
    inverse = (I2+V.copy().T@A_1@U)
    # print('........inverse:',inverse)
    inverse = la.inv(inverse)
    # psie,psii= inverse[0,0], inverse[-1,-1]
    # diagpsi = np.eye(N)
    # for i in range(NE):
    #     diagpsi[i,i] = 1/psie 
    # for j in range(NE,N):
    #     diagpsi[j,j] = 1/psii
    # inverse = diagpsi.copy()
    inverse = -A_1@U@inverse@V.copy().T@A_1
    inverse = A_1+ inverse
    # inverse = A_1-A_1@U@(I2+V.copy().T@A_1@U)^{-1}@V.copy().T@A_1 
    recal = np.squeeze(Nbar.copy().T@inverse@Mbar.copy())#*(x**2)/N
    recal_r,recal_i = recal.real,recal.imag 
    x2_r, x2_i = x[0]**2-x[1]**2, 2*x[0]*x[1]
    result_r, result_i = recal_r*x2_r-recal_i*x2_i, recal_r*x2_i+recal_i*x2_r
    results = [result_r/N - x[0],result_i/N-x[1]]
        
    return results
### SPARSE NETWORK WITH IDENTICALLY INDEPENDENT CONNECTIVITY 
''' sparse_independent():'''
# generate mean matrix
nrank,nn,gnn = 1,200,4
Cs,kE,kI     = 60,gnn,1
Nt    = np.array([gnn*nn,nn])
NE,NI = Nt[0],Nt[1]
N     = NE+NI
## useful for determining g
Nparams  = np.array([NE,NI])
Npercent = Nparams/N
nrank,ntrial,neta,nvec,nmaxFP = 1,30,1,2,3#20,1,2,3
cuttrials = int(ntrial/2)
iktrial   = 0

#### network settings
ngEI      = 21 # iid Bernoulli random variables -- change g 
gEIseries = np.linspace(2,4,ngEI)
shiftx    = 1.5

RERUN     = 1
if RERUN == 0:
    data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_spr_PositiveTF_kappa_1.npz"
    # np.savez(data_name, **data)
    data      = np.load(data_name)

## heterogeneous degree of symmetry: amplitudes and signs
coeffetaEsub  = np.array([1.0,1.0,1.0])#
coeffetaTotal = np.array([1.0,1.0,1.0])#
# coeffetaTotal = np.zeros(3)
signetaEsub  = np.ones(3)
signetaTotal = np.ones(3)

ppercentEsub    = np.ones(2)
ppercentEsub[0] = 0.5
ppercentEsub[1] = 1.0-ppercentEsub[0]
## E->total ppercentEsub[0]/2.0,ppercentEsub[1]/2.0, (I) 1/2.0

#### recording variables 
RAmvecseries,LAmvecseries = np.zeros((ngEI,N)),np.zeros((ngEI,N))
BAmvseries = np.zeros((ngEI,N),dtype=complex)

#### the original Sparse Model, eigenvectors, decomposing Sparse Matrix (Bernoulli)
Rsprvecseries,Lsprvecseries=np.zeros((ngEI,ntrial,N,nvec*2),dtype=complex),np.zeros((ngEI,ntrial,N,nvec*2),dtype=complex)
Bsprvseries  = np.zeros((ngEI,ntrial,N),dtype=complex)
Radiusseries = np.zeros(ngEI)
#### estimated rank-one vectors according to perturbation theory (reconstruction)
RsprvecTseries,LsprvecTseries=np.zeros((ngEI,ntrial,N,nvec*2)),np.zeros((ngEI,ntrial,N,nvec*2))
#### Gaussian iid random connections (the equivalent Gaussian connectivity)
Riidvecseries,Liidvecseries   = np.zeros((ngEI,ntrial,N,nvec*2)),np.zeros((ngEI,ntrial,N,nvec*2))
RiidvecTseries,LiidvecTseries = np.zeros((ngEI,ntrial,N,nvec*2)),np.zeros((ngEI,ntrial,N,nvec*2))
Biidvseries = np.zeros((ngEI,ntrial,N),dtype=complex)
#### statistical properties
armu,sigrcov = np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2)) # 2 for E and I
almu,siglcov = np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))
siglr = np.zeros((ngEI,ntrial,2))

if(RERUN==0):
    Rsprvecseries[:,:,:,0] = data['Rsprvecseries']
    Lsprvecseries[:,:,:,0] = data['Lsprvecseries']
    RsprvecTseries[:,:,:,0] = data['RsprvecTseries']
    LsprvecTseries[:,:,:,0] = data['LsprvecTseries']
    Bsprvseries[:,:,0]      = data['Bsprvseries']
    sigrcov = data['sigrcov']
    siglcov = data['siglcov']


if(RERUN==0):
    kappaintersect_Sparse = data['kappaintersect_Sparse']
    kappaintersect_Full   = data['kappaintersect_Full']
    kappaintersect_R1     = data['kappaintersect_R1']


#### Dynamics --- temporal evolution 
tt  = np.linspace(0,100,500)
dt  = tt[2]-tt[1]
ntt = len(tt)
## compare three neuron population activities
xfpseries_Sparse = np.zeros((ngEI,ntrial,N,ntt))
xfpseries_Full   = np.zeros((ngEI,ntrial,N,ntt))
xfpseries_R1     = np.zeros((ngEI,ntrial,N,ntt))

#### fixed points of dynamical variable kappa
kappaintersect_Sparse = np.zeros((ngEI,ntrial,nmaxFP*2))
kappaintersect_Full   = np.zeros((ngEI,ntrial,nmaxFP*2))
kappaintersect_R1     = np.zeros((ngEI,ntrial,nmaxFP*2))
if(RERUN==0):
    xfpseries_Sparse[:,:,:,-1] = data['xfpseries_Sparse']
    xfpseries_Full[:,:,:,-1]   = data['xfpseries_Full']
    xfpseries_R1[:,:,:,-1]     = data['xfpseries_R1']
#### @YX  temporal evolution of dynamical variable kappa
kappaseries_R1   = np.zeros((ngEI,ntrial,ntt))


#### EPSP ---- A_E
EPSP,a,b=60*4*0.025/kE/Cs,0,0#0.025,0,0# JE PER EPSP 

if(RERUN):
    ''' Iterative Processing '''
    for ig, gEI in enumerate (gEIseries):
        print('>>>>>>> simulating neuronal activity ... >>>>>>')
        ## mean matrix, reference eigenvalue and eigenvectors
        Am   = np.zeros((N,N))
        IPSP = gEI*EPSP # gJ, g is gradually changing, lambda0 is changing along with g
        JE0  = EPSP*kE*Cs
        JI0  = IPSP*kI*Cs
        Am[:,:NE],Am[:,NE:] = JE0/NE,-JI0/NI
        xAm,yAm = np.zeros((N,1)),np.zeros((N,1))
        eigvAm  = np.zeros(N)
        eigvAm[0] = (JE0-JI0)
        xAm[:NE,0],xAm[NE:,0] = 1,1 #consider m&n rather than r&l
        yAm[:NE,0],yAm[NE:,0] = N*JE0/NE,-N*JI0/NI 
        RAmvecseries[ig,:],LAmvecseries[ig,:] = xAm[:,0],yAm[:,0]
        BAmvseries[ig,:] = eigvAm[:]
        print('eigvAm:',eigvAm[0])

        for iktrial in range(ntrial):
            ## dilute/sparse matrix 
            Jdilute = np.zeros((N,N))
            JE,JI = EPSP,IPSP#JE0/Cs/kE,JI0/Cs/kI
            for inn in range(NE+NI):
                # for NE
                Einput = randbin(1,NE,1-kE*Cs/NE)
                Jdilute[inn,:NE] = Einput*JE
                # for NE
                Iinput = randbin(1,NI,1-kI*Cs/NI)
                Jdilute[inn,NE:] = Iinput*(-JI)

            # overall
            J = Jdilute.copy()
            ## subtract mean value
            # X = Jdilute.copy() - Am.copy() # random matrix, with zero mean 
            X = Jdilute.copy() # random matrix, with zero mean 
            X[:NE,:NE] = X[:NE,:NE] - np.mean(X[:NE,:NE].flatten())
            X[:NE,NE:] = X[:NE,NE:] - np.mean(X[:NE,NE:].flatten())
            X[NE:,:NE] = X[NE:,:NE] - np.mean(X[NE:,:NE].flatten())
            X[NE:,NE:] = X[NE:,NE:] - np.mean(X[NE:,NE:].flatten())
            ''' Original Sparse Network '''
            # #### @YX 1109 Note -- extreme value because of sort=1?
            # eigvJ,leigvec,reigvec,xnorm0,ynorm0=decompNormalization(J,xAm,yAm,xAm,yAm,nparams = Nparams,sort=1)
            #### @YX 1209 NOTE --- SORT 1 -- TO MAKE EIGENVALUES SPECIAL
            eigvJ,leigvec,reigvec,xnorm0,ynorm0=decompNormalization(J,xAm,yAm,xAm,yAm,nparams = Nparams,sort=1)
            Rsprvecseries[ig,iktrial,:,0],Lsprvecseries[ig,iktrial,:,0] = xnorm0[:,0].copy(),ynorm0[:,0].copy()
            Bsprvseries[ig,iktrial,:] = eigvJ.copy()
            # print('Sparse eigenvalue:',eigvJ[0],', ', np.sum(xnorm0*ynorm0)/N)


            #### statistical properties of the elements on the unit rank eigenvectors
            axrmu,aylmu,sigxr,sigyl,sigcov = numerical_stats(xnorm0,ynorm0,xAm,yAm,eigvJ,nrank,2,ppercent=Npercent)
            armu[ig,iktrial,:],almu[ig,iktrial,:]       = axrmu[:,0],aylmu[:,0]
            sigrcov[ig,iktrial,:],siglcov[ig,iktrial,:] = sigxr[:,0],sigyl[:,0]
            siglr[ig,iktrial,:] = sigcov[:,0,0]
            
            xnorm0, ynorm0 = Rsprvecseries[ig,iktrial,:,0].copy(),Lsprvecseries[ig,iktrial,:,0]
            xnorm0, ynorm0 = np.reshape(xnorm0,(N,1)),np.reshape(ynorm0,(N,1))
            xmu,ymu   = armu[ig,iktrial,:].copy(),almu[ig,iktrial,:].copy()
            xsig,ysig = sigrcov[ig,iktrial,:].copy(),siglcov[ig,iktrial,:].copy()
            yxcov     = siglr[ig,iktrial,:].copy()


            #### dynamics of the original-defined E-I network with Bernoulli connections
            Jpt = J.copy()
            if iktrial<cuttrials:
                xinit = np.random.normal(5,1e-1,(1,N))
            else:
                xinit = np.random.normal(0,0,(1,N))
            xinit     = np.squeeze(xinit)
            xtemporal = odesimulationP(tt,xinit,Jpt,0)
            xfpseries_Sparse[ig,iktrial,:,:] = xtemporal.T.copy()

            kappanum = np.zeros(3)
            xact     = np.squeeze(xfpseries_Sparse[ig,iktrial,:,-1])
            ## <n,phi(x)> -- kappa_eq (unperturbed mean, perturbed check below)
            if(1):#(ymu[0])*(yAm[0,0])>0):##((ymu[0]*eigvJ[0])*(yAm[0,0]*eigvAm[0])>0):
                kappanum[0] = yAm[0,0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0]+yAm[NE,0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            else:
                kappanum[0] = -yAm[0,0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0]-yAm[NE,0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]

            kappaintersect_Sparse[ig,iktrial,:3]=kappanum[:].copy()

            #### dynamics of the equivalent Gaussian E-I network
            ## variance
            gE2,gI2  = EPSP**2*(1-kE*Cs/NE)*kE*Cs*N/NE,IPSP**2*(1-kI*Cs/NI)*kI*Cs*N/NI
            alphaNEI = Npercent.copy()
            gMmat    = np.array([[gE2*alphaNEI[0],gI2*alphaNEI[1]],[gE2*alphaNEI[0],gI2*alphaNEI[1]]])
            eigvgm,eigvecgm = la.eig(gMmat) 
            r_g2 = np.max(eigvgm)
            r_g  = np.sqrt(r_g2)
            Radiusseries[ig] = r_g # effective radius of the eigenvalue bulk
            Xgiid = iidGaussian([0,1.0/np.sqrt(N)],[N,N]) #### use the normalized random connectivity (homogeneous random gain g_{benchmark})
            Xgiid[:NE,:NE]*=np.sqrt(gE2)
            Xgiid[:NE,NE:]*=np.sqrt(gI2)
            Xgiid[NE:,:NE]*=np.sqrt(gE2)
            Xgiid[NE:,NE:]*=np.sqrt(gI2)

            Jgiid = Xgiid.copy()+Am.copy()

            #### connectivity statistics of the equivalent Gaussian random network
            eigvJiid,leigveciid,reigveciid,xnorm0iid,ynorm0iid=decompNormalization(Jgiid,xAm,yAm,xAm,yAm,nparams = Nparams,sort=1)
            print('Gaussian eigenvalue:',eigvJiid[0],', ', np.sum(xnorm0iid*ynorm0iid)/N)
            Riidvecseries[ig,iktrial,:,0],Liidvecseries[ig,iktrial,:,0] = xnorm0iid[:,0].copy(),ynorm0iid[:,0].copy()
            Biidvseries[ig,iktrial,:] = eigvJiid.copy()
            axrmuiid,aylmuiid,sigxriid,sigyliid,sigcoviid = numerical_stats(xnorm0iid,ynorm0iid,xAm,yAm,eigvJiid,nrank,2,ppercent=Npercent)

            xtemporal = odesimulationP(tt,xinit,Jgiid,0)
            xfpseries_Full[ig,iktrial,:,:] = xtemporal.T.copy()
            ## 2 populations 
            kappanum = np.zeros(3)
            xact = np.squeeze(xfpseries_Full[ig,iktrial,:,-1])
            ## use <n,phi(x)_{Gau}>
            if(1):#(aylmuiid[0,0])*(yAm[0,0])>0):##if((aylmuiid[0,0]*eigvJiid[0])*(yAm[0,0]*eigvAm[0])>0): ## already considered
                kappanum[0] = yAm[0,0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0]+yAm[NE,0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            else:
                kappanum[0] = -yAm[0,0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0]-yAm[NE,0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            kappaintersect_Full[ig,iktrial,:3]=kappanum[:].copy()


            #### rank-one approximation 
            # # from the equivalent Gaussian network
            # xnormt, ynormt = Xgiid.copy()@xAm.copy()/eigvAm[0], Xgiid.copy().T@yAm.copy()/eigvAm[0] # use Gaussian random connectivity
            # xnormt, ynormt = xnormt+xAm.copy(), ynormt+yAm.copy()
            ### use sparse random network
            eigvnorm = np.real(Bsprvseries[ig,iktrial,0])#eigvAm[0]#
            xnormt, ynormt = (X.copy())@xAm.copy()/eigvnorm, (X.copy()).T@yAm.copy()/eigvnorm
            xnormt, ynormt = xnormt+xAm.copy(), ynormt+yAm.copy()
            ### ------------ renew 
            kappanum = np.zeros(3)
            if(1):
                xact = np.reshape(np.squeeze(xfpseries_Sparse[ig,iktrial,:,-1]),(N,1))
                kappanum[0] = np.squeeze(ynormt.copy().T@(1.0+np.tanh(xact.copy()-shiftx)))/N
            else:
                xact = np.squeeze(xfpseries_Sparse[ig,iktrial,:,-1])
                mvec_norm = np.reshape(xnormt,(N,1))
                kappanum[0] = np.squeeze(np.reshape(np.squeeze(xact),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            kappaintersect_Sparse[ig, iktrial, :3] = kappanum[:].copy()

            # # normalize -- unnecessary 
            # # xnormt = xnormt.copy()/np.linalg.norm(xnormt.copy())
            # # ynormt = ynormt.copy()
            # #### ---------- @YX  ---------------
            # #### @YX: make sure that ynormt.T@xnormt = Bsprvseries[ig,iktrial,:]*N 
            # check_lambda  = np.squeeze(np.reshape(ynormt,(1,N))@np.reshape(xnormt,(N,1)))
            # should_lambda = np.real(Bsprvseries[ig,iktrial,0]*N)
            # ynormt = ynormt/check_lambda*should_lambda
            # #### -------------------------------------
            # _,_,xnormt,ynormt=Normalization(xnormt.copy(),ynormt.copy(),xAm.copy(),yAm.copy(),nparams=Nparams,sort=0,nrank=1)
            RsprvecTseries[ig,iktrial,:,0],LsprvecTseries[ig,iktrial,:,0]=xnormt[:,0].copy(),ynormt[:,0].copy()

            r1Matt = np.real(xnormt@ynormt.T)
            r1Matt = r1Matt/N    
            # use the same initial values
            xtemporal = odesimulationP(tt,xinit,r1Matt,0)
            xfpseries_R1[ig,iktrial,:,:] = xtemporal.T.copy()
            kappanum = np.zeros(3)
            xact = np.squeeze(xfpseries_R1[ig,iktrial,:,-1])
            ### @YX DEC
            kappaseries_R1[ig, iktrial, :] = np.squeeze(
    xtemporal.copy()@np.reshape(xnormt, (-1, 1)))/np.sum(xnormt**2)
            ## use yAm -- unperturbed centre.
            if(1):
                # do not change
                kappanum[0] = yAm[0,0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0]+yAm[NE,0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            else:
                kappanum[0] = -yAm[0,0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0]-yAm[NE,0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            kappaintersect_R1[ig,iktrial,:3]=kappanum[:].copy()

def analysis():

    # def criticaltrans_radius():
    cinit = 1
    KE,KI = kE*Cs,kI*Cs
    a = KE/NE #### represent sparsity
    CriticCradl= fsolve(criticcRad,[cinit],args=(EPSP,KE,KI,a))
    print("critical c for transition:",CriticCradl)

    cinit = 5
    KE,KI = kE*Cs,kI*Cs
    a = KE/NE #### represent sparsity
    CriticCradh= fsolve(criticcRad,[cinit],args=(EPSP,KE,KI,a))
    print("critical c for transition:",CriticCradh)
    
    xticks = np.linspace(-50,20,3)
    xlims  = [-50,20]
    yticks = [-0.5,0,3]#np.linspace(0,2,2)
    ylims  = [-0.5,3]#[-2,3]
    theta  = np.linspace(0, 2 * np.pi, 200)
    ''' how eigenvalue outlier changes with g_bar '''

    iktrial=3
    ig=10#5#3.5~~~~#ngEI-1##6
    alphaval=0.10
    edgv='black'
    cm='br'
    fig,ax=plt.subplots(figsize=(5,3))  
    nrank=1
    ntrial = np.shape(Rsprvecseries)[1]

    
    meanmE,meanmI,meannE,meannI=np.zeros(nrank),np.zeros(nrank),np.zeros(nrank),np.zeros(nrank)
    mEvec,mIvec,nEvec,nIvec=np.squeeze(Rsprvecseries[ig,:,:NE,0]),np.squeeze(Rsprvecseries[ig,:,NE:,0]),np.squeeze(Lsprvecseries[ig,:,:NE,0]),np.squeeze(Lsprvecseries[ig,:,NE:,0])
    mEvec,mIvec,nEvec,nIvec=mEvec.flatten(),mIvec.flatten(),nEvec.flatten(),nIvec.flatten()

    scale_std=3.0
    for irank in range(nrank):
        meanmEtotal,stdmEtotal = np.mean(mEvec),np.std(mEvec)
        varmE = mEvec - meanmEtotal
        idxwhere = np.where(np.abs(varmE)>scale_std*stdmEtotal)
        mEvec[idxwhere]=meanmEtotal
        meanmE[irank]=np.mean(mEvec)

        # pruning I
        meanmItotal,stdmItotal = np.mean(mIvec),np.std(mIvec)
        varmI    = mIvec - meanmItotal
        idxwhere = np.where(np.abs(varmI)>scale_std*stdmItotal)
        mIvec[idxwhere] = meanmItotal
        meanmI[irank]   = np.mean(mIvec)
        
        # n vector
        # pruning E
        meannEtotal,stdnEtotal = np.mean(nEvec),np.std(nEvec)
        varnE    = nEvec - meannEtotal
        idxwhere = np.where(np.abs(varnE)>scale_std*stdnEtotal)
        nEvec[idxwhere] = meannEtotal
        meannE[irank]   = np.mean(nEvec)

        # pruning I
        meannItotal,stdnItotal = np.mean(nIvec),np.std(nIvec)
        varnI    = nIvec - meannItotal
        idxwhere = np.where(np.abs(varnI)>scale_std*stdnItotal)
        nIvec[idxwhere] = meannItotal
        meannI[irank]   = np.mean(nIvec)

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

    idrands=np.random.choice(np.arange(0,len(nIvec)),size=300,replace=False)
    ax.scatter(nIvec[idrands],mIvec[idrands],s=5.0,c='tab:blue',alpha=0.2)#alphaval)#cm[1],alpha=alphaval)
    ax.scatter(nEvec[idrands],mEvec[idrands],s=5.0,c='tab:red',alpha=0.2)#alphaval)#cm[0],alpha=alphaval)

    ax.plot([meannE[0],meannE[0]+np.sqrt(N)*dirvecE[0,0]],[meanmE[0],meanmE[0]+np.sqrt(N)*dirvecE[1,0]],color='gray',linestyle='--',linewidth=1.5)
    ax.plot([meannE[0],meannE[0]+np.sqrt(N)*dirvecE[0,1]],[meanmE[0],meanmE[0]+np.sqrt(N)*dirvecE[1,1]],color='gray',linestyle='--',linewidth=1.5)
    ax.plot([meannI[0],meannI[0]+np.sqrt(N)*dirvecI[0,0]],[meanmI[0],meanmI[0]+np.sqrt(N)*dirvecI[1,0]],color=edgv,linestyle='--',linewidth=1.5)
    ax.plot([meannI[0],meannI[0]+np.sqrt(N)*dirvecI[0,1]],[meanmI[0],meanmI[0]+np.sqrt(N)*dirvecI[1,1]],color=edgv,linestyle='--',linewidth=1.5)
    ax.scatter(meannE[0],meanmE[0],s=50,color='white')
    ax.scatter(meannI[0],meanmI[0],s=50,color='white')
    # confidence_ellipse(np.real(nEvec),np.real(mEvec),ax,edgecolor=edgv)
    # confidence_ellipse(np.real(nIvec),np.real(mIvec),ax,edgecolor=edgv)


    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # ax.set_aspect('equal')

    ### C. Gaussian mixture model
    from scipy import stats
    np.random.seed(41)
    xticks = np.array([-50,0,20])#np.linspace(-50,5,5)
    xlims = [-50,20]
    
    yticks = [-0.5,0,3]#np.linspace(-0.2,2,3)
    ylims = [-0.5,3]
    nvbins = np.linspace(-50,20,40)
    mvbins = np.linspace(-0.5,3,40)

    nEkde = stats.gaussian_kde(np.real(nEvec))
    nIkde = stats.gaussian_kde(np.real(nIvec))
    mEkde = stats.gaussian_kde(np.real(mEvec))
    mIkde = stats.gaussian_kde(np.real(mIvec))
    xx = np.linspace(-50, 20, 100)
    fign, axn = plt.subplots(figsize=(5,1))
    axn.hist(np.real(nEvec), density=True, bins=nvbins, facecolor='tab:red', alpha=0.3)
    axn.plot(xx, nEkde(xx),c='tab:red');
    axn.hist(np.real(nIvec), density=True, bins=nvbins, facecolor='tab:blue',alpha=0.3)
    axn.plot(xx, nIkde(xx),c='tab:blue')
    axn.set_xlim(xlims)
    axn.set_xticks(xticks)

    yy  = np.linspace(-0.5,3,20)
    figm, axm = plt.subplots(figsize=(3,1))
    axm.hist(np.real(mEvec), density=True, bins=mvbins,facecolor='tab:red', alpha=0.3)
    axm.plot(yy, mEkde(yy),c='tab:red')

    axm.hist(np.real(mIvec), density=True, bins=mvbins,facecolor='tab:blue' ,alpha=0.3)
    axm.plot(yy, mIkde(yy),c='tab:blue')
    axm.set_xlim(ylims)
    axm.set_xticks(ylims)

    # def statseigenvalue():
    ### WITH CRITICAL TRANSITION POINTS
    mean_theo_eig,mean_num_eig,std_num_eig = np.zeros((ngEI,1)),np.zeros((ngEI,1)),np.zeros((ngEI,1))
    mean_theo_eig[:,0] = ((EPSP*kE*Cs)*(1-gEIseries*kI/kE))
    mean_num_eig[:,0]  = np.mean(Bsprvseries[:,:,0],axis=1)
    std_num_eig[:,0]   = np.std(Bsprvseries[:,:,0],axis=1)

    ### standard deviation of the eigenvalue outliers
    std_theo_eig = np.zeros((ngEI,1))
    std_theo_eig = (EPSP**2*NE+(EPSP*gEIseries)**2*NI)**2*(1-Cs/nn)*(Cs/nn)**3
    std_theo_eig = np.squeeze(std_theo_eig)/np.squeeze(mean_theo_eig**2)
    std_theo_eig = np.sqrt(std_theo_eig)

    figstd, axstd = plt.subplots(figsize=(3,3))
    
    ylims  = [0,0.6]
    
    axstd.set_xlim(ylims)
    axstd.set_xticks(ylims)
    axstd.set_ylim(ylims)
    axstd.set_yticks(ylims)
    axstd.scatter(std_theo_eig,std_num_eig,c=gEIseries,cmap=matplotlib.cm.Blues_r,s=20,marker='o')
    axstd.scatter(std_theo_eig[15:17],std_num_eig[15:17],c= '#FF000000',edgecolor='k',s=50,marker='o')
    axstd.set_aspect('equal')
    axstd.plot(ylims,ylims,linestyle='--',c='gray')


    figeig, axeig = plt.subplots(figsize=(6,2))
    xticks = [gEIseries[0],gEIseries[-1]]#[2,5.5]
    xlims  = [gEIseries[0],gEIseries[-1]]#[2,5.5]
    yticks = [-2,0,4]
    ylims  = [-2,4]


    axeig.plot(gEIseries,mean_theo_eig[:,0],color='black',linewidth=1.5,linestyle='--')
    axeig.plot(gEIseries,mean_theo_eig[:,0]+std_theo_eig[:],color='tab:red',linewidth=1,linestyle='--')
    axeig.plot(gEIseries,mean_theo_eig[:,0]-std_theo_eig[:],color='tab:red',linewidth=1,linestyle='--')


    peigv_mean,peigv_std,neigv_mean,neigv_std = np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)
    for ig in range(ngEI):
        peigv_mean[ig] = np.mean(Bsprvseries[ig,np.where(Bsprvseries[ig,:,0]>0)[0],0])
        neigv_mean[ig] = np.mean(Bsprvseries[ig,np.where(Bsprvseries[ig,:,0]<0)[0],0])
        peigv_std[ig]  = np.std(Bsprvseries[ig,np.where(Bsprvseries[ig,:,0]>0)[0],0])
        neigv_std[ig]  = np.std(Bsprvseries[ig,np.where(Bsprvseries[ig,:,0]<0)[0],0])
    axeig.fill_between(gEIseries,peigv_mean-peigv_std,peigv_mean+peigv_std,alpha=0.3,facecolor='red')#color='gray',alpha=0.5)
    axeig.fill_between(gEIseries,neigv_mean-neigv_std,neigv_mean+neigv_std,alpha=0.3,facecolor='red')#color='gray',alpha=0.5)

    axeig.plot(CriticCradl*np.ones(2),np.array([ylims[0],ylims[-1]]),c='gray',linewidth=1.5,linestyle='--')
    axeig.plot(CriticCradh*np.ones(2),np.array([ylims[0],ylims[-1]]),c='gray',linewidth=1.5,linestyle='--')

    axeig.set_xlim(xlims)
    axeig.set_xticks(xticks)
    axeig.set_ylim(ylims)
    axeig.set_yticks(yticks)

    ### >>>>>>>>>>>> RADIUS >>>>>>>>>>>>>>>
    # for ig,gEI in enumerate (gEIseries[-3:]):
    radius_theo = np.zeros_like(gEIseries)
    for ig in range(len(gEIseries)):
        gEI  = gEIseries[ig]
        IPSP = gEI*EPSP
        JE0  = EPSP*kE*Cs
        JI0  = IPSP*kI*Cs
        ## variance
        gE2,gI2  = EPSP**2*(1-kE*Cs/NE)*kE*Cs/NE,IPSP**2*(1-kI*Cs/NI)*kI*Cs/NI
        gE2,gI2  = gE2*N,gI2*N
        alphaNEI = Npercent.copy()
        gMmat = np.array([[gE2*alphaNEI[0],gI2*alphaNEI[1]],[gE2*alphaNEI[0],gI2*alphaNEI[1]]])
        eigvgm,eigvecgm = la.eig(gMmat) 
        r_g2  = np.max(eigvgm)
        radius_theo[ig] = np.sqrt(r_g2)
    axeig.fill_between(gEIseries,radius_theo,-radius_theo,color='gray',alpha=0.2)

    #### statistical properties of the elements on the unit rank eigenvectors
    ntrial = np.shape(Rsprvecseries)[1]
    mean_theo_mvec,mean_theo_nvec = np.zeros((ngEI,2)),np.zeros((ngEI,2))
    mean_num_mvec,mean_num_nvec   = np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))
    var_theo_mvec,var_theo_nvec   = np.zeros((ngEI,2)),np.zeros((ngEI,2))
    var_num_mvec,var_num_nvec     = np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))
    for ig, gEI in enumerate (gEIseries): 
        IPSP = gEI*EPSP
        JE0  = EPSP*kE*Cs
        JI0  = IPSP*kI*Cs
        ## variance
        gE2,gI2  = EPSP**2*(1-kE*Cs/NE)*kE*Cs/NE,IPSP**2*(1-kI*Cs/NI)*kI*Cs/NI
        gE2,gI2  = gE2*N,gI2*N
        alphaNEI = Npercent.copy()
        gMmat   = np.array([[gE2*alphaNEI[0],gI2*alphaNEI[1]],[gE2*alphaNEI[0],gI2*alphaNEI[1]]])
        eigvgm,eigvecgm = la.eig(gMmat) 

        # theoretical variances
        mean_theo_mvec[ig,:] = 1
        mean_theo_nvec[ig,0],mean_theo_nvec[ig,1] = 2*JE0,-2*JI0
        
        gQE,gQI = EPSP*np.sqrt(kE*Cs/NE*(1-kE*Cs/NE))*np.sqrt(N),IPSP*np.sqrt(kI*Cs/NI*(1-kI*Cs/NI))*np.sqrt(N)
        lambda0=(JE0-JI0)
        var_theo_mvec[ig,0],var_theo_mvec[ig,1] = (gQE**2*NE/N+gQI**2*NI/N)/lambda0**2,(gQE**2*NE/N+gQI**2*NI/N)/lambda0**2
        var_theo_nvec[ig,0],var_theo_nvec[ig,1] = ((N*JE0/NE)**2*NE*(gQE**2/N)+(N*JI0/NI)**2*NI*(gQE**2/N))/lambda0**2,((N*JE0/NE)**2*NE*(gQI**2/N)+(N*JI0/NI)**2*NI*(gQI**2/N))/lambda0**2

    #### statistics of sigma_{m^2} and sigma_{n^2}
    mean_sigma_m,mean_sigma_n = np.zeros((ngEI,2)),np.zeros((ngEI,2))
    std_sigma_m,std_sigma_n   = np.zeros((ngEI,2)),np.zeros((ngEI,2))

    mean_sigma_m[:,0],mean_sigma_m[:,1] = np.mean(sigrcov[:,:,0],axis=1),np.mean(sigrcov[:,:,1],axis=1),
    mean_sigma_n[:,0],mean_sigma_n[:,1] = np.mean(siglcov[:,:,0],axis=1),np.mean(siglcov[:,:,1],axis=1),

    std_sigma_m[:,0],std_sigma_m[:,1] = np.std(sigrcov[:,:,0],axis=1),np.std(sigrcov[:,:,1],axis=1),
    std_sigma_n[:,0],std_sigma_n[:,1] = np.std(siglcov[:,:,0],axis=1),np.std(siglcov[:,:,1],axis=1),


    fign, axn = plt.subplots(2,1,figsize=(6,3),sharex=True,tight_layout=True)
    xticks    = [gEIseries[0],gEIseries[-1]]#[2,5.5]
    xlims     = [gEIseries[0],gEIseries[-1]]#[2,5.5]
    yticks0   = [0,150]#[0,80]
    ylims0    = [0,150]#[0,80]#[-5,30]
    yticks1   = [0,1500]#[0,1000]
    ylims1    = [0,1500]#[0,1000]#[-50,500]
    axn[0].plot(gEIseries,var_theo_nvec[:,0],c='tab:red',lw=1.5)
    axn[1].plot(gEIseries,var_theo_nvec[:,1],c='tab:blue',lw=1.5)

    axn[0].plot(CriticCradl*np.ones(2),np.array([ylims0[0],ylims0[-1]]),c='gray',linewidth=1.5,linestyle='--')
    axn[0].plot(CriticCradh*np.ones(2),np.array([ylims0[0],ylims0[-1]]),c='gray',linewidth=1.5,linestyle='--')
    axn[1].plot(CriticCradl*np.ones(2),np.array([ylims1[0],ylims1[-1]]),c='gray',linewidth=1.5,linestyle='--')
    axn[1].plot(CriticCradh*np.ones(2),np.array([ylims1[0],ylims1[-1]]),c='gray',linewidth=1.5,linestyle='--')

    #### excitatory
    ### do not plot numerical mean
    axn[0].plot(gEIseries,mean_sigma_n[:,0],color='tab:red',linewidth=1.5,linestyle='--')
    # axn[0].fill_between(gEIseries,mean_sigma_n[:,0]+std_sigma_n[:,0],mean_sigma_n[:,0]-std_sigma_n[:,0], alpha=0.3,facecolor='red')
    #### inhibitory
    axn[1].plot(gEIseries,mean_sigma_n[:,1],color='tab:blue',linewidth=1.5,linestyle='--')
    # axn[1].fill_between(gEIseries,mean_sigma_n[:,1]+std_sigma_n[:,1],mean_sigma_n[:,1]-std_sigma_n[:,1], alpha=0.3,facecolor='blue')
    axn[0].set_title('left vector n\n (main code record)')

    for i in range(2):
        axn[i].set_xlim(xlims)
        axn[i].set_xticks(xticks)
    axn[0].set_ylim(ylims0)
    axn[0].set_yticks(yticks0)
    axn[1].set_ylim(ylims1)
    axn[1].set_yticks(yticks1)

    figm, axm = plt.subplots(2,1,figsize=(6,3),sharex=True,tight_layout=True)
    yticks    = [0,2]#np.linspace(0,1,2)
    ylims     = [0,2]

    axm[0].plot(gEIseries,var_theo_mvec[:,0],c='tab:red')
    axm[1].plot(gEIseries,var_theo_mvec[:,1],c='tab:blue')

    #### excitatory
    axm[0].plot(gEIseries,mean_sigma_m[:,0],color='tab:red',linewidth=1.5,linestyle='--')
    # axm[0].fill_between(gEIseries,mean_sigma_m[:,0]+std_sigma_m[:,0],mean_sigma_m[:,0]-std_sigma_m[:,0], alpha=0.3,facecolor='red')
    #### inhibitory
    axm[1].plot(gEIseries,mean_sigma_m[:,1],color='tab:blue',linewidth=1.5,linestyle='--')
    # axm[1].fill_between(gEIseries,mean_sigma_m[:,1]+std_sigma_m[:,1],mean_sigma_m[:,1]-std_sigma_m[:,1], alpha=0.3,facecolor='blue')

    axm[0].plot(CriticCradl*np.ones(2),np.array([ylims[0],ylims[-1]]),c='gray',linewidth=1.5,linestyle='--')
    axm[0].plot(CriticCradh*np.ones(2),np.array([ylims[0],ylims[-1]]),c='gray',linewidth=1.5,linestyle='--')
    axm[1].plot(CriticCradl*np.ones(2),np.array([ylims[0],ylims[-1]]),c='gray',linewidth=1.5,linestyle='--')
    axm[1].plot(CriticCradh*np.ones(2),np.array([ylims[0],ylims[-1]]),c='gray',linewidth=1.5,linestyle='--')

    for i in range(2):
        axm[i].set_xlim(xlims)
        axm[i].set_xticks(xticks)
        axm[i].set_ylim(ylims)
        axm[i].set_yticks(yticks)
    axm[0].set_title('right vector m\n (main code record)')

    '''
    axrmu,aylmu,sigxr,sigyl,sigcov = numerical_stats(xnorm0,ynorm0,xAm,yAm,eigvJ,nrank,2,ppercent=Npercent)
    armu[ig,iktrial,:],almu[ig,iktrial,:] = axrmu[:,0],aylmu[:,0]
    sigrcov[ig,iktrial,:],siglcov[ig,iktrial,:]= sigxr[:,0],sigyl[:,0]
    siglr[ig,iktrial,:] = sigcov[:,0,0]
    '''
    ntrial = np.shape(Rsprvecseries)[1]
    mean_theo_mvec,mean_theo_nvec = np.zeros((ngEI,2)),np.zeros((ngEI,2))
    mean_num_mvec,mean_num_nvec = np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))
    var_theo_mvec,var_theo_nvec = np.zeros((ngEI,2)),np.zeros((ngEI,2))
    var_num_mvec,var_num_nvec = np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))
    for ig, gEI in enumerate (gEIseries): 
        IPSP = gEI*EPSP
        JE0  = EPSP*kE*Cs
        JI0  = IPSP*kI*Cs
        ## variance
        gE2,gI2  = EPSP**2*(1-kE*Cs/NE)*kE*Cs/NE,IPSP**2*(1-kI*Cs/NI)*kI*Cs/NI
        gE2,gI2  = gE2*N,gI2*N
        alphaNEI = Npercent.copy()
        gMmat = np.array([[gE2*alphaNEI[0],gI2*alphaNEI[1]],[gE2*alphaNEI[0],gI2*alphaNEI[1]]])
        eigvgm,eigvecgm=la.eig(gMmat) 
        for iktrial in range(ntrial):
            ## numerical variances
            mEvec,mIvec,nEvec,nIvec=np.squeeze(Rsprvecseries[ig,iktrial,:NE,0]),np.squeeze(Rsprvecseries[ig,iktrial,NE:,0]),np.squeeze(Lsprvecseries[ig,iktrial,:NE,0]),np.squeeze(Lsprvecseries[ig,iktrial,NE:,0])
            mEvec,mIvec,nEvec,nIvec=mEvec.flatten(),mIvec.flatten(),nEvec.flatten(),nIvec.flatten()
            scale_std=3.0
            for irank in range(nrank):
                mean_num_mvec[ig,iktrial,0]=np.mean(mEvec)
                var_num_mvec[ig,iktrial,0] = np.std(mEvec)**2

                mean_num_mvec[ig,iktrial,1]=np.mean(mIvec)
                var_num_mvec[ig,iktrial,1] = np.std(mIvec)**2
                
                mean_num_nvec[ig,iktrial,0]=np.mean(nEvec)
                var_num_nvec[ig,iktrial,0] = np.std(nEvec)**2

                mean_num_nvec[ig,iktrial,1]=np.mean(nIvec)
                var_num_nvec[ig,iktrial,1] = np.std(nIvec)**2

        ## theoretical variances
        mean_theo_mvec[ig,:] = 1
        mean_theo_nvec[ig,0],mean_theo_nvec[ig,1] = 2*JE0,-2*JI0
        
        gQE,gQI = EPSP*np.sqrt(kE*Cs/NE*(1-kE*Cs/NE))*np.sqrt(N),IPSP*np.sqrt(kI*Cs/NI*(1-kI*Cs/NI))*np.sqrt(N)
        lambda0=(JE0-JI0)
        var_theo_mvec[ig,0],var_theo_mvec[ig,1] = (gQE**2*NE/N+gQI**2*NI/N)/lambda0**2,(gQE**2*NE/N+gQI**2*NI/N)/lambda0**2
        var_theo_nvec[ig,0],var_theo_nvec[ig,1] = ((N*JE0/NE)**2*NE*(gQE**2/N)+(N*JI0/NI)**2*NI*(gQE**2/N))/lambda0**2,((N*JE0/NE)**2*NE*(gQI**2/N)+(N*JI0/NI)**2*NI*(gQI**2/N))/lambda0**2
        # print('varn:',var_theo_nvec[ig,:],gQI/gQE)

    #### statistics of sigma_{m^2} and sigma_{n^2}
    mean_sigma_m,mean_sigma_n = np.zeros((ngEI,2)),np.zeros((ngEI,2))
    std_sigma_m,std_sigma_n   = np.zeros((ngEI,2)),np.zeros((ngEI,2))

    mean_sigma_m[:,0],mean_sigma_m[:,1] = np.mean(var_num_mvec[:,:,0],axis=1),np.mean(var_num_mvec[:,:,1],axis=1),
    mean_sigma_n[:,0],mean_sigma_n[:,1] = np.mean(var_num_nvec[:,:,0],axis=1),np.mean(var_num_nvec[:,:,1],axis=1),

    std_sigma_m[:,0],std_sigma_m[:,1] = np.std(var_num_mvec[:,:,0],axis=1),np.std(var_num_mvec[:,:,1],axis=1),
    std_sigma_n[:,0],std_sigma_n[:,1] = np.std(var_num_nvec[:,:,0],axis=1),np.std(var_num_nvec[:,:,1],axis=1),


    ### C.  COMPARE EACH ENTRY ON THE EIGENVECTORS (RECONSTRUCT V.S. EIGENDECOMPOSITION)
    xtickms = np.linspace(-2,4,2)
    xlimms = [-2,4]
    ytickms = np.linspace(-2,4,2)
    ylimms = [-2,4]

    xticks = np.linspace(-50,20,2)
    xlims  = [-50,20]
    yticks = np.linspace(-50,20,2)
    ylims  = [-50,20]
    '''# CHOOSE ONE TRIAL'''
    idxtrial,idxtrial_ = 8,3 # ### @YX original
    fig,ax=plt.subplots(2,2,figsize=(4,4))
    idxtrial=0
    ###  c SAMPLE
    gEIsamples=np.array([6,10])

    for i in range(len(gEIsamples)):
        ax[0,i].plot(xticks,yticks,color='darkred',linestyle='--')
        ax[1,i].plot(xticks,yticks,color='darkred',linestyle='--')
        #### @YX modify 2508 -- redundancy
        #### @YX modify 2508 -- from Reigvecseries[i,...] to Reigvecseries[gEIsamples[i]...]
        idrandsE=np.random.choice(np.arange(0,NE),size=100,replace=False)
        idrandsI=np.random.choice(np.arange(NE,N),size=100,replace=False)

        ax[0,i].scatter(np.real(Rsprvecseries[gEIsamples[i],idxtrial,idrandsE]),np.real(RsprvecTseries[gEIsamples[i],idxtrial,idrandsE]),s=2,c='red',alpha=0.5)
        ax[1,i].scatter(np.real(Lsprvecseries[gEIsamples[i],idxtrial,idrandsE]),np.real(LsprvecTseries[gEIsamples[i],idxtrial,idrandsE]),s=2,c='red',alpha=0.5)
        
        ax[0,i].scatter(np.real(Rsprvecseries[gEIsamples[i],idxtrial,idrandsI]),np.real(RsprvecTseries[gEIsamples[i],idxtrial,idrandsI]),s=2,c='tab:blue',alpha=0.5)
        ax[1,i].scatter(np.real(Lsprvecseries[gEIsamples[i],idxtrial,idrandsI]),np.real(LsprvecTseries[gEIsamples[i],idxtrial,idrandsI]),s=2,c='tab:blue',alpha=0.5)
        # ax[0,i].scatter(np.real(Reigvecseries[idxgsamples[i],idxtrial,:int(NE/2)]),np.real(ReigvecTseries[idxgsamples[i],idxtrial,:int(NE/2)]),s=2,c='red',alpha=0.5)
        # ax[1,i].scatter(np.real(Leigvecseries[idxgsamples[i],idxtrial,:int(NE/2)]),np.real(LeigvecTseries[idxgsamples[i],idxtrial,:int(NE/2)]),s=2,c='red',alpha=0.5)
        
        # ax[0,i].scatter(np.real(Reigvecseries[idxgsamples[i],idxtrial,int(NE*3/2):]),np.real(ReigvecTseries[idxgsamples[i],idxtrial,int(NE*3/2):]),s=2,c='blue',alpha=0.5)
        # ax[1,i].scatter(np.real(Leigvecseries[idxgsamples[i],idxtrial,int(NE*3/2):]),np.real(LeigvecTseries[idxgsamples[i],idxtrial,int(NE*3/2):]),s=2,c='blue',alpha=0.5)
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
 
    

    #### ------------ Dynamics -----------
    dshift = 0.0
    kappa_theo_iid = np.zeros((ngEI,3))
    for ig,gEI in enumerate (gEIseries):
        alphaa = ig*1.0/ngEI
        IPSP   = gEI*EPSP
        JE0    = EPSP*kE*Cs
        JI0    = IPSP*kI*Cs
        gE2,gI2 = EPSP**2*(1-kE*Cs/NE)*kE*Cs*N/NE,IPSP**2*(1-kI*Cs/NI)*kI*Cs*N/NI
        gE,gI   = np.sqrt(gE2),np.sqrt(gI2)
        gmat    = np.array([gE,gI,gE,gI])

        #### the smallest
        init_k  = np.max(kappaintersect_Full[ig,:, 0])
        kappa_max = fsolve(iidperturbationP,init_k,args=(JE0,JI0,gmat,Nparams),xtol=1e-6,maxfev=800)
        residual0 = np.abs(iidperturbationP(kappa_max,JE0,JI0,gmat,Nparams))

        #### the middle (unstable) one
        init_k    = (alphaa*np.max(kappaintersect_Full[ig,:, 0])+(1-alphaa)*np.min(kappaintersect_Full[ig,:, 0]))
        kappa_middle = fsolve(iidperturbationP,init_k,args=(JE0,JI0,gmat,Nparams),xtol=1e-6,maxfev=800)
        residual1    = np.abs(iidperturbationP(kappa_middle,JE0,JI0,gmat,Nparams))

        #### the largest (stable) one
        init_k       = np.min(kappaintersect_Full[ig,:, 0])
        kappa_theo_iid[ig,2] = fsolve(iidperturbationP,init_k,args=(JE0,JI0,gmat,Nparams),xtol=1e-6,maxfev=800)

        if(residual0>1e-3):
            kappa_theo_iid[ig,0]=kappa_theo_iid[ig,2]
        else:
            kappa_theo_iid[ig,0]=kappa_max
        if(residual1>1e-3):
            kappa_theo_iid[ig,1]=kappa_theo_iid[ig,2]
        else:
            kappa_theo_iid[ig,1]=kappa_middle
    ## plot kappa change with JE ##
    fig,ax = plt.subplots(figsize=(6,2))
    xticks = [gEIseries[0],gEIseries[-1]]#[2,5.5]
    xlims  = [gEIseries[0],gEIseries[-1]]# [2,5.5]
    yticks = [0,6.5]#np.linspace(0,6.5,3)
    ylims  = [0,6.5]
    ax.plot(gEIseries,kappa_theo_iid[:,0],color='purple',linewidth=1.5)
    ax.plot(gEIseries,kappa_theo_iid[:,2],color='purple',linewidth=1.5)
       

    #### @YX--calculate the shaded kappa
    mean_pos_kappa_full  = np.zeros(ngEI)
    std_pos_kappa_full   = np.zeros(ngEI)
    mean_pos_kappa_r1    = np.zeros(ngEI)
    std_pos_kappa_r1     = np.zeros(ngEI)
    mean_pos_kappa_spr   = np.zeros(ngEI)
    std_pos_kappa_spr    = np.zeros(ngEI)
    pos_t0=0.7
    for ig in range(ngEI):
        pos_full = kappaintersect_Full[ig,np.where(kappaintersect_Full[ig,:,0]>=pos_t0),0]
        mean_pos_kappa_full[ig] = np.mean(pos_full)
        std_pos_kappa_full[ig]  = np.std(pos_full)
        pos_r1 = kappaintersect_R1[ig,np.where(kappaintersect_R1[ig,:,0]>=pos_t0),0]
        mean_pos_kappa_r1[ig] = np.mean(pos_r1)
        std_pos_kappa_r1[ig]  = np.std(pos_r1)
        pos_spr = kappaintersect_Sparse[ig,np.where(kappaintersect_Sparse[ig,:,0]>=pos_t0),0]
        mean_pos_kappa_spr[ig] = np.mean(pos_spr)
        std_pos_kappa_spr[ig]  = np.std(pos_spr)
    ax.fill_between(gEIseries,mean_pos_kappa_spr+std_pos_kappa_spr,mean_pos_kappa_spr-std_pos_kappa_spr, alpha=0.3,facecolor='black')


    for ig in range(ngEI):
        pos_full = kappaintersect_Full[ig,np.where(kappaintersect_Full[ig,:,0]<pos_t0),0]
        mean_pos_kappa_full[ig] = np.mean(pos_full)
        std_pos_kappa_full[ig]  = np.std(pos_full)
        pos_r1 = kappaintersect_R1[ig,np.where(kappaintersect_R1[ig,:,0]<pos_t0),0]
        mean_pos_kappa_r1[ig] = np.mean(pos_r1)
        std_pos_kappa_r1[ig]  = np.std(pos_r1)
        pos_spr = kappaintersect_Sparse[ig,np.where(kappaintersect_Sparse[ig,:,0]<pos_t0),0]
        mean_pos_kappa_spr[ig] = np.mean(pos_spr)
        std_pos_kappa_spr[ig]  = np.std(pos_spr)
    ax.fill_between(gEIseries,mean_pos_kappa_spr+std_pos_kappa_spr,mean_pos_kappa_spr-std_pos_kappa_spr, alpha=0.3,facecolor='black')

    ax.plot(CriticCradl*np.ones(2),np.array([yticks[0],yticks[-1]]),c='gray',linewidth=1.5,linestyle='--')
    ax.plot(CriticCradh*np.ones(2),np.array([yticks[0],yticks[-1]]),c='gray',linewidth=1.5,linestyle='--')


    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)


    ### ~~~~~~~~~ DEFINITION OF numerical ~~~~~~~~~
    numkappa= np.zeros((ngEI,ntrial))
    pavgkappa,pstdkappa = np.zeros(ngEI),np.zeros(ngEI)
    navgkappa,nstdkappa = np.zeros(ngEI),np.zeros(ngEI)
    for iJE in range(ngEI):
        signXE = np.mean(1.0+np.tanh(np.squeeze(xfpseries_Sparse[iJE,:,:NE,-1])-shiftx),axis=1)
        signXI = np.mean(1.0+np.tanh(np.squeeze(xfpseries_Sparse[iJE,:,NE:,-1])-shiftx),axis=1)
        ptrialXE,ntrialXE = np.where(signXE>=pos_t0)[0],np.where(signXE<pos_t0)[0]
        print('>>>P/N TRIAL',(ptrialXE),ntrialXE)
        ### CALCULATE [PHIXE/I]
        if len(ptrialXE)>0:
            for iktrial in ptrialXE:
                mvec_norm = np.reshape(RsprvecTseries[iJE,iktrial,:,0],(N,1))
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(np.squeeze(xfpseries_Sparse[iJE,iktrial,:,-1]),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            pavgkappa[iJE],pstdkappa[iJE] = np.mean(numkappa[iJE,ptrialXE]),np.std(numkappa[iJE,ptrialXE])
        if len(ntrialXE)>0:
            for iktrial in ntrialXE:
                mvec_norm = np.reshape(RsprvecTseries[iJE,iktrial,:,0],(N,1))
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(np.squeeze(xfpseries_Sparse[iJE,iktrial,:,-1]),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            navgkappa[iJE],nstdkappa[iJE] = np.mean(numkappa[iJE,ntrialXE]),np.std(numkappa[iJE,ntrialXE])
    ax.plot(gEIseries, pavgkappa,c='black', linewidth=1.5, linestyle='--')
    ax.plot(gEIseries, navgkappa,c='black', linewidth=1.5, linestyle='--')
    ax.fill_between(gEIseries, pavgkappa+pstdkappa,pavgkappa-pstdkappa, alpha=.5, facecolor='blue')


    numkappa= np.zeros((ngEI,ntrial))
    pavgkappa,pstdkappa = np.zeros(ngEI),np.zeros(ngEI)
    navgkappa,nstdkappa = np.zeros(ngEI),np.zeros(ngEI)
    for iJE in range(ngEI):
        signXE = np.mean(1.0+np.tanh(np.squeeze(xfpseries_Sparse[iJE,:,:NE,-1])-shiftx),axis=1)
        signXI = np.mean(1.0+np.tanh(np.squeeze(xfpseries_Sparse[iJE,:,NE:,-1])-shiftx),axis=1)
        ptrialXE,ntrialXE = np.where(signXE>=pos_t0)[0],np.where(signXE<pos_t0)[0]
        print('>>>P/N TRIAL',(ptrialXE),ntrialXE)
        ### CALCULATE [PHIXE/I]
        if len(ptrialXE)>0:
            for iktrial in ptrialXE:
                nvec_norm = np.reshape(LsprvecTseries[iJE,iktrial,:,0],(N,1))
                phix      = 1.0+np.tanh(np.squeeze(xfpseries_Sparse[iJE,iktrial,:,-1])-shiftx)
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(phix,(1,N))@nvec_norm)/N
            pavgkappa[iJE],pstdkappa[iJE] = np.mean(numkappa[iJE,ptrialXE]),np.std(numkappa[iJE,ptrialXE])
        if len(ntrialXE)>0:
            for iktrial in ntrialXE:
                nvec_norm = np.reshape(LsprvecTseries[iJE,iktrial,:,0],(N,1))
                phix      = 1.0+np.tanh(np.squeeze(xfpseries_Sparse[iJE,iktrial,:,-1])-shiftx)
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(phix,(1,N))@nvec_norm)/N
            navgkappa[iJE],nstdkappa[iJE] = np.mean(numkappa[iJE,ntrialXE]),np.std(numkappa[iJE,ntrialXE])

    ax.plot(gEIseries, pavgkappa,c='red', linewidth=1.5, linestyle='--')
    ax.plot(gEIseries, navgkappa,c='red', linewidth=1.5, linestyle='--')
    ax.fill_between(gEIseries, pavgkappa+pstdkappa,pavgkappa-pstdkappa, alpha=.5, facecolor='green')


    ## numerically calculate the variance and mean, using xfpseries_Full and xfpseries_R1
    variance_sparse_num, mu_sparse_num =np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))
    variance_full_num, mu_full_num =np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))
    variance_R1_num, mu_R1_num = np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))

    for ig,gEI in enumerate (gEIseries):
        for iktrial in range(ntrial):
            ## numerical results for Full Mat
            variance_full_num[ig,iktrial,0],variance_full_num[ig,iktrial,1]=np.std(xfpseries_Full[ig,iktrial,:NE,-1])**2,np.std(xfpseries_Full[ig,iktrial,NE:N,-1])**2
            mu_full_num[ig,iktrial,0],mu_full_num[ig,iktrial,1]=np.mean(xfpseries_Full[ig,iktrial,:NE,-1]),np.mean(xfpseries_Full[ig,iktrial,NE:N,-1])
            ## numerical results for Sparse Mat
            variance_sparse_num[ig,iktrial,0],variance_sparse_num[ig,iktrial,1]=np.std(xfpseries_Sparse[ig,iktrial,:NE,-1])**2,np.std(xfpseries_Sparse[ig,iktrial,NE:N,-1])**2
            mu_sparse_num[ig,iktrial,0],mu_sparse_num[ig,iktrial,1]=np.mean(xfpseries_Sparse[ig,iktrial,:NE,-1]),np.mean(xfpseries_Sparse[ig,iktrial,NE:N,-1])
            ## numerical results for Rank one Appriximation Mat
            variance_R1_num[ig,iktrial,0],variance_R1_num[ig,iktrial,1]=np.std(xfpseries_R1[ig,iktrial,:NE,-1])**2,np.std(xfpseries_R1[ig,iktrial,NE:N,-1])**2
            mu_R1_num[ig,iktrial,0],mu_R1_num[ig,iktrial,1]=np.mean(xfpseries_R1[ig,iktrial,:NE,-1]),np.mean(xfpseries_R1[ig,iktrial,NE:N,-1])

    #### ----------------- theoretical values ---------------------------------------------
    variance_full_theo, mu_full_theo = np.zeros((ngEI,3,2)),np.zeros((ngEI,3,2))
    variance_R1_theo, mu_R1_theo     = np.zeros((ngEI,3,2)),np.zeros((ngEI,3,2))
    variance_R1_theo_, mu_R1_theo_   = np.zeros((ngEI,3,2)),np.zeros((ngEI,3,2))
    for ig,gEI in enumerate (gEIseries):
        IPSP = gEI*EPSP
        JE0  = EPSP*kE*Cs
        JI0  = IPSP*kI*Cs
        gE2,gI2 = EPSP**2*(1-kE*Cs/NE)*kE*Cs*N/NE,IPSP**2*(1-kI*Cs/NI)*kI*Cs*N/NI
        gE,gI   = np.sqrt(gE2),np.sqrt(gI2)
        gmat    = np.array([gE,gI,gE,gI])
        # ## theoretical    
        deltainit,meaninit = np.mean(variance_full_num[ig,:cuttrials,0]),kappa_theo_iid[ig,0]#np.mean(kappaintersect[idxg,:,0])
        statsfull = fsolve(iidfull_mudelta_consistencyP,[meaninit,meaninit,deltainit,deltainit],args=(JE0,JI0,gmat,Nparams))
        mu_full_theo[ig,0,0],variance_full_theo[ig,0,0]= statsfull[0],statsfull[2]
        mu_full_theo[ig,0,1],variance_full_theo[ig,0,1]= statsfull[1],statsfull[3]

        statsR1 = fsolve(iidR1_mudelta_consistencyP,[meaninit,meaninit,deltainit,deltainit],args=(JE0,JI0,gmat,Nparams))
        mu_R1_theo[ig,0,0],variance_R1_theo[ig,0,0]= statsR1[0],statsR1[2]
        mu_R1_theo[ig,0,1],variance_R1_theo[ig,0,1]= statsR1[1],statsR1[3]

        # ## theoretical    
        deltainit,meaninit = np.mean(variance_full_num[ig,cuttrials:,0]),kappa_theo_iid[ig,2]#np.mean(kappaintersect[idxg,:,0])
        ## JEI0 are effective mean connectivity weights, the nonzero connectivity strengths are JE0/(Cs/NE),JI0/(Cs/NI)
        statsfull = fsolve(iidfull_mudelta_consistencyP,[meaninit,meaninit,deltainit,deltainit],args=(JE0,JI0,gmat,Nparams))
        mu_full_theo[ig,2,0],variance_full_theo[ig,2,0]= statsfull[0],statsfull[2]
        mu_full_theo[ig,2,1],variance_full_theo[ig,2,1]= statsfull[1],statsfull[3]

        statsR1 = fsolve(iidR1_mudelta_consistencyP,[meaninit,meaninit,deltainit,deltainit],args=(JE0,JI0,gmat,Nparams))
        mu_R1_theo[ig,2,0],variance_R1_theo[ig,2,0]= statsR1[0],statsR1[2]
        mu_R1_theo[ig,2,1],variance_R1_theo[ig,2,1]= statsR1[1],statsR1[3]

        ## Delta m2
        p = kE*Cs/NE
        IPSP = gEI*EPSP
        lambda0 = (EPSP*p*NE - IPSP*p*NI)#np.mean(np.mean(Biidvseries[ig,:,0]))
        lambda0 = np.real(lambda0)
        sigm2 = EPSP**2*kE*Cs*(1-p)/lambda0**2+IPSP**2*kI*Cs*(1-p)/lambda0**2
        mu_R1_theo_[ig,:,0],mu_R1_theo_[ig,:,1] = kappa_theo_iid[ig,:]*1,kappa_theo_iid[ig,:]*1
        variance_R1_theo_[ig,:,0],variance_R1_theo_[ig,:,1] = kappa_theo_iid[ig,:]**2*sigm2,kappa_theo_iid[ig,:]**2*sigm2

    #### moments
    ## Figures reflecting how mean and variance change with random gain of iid Gaussian matrix
    #### @YX 1109 NOTE !!! MATCH KAPPA
    clrs = ['b','g','c']
    fig,ax2 = plt.subplots(2,1,figsize=(6,4),sharex=True,tight_layout=True)
    #### -----------------  mean act------------------------------------------------------------
    pmean_full_numE,pmean_R1_numE,pmean_full_numI,pmean_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)
    pstd_full_numE,pstd_R1_numE,pstd_full_numI,pstd_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)

    nmean_full_numE,nmean_R1_numE,nmean_full_numI,nmean_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)
    nstd_full_numE,nstd_R1_numE,nstd_full_numI,nstd_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)

    #### @YX 2908 add gavg -- main-text: IGNORE
    pmean_spr_numE,pmean_spr_numI=np.zeros(ngEI),np.zeros(ngEI)
    pstd_spr_numE,pstd_spr_numI  =np.zeros(ngEI),np.zeros(ngEI)
    nmean_spr_numE,nmean_spr_numI=np.zeros(ngEI),np.zeros(ngEI)
    nstd_spr_numE,nstd_spr_numI  =np.zeros(ngEI),np.zeros(ngEI)
    pos_t0 = 0.7

    for i in range(ngEI):
        pmean_full_numE[i], pmean_R1_numE[i] = np.mean(mu_full_num[i,np.where(mu_full_num[i,:,0]>=pos_t0),0]),np.mean(mu_R1_num[i,np.where(mu_R1_num[i,:,0]>=pos_t0),0])
        pstd_full_numE[i], pstd_R1_numE[i]   = np.std(mu_full_num[i,np.where(mu_full_num[i,:,0]>=pos_t0),0]),np.std(mu_R1_num[i,np.where(mu_R1_num[i,:,0]>=pos_t0),0])

        pmean_full_numI[i], pmean_R1_numI[i] = np.mean(mu_full_num[i,np.where(mu_full_num[i,:,1]>=pos_t0),1]),np.mean(mu_R1_num[i,np.where(mu_R1_num[i,:,1]>=pos_t0),1])
        pstd_full_numI[i], pstd_R1_numI[i]   = np.std(mu_full_num[i,np.where(mu_full_num[i,:,1]>=pos_t0),1]),np.std(mu_R1_num[i,np.where(mu_R1_num[i,:,1]>=pos_t0),1])

        nmean_full_numE[i], nmean_R1_numE[i] = np.mean(mu_full_num[i,np.where(mu_full_num[i,:,0]<pos_t0),0]),np.mean(mu_R1_num[i,np.where(mu_R1_num[i,:,0]<pos_t0),0])
        nstd_full_numE[i], nstd_R1_numE[i]   = np.std(mu_full_num[i,np.where(mu_full_num[i,:,0]<pos_t0),0]),np.std(mu_R1_num[i,np.where(mu_R1_num[i,:,0]<pos_t0),0])

        nmean_full_numI[i], nmean_R1_numI[i] = np.mean(mu_full_num[i,np.where(mu_full_num[i,:,1]<pos_t0),1]),np.mean(mu_R1_num[i,np.where(mu_R1_num[i,:,1]<pos_t0),1])
        nstd_full_numI[i], nstd_R1_numI[i]   = np.std(mu_full_num[i,np.where(mu_full_num[i,:,1]<pos_t0),1]),np.std(mu_R1_num[i,np.where(mu_R1_num[i,:,1]<pos_t0),1])

        #### @YX 0109 add dynamics of sparse E-I network
        pmean_spr_numE[i] = np.mean(mu_sparse_num[i,np.where(mu_sparse_num[i,:,0]>=pos_t0),0])
        pstd_spr_numE[i]  = np.std(mu_sparse_num[i,np.where(mu_sparse_num[i,:,0]>=pos_t0),0])

        pmean_spr_numI[i] = np.mean(mu_sparse_num[i,np.where(mu_sparse_num[i,:,1]>=pos_t0),1])
        pstd_spr_numI[i]  = np.std(mu_sparse_num[i,np.where(mu_sparse_num[i,:,1]>=pos_t0),1])

        nmean_spr_numE[i] = np.mean(mu_sparse_num[i,np.where(mu_sparse_num[i,:,0]<pos_t0),0])
        nstd_spr_numE[i]  = np.std(mu_sparse_num[i,np.where(mu_sparse_num[i,:,0]<pos_t0),0])

        nmean_spr_numI[i] = np.mean(mu_sparse_num[i,np.where(mu_sparse_num[i,:,1]<pos_t0),1])
        nstd_spr_numI[i]  = np.std(mu_sparse_num[i,np.where(mu_sparse_num[i,:,1]<pos_t0),1])
    #### ----------------------------- sparse   --------------------------------------------
    ax2[0].fill_between(gEIseries,pmean_spr_numE+pstd_spr_numE,pmean_spr_numE-pstd_spr_numE, alpha=0.5,facecolor='red')
    ax2[1].fill_between(gEIseries,pmean_spr_numI+pstd_spr_numI,pmean_spr_numI-pstd_spr_numI, alpha=0.5,facecolor='blue')

    ax2[0].fill_between(gEIseries,nmean_spr_numE+nstd_spr_numE,nmean_spr_numE-nstd_spr_numE, alpha=0.5,facecolor='red')

    ax2[1].fill_between(gEIseries,nmean_spr_numI+nstd_spr_numI,nmean_spr_numI-nstd_spr_numI, alpha=0.5,facecolor='blue')
    ax2[0].plot(CriticCradl*np.ones(2),np.array([yticks[0],yticks[-1]]),c='gray',linewidth=1.5,linestyle='--')
    ax2[1].plot(CriticCradl*np.ones(2),np.array([yticks[0],yticks[-1]]),c='gray',linewidth=1.5,linestyle='--')
    ax2[0].plot(CriticCradh*np.ones(2),np.array([yticks[0],yticks[-1]]),c='gray',linewidth=1.5,linestyle='--')
    ax2[1].plot(CriticCradh*np.ones(2),np.array([yticks[0],yticks[-1]]),c='gray',linewidth=1.5,linestyle='--')

    # ax2[0].plot(CriticCdyn*np.ones(2),np.array([yticks[0],yticks[-1]]),c='tab:green',linewidth=1.5,linestyle='--')
    # ax2[1].plot(CriticCdyn*np.ones(2),np.array([yticks[0],yticks[-1]]),c='tab:green',linewidth=1.5,linestyle='--')

    ax2[0].plot(gEIseries,(mu_R1_theo_[:,0,0]),color='tab:red',lw=1.5)
    ax2[1].plot(gEIseries,(mu_R1_theo_[:,0,1]),color='tab:blue',lw=1.5)
    ax2[0].plot(gEIseries,(mu_R1_theo_[:,2,0]),color='tab:red',lw=1.5)
    ax2[1].plot(gEIseries,(mu_R1_theo_[:,2,1]),color='tab:blue',lw=1.5)

    # ax2[0].plot(gEIseries,(mu_R1_theo[:,0,0]),linestyle='--',color='black',alpha=0.5)
    # ax2[1].plot(gEIseries,(mu_R1_theo[:,1,0]),linestyle='--',color='black',alpha=0.5)
    # ax2[0].plot(gEIseries,(mu_R1_theo[:,0,1]),linestyle='--',color='black',alpha=0.5)
    # ax2[1].plot(gEIseries,(mu_R1_theo[:,1,1]),linestyle='--',color='black',alpha=0.5)

    for i in range(2):     
            ax2[i].set_xlim(xlims)
            ax2[i].set_ylim(ylims)
            ax2[i].set_xticks(xticks)
            ax2[i].set_yticks(yticks)
    ax2[1].set_xlabel(r'ratio c')


xticks = [-0.5,0,1.0]#np.linspace(-1.0,1.5,1)
xlims  = [-0.6,1.5]
yticks = np.linspace(-0.5,0.5,2)
ylims  = [-0.6,0.6]
theta  = np.linspace(0, 2 * np.pi, 200)
idxc=0
for iktrial in np.range(0,ntrial,3):
    # iktrial = 4
    cm='rbg'
    # for ig,gEI in enumerate (gEIseries[-3:]):
    ig = 10
    gEI = gEIseries[ig]
    figtspt,axtspt=plt.subplots(figsize=(6,3))
    IPSP = gEI*EPSP
    JE0  = EPSP*kE*Cs
    JI0  = IPSP*kI*Cs
    ## variance
    gE2,gI2 = EPSP**2*(1-kE*Cs/NE)*kE*Cs/NE,IPSP**2*(1-kI*Cs/NI)*kI*Cs/NI
    gE2,gI2 = gE2*N,gI2*N
    alphaNEI = Npercent.copy()
    gMmat = np.array([[gE2*alphaNEI[0],gI2*alphaNEI[1]],[gE2*alphaNEI[0],gI2*alphaNEI[1]]])
    eigvgm,eigvecgm=la.eig(gMmat) 
    r_g2=np.max(eigvgm)
    r_g = np.sqrt(r_g2)
    xr = r_g*(1)*np.cos(theta)
    yr = r_g*(1)*np.sin(theta)
    axtspt.plot(xr, yr, color="gray", linewidth=0.5,linestyle='--') # >>>>>
    axtspt.scatter(np.real(Bsprvseries[ig,iktrial,1:]),np.imag(Bsprvseries[ig,iktrial,1:]),s=5,c='tab:blue',alpha=0.25) # >>>>>>>>>>>>>>
    axtspt.scatter(np.real(Bsprvseries[ig,iktrial,0]),np.imag(Bsprvseries[ig,iktrial,0]),s=20,c='tab:blue',alpha=0.5) # >>>>>>>>>>>> Only one realization
    axtspt.set_aspect('equal')
    axtspt.scatter(np.real(BAmvseries[ig,0]),0,s=80,c= '#FF000000',edgecolor='tab:red') # >>>>>>>>>>>>
    axtspt.spines['right'].set_color('none')
    axtspt.spines['top'].set_color('none')
    axtspt.xaxis.set_ticks_position('bottom')
    axtspt.spines['bottom'].set_position(('data', 0))
    axtspt.set_xlim(xlims)
    axtspt.set_ylim(ylims)
    axtspt.set_xticks(xticks)
    axtspt.set_yticks(yticks)
    axtspt.set_aspect('equal')
    axtspt.set_title(r'$c=$'+str("{:.2f}".format(gEI)))

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
# if RERUN == 1:

#     lst = [kappaintersect_Sparse, kappaintersect_Full, kappaintersect_R1,
#            xfpseries_Full[:,:,:,-1], xfpseries_R1[:,:,:,-1], xfpseries_Sparse[:,:,:,-1],
#            RsprvecTseries[:,:,:,0], Rsprvecseries[:,:,:,0],
#            LsprvecTseries[:,:,:,0], Lsprvecseries[:,:,:,0],
#            Bsprvseries[:,:,:],
#            sigrcov, siglcov]
#     stg = ["kappaintersect_Sparse, kappaintersect_Full, kappaintersect_R1,"
#            "xfpseries_Full, xfpseries_R1, xfpseries_Sparse,"
#            "RsprvecTseries, Rsprvecseries,"
#            "LsprvecTseries, Lsprvecseries,"
#            "Bsprvseries,"
#            "sigrcov, siglcov"]
#     data = list_to_dict(lst=lst, string=stg)
#     data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_spr_PositiveTF_kappa_1.npz"
#     np.savez(data_name, **data)


# #### A. intersections of the equilibrium of kappa
# kappa_x    = np.linspace(-1, 6, 30)
# xintersect = np.linspace(1.5, 2.4, 2)#np.linspace(1.2, 1.9, 2)
# Sx         = np.zeros((len(gEIseries), len(kappa_x)))
# F0         = np.zeros_like(Sx)
# F1         = np.zeros_like(Sx)
# for idx, gEI in enumerate(gEIseries):
#     IPSP = gEI*EPSP
#     JE0  = EPSP*kE*Cs
#     JI0  = IPSP*kI*Cs
#     gE2,gI2 = EPSP**2*(1-kE*Cs/NE)*kE*Cs*N/NE,IPSP**2*(1-kI*Cs/NI)*kI*Cs*N/NI
#     gE,gI   = np.sqrt(gE2),np.sqrt(gI2)
#     gmat    = np.array([gE,gI,gE,gI])
#     gee, gei, gie, gii = gmat[0], gmat[1], gmat[2], gmat[3]
#     if(len(Nparams) == 2):
#         NE, NI = Nparams[0], Nparams[1]
#     else:
#         NE1, NE2, NI = Nparams[0], Nparams[1], Nparams[2]
#         NE = NE1+NE2
#     N = NE+NI
#     for idxx, x in enumerate(kappa_x):
#         muphi, delta0phiE, delta0phiI = x, x**2 * \
#             (gee**2*NE/N+gei**2*NI/N)/(JE0-JI0)**2, x**2 * \
#             (gie**2*NE/N+gii**2*NI/N)/(JE0-JI0)**2
#         Sx[idx, idxx] = -x+(JE0*PhiP(muphi, delta0phiE) -
#                             JI0*PhiP(muphi, delta0phiI))
#         F0[idx,idxx]  = x 
#         F1[idx,idxx]  = (JE0*PhiP(muphi, delta0phiE) -JI0*PhiP(muphi, delta0phiI))
#     fig, ax = plt.subplots(figsize=(4, 4))
#     xticks  = [-1,0,6]#np.linspace(-1, 4, 3)
#     xlims   = [-1, 6]
#     yticks  = [-1,0,2]#np.linspace(-1.0, 1.0, 3)
#     ylims   = [-1,2]
#     ax.plot(kappa_x, Sx[idx, :], c='tab:purple', lw=1.0)
#     ax.set_xlim(xlims)
#     ax.set_ylim(ylims)
#     ax.set_xticks(xticks)
#     ax.set_yticks(yticks)


#     fig, ax = plt.subplots(figsize=(4, 4))
#     xticks  = [-1,0,6]#np.linspace(-1, 5, 2)
#     xlims   = [-1, 6]
#     yticks  = [-1,0,4]#np.linspace(-1., 5, 2)
#     ylims   = [-1.0,4]
#     # ax.plot(kappa_x,np.zeros_like(kappa_x),c='k',lw=1.0)
#     ax.plot(kappa_x, F0[idx, :], c='gray', lw=1)
#     ax.plot(kappa_x, F1[idx, :], c='tab:purple',linestyle='--',alpha=0.75)
#     # ax.set_xlim(xlims)
#     # ax.set_ylim(ylims)
#     # ax.set_xticks(xticks)
#     # ax.set_yticks(yticks)

# fig,ax = plt.subplots(figsize=(6,2))
# xticks = [gEIseries[0],gEIseries[-1]]#[2,5.5]
# xlims  = [gEIseries[0],gEIseries[-1]]# [2,5.5]
# yticks = [0,6.5]#np.linspace(0,6.5,3)
# ylims  = [0,6.5]
# for iktrial in range(ntrial):
#     ax.scatter(gEIseries,kappaintersect_Sparse[:,iktrial,0],s=10,color='tab:blue',alpha=0.75)

# ax.set_xlim(xlims)
# ax.set_ylim(ylims)
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)
