# -*- coding: utf-8 -*-
"""
@author: Yuxiu Shao
Sparse EI Network with reciprocal connectivity, Full-rank Gaussian Network and Rank-one Mixture of Gaussian Approximation
Connectivity, Dynamics
"""

''' HELP FUNCTIONS '''
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
plt.rc('axes', labelweight='bold', labelsize='large',titleweight='bold', titlesize=18, titlepad=10)
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

## network params
nn,gnn=200,4
gEI,Cs,kE,kI = 3.35,60,gnn,1
Nt    = np.array([gnn*nn,nn])
NE,NI = Nt[0],Nt[1]
N     = NE+NI

Nparams  = np.array([NE,NI])
Npercent = Nparams/N
nrank,ntrial,neta,nvec,nmaxFP=1,30,1,2,3
shiftx   = 1.5
cuttrials=int(ntrial/2)

RERUN = 1
if RERUN == 0:
    data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_spr_PositiveTF_reciprocal_kappa_1.npz"
    # np.savez(data_name, **data)
    data  = np.load(data_name)


### sparse params
EPSP,a,b = 0.023,0,0#0.025,0,0# JE PER EPSP
ngEI     = 21# number of eta(s)
# cell-type specific reciprocal connectivity
etaseries    = np.linspace(0.0,1.0,ngEI)#
etassrecord  = np.zeros((ngEI,3))
etauserecord = np.zeros((ngEI,3))

### recording variables
RAmvecseries,LAmvecseries = np.zeros((ngEI,N)),np.zeros((ngEI,N))
BAmvseries = np.zeros((ngEI,N),dtype=complex)
### original Sparse Matrix
Rsprvecseries,Lsprvecseries  =np.zeros((ngEI,ntrial,N,nvec*2)),np.zeros((ngEI,ntrial,N,nvec*2))
RsprvecTseries,LsprvecTseries=np.zeros((ngEI,ntrial,N,nvec*2)),np.zeros((ngEI,ntrial,N,nvec*2))
Bsprvseries  = np.zeros((ngEI,ntrial,N),dtype=complex)
Radiusseries = np.zeros(ngEI)
### eigenvector predictions obtained using perturbation theory
Restvecseries,Lestvecseries=np.zeros((ngEI,ntrial,N,nvec*2)),np.zeros((ngEI,ntrial,N,nvec*2))

### the equivalent gaussian connectivity
Riidvecseries,Liidvecseries  =np.zeros((ngEI,ntrial,N,nvec*2)),np.zeros((ngEI,ntrial,N,nvec*2))
RiidvecTseries,LiidvecTseries=np.zeros((ngEI,ntrial,N,nvec*2)),np.zeros((ngEI,ntrial,N,nvec*2))
Biidvseries  = np.zeros((ngEI,ntrial,N),dtype=complex)

### statistical properties
armu,sigrcov = np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2)) # 2 for E and I
almu,siglcov = np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))
siglr        = np.zeros((ngEI,ntrial,2))

if(RERUN==0):
    Rsprvecseries[:,:,:,0] = data['Rsprvecseries']
    Lsprvecseries[:,:,:,0] = data['Lsprvecseries']
    Restvecseries[:,:,:,0] = data['Restvecseries']
    Lestvecseries[:,:,:,0] = data['Lestvecseries']
    Bsprvseries[:,:,0]     = data['Bsprvseries']
    sigrcov = data['sigrcov']
    siglcov = data['siglcov']

    for ig, eta in enumerate (etaseries):              
        Jbinary = np.zeros((N,N))
        ### GENERATE SYMMETRIC, SPARSE SUBMATRIX.
        p = kE*Cs/NE
        # for NE
        etaEE  = eta#1.0
        # for NI
        etaII  = 0# eta#0#1.0#0#
        # for EI
        etaEI  = eta#eta#0.0#1.0
        etassrecord[ig,0],etassrecord[ig,1],etassrecord[ig,2] = (etaEE)/(1),(etaEI)/(1),(etaII)/(1)
        ### generate corresponding gaussian random network with reciprocal connectivity 
        etaset = np.zeros(3)
        etaset[0],etaset[1],etaset[2] = (etaEE-p)/(1-p),-(etaEI-p)/(1-p),(etaII-p)/(1-p)
        etauserecord[ig,:]=etaset[:]


### covariance of eigenvectors
Sigmn_num    = np.zeros((ngEI,ntrial,3)) ## sparse
Sigmniid_num = np.zeros((ngEI,ntrial,3)) ## Gaussian
Sigmn_theo   = np.zeros((ngEI,ntrial,3)) ## theoretical 
### cell-type-dependent reciprocal statistics
etaMat  = np.zeros((ngEI,ntrial,2,2))
gRandom = np.zeros((ngEI,ntrial,2))

### fixed points of dynamical variable kappa
kappaintersect_Sparse = np.zeros((ngEI,ntrial,nmaxFP*2))
kappaintersect_Full   = np.zeros((ngEI,ntrial,nmaxFP*2))
kappaintersect_R1     = np.zeros((ngEI,ntrial,nmaxFP*2))

if(RERUN==0):
    kappaintersect_Sparse = data['kappaintersect_Sparse']
    kappaintersect_Full   = data['kappaintersect_Full']
    kappaintersect_R1     = data['kappaintersect_R1']

### params for temporal dynamics
tt  = np.linspace(0,100,500)
dt  = tt[2]-tt[1]
ntt = len(tt)
## compare three neuron population activities
xfpseries_Sparse = np.zeros((ngEI,ntrial,N,ntt))
xfpseries_Full   = np.zeros((ngEI,ntrial,N,ntt))
xfpseries_R1     = np.zeros((ngEI,ntrial,N,ntt))
if(RERUN==0):
    xfpseries_Sparse[:,:,:,-1] = data['xfpseries_Sparse']
    xfpseries_Full[:,:,:,-1]   = data['xfpseries_Full']
    xfpseries_R1[:,:,:,-1]     = data['xfpseries_R1']

kappaseries_R1   = np.zeros((ngEI,ntrial,ntt))

### Am is fixed, mean matrix, reference eigenvalue and eigenvectors
Am   = np.zeros((N,N))
IPSP = gEI*EPSP
JE0  = EPSP*kE*Cs
JI0  = IPSP*kI*Cs
Am[:,:NE],Am[:,NE:] = JE0/NE,-JI0/NI
xAm,yAm = np.zeros((N,1)),np.zeros((N,1))
eigvAm  = np.zeros(N)
eigvAm[0]=(JE0-JI0)
xAm[:NE,0],xAm[NE:,0] = 1,1 
yAm[:NE,0],yAm[NE:,0] = N*JE0/NE,-N*JI0/NI 
RAmvecseries[0,:],LAmvecseries[0,:] = xAm[:,0],yAm[:,0]
BAmvseries[0,:] = eigvAm[:]
print('eigvAm:',eigvAm[0])

### preparation for the equivalent gaussian connectivity
gE2,gI2  = EPSP**2*(1-kE*Cs/NE)*kE*Cs*N/NE,IPSP**2*(1-kI*Cs/NI)*kI*Cs*N/NI
# print("ratio gE/gI ",np.sqrt(gE2/gI2),', ',1/gEI)
alphaNEI = Npercent.copy()
gMmat    = np.array([[gE2*alphaNEI[0],gI2*alphaNEI[1]],[gE2*alphaNEI[0],gI2*alphaNEI[1]]])
eigvgm,eigvecgm=la.eig(gMmat) 
r_g2 = np.max(eigvgm)
r_g  = np.sqrt(r_g2)


if(RERUN):
    ''' Iterative Processing '''
    for iktrial in range(ntrial):
        print('---- simulating neuronal activity #trial-',iktrial,'------')
        #### the equivalent gaussian connectivity
        Xsym  = iidGaussian([0,1.0/np.sqrt(N)],[N,N]) #### within the same trial, share the same Y
        XsymT = Xsym.copy().T
        Xgiid = Xsym.copy()
        Xgiid[:NE,:NE]*=np.sqrt(gE2)
        Xgiid[:NE,NE:]*=np.sqrt(gI2)
        Xgiid[NE:,:NE]*=np.sqrt(gE2)
        Xgiid[NE:,NE:]*=np.sqrt(gI2)
        Jgiid = Xgiid.copy()+Am.copy()
        
        #### the original Sparse Network
        eigvJiid,leigveciid,reigveciid,xnorm0iid,ynorm0iid=decompNormalization(Jgiid,xAm,yAm,xAm,yAm,nparams = Nparams,sort=1)
        ## same sparse Random Connectivity for each trial
        ## EE-trial
        n1,n2,prob= NE,NE,kE*Cs/NE
        subProb   = randbin(1,n1*n2,1-prob)
        JEEsub    = np.reshape(subProb,(n1,n2))

        ## EI-trial
        n1,n2,prob= NI,NE,kE*Cs/NE
        subProb   = randbin(1,n1*n2,1-prob)
        JEIsub    = np.reshape(subProb,(n1,n2))

        ## II-trial
        n1,n2,prob= NI,NI,kI*Cs/NI
        subProb   = randbin(1,n1*n2,1-prob)
        JIIsub    = np.reshape(subProb,(n1,n2))
        ## dilute/sparse matrix /attic
        for ig, eta in enumerate (etaseries):                
            Jbinary = np.zeros((N,N))

            #### setting reciprocal statistics
            p = kE*Cs/NE
            # for NE
            etaEE  = eta#1.0
            # for NI
            etaII  = 0# eta#0#1.0#0#
            # for EI
            etaEI  = eta#eta#0.0#1.0

            etassrecord[ig,0],etassrecord[ig,1],etassrecord[ig,2] = (etaEE)/(1),(etaEI)/(1),(etaII)/(1)
            
            #### the equivalent gaussian connectivity with reciprocal motifs
            etaset = np.zeros(3)
            etaset[0],etaset[1],etaset[2] = (etaEE-p)/(1-p),-(etaEI-p)/(1-p),(etaII-p)/(1-p)
            etauserecord[ig,:]=etaset[:]
            #### check the sign of the statistics(negative)
            signeta_use = np.ones(3)
            for i in range(3):
                if etaset[i]<0:
                    signeta_use[i] *=(-1)

            ### E-E
            probEE = kE*Cs/NE
            JbEE,JbEET_=generate_SymSpr_Mat_trial(JEEsub,etaEE,kE*Cs/NE,np.array([NE,NE]))
            JbEE_=JbEET_.copy().T
            ## combine two to get EE sub-matrix
            JbEEt = np.zeros((NE,NE))
            for i in range(NE):
                JbEEt[i,i]=JbEE[i,i]
            idxEEd, idyEEd = np.where(JbEE>0)
            iddown = np.where(idxEEd>idyEEd)
            idxEEd,idyEEd=idxEEd[iddown],idyEEd[iddown]
            JbEEt[idxEEd,idyEEd]=JbEE[idxEEd,idyEEd]
            idxEEu, idyEEu = np.where(JbEET_>0)
            idup = np.where(idxEEu<idyEEu)
            idxEEu,idyEEu=idxEEu[idup],idyEEu[idup]
            JbEEt[idxEEu,idyEEu]=JbEET_[idxEEu,idyEEu]
            # print('symmetryEE:',np.sum(np.sum(JbEEt*JbEEt.T))/kE/Cs/NE,etaEE)
            # print('probEE:',np.sum(np.sum(JbEEt))/NE**2,kE*Cs/NE)

            ### I-I
            probII = kE*Cs/NE
            JbII,JbIIT_=generate_SymSpr_Mat_trial(JIIsub,etaII,kI*Cs/NI,np.array([NI,NI]))
            JbII_=JbIIT_.copy().T
            ## combine two to get EE sub-matrix
            JbIIt = np.zeros((NI,NI))
            for i in range(NI):
                JbIIt[i,i]=JbII[i,i]
            idxIId, idyIId = np.where(JbII>0)
            iddown = np.where(idxIId>idyIId)
            idxIId,idyIId=idxIId[iddown],idyIId[iddown]
            JbIIt[idxIId,idyIId]=JbII[idxIId,idyIId]
            idxIIu, idyIIu = np.where(JbIIT_>0)
            idup = np.where(idxIIu<idyIIu)##
            idxIIu,idyIIu=idxIIu[idup],idyIIu[idup]
            JbIIt[idxIIu,idyIIu]=JbIIT_[idxIIu,idyIIu]

            #
            JbIE,JbIET_=generate_SymSpr_MatEI_trial(JEIsub,etaEI,np.array([kE*Cs/NE,kI*Cs/NI]),np.array([NI,NE]))
            ## combine two to get EE sub-matrix
            JbIEt,JbEIt = np.zeros((NI,NE)),np.zeros((NE,NI))
            JbIEt,JbEIt = JbIE.copy(),JbIET_.copy()

            ### dalean network
            Jbinary[:NE,:NE],Jbinary[NE:,NE:]=JbEEt,-JbIIt
            Jbinary[NE:,:NE],Jbinary[:NE,NE:]=JbIEt,-JbEIt

            ### from attic to sparse connectivity
            JE,JI   = EPSP,IPSP
            Jdilute = Jbinary.copy()
            Jdilute[:,:NE] *= JE
            Jdilute[:,NE:] *= JI
            ### overall sparse network
            J  = Jdilute.copy()
            JT = Jdilute.copy().T


            ### subtracting thee mean connectivity
            # X = Jdilute.copy() - Am.copy() # random matrix, with zero mean 
            X = Jdilute.copy() # random matrix, with zero mean 
            X[:NE,:NE] = X[:NE,:NE] - np.mean(X[:NE,:NE].flatten())
            X[:NE,NE:] = X[:NE,NE:] - np.mean(X[:NE,NE:].flatten())
            X[NE:,:NE] = X[NE:,:NE] - np.mean(X[NE:,:NE].flatten())
            X[NE:,NE:] = X[NE:,NE:] - np.mean(X[NE:,NE:].flatten())
            ''' Original Sparse Network '''
            eigvJ,leigvec,reigvec,xnorm0,ynorm0=decompNormalization(J,xnorm0iid,ynorm0iid,xAm,yAm,nparams = Nparams,sort=1)
            # eigvJ,leigvec,reigvec,xnorm0,ynorm0=decompNormalization(J,xAm,yAm,xAm,yAm,nparams = Nparams,sort=1)

            Rsprvecseries[ig,iktrial,:,0],Lsprvecseries[ig,iktrial,:,0]=xnorm0[:,0].copy(),ynorm0[:,0].copy()
            Bsprvseries[ig,iktrial,:]=eigvJ.copy()
            print('Sparse eigenvalue:',eigvJ[0],', ', np.sum(xnorm0*ynorm0)/N)

            ####
            eigvnorm = np.real(Bsprvseries[ig,iktrial,0])#eigvAm[0]#
            Zss      = X.copy()# J.copy()-Am.copy()
            Restvec,LestvecT =  np.reshape(Zss@xAm/eigvnorm,(N,1)),np.reshape(yAm.copy().T@Zss/eigvnorm,(1,N))
            Restvec,LestvecT = Restvec+xAm,LestvecT+yAm.T
            Restvecseries[ig,iktrial,:,0],Lestvecseries[ig,iktrial,:,0] =np.squeeze(Restvec),np.squeeze(LestvecT)

            # #### normalization(not necessary)
            # Lestvec  = np.reshape(LestvecT.T,(N,1))
            # check_lambda  = np.squeeze(np.reshape(Lestvec,(1,N))@np.reshape(Restvec,(N,1)))
            # should_lambda = np.real(Bsprvseries[ig,iktrial,0]*N)
            # Lestvec = Lestvec/check_lambda*should_lambda
            # _,_,xnormt,ynormt=Normalization(Restvec.copy(),Lestvec.copy(),xAm.copy(),yAm.copy(),nparams=Nparams,sort=0,nrank=1)
            # ### RECONSTRUCTED EIGENVECTORS
            # Restvecseries[ig,iktrial,:,0],Lestvecseries[ig,iktrial,:,0] =xnormt[:,0],ynormt[:,0]

            

            #### statistics of the eigenvectors
            axrmu,aylmu,sigxr,sigyl,sigcov = numerical_stats(xnorm0,ynorm0,xAm,yAm,eigvJ,nrank,2,ppercent=Npercent)
            armu[ig,iktrial,:],almu[ig,iktrial,:] = axrmu[:,0],aylmu[:,0]
            sigrcov[ig,iktrial,:],siglcov[ig,iktrial,:]= sigxr[:,0],sigyl[:,0]
            siglr[ig,iktrial,:] = sigcov[:,0,0]
            Sigmn_num[ig,iktrial,:2] = siglr[ig,iktrial,:] #*np.real(eigvJ[0])#eigvAm[0] ## already considered
            Sigmn_num[ig,iktrial,2]  = Sigmn_num[ig,iktrial,0]*Npercent[0]+Sigmn_num[ig,iktrial,1]*Npercent[1]
               
            ## Rank One Approximation, Nullclines of \kappa
            xnorm0, ynorm0 = Rsprvecseries[ig,iktrial,:,0].copy(),Lsprvecseries[ig,iktrial,:,0]
            xnorm0, ynorm0 = np.reshape(xnorm0,(N,1)),np.reshape(ynorm0,(N,1))
            xmu,ymu = armu[ig,iktrial,:].copy(),almu[ig,iktrial,:].copy()
            xsig,ysig = sigrcov[ig,iktrial,:].copy(),siglcov[ig,iktrial,:].copy()
            yxcov = siglr[ig,iktrial,:].copy()

            ### the equivalent gaussian connectivity
            Xinit = Xsym.copy() 
            ### >>>>>>>>.Subcircuit Exc sym >>>>>>>>>.
            ## remaining EE
            asqr=(1-np.sqrt(1-etaset[0]**2))/2.0 ## when eta = 0, asqr = 0, aamp = 0, XT-0, X-1
            aamp=np.sqrt(asqr)
            Xinit[:NE,:NE]=signeta_use[0]*aamp*XsymT[:NE,:NE].copy()+np.sqrt(1-aamp**2)*Xsym[:NE,:NE].copy()
            ### >>>>>>> Total >>>>>
            # EI
            asqr=(1-np.sqrt(1-(etaset[1])**2))/2.0 ## when eta = 0, asqr = 0, aamp = 0, XT-0, X-1
            aamp=np.sqrt(asqr)
            Xinit[NE:,:NE]=signeta_use[1]*aamp*XsymT[NE:,:NE].copy()+np.sqrt(1-aamp**2)*Xsym[NE:,:NE].copy()
            Xinit[:NE,NE:]=signeta_use[1]*aamp*XsymT[:NE,NE:].copy()+np.sqrt(1-aamp**2)*Xsym[:NE,NE:].copy()
            ## II ##
            asqr=(1-np.sqrt(1-etaset[2]**2))/2.0
            aamp=np.sqrt(asqr)
            Xinit[NE:,NE:]=signeta_use[2]*aamp*XsymT[NE:,NE:].copy()+np.sqrt(1-aamp**2)*Xsym[NE:,NE:].copy()

            Xinit[:NE,:NE]*=np.sqrt(gE2)
            Xinit[:NE,NE:]*=np.sqrt(gI2)
            Xinit[NE:,:NE]*=np.sqrt(gE2)
            Xinit[NE:,NE:]*=np.sqrt(gI2)
            Jfull = Xinit.copy()+Am.copy()

            eigvJiid,leigveciid,reigveciid,xnormgau,ynormgau=decompNormalization(Jfull,xnorm0iid,ynorm0iid,xAm,yAm,nparams = Nparams,sort=1)
            # eigvJiid,leigveciid,reigveciid,xnormgau,ynormgau=decompNormalization(Jfull,xAm,yAm,xAm,yAm,nparams = Nparams,sort=1)
            print('Gaussian eigenvalue:',eigvJiid[0],', ', np.sum(xnormgau*ynormgau)/N)
            Riidvecseries[ig,iktrial,:,0],Liidvecseries[ig,iktrial,:,0]=xnormgau[:,0].copy(),ynormgau[:,0].copy()
            Biidvseries[ig,iktrial,:]=eigvJiid.copy()

            ## lambda_change
            KE,KI=kE*Cs,kI*Cs
            p=KE/NE ## KI/NI
            forJEE=EPSP**2*KE*etassrecord[ig,0]*(1-p)**2-2*EPSP**2*KE*(1-etassrecord[ig,0])*(1-p)*p+EPSP**2*p**2*(NE-KE*2+KE*etassrecord[ig,0])## sigma^E
            forJEI=-EPSP*IPSP*KI*etassrecord[ig,1]*(1-p)**2+2*EPSP*IPSP*KI*(1-etassrecord[ig,1])*(1-p)*p-EPSP*IPSP*p**2*(NI-KI*2+KI*etassrecord[ig,1])## sigma^I
            forJIE=-EPSP*IPSP*KE*etassrecord[ig,1]*(1-p)**2+2*EPSP*IPSP*KE*(1-etassrecord[ig,1])*(1-p)*p-EPSP*IPSP*p**2*(NE-KE*2+KE*etassrecord[ig,1])## signa^E
            forJII=IPSP**2*KI*etassrecord[ig,2]*(1-p)**2-2*IPSP**2*KI*(1-etassrecord[ig,2])*(1-p)*p+IPSP**2*p**2*(NI-KI*2+KI*etassrecord[ig,2])
            ssigE,ssigI = (forJEE*N*JE0/NE*NE+forJIE*(-N*JI0/NI*NI))/NE,(forJEI*N*JE0/NE*NE+forJII*(-N*JI0/NI*NI))/NI
            # gsigE = JE0*EPSP**2*N*p*(1-p)*etassrecord[ig,0]-JI0*EPSP*IPSP*N*(1-p)*p*etassrecord[ig,1]
            # gsigI = JE0*EPSP*IPSP*N*p*(1-p)*etassrecord[ig,1]-JI0*IPSP*IPSP*N*(1-p)*p*etassrecord[ig,2]
            gsigE = JE0*EPSP**2*p*(etassrecord[ig,0]-p)*(1)*N+JE0*EPSP**2*p*(etassrecord[ig,1]-p)*(KI/KE*gEI**2)*N
            gsigI = -(gEI*JE0*EPSP**2*p*(etassrecord[ig,1]-p)*(1)*N + gEI*JE0*EPSP**2*p*(etassrecord[ig,2]-p)*(KI/KE*gEI**2)*N)
            # print('COMPARE-SPARSE:',ssigE,ssigI,ssigE*Npercent[0]+ssigI*Npercent[1])
            # print('COMPARE-gau:',gsigE,gsigI)

            # Sigmn_theo[ig,iktrial,0],Sigmn_theo[ig,iktrial,1]=ssigE/eigvJiid[0]**2, ssigI/eigvJiid[0]**2#ssigE/eigvAm[0]**2, ssigI/eigvAm[0]**2

            ### normalized by lambda_0
            Sigmn_theo[ig,iktrial,0],Sigmn_theo[ig,iktrial,1]=ssigE/eigvAm[0]**2, ssigI/eigvAm[0]**2
            Sigmn_theo[ig,iktrial,2] = Sigmn_theo[ig,iktrial,0]*Npercent[0]+Sigmn_theo[ig,iktrial,1]*Npercent[1]

            axrmuiid,aylmuiid,sigxriid,sigyliid,sigcoviid = numerical_stats(xnormgau,ynormgau,xAm,yAm,eigvJiid,nrank,2,ppercent=Npercent)

            Sigmniid_num[ig,iktrial,:2] = sigcoviid[:,0,0] # *np.real(eigvJiid[0])#eigvAm[0] ## already considered
            Sigmniid_num[ig,iktrial,2]  = Sigmniid_num[ig,iktrial,0]*Npercent[0]+Sigmniid_num[ig,iktrial,1]*Npercent[1]
            

            ### sparse connectivity - dynamics
            Jpt   = J.copy()
            if(iktrial<cuttrials):
                xinit = np.random.normal(5,1E-1,(1,N))
            else:
                xinit = np.random.normal(0,1E-2,(1,N))
            xinit = np.squeeze(xinit)
            xtemporal = odesimulationP(tt,xinit,Jpt,0)
            xfpseries_Sparse[ig,iktrial,:,:] = xtemporal.T.copy()
            
            ### ------------ renew 
            kappanum = np.zeros(3)
            if(1):
                xact = np.squeeze(xfpseries_Sparse[ig,iktrial,:,-1])
                kappanum[0] = np.squeeze(LestvecT.copy()@(1.0+np.tanh(xact.copy()-shiftx)))/N
            else:
                xact = np.squeeze(xfpseries_Sparse[ig,iktrial,:,-1])
                mvec_norm = np.reshape(Restvec,(N,1))
                kappanum[0] = np.squeeze(np.reshape(np.squeeze(xact),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            kappaintersect_Sparse[ig, iktrial, :3] = kappanum[:].copy()
            
            ''' Gaussian full Connectivity -- Dynamics '''
            Jpt       = Jfull.copy()
            xtemporal = odesimulationP(tt,xinit,Jpt,0)
            xfpseries_Full[ig,iktrial,:,:] = xtemporal.T.copy()
            ''' kappa dynamics    '''
            ## 2 populations 
            kappanum = np.zeros(3)
            xact = np.squeeze(xfpseries_Full[ig,iktrial,:,-1])
            ## use yAm -- unperturbed centre.
            if(1):#(aylmuiid[0,0])*(yAm[0,0])>0): ## ((aylmuiid[0,0]*eigvJiid[0])*(yAm[0,0]*eigvAm[0])>0):
                # do not change
                kappanum[0] = yAm[0,0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0]+yAm[NE,0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            else:
                kappanum[0] = -yAm[0,0]*np.mean(1.0+np.tanh(xact[:NE]-shiftx))*Npercent[0]-yAm[NE,0]*np.mean(1.0+np.tanh(xact[NE:]-shiftx))*Npercent[1]
            kappanum[0]=kappanum[0]#*np.sqrt(N)*eigvAm[0] already considered
            kappaintersect_Full[ig,iktrial,:3]=kappanum[:].copy()

            

            ### rank one approximation dynamics
            r1Matt = np.real(Restvec@LestvecT)
            r1Matt = r1Matt/N 
            ## use the same initial values
            xtemporal = odesimulationP(tt,xinit,r1Matt,0)
            xfpseries_R1[ig,iktrial,:,:]   = xtemporal.T.copy()
            kappaseries_R1[ig, iktrial, :] = np.squeeze(
                xtemporal.copy()@np.reshape(xnorm0, (-1, 1)))/np.sum(xnorm0**2)
            ### temporal evolution of kappa
            kappanum = np.zeros(3)
            xact = np.reshape(np.squeeze(xfpseries_R1[ig,iktrial,:,-1]),(N,1))
            #### @18/08 - error
            kappanum[0] = np.squeeze(LestvecT.copy()@(1.0+np.tanh(xact.copy()-shiftx)))/N#@np.tanh(xact.copy()))/N #
            kappaintersect_R1[ig,iktrial,:3]=kappanum[:].copy()
        
def analysis():
    '''
    with symmetry, subcircuit
    eigvJ0
    '''
    xticks = np.linspace(-1.0,1.0,3)
    xlims  = [-1.,1.8]
    yticks = np.linspace(-1,1,2)
    ylims  = [-1.2,1.2]
    theta  = np.linspace(0, 2 * np.pi, 200)
    idxc=0
    cm='rbg'
    Bsprvseries_theo = np.zeros((ngEI,3),dtype=complex)
    Bsprvseries_theo_general = np.zeros((ngEI,3),dtype=complex) ### infinite -- first round revision
    ig,iiitrial  = 16,2#eta==0.8#8,2#3,2#
    eta = etaseries[ig] 
    figtspt,axtspt=plt.subplots(2,1,figsize=(4,5),gridspec_kw = {'wspace':0, 'hspace':0.2})
    for i, ax in enumerate(figtspt.axes):
        ax.grid('on', linestyle='--')
        # ax.set_xticklabels([])
        ax.set_yticklabels([])

    IPSP = gEI*EPSP
    JE0=EPSP*kE*Cs
    JI0=IPSP*kI*Cs
    KE,KI=kE*Cs,kI*Cs
    p=KE/NE
    etaset = etassrecord[ig,:]
    forJEE = EPSP**2*KE*etaset[0]*(1-p)**2-2*EPSP**2*KE*(1-etaset[0])*(1-p)*p+EPSP**2*p**2*(NE-KE*2+KE*etaset[0])## sigma^E
    forJEI = -EPSP*IPSP*KI*etaset[1]*(1-p)**2+2*EPSP*IPSP*KI*(1-etaset[1])*(1-p)*p-EPSP*IPSP*p**2*(NI-KI*2+KI*etaset[1])## sigma^I

    forJIE=-EPSP*IPSP*KE*etaset[1]*(1-p)**2+EPSP*IPSP*KE*(1-etaset[1])*(1-p)*p+EPSP*IPSP*KE*(1-etaset[1])*(1-p)*p-EPSP*IPSP*p**2*(NE-KE*2+KE*etaset[1])## signa^E
    forJII=IPSP**2*KI*etaset[2]*(1-p)**2-2*IPSP**2*KI*(1-etaset[2])*(1-p)*p+IPSP**2*p**2*(NI-KI*2+KI*etaset[2])
    totallambda=(forJEE+forJEI)*N*JE0/NE*NE/N-(forJIE+forJII)*JI0
    Janalytical = JE0*EPSP**2*p*(etaset[0]-p)*(1)*NE+JE0*EPSP**2*p*(etaset[1]-p)*(KI/KE*gEI**2)*NE#JE0*EPSP**2*p*(eta-p)*(1+KI/KE*gEI**2)*NE
    Janalytical -= (gEI*JE0*EPSP**2*p*(etaset[1]-p)*(1)*NI + gEI*JE0*EPSP**2*p*(etaset[2]-p)*(KI/KE*gEI**2)*NI)#gEI*JE0*EPSP**2*p*(eta-p)*(1+KI/KE*gEI**2)*NI
    B2=totallambda
    roots = Cubic([1,-BAmvseries[0,0],0,-B2])
    Bsprvseries_theo[ig,:] = roots[:3] 
    #### @YX Oct 22 infinite series summing
    tauset_series = {}
    tau_set       = np.zeros(14)
    # # ~~~~~ reciprocal motisf ~~~~~~~~~~
    tau_set[10:]  = etauserecord[ig,0], etauserecord[ig,1],etauserecord[ig,1], etauserecord[ig,2]
    
    tdiv, tcon, tchn, trec = tau_set[:2],np.reshape(tau_set[2:6],(2,2)), np.reshape(tau_set[6:10],(2,2)),np.reshape(tau_set[10:],(2,2))
    
    tauset_series = {'tdiv':tdiv,
                    'tcon':tcon,
                    'tchn':tchn,
                    'trec':trec
                    }
    Jparams = [JE0, JI0]
    Nparams = [NE,NI]
    gmat    = np.array([np.sqrt(gE2),np.sqrt(gI2),np.sqrt(gE2),np.sqrt(gI2)])
    
    INIT    = [-0.5,1] 
    realg,imagg = fsolve(cal_outliers_general_complex,INIT,args=(tauset_series, [], Jparams, Nparams, gmat, eigvAm),xtol=10e-9,maxfev=1000)   
    Bsprvseries_theo_general[ig,1] = complex(realg, imagg)
    
    INIT = [-0.5,-1] ### heterogeneous g
    # INIT = [gaverage*np.sqrt(etaE),-0.] ### Fig3.1
    # INIT = [-0.05,-0.3] ### Fig3.-1
    
    realg,imagg = fsolve(cal_outliers_general_complex,INIT,args=(tauset_series, [], Jparams, Nparams, gmat, eigvAm),xtol=10e-9,maxfev=1000)   
    Bsprvseries_theo_general[ig,2] = complex(realg, imagg)
    
    
    INIT = [1.5,0]
    realg,imagg = fsolve(cal_outliers_general_complex,INIT,args=(tauset_series, [], Jparams, Nparams, gmat, eigvAm))#,xtol=10e-9,maxfev=1000)   
    Bsprvseries_theo_general[ig,0] = complex(realg, imagg)
    ### variance
    alphaNEI = Npercent.copy()
    gMmat = np.array([[gE2*alphaNEI[0],gI2*alphaNEI[1]],[gE2*alphaNEI[0],gI2*alphaNEI[1]]])
    eigvgm,eigvecgm=la.eig(gMmat) 
    r_g2= np.max(eigvgm)
    r_g = np.sqrt(r_g2)
    ### reciprocity 
    for idxtrial in range(ntrial):
        imagsort = np.argsort(np.imag(Bsprvseries[ig,idxtrial,:]))
        mostIm   = Bsprvseries[ig,idxtrial,imagsort[0]]#(imagsort[:3])
        axtspt[0].scatter(np.real(mostIm),np.imag(mostIm),s=2,c='blue',alpha=0.75) # 
        mostIm   = Bsprvseries[ig,idxtrial,imagsort[-1]]#(imagsort[:3])
        axtspt[0].scatter(np.real(mostIm),np.imag(mostIm),s=2,c='blue',alpha=0.75) # 

        imagsort = np.argsort(np.imag(Biidvseries[ig,idxtrial,:]))
        mostIm   = Biidvseries[ig,idxtrial,imagsort[0]]#np.mean(imagsort[:3])
        axtspt[1].scatter(np.real(mostIm),np.imag(mostIm),s=2,c='green',alpha=0.75) 
        mostIm   = Biidvseries[ig,idxtrial,imagsort[-1]]#np.mean(imagsort[:3])
        axtspt[1].scatter(np.real(mostIm),np.imag(mostIm),s=2,c='green',alpha=0.75) 

    # ### original sparse
    # axtspt[0].scatter(np.real(Bsprvseries[ig,:,0]),np.imag(Bsprvseries[ig,:,0])-0.2*idxc,s=10,c='blue',marker='o',alpha=0.75) # 
    # axtspt[1].scatter(np.real(Biidvseries[ig,:,0]),np.imag(Biidvseries[ig,:,0])-0.2*idxc,s=10,c='green',marker='o',alpha=0.75)


    axtspt[1].scatter(np.real(roots[0]),np.imag(roots[0]),s=50,c='',marker='o',edgecolor='purple')
    axtspt[1].scatter(np.real(roots[1]),np.imag(roots[1]),s=50,c='',marker='o',edgecolor='red') # 
    axtspt[1].scatter(np.real(roots[2]),np.imag(roots[2]),s=50,c='',marker='o',edgecolor='orange') #  # 
    axtspt[0].scatter(np.real(roots[0]),np.imag(roots[0]),s=50,c='',marker='o',edgecolor='purple')
    axtspt[0].scatter(np.real(roots[1]),np.imag(roots[1]),s=50,c='',marker='o',edgecolor='red')
    axtspt[0].scatter(np.real(roots[2]),np.imag(roots[2]),s=50,c='',marker='o',edgecolor='orange') # # 

    idrands=np.random.choice(np.arange(nrank,N),size=600,replace=False)
    axtspt[0].scatter(np.real(Bsprvseries[ig,iiitrial,idrands]),np.imag(Bsprvseries[ig,iiitrial,idrands]),s=2,c='blue',alpha=0.25) # 
    axtspt[1].scatter(np.real(Biidvseries[ig,iiitrial,idrands]),np.imag(Biidvseries[ig,iiitrial,idrands]),s=2,c='green',alpha=0.25) 
    
    ### first round revision -- S6 Appendix
    axtspt[0].scatter(np.real(Bsprvseries_theo_general[ig,0]),np.imag(Bsprvseries_theo_general[ig,0]),s=50,c='',marker='^',edgecolor='purple')
    axtspt[0].scatter(np.real(Bsprvseries_theo_general[ig,1]),np.imag(Bsprvseries_theo_general[ig,1]),s=50,c='',marker='^',edgecolor='red')
    axtspt[0].scatter(np.real(Bsprvseries_theo_general[ig,2]),np.imag(Bsprvseries_theo_general[ig,2]),s=50,c='',marker='^',edgecolor='orange')
    axtspt[1].scatter(np.real(Bsprvseries_theo_general[ig,0]),np.imag(Bsprvseries_theo_general[ig,0]),s=50,c='',marker='^',edgecolor='purple')
    axtspt[1].scatter(np.real(Bsprvseries_theo_general[ig,1]),np.imag(Bsprvseries_theo_general[ig,1]),s=50,c='',marker='^',edgecolor='red')
    axtspt[1].scatter(np.real(Bsprvseries_theo_general[ig,2]),np.imag(Bsprvseries_theo_general[ig,2]),s=50,c='',marker='^',edgecolor='orange')

    for i in range(2):
        axtspt[i].set_aspect('equal')
        axtspt[i].scatter(np.ones(1)*BAmvseries[0,0],np.zeros(1),s=50,c='',marker='o',edgecolor='red') #
        axtspt[i].spines['right'].set_color('none')
        axtspt[i].spines['top'].set_color('none')
        axtspt[i].xaxis.set_ticks_position('bottom')
        axtspt[i].spines['bottom'].set_position(('data', 0))
        axtspt[i].set_xlim(xlims)
        axtspt[i].set_ylim(ylims)
        axtspt[i].set_xticks(xticks)
        axtspt[i].set_yticks(yticks)
        axtspt[i].set_aspect('equal')

    ### ~~~~~~~~~~~~~~ raw iid sparse ~~~~~~~~~~~
    # eta = 0.0
    # longaxis,shortaxis=(1+eta)*r_g ,(1-eta)*r_g 
    # xr = longaxis*np.cos(theta)
    # yr = shortaxis*np.sin(theta)
    # axtspt[0].plot(xr, yr, color="gray", linewidth=0.5,linestyle='--')
    # axtspt[1].plot(xr, yr, color="gray", linewidth=0.5,linestyle='--')

    ### transition dynamics
    IPSP = gEI*EPSP
    JE0=EPSP*kE*Cs
    JI0=IPSP*kI*Cs
    KE,KI=kE*Cs,kI*Cs
    p=KE/NE

    if(etassrecord[-1,2]==0):
        coeffs= [1, 1, 0]
    else:
        coeffs=[1,1,1]
    # gee,gei,gie,gii
    gmat = np.array([np.sqrt(gE2), np.sqrt(gI2), np.sqrt(gE2), np.sqrt(gI2)])
    Jparams = np.array([JE0, JI0])
    aprob=p

    int_eta = 0.1
    transitioneta = fsolve(TransitionEta_spr, int_eta, args=(
        Jparams, yAm, xAm, eigvAm, gmat, coeffs, aprob, Nparams,))
    targetvalue = (1.0-eigvAm[0])
    transitioneta_ = fsolve(TransitionOver_spr, int_eta, args=(
        targetvalue, Jparams, yAm, xAm,  eigvAm, gmat, coeffs, aprob, Nparams,))
    print("transition eta LAMBDA:", transitioneta)
    print("transition eta OVERLAP:", transitioneta_)

    #### B. theoretical value of $\lambda$ v.s. numerical of the original sparse network
    xticks = np.linspace(0,1.0,2)
    xlims  = [-0.,1.]
    theta  = np.linspace(0, 2 * np.pi, 200)
    idxc=0
    cm=['b','g','c']#'bgc'
    envelopeRe = np.zeros(ngEI)
    envelopeIm = np.zeros(ngEI)
    # Bsprvseries_theo = np.zeros((ngEI,3),dtype=complex)
    IPSP = gEI*EPSP
    JE0=EPSP*kE*Cs
    JI0=IPSP*kI*Cs
    KE,KI=kE*Cs,kI*Cs
    p=KE/NE
    for ig,eta in enumerate (etaseries): 
        etaset = etassrecord[ig,:]
        forJEE=EPSP**2*KE*etaset[0]*(1-p)**2-2*EPSP**2*KE*(1-etaset[0])*(1-p)*p+EPSP**2*p**2*(NE-KE*2+KE*etaset[0])## sigma^E
        forJEI=-EPSP*IPSP*KI*etaset[1]*(1-p)**2+2*EPSP*IPSP*KI*(1-etaset[1])*(1-p)*p-EPSP*IPSP*p**2*(NI-KI*2+KI*etaset[1])## sigma^I

        forJIE=-EPSP*IPSP*KE*etaset[1]*(1-p)**2+EPSP*IPSP*KE*(1-etaset[1])*(1-p)*p+EPSP*IPSP*KE*(1-etaset[1])*(1-p)*p-EPSP*IPSP*p**2*(NE-KE*2+KE*etaset[1])## signa^E
        forJII=IPSP**2*KI*etaset[2]*(1-p)**2-2*IPSP**2*KI*(1-etaset[2])*(1-p)*p+IPSP**2*p**2*(NI-KI*2+KI*etaset[2])
        totallambda=(forJEE+forJEI)*N*JE0/NE*NE/N-(forJIE+forJII)*JI0
        Janalytical = JE0*EPSP**2*p*(etaset[0]-p)*(1)*NE+JE0*EPSP**2*p*(etaset[1]-p)*(KI/KE*gEI**2)*NE#JE0*EPSP**2*p*(eta-p)*(1+KI/KE*gEI**2)*NE
        Janalytical -= (gEI*JE0*EPSP**2*p*(etaset[1]-p)*(1)*NI + gEI*JE0*EPSP**2*p*(etaset[2]-p)*(KI/KE*gEI**2)*NI)#gEI*JE0*EPSP**2*p*(eta-p)*(1+KI/KE*gEI**2)*NI
        B2=totallambda
        roots = Cubic([1,-BAmvseries[0,0],0,-B2])
        #### SORT THE ROOTS @YX 07DEC
        realroots=np.real(roots)
        sidx = np.argsort(realroots)
        roots_uno = roots.copy()
        for i in range(len(sidx)):
            roots[i]=roots_uno[sidx[2-i]]
        ###--------------------------------
        Bsprvseries_theo[ig,:] = roots[:3]
        ### @YX 07DEC --- note here we use abs??? np.sqrt(re^2+im^2)
        # envelopeRe[idxeta]=np.mean(np.abs(np.squeeze(Beigvseries[idxeta,:,1])))
        # envelopeRe[ig]=np.mean(np.abs(np.real(Bsprvseries[ig,:,1])))
        ##### @YX  SORT AND THEN FIND ENVELOPEre and imag
        realsort = np.sort(np.real(Bsprvseries[ig,:,:]),axis=1)
        envelopeRe[ig]=np.mean(realsort[:,2])
        # amax_value = np.amax(arr, axis=0)
        imagsort = np.sort(np.imag(Bsprvseries[ig,:,:]),axis=1)
        envelopeIm[ig]=np.mean(imagsort[:,2])


    figtspt,axtspt=plt.subplots(figsize=(5,3))
    axtspt.plot(etaseries,np.real(Bsprvseries_theo[:,0]),color='black',linestyle='--',linewidth=1.5)
    
    #### @YX variance of the eigenvalue outlier
    mean_eigv_Jss,std_eigv_Jss = np.mean(np.real(Bsprvseries[:,:,0]),axis=1),np.std(np.real(Bsprvseries[:,:,0]),axis=1)
    # #### sparse symmetry
    axtspt.fill_between(etaseries,mean_eigv_Jss+std_eigv_Jss,mean_eigv_Jss-std_eigv_Jss, alpha=0.3,facecolor='tab:green')

    axtspt.plot(etaseries,np.ones(ngEI)*BAmvseries[0,0],linestyle='--',linewidth=1.5,color='grey')
    axtspt.set_xlim(xlims)
    axtspt.set_ylim(ylims)
    axtspt.set_xticks(xticks)
    axtspt.set_yticks(yticks)
    axtspt.legend()
    cm = ['purple','red','orange']
    # figR,axR=plt.subplots(figsize=(5,3))
    figR,axR=plt.subplots(2,1,figsize=(4,4),sharex=True, tight_layout=True)
    #### @YX mean and variance of the eigenvalue outlier
    for i in range(3):
        axR[0].plot(etaseries,Bsprvseries_theo[:,i].real,color=cm[i],linewidth=1.5,label='theo')
    mean_eigv_Jss,std_eigv_Jss = np.mean(np.real(Bsprvseries[:,:,0]),axis=1),np.std(np.real(Bsprvseries[:,:,0]),axis=1)
    axR[0].fill_between(etaseries,mean_eigv_Jss+std_eigv_Jss,mean_eigv_Jss-std_eigv_Jss, alpha=0.25,facecolor='purple')
    axR[0].fill_between(etaseries,-(envelopeRe),envelopeRe, color='gray',alpha=0.2)
    axR[0].plot(etaseries,np.ones(ngEI)*BAmvseries[0,0],linestyle='--',linewidth=1.5,color='black')

    # yticks = np.linspace(-1.0,1.5,3)
    # ylims = [-1.0,1.5]

    yticks = [-0.6,0,2]#np.linspace(-0.6,2.0,3)
    ylims = [-0.6,2.0]

    axR[0].set_xlim(xlims)
    axR[0].set_ylim(ylims)
    axR[0].set_xticks(xticks)
    axR[0].set_yticks(yticks)

    # yticks = np.linspace(-1.5,1.5,2)
    # ylims = [-1.5,1.5]
    yticks = np.linspace(-1.5,1.5,2)
    ylims = [-1.5,1.5]
    #### @YX 0409 modify -- imaginary part
    # figI,axI=plt.subplots(figsize=(5,3)) 
    for i in range(3):
        axR[1].plot(etaseries,np.imag(Bsprvseries_theo[:,i]),color=cm[i],linewidth=1.5,label='theo')
    axR[1].fill_between(etaseries,-(envelopeIm),envelopeIm, color='gray',alpha=0.2)

    mean_lambda_num = np.mean(Bsprvseries[:,:,:2].imag,axis=1)
    std_lambda_num  = np.std(Bsprvseries[:,:,:2].imag,axis=1)
    # axI.plot(etaseries,mean_lambda_num[:,0],c=cm[0],linewidth=1.5,)
    axR[1].fill_between(etaseries,mean_lambda_num[:,0]+std_lambda_num[:,0],mean_lambda_num[:,0]-std_lambda_num[:,0],alpha=0.25,facecolor='tab:blue')

    axR[1].set_xlim(xlims)
    axR[1].set_ylim(ylims)
    axR[1].set_xticks(xticks)
    axR[1].set_yticks(yticks)
     
    # ### scatter with colormap
    # xticks = np.linspace(0.5,1.5,2)
    # xlims = [0.5,1.5]
    # yticks = np.linspace(0.5,1.5,2)
    # ylims = [0.5,1.5]

    ### ~~~~~~~~~~ NEW HIGH ~~~~~~~~
    ### scatter with colormap
    xticks = [0.5,2.0]
    xlims  = [0.5,2.0]
    yticks = [0.5,2.0]
    ylims  = [0.5,2.0]

    fig,ax=plt.subplots(figsize=(4,4))
    # scatter(X, Y, c = angle, cmap = cm.hsv)
    for i in range(ntrial):
        ax.scatter(Bsprvseries[:,i,0].real,Biidvseries[:,i,0].real,c=etaseries,cmap=matplotlib.cm.Blues,s=22,marker='o')
    ax.set_xlim(ylims)
    ax.set_ylim(ylims)
    ax.set_xticks(yticks)
    ax.set_yticks(yticks)

    fig,ax=plt.subplots(figsize=(4,4))
    # xticks = np.linspace(0.0,1.5,2)
    # xlims = [-0.2,1.7]
    # yticks = np.linspace(0.0,1.5,2)
    # ylims = [-0.2,1.7]
    ###~~~~~~~~~~~ new one
    xticks = [-0.5,1.0]
    xlims  = [-0.5,1.0]
    yticks = [-0.5,1.0]
    ylims  = [-0.5,0.8]
    mean_DeltaLR_num,std_DeltaLR_num = np.mean(Sigmn_num[:,:,2],axis=1),np.std(Sigmn_num[:,:,2],axis=1)    
    x  = np.real(Bsprvseries_theo[:,0])-eigvAm[0]
    y1 = mean_DeltaLR_num+std_DeltaLR_num
    y2 = mean_DeltaLR_num-std_DeltaLR_num

    polygon  = ax.fill_between(x, y1, y2, lw=0, color='none')
    xlim     = plt.xlim()
    ylim     = plt.ylim()
    verts    = plt.vstack([p.vertices for p in polygon.get_paths()])
    gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='Blues', aspect='auto',
                          extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)


    ax.set_xlim(ylims)
    ax.set_ylim(ylims)
    ax.set_xticks(yticks)
    ax.set_yticks(yticks)
    
    ### D. overlap between m and n
    ### eigenvalue 
    xticks = np.linspace(0.0,1.0,2)
    xlims = [0.0,1.0]
    yticks = np.linspace(0.0,1.0,3)
    ylims = [-0.2,1.20]
    theta = np.linspace(0, 2 * np.pi, 200)
    idxc=0
    cm=['r','b','g']#rbg'

    Sigmn_theo_ = np.zeros((ngEI,3))
    IPSP = gEI*EPSP
    JE0=EPSP*kE*Cs
    JI0=IPSP*kI*Cs
    KE,KI=kE*Cs,kI*Cs
    p=KE/NE
    for ig,eta in enumerate (etaseries): 
        
        etaset = etassrecord[ig,:]#eta*np.ones(3)#
        # etaset[2:]=0
        forJEE=EPSP**2*KE*etaset[0]*(1-p)**2-2*EPSP**2*KE*(1-etaset[0])*(1-p)*p+EPSP**2*p**2*(NE-KE*2+KE*etaset[0])## sigma^E
        forJEI=-EPSP*IPSP*KI*etaset[1]*(1-p)**2+2*EPSP*IPSP*KI*(1-etaset[1])*(1-p)*p-EPSP*IPSP*p**2*(NI-KI*2+KI*etaset[1])## sigma^I

        # forJIE=-EPSP*IPSP*KE*eta*(1-p)**2+2*EPSP*IPSP*KE*(1-eta)*(1-p)*p-EPSP*IPSP*p**2*(NE-KE*2+KE*eta)## ERROR!!!
        forJIE=-EPSP*IPSP*KE*etaset[1]*(1-p)**2+EPSP*IPSP*KE*(1-etaset[1])*(1-p)*p+EPSP*IPSP*KE*(1-etaset[1])*(1-p)*p-EPSP*IPSP*p**2*(NE-KE*2+KE*etaset[1])## signa^E
        forJII=IPSP**2*KI*etaset[2]*(1-p)**2-2*IPSP**2*KI*(1-etaset[2])*(1-p)*p+IPSP**2*p**2*(NI-KI*2+KI*etaset[2])
        totallambda=(forJEE+forJEI)*N*JE0/NE*NE/N-(forJIE+forJII)*JI0
        B2=totallambda
        roots = Cubic([1,-BAmvseries[0,0],0,-B2])
        
        ### @YX  -- SORT THE ROOTS
        realroots= [np.real(x) for x in roots]
        # print(realroots)
        sidx = np.argsort(realroots)
        roots_uno = roots.copy()
        for i in range(len(sidx)):
            roots[i]=roots_uno[sidx[2-i]]
        ###--------------------------------
        JanalyticalE = (forJEE*N*JE0/NE*NE+forJIE*(-N*JI0/NI*NI))/NE
        JanalyticalI = (forJEI*N*JE0/NE*NE+forJII*(-N*JI0/NI*NI))/NI
        Sigmn_theo_[ig,0],Sigmn_theo_[ig,1] = JanalyticalE/roots[0]**2,JanalyticalI/roots[0]**2#JanalyticalE/eigvAm[0]**2,JanalyticalI/eigvAm[0]**2#JanalyticalE/Biidvseries[ig,0,0]**2,JanalyticalI/Biidvseries[ig,0,0]**2#
        Sigmn_theo_[ig,2] = Sigmn_theo_[ig,0]*Npercent[0]+Sigmn_theo_[ig,1]*Npercent[1]
        
    #### ~~~~~ previous scatter plot
    # ax.plot(np.real(Bsprvseries_theo[:,0])-eigvAm[0],Sigmn_theo[:,0,2],c='black',linewidth=2.0)
    ax.plot(np.real(Bsprvseries_theo[:,0])-eigvAm[0],Sigmn_theo_[:,2],c='black',linewidth=2.0)
    
    titlestr='unperturbed eigenvalue'

    fig,ax2 = plt.subplots(figsize=(5,3))
    ax2.fill_between(etaseries,np.mean(Sigmn_num[:,:,0],axis=1)+np.std(Sigmn_num[:,:,0],axis=1),\
        np.mean(Sigmn_num[:,:,0],axis=1)-np.std(Sigmn_num[:,:,0],axis=1),color='tab:red',alpha=0.3,label=r'numerical')
    ax2.plot(etaseries,Sigmn_theo_[:,0],'tab:red',linewidth =1.5,label=r'theoretical')
    
    ax2.fill_between(etaseries,np.mean(Sigmn_num[:,:,1],axis=1)+np.std(Sigmn_num[:,:,1],axis=1),\
        np.mean(Sigmn_num[:,:,1],axis=1)-np.std(Sigmn_num[:,:,1],axis=1),color='tab:blue',alpha=0.3,label=r'numerical')
    ax2.plot(etaseries,Sigmn_theo_[:,1],'tab:blue',linewidth =1.5,label=r'theoretical')

    ax2.fill_between(etaseries,np.mean(Sigmn_num[:,:,2],axis=1)+np.std(Sigmn_num[:,:,2],axis=1),\
        np.mean(Sigmn_num[:,:,2],axis=1)-np.std(Sigmn_num[:,:,2],axis=1),color='black',alpha=0.3,label=r'numerical')
    ax2.plot(etaseries,Sigmn_theo_[:,2],'black',linewidth =1.5,label=r'theoretical')

    yticks = [-2,0,3]#np.linspace(-2,5,7)
    ylims = [-2,3]
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)

   
    ##### ----------------------- 3*1 subplots
    yticks = np.linspace(-2,1,3)
    ylims = [-2,1]
    fig,ax2 = plt.subplots(3,1,figsize=(5,4),sharex=True,tight_layout=True)
    ax2[0].fill_between(etaseries,np.mean(Sigmn_num[:,:,0],axis=1)+np.std(Sigmn_num[:,:,0],axis=1),\
        np.mean(Sigmn_num[:,:,0],axis=1)-np.std(Sigmn_num[:,:,0],axis=1),color='tab:red',alpha=0.25,label=r'numerical')
    ax2[0].plot(etaseries,Sigmn_theo_[:,0],'tab:red',linewidth =0.75,linestyle='--',label=r'theoretical')
    # yticks = np.linspace(-5,10,4)
    # ylims = [-5,10]
    ax2[0].set_xlim(xlims)
    ax2[0].set_ylim(ylims)
    ax2[0].set_xticks(xticks)
    ax2[0].set_yticks(yticks) 

    # fig,ax2 = plt.subplots(figsize=(5,2))
    yticks = np.linspace(-0.5,1,2)
    ylims = [-0.5,1]
    ax2[2].fill_between(etaseries,np.mean(Sigmn_num[:,:,2],axis=1)+np.std(Sigmn_num[:,:,2],axis=1),\
        np.mean(Sigmn_num[:,:,2],axis=1)-np.std(Sigmn_num[:,:,2],axis=1),color='black',alpha=0.25,label=r'numerical')
    # ax2[2].plot(etaseries,Sigmn_theo_[:,2],'k',linewidth =0.75,label=r'theoretical')
    # # ax2.plot(etaseries,np.mean(Bsprvseries[:,:,0],axis=1)-eigvAm[0],'orange',linewidth =1.5,label=r'theoretical')
    ax2[2].plot(etaseries,Sigmn_theo_[:,1]*1.0/5+Sigmn_theo_[:,0]*4.0/5,'k',linestyle='--',linewidth =1.5,label=r'theoretical')

    ax2[2].set_xlim(xlims)
    ax2[2].set_ylim(ylims)
    ax2[2].set_xticks(xticks)
    ax2[2].set_yticks(yticks) 

    yticks = np.linspace(-0.5,4,2)
    ylims = [-0.5,4]

    # fig,ax2 = plt.subplots(figsize=(5,2))
    ax2[1].fill_between(etaseries,np.mean(Sigmn_num[:,:,1],axis=1)+np.std(Sigmn_num[:,:,1],axis=1),\
        np.mean(Sigmn_num[:,:,1],axis=1)-np.std(Sigmn_num[:,:,1],axis=1),color='tab:blue',alpha=0.25,label=r'numerical')
    ax2[1].plot(etaseries,Sigmn_theo_[:,1],'tab:blue',linewidth =0.75,linestyle='--',label=r'theoretical')
    ax2[1].set_xlim(xlims)
    ax2[1].set_ylim(ylims)
    ax2[1].set_xticks(xticks)
    ax2[1].set_yticks(yticks) 

    #### >>>>>>>>> individual perturbations >>>>>>>>>
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
    gEIsamples=np.array([16,20])

    for i in range(len(gEIsamples)):
        ax[0,i].plot(xticks,yticks,color='darkred',linestyle='--')
        ax[1,i].plot(xticks,yticks,color='darkred',linestyle='--')
        #### @YX modify 2508 -- redundancy
        #### @YX modify 2508 -- from Reigvecseries[i,...] to Reigvecseries[etaseriesples[i]...]
        idrandsE=np.random.choice(np.arange(0,NE),size=100,replace=False)
        idrandsI=np.random.choice(np.arange(NE,N),size=100,replace=False)

        ax[0,i].scatter(np.real(Rsprvecseries[gEIsamples[i],idxtrial,idrandsE]),np.real(Restvecseries[gEIsamples[i],idxtrial,idrandsE]),s=2,c='red',alpha=0.5)
        ax[1,i].scatter(np.real(Lsprvecseries[gEIsamples[i],idxtrial,idrandsE]),np.real(Lestvecseries[gEIsamples[i],idxtrial,idrandsE]),s=2,c='red',alpha=0.5)
        
        ax[0,i].scatter(np.real(Rsprvecseries[gEIsamples[i],idxtrial,idrandsI]),np.real(Restvecseries[gEIsamples[i],idxtrial,idrandsI]),s=2,c='tab:blue',alpha=0.5)
        ax[1,i].scatter(np.real(Lsprvecseries[gEIsamples[i],idxtrial,idrandsI]),np.real(Lestvecseries[gEIsamples[i],idxtrial,idrandsI]),s=2,c='tab:blue',alpha=0.5)
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

    
    
    # def dyn_analysis():
    ## self-consistency in \kappa, for rank one approximation 
    kappa_theo_ss = np.zeros((ngEI,3))
    for ig,eta in enumerate (etaseries): 
        alphaa = ig*1.0/len(etaseries)
        IPSP = gEI*EPSP
        JE0=EPSP*kE*Cs
        JI0=IPSP*kI*Cs
        Kparams = np.array([kE,kI,Cs])
        # print(Nparams,etaseries,gEI)
        etaset = etassrecord[ig,:]
        init_k=np.max(kappaintersect_Full[ig,:, 0])
        meanvv = (np.max(kappaintersect_Full[ig,:, 0])+np.min(kappaintersect_Full[ig,:, 0]))/2.0
        #### >>>>>>>>>>>>> lambda 0???
        lambda0= np.real(Bsprvseries_theo[ig,0])#eigvAm[0]#
        # kappa_theo_ss[ig,0]
        kappa_max= fsolve(symsparse_kappaP,init_k,args=(gEI,JE0,JI0,EPSP,Kparams,Nparams,etaset,lambda0),xtol=1e-6,maxfev=800)#eigvAm[0]))#
        residual0=np.abs(symsparse_kappaP(kappa_max,gEI,JE0,JI0,EPSP,Kparams,Nparams,etaset,lambda0))
        init_k=0.5*init_k#((1.0-alphaa)*np.max(kappaintersect_Full[ig,:, 0])+(alphaa)*meanvv)
        kappa_theo_ss[ig,1]
        kappa_middle= fsolve(symsparse_kappaP,init_k,args=(gEI,JE0,JI0,EPSP,Kparams,Nparams,etaset,lambda0),xtol=1e-6,maxfev=800)#eigvAm[0]))#
        residual1=np.abs(symsparse_kappaP(kappa_middle,gEI,JE0,JI0,EPSP,Kparams,Nparams,etaset,lambda0))
        init_k=np.min(kappaintersect_Full[ig,:, 0])
        kappa_theo_ss[ig,2]= fsolve(symsparse_kappaP,init_k,args=(gEI,JE0,JI0,EPSP,Kparams,Nparams,etaset,lambda0),xtol=1e-6,maxfev=800)#eigvAm[0]))#
        if(residual0>1e-3):
            kappa_theo_ss[ig,0]=kappa_theo_ss[ig,2]
        else:
            kappa_theo_ss[ig,0]=kappa_max
        if(residual1>1e-3):
            kappa_theo_ss[ig,1]=kappa_theo_ss[ig,2]
        else:
            kappa_theo_ss[ig,1]=kappa_middle

    ### ----------------- kappa fill_between
    fig,ax = plt.subplots(figsize=(6,3))
    ax.plot(etaseries,kappa_theo_ss[:,0],color='tab:purple',linewidth=1.5)
    ax.plot(etaseries,kappa_theo_ss[:,1],color='tab:purple',linewidth=1.5)
    ax.plot(etaseries,kappa_theo_ss[:,2],color='tab:purple',linewidth=1.5)
   

    #### @YX 3108 add--calculate the shaded kappa
    mean_pos_kappa_full,mean_neg_kappa_full = np.zeros(ngEI),np.zeros(ngEI)
    std_pos_kappa_full,std_neg_kappa_full   = np.zeros(ngEI),np.zeros(ngEI)
    mean_pos_kappa_r1,mean_neg_kappa_r1     = np.zeros(ngEI),np.zeros(ngEI)
    std_pos_kappa_r1,std_neg_kappa_r1       = np.zeros(ngEI),np.zeros(ngEI)
    mean_pos_kappa_spr,mean_neg_kappa_spr   = np.zeros(ngEI),np.zeros(ngEI)
    std_pos_kappa_spr,std_neg_kappa_spr     = np.zeros(ngEI),np.zeros(ngEI)

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

        neg_full = kappaintersect_Full[ig,np.where(kappaintersect_Full[ig,:,0]<pos_t0),0]
        mean_neg_kappa_full[ig] = np.mean(neg_full)
        std_neg_kappa_full[ig]  = np.std(neg_full)
        neg_r1 = kappaintersect_R1[ig,np.where(kappaintersect_R1[ig,:,0]<pos_t0),0]
        mean_neg_kappa_r1[ig] = np.mean(neg_r1)
        std_neg_kappa_r1[ig]  = np.std(neg_r1)
        neg_spr = kappaintersect_Sparse[ig,np.where(kappaintersect_Sparse[ig,:,0]<pos_t0),0]
        mean_neg_kappa_spr[ig] = np.mean(neg_spr)
        std_neg_kappa_spr[ig]  = np.std(neg_spr)

    low_bound = np.zeros_like(mean_pos_kappa_spr)
    for imax in range(len(mean_pos_kappa_spr)):
        low_bound[imax]=max(0,mean_pos_kappa_spr[imax]-std_pos_kappa_spr[imax])
    ax.fill_between(etaseries,mean_pos_kappa_spr+std_pos_kappa_spr,low_bound, alpha=0.3,facecolor='black')

    low_bound = np.zeros_like(mean_neg_kappa_spr)
    for imax in range(len(mean_neg_kappa_spr)):
        low_bound[imax]=max(0,mean_neg_kappa_spr[imax]-std_neg_kappa_spr[imax])
    ax.fill_between(etaseries,mean_neg_kappa_spr+std_neg_kappa_spr,low_bound, alpha=0.3,facecolor='black')
    ### ~~~~~~~~~~~ new one
    yticks = [0,2.5]
    ylims  = [0,2.5]
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
                mvec_norm = np.reshape(Restvecseries[iJE,iktrial,:,0],(N,1))
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(np.squeeze(xfpseries_Sparse[iJE,iktrial,:,-1]),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            pavgkappa[iJE],pstdkappa[iJE] = np.mean(numkappa[iJE,ptrialXE]),np.std(numkappa[iJE,ptrialXE])
        if len(ntrialXE)>0:
            for iktrial in ntrialXE:
                mvec_norm = np.reshape(Restvecseries[iJE,iktrial,:,0],(N,1))
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(np.squeeze(xfpseries_Sparse[iJE,iktrial,:,-1]),(1,N))@mvec_norm)/np.squeeze(mvec_norm.T@mvec_norm)
            navgkappa[iJE],nstdkappa[iJE] = np.mean(numkappa[iJE,ntrialXE]),np.std(numkappa[iJE,ntrialXE])
    ax.plot(etaseries, pavgkappa,c='black', linewidth=1.5, linestyle='--')
    ax.plot(etaseries, navgkappa,c='black', linewidth=1.5, linestyle='--')
    ax.fill_between(etaseries, pavgkappa+pstdkappa,pavgkappa-pstdkappa, alpha=.5, facecolor='blue')


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
                nvec_norm = np.reshape(Lestvecseries[iJE,iktrial,:,0],(N,1))
                phix      = 1.0+np.tanh(np.squeeze(xfpseries_Sparse[iJE,iktrial,:,-1])-shiftx)
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(phix,(1,N))@nvec_norm)/N
            pavgkappa[iJE],pstdkappa[iJE] = np.mean(numkappa[iJE,ptrialXE]),np.std(numkappa[iJE,ptrialXE])
        if len(ntrialXE)>0:
            for iktrial in ntrialXE:
                nvec_norm = np.reshape(Lestvecseries[iJE,iktrial,:,0],(N,1))
                phix      = 1.0+np.tanh(np.squeeze(xfpseries_Sparse[iJE,iktrial,:,-1])-shiftx)
                numkappa[iJE,iktrial] =np.squeeze(np.reshape(phix,(1,N))@nvec_norm)/N
            navgkappa[iJE],nstdkappa[iJE] = np.mean(numkappa[iJE,ntrialXE]),np.std(numkappa[iJE,ntrialXE])

    ax.plot(etaseries, pavgkappa,c='red', linewidth=1.5, linestyle='--')
    ax.plot(etaseries, navgkappa,c='red', linewidth=1.5, linestyle='--')

    mu_ss_theo,variance_ss_theo = np.zeros((ngEI,2,3)),np.zeros((ngEI,2,3))
    for ig in range(ngEI):
        ## Delta m2
        p = kE*Cs/NE
        lambda0 = np.real(Bsprvseries_theo[ig,0])#eigvAm[0]#
        lambda0 = np.real(lambda0)
        sigm2 = EPSP**2*kE*Cs*(1-p)/lambda0**2+IPSP**2*kI*Cs*(1-p)/lambda0**2
        mu_ss_theo[ig,0,:],mu_ss_theo[ig,1,:] = kappa_theo_ss[ig,:]*1,kappa_theo_ss[ig,:]*1
        variance_ss_theo[ig,0,:],variance_ss_theo[ig,1,:] = kappa_theo_ss[ig,:]**2*sigm2,kappa_theo_ss[ig,:]**2*sigm2

    ### mu and variance for each realization
    variance_full_num,variance_sparse_num,variance_R1_num= np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))
    mu_full_num,mu_sparse_num,mu_R1_num = np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2)),np.zeros((ngEI,ntrial,2))
    for igEI in range(ngEI):
        for iktrial in range(ntrial):
            ## numerical results for Full Mat
            variance_full_num[igEI,iktrial,0],variance_full_num[igEI,iktrial,1]=np.std(xfpseries_Full[igEI,iktrial,:NE,-1])**2,np.std(xfpseries_Full[igEI,iktrial,NE:,-1])**2
            mu_full_num[igEI,iktrial,0],mu_full_num[igEI,iktrial,1]=np.mean(xfpseries_Full[igEI,iktrial,:NE,-1]),np.mean(xfpseries_Full[igEI,iktrial,NE:,-1])
            
            variance_sparse_num[igEI,iktrial,0],variance_sparse_num[igEI,iktrial,1]=np.std(xfpseries_Sparse[igEI,iktrial,:NE,-1])**2,np.std(xfpseries_Sparse[igEI,iktrial,NE:,-1])**2
            mu_sparse_num[igEI,iktrial,0],mu_sparse_num[igEI,iktrial,1]=np.mean(xfpseries_Sparse[igEI,iktrial,:NE,-1]),np.mean(xfpseries_Sparse[igEI,iktrial,NE:,-1])

            variance_R1_num[igEI,iktrial,0],variance_R1_num[igEI,iktrial,1]=np.std(xfpseries_R1[igEI,iktrial,:NE,-1])**2,np.std(xfpseries_R1[igEI,iktrial,NE:,-1])**2
            mu_R1_num[igEI,iktrial,0],mu_R1_num[igEI,iktrial,1]=np.mean(xfpseries_R1[igEI,iktrial,:NE,-1]),np.mean(xfpseries_R1[igEI,iktrial,NE:,-1])

    iktrial=0
    dshift = 0.0

    ### E. fill in \mu and \Delta for x dynamics
    clrs = ['b','g','c']#sns.color_palette("husl", 5)

    xticks = np.linspace(0,1.0,2)# np.linspace(0,0.8,2)
    xlims = [0,1.0]#[-0.1,1.1]#[-0.1,0.9]

    # yticks = np.linspace(-3,3,2)
    # ylims = [-3.0,3.0]
    
    # yticks = np.linspace(-1.5,1.5,2)
    # ylims = [-1.6,1.6]
    ###~~~~~~~~~~ new one
    yticks = np.linspace(-1.0,1.0,3)
    ylims = [-1.0,1.0]

    fig,ax2 = plt.subplots(2,1,figsize=(6,4),sharex=True,tight_layout=True)
    #### -----------------  mean act------------------------------------------------------------
    pmean_full_numE,pmean_R1_numE,pmean_full_numI,pmean_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)
    pstd_full_numE,pstd_R1_numE,pstd_full_numI,pstd_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)

    nmean_full_numE,nmean_R1_numE,nmean_full_numI,nmean_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)
    nstd_full_numE,nstd_R1_numE,nstd_full_numI,nstd_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)

    #### @YX 2908 add gavg -- main-text: IGNORE
    pmean_spr_numE,pmean_spr_numI=np.zeros(ngEI),np.zeros(ngEI)
    pstd_spr_numE,pstd_spr_numI=np.zeros(ngEI),np.zeros(ngEI)
    nmean_spr_numE,nmean_spr_numI=np.zeros(ngEI),np.zeros(ngEI)
    nstd_spr_numE,nstd_spr_numI=np.zeros(ngEI),np.zeros(ngEI)

    pos_t0 = 0.7

    #### @YX 08DEC --- THEORETICAL MU 
    for i in range(3):
        ax2[0].plot(etaseries,mu_ss_theo[:,0,i],color='tab:red',lw=1.0,label='R1 theo')
        ax2[1].plot(etaseries,mu_ss_theo[:,1,i],color='tab:blue',lw=1.0,label='R1 theo')

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
        pmean_spr_numE[i]  = np.mean(mu_sparse_num[i,np.where(mu_sparse_num[i,:,0]>=pos_t0),0])
        pstd_spr_numE[i]   = np.std(mu_sparse_num[i,np.where(mu_sparse_num[i,:,0]>=pos_t0),0])

        pmean_spr_numI[i]  = np.mean(mu_sparse_num[i,np.where(mu_sparse_num[i,:,1]>=pos_t0),1])
        pstd_spr_numI[i]   = np.std(mu_sparse_num[i,np.where(mu_sparse_num[i,:,1]>=pos_t0),1])

        nmean_spr_numE[i]  = np.mean(mu_sparse_num[i,np.where(mu_sparse_num[i,:,0]<pos_t0),0])
        nstd_spr_numE[i]   = np.std(mu_sparse_num[i,np.where(mu_sparse_num[i,:,0]<pos_t0),0])

        nmean_spr_numI[i]  = np.mean(mu_sparse_num[i,np.where(mu_sparse_num[i,:,1]<pos_t0),1])
        nstd_spr_numI[i]   = np.std(mu_sparse_num[i,np.where(mu_sparse_num[i,:,1]<pos_t0),1])
    #### ----------------------------- sparse   --------------------------------------------
    low_bound = np.zeros_like(pmean_spr_numE)
    for imax in range(len(pmean_spr_numE)):
        low_bound[imax]=max(0,pmean_spr_numE[imax]-pstd_spr_numE[imax])
    ax2[0].fill_between(etaseries,pmean_spr_numE+pstd_spr_numE,low_bound, alpha=0.25,facecolor='tab:red')

    low_bound = np.zeros_like(pmean_spr_numI)
    for imax in range(len(pmean_spr_numI)):
        low_bound[imax]=max(0,pmean_spr_numI[imax]-pstd_spr_numI[imax])
    ax2[1].fill_between(etaseries,pmean_spr_numI+pstd_spr_numI,low_bound, alpha=0.25,facecolor='tab:blue')

    low_bound = np.zeros_like(nmean_spr_numE)
    for imax in range(len(nmean_spr_numE)):
        low_bound[imax]=max(0,nmean_spr_numE[imax]-nstd_spr_numE[imax])
    ax2[0].fill_between(etaseries,nmean_spr_numE+nstd_spr_numE,low_bound, alpha=0.25,facecolor='tab:red')

    low_bound = np.zeros_like(nmean_spr_numI)
    for imax in range(len(nmean_spr_numI)):
        low_bound[imax]=max(0,nmean_spr_numI[imax]-nstd_spr_numI[imax])
    ax2[1].fill_between(etaseries,nmean_spr_numI+nstd_spr_numI,low_bound, alpha=0.25,facecolor='tab:blue')

    # for i in range(2):     
    #         ax2[i].set_xlim(xlims)
    #         ax2[i].set_ylim(ylims)
    #         ax2[i].set_xticks(xticks)
    #         ax2[i].set_yticks(yticks)
            # ax2[i].legend()
    # ax2[1].set_xlabel(r'reciprocal connectivity $\eta$')


    #### variance 
    clrs = ['b','g','c']#sns.color_palette("husl", 5)

    xticks = np.linspace(0,1.0,2)# np.linspace(0,0.8,2)
    xlims = [0,1.0]#[-0.1,1.1]#[-0.1,0.9]

    # yticks = np.linspace(-0.0,0.5,2)
    # ylims = [-0.0,0.5]
    ### ~~~~~ new one
    yticks = [0,1.0]
    ylims  = [0,1.0]

    fig,ax2    = plt.subplots(2,1,figsize=(6,4),sharex=True,tight_layout=True)
    fige, ax2e = plt.subplots(2,1,figsize=(6,4),sharex=True,tight_layout=True) ### ERROR BAR
    ax2[0].plot(etaseries,variance_ss_theo[:,0,0],color='tab:red',lw=1.0,label='R1 theo')
    ax2[1].plot(etaseries,variance_ss_theo[:,1,0],color='tab:blue',lw=1.0,label='R1 theo')
    ax2[0].plot(etaseries,variance_ss_theo[:,0,2],color='tab:red',lw=1.0,label='R1 theo')
    ax2[1].plot(etaseries,variance_ss_theo[:,1,2],color='tab:blue',lw=1.0,label='R1 theo')

    ax2e[0].plot(etaseries,variance_ss_theo[:,0,0],color='tab:red',lw=1.0,label='R1 theo')
    ax2e[1].plot(etaseries,variance_ss_theo[:,1,0],color='tab:blue',lw=1.0,label='R1 theo')
    ax2e[0].plot(etaseries,variance_ss_theo[:,0,2],color='tab:red',lw=1.0,label='R1 theo')
    ax2e[1].plot(etaseries,variance_ss_theo[:,1,2],color='tab:blue',lw=1.0,label='R1 theo')
    #### -----------------  mean act------------------------------------------------------------
    pmean_full_numE,pmean_R1_numE,pmean_full_numI,pmean_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)
    pstd_full_numE,pstd_R1_numE,pstd_full_numI,pstd_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)

    nmean_full_numE,nmean_R1_numE,nmean_full_numI,nmean_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)
    nstd_full_numE,nstd_R1_numE,nstd_full_numI,nstd_R1_numI=np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI),np.zeros(ngEI)

    #### @YX 2908 add gavg -- main-text: IGNORE
    pmean_spr_numE,pmean_spr_numI=np.zeros(ngEI),np.zeros(ngEI)
    pstd_spr_numE,pstd_spr_numI=np.zeros(ngEI),np.zeros(ngEI)
    nmean_spr_numE,nmean_spr_numI=np.zeros(ngEI),np.zeros(ngEI)
    nstd_spr_numE,nstd_spr_numI=np.zeros(ngEI),np.zeros(ngEI)

    pos_t0=0.15
    for i in range(ngEI):
        pmean_full_numE[i], pmean_R1_numE[i] = np.mean(variance_full_num[i,np.where(variance_full_num[i,:,0]>=pos_t0),0]),np.mean(variance_R1_num[i,np.where(variance_R1_num[i,:,0]>=pos_t0),0])
        pstd_full_numE[i], pstd_R1_numE[i]   = np.std(variance_full_num[i,np.where(variance_full_num[i,:,0]>=pos_t0),0]),np.std(variance_R1_num[i,np.where(variance_R1_num[i,:,0]>=pos_t0),0])

        pmean_full_numI[i], pmean_R1_numI[i] = np.mean(variance_full_num[i,np.where(variance_full_num[i,:,1]>=pos_t0),1]),np.mean(variance_R1_num[i,np.where(variance_R1_num[i,:,1]>=pos_t0),1])
        pstd_full_numI[i], pstd_R1_numI[i]   = np.std(variance_full_num[i,np.where(variance_full_num[i,:,1]>=pos_t0),1]),np.std(variance_R1_num[i,np.where(variance_R1_num[i,:,1]>=pos_t0),1])

        #### @YX 0109 add dynamics of sparse E-I network
        pmean_spr_numE[i] = np.mean(variance_sparse_num[i,np.where(variance_sparse_num[i,:,0]>=pos_t0),0])
        pstd_spr_numE[i]   = np.std(variance_sparse_num[i,np.where(variance_sparse_num[i,:,0]>=pos_t0),0])

        pmean_spr_numI[i] = np.mean(variance_sparse_num[i,np.where(variance_sparse_num[i,:,1]>=pos_t0),1])
        pstd_spr_numI[i]  = np.std(variance_sparse_num[i,np.where(variance_sparse_num[i,:,1]>=pos_t0),1])

    #### ----------------------------- sparse   -------------------------------------------
    ax2[0].plot(etaseries,pmean_spr_numE,color='tab:red',lw=1.0,linestyle='--')
    ax2[1].plot(etaseries,pmean_spr_numI,color='tab:blue',lw=1.0,linestyle='--')

    low_bound = np.zeros_like(pmean_spr_numE)
    for imax in range(len(pmean_spr_numE)):
        low_bound[imax]=max(0,pmean_spr_numE[imax]-pstd_spr_numE[imax])
    ax2[0].fill_between(etaseries, pmean_spr_numE+pstd_spr_numE,low_bound, alpha=0.3, facecolor='tab:red')
    low_bound = np.zeros_like(pmean_spr_numI)
    for imax in range(len(pmean_spr_numI)):
        low_bound[imax]=max(0,pmean_spr_numI[imax]-pstd_spr_numI[imax])
    ax2[1].fill_between(etaseries, pmean_spr_numI+pstd_spr_numI,low_bound, alpha=0.3, facecolor='tab:blue')
    ### error bar 
    ax2e[0].errorbar(etaseries, pmean_spr_numE,pstd_spr_numE/np.sqrt(ntrial),c='tab:red')
    ax2e[1].errorbar(etaseries, pmean_spr_numI,pstd_spr_numI/np.sqrt(ntrial), c='tab:blue')


    for i in range(ngEI):
        pmean_full_numE[i], pmean_R1_numE[i] = np.mean(variance_full_num[i,np.where(variance_full_num[i,:,0]<pos_t0),0]),np.mean(variance_R1_num[i,np.where(variance_R1_num[i,:,0]<pos_t0),0])
        pstd_full_numE[i], pstd_R1_numE[i]   = np.std(variance_full_num[i,np.where(variance_full_num[i,:,0]<pos_t0),0]),np.std(variance_R1_num[i,np.where(variance_R1_num[i,:,0]<pos_t0),0])

        pmean_full_numI[i], pmean_R1_numI[i] = np.mean(variance_full_num[i,np.where(variance_full_num[i,:,1]<pos_t0),1]),np.mean(variance_R1_num[i,np.where(variance_R1_num[i,:,1]<pos_t0),1])
        pstd_full_numI[i], pstd_R1_numI[i]   = np.std(variance_full_num[i,np.where(variance_full_num[i,:,1]<pos_t0),1]),np.std(variance_R1_num[i,np.where(variance_R1_num[i,:,1]<pos_t0),1])

        #### @YX 0109 add dynamics of sparse E-I network
        pmean_spr_numE[i] = np.mean(variance_sparse_num[i,np.where(variance_sparse_num[i,:,0]<pos_t0),0])
        pstd_spr_numE[i]   = np.std(variance_sparse_num[i,np.where(variance_sparse_num[i,:,0]<pos_t0),0])

        pmean_spr_numI[i] = np.mean(variance_sparse_num[i,np.where(variance_sparse_num[i,:,1]<pos_t0),1])
        pstd_spr_numI[i]  = np.std(variance_sparse_num[i,np.where(variance_sparse_num[i,:,1]<pos_t0),1])

    #### ----------------------------- sparse   -------------------------------------------
    ax2[0].plot(etaseries,pmean_spr_numE,color='tab:red',lw=1.0,linestyle='--')
    ax2[1].plot(etaseries,pmean_spr_numI,color='tab:blue',lw=1.0,linestyle='--')
    low_bound = np.zeros_like(pmean_spr_numE)
    for imax in range(len(pmean_spr_numE)):
        low_bound[imax]=max(0,pmean_spr_numE[imax]-pstd_spr_numE[imax])
    ax2[0].fill_between(etaseries, pmean_spr_numE+pstd_spr_numE,low_bound, alpha=0.3, facecolor='tab:red')
    low_bound = np.zeros_like(pmean_spr_numI)
    for imax in range(len(pmean_spr_numI)):
        low_bound[imax]=max(0,pmean_spr_numI[imax]-pstd_spr_numI[imax])
    ax2[1].fill_between(etaseries, pmean_spr_numI+pstd_spr_numI,low_bound, alpha=0.3, facecolor='tab:blue')
    ### error bar 
    ax2e[0].errorbar(etaseries, pmean_spr_numE,pstd_spr_numE/np.sqrt(ntrial),c='tab:red')
    ax2e[1].errorbar(etaseries, pmean_spr_numI,pstd_spr_numI/np.sqrt(ntrial), c='tab:blue')

    for i in range(2):     
            ax2[i].set_xlim(xlims)
            ax2[i].set_ylim(ylims)
            ax2[i].set_xticks(xticks)
            ax2[i].set_yticks(yticks)
    for i in range(2):     
            ax2e[i].set_xlim(xlims)
            ax2e[i].set_ylim(ylims)
            ax2e[i].set_xticks(xticks)
            ax2e[i].set_yticks(yticks)

    
    ### normalized by lambda (nvec)
    # xticks = np.linspace(-15,5,5)
    # xlims = [-20,6]
    xticks = [-50,0,20]
    xlims  = [-50,20]
    
    yticks = [0,2]#np.linspace(-5,5,3)
    ylims  = [-0.5,2]#[-5,5]
    
    
    iktrial=0
    ig=16#same as eta ==0.8
    alphaval=0.10
    edgv='black'
    cm='br'
    fig,ax=plt.subplots(figsize=(4,2))  
    
    '''loading vector changing'''
    meanmE,meanmI,meannE,meannI=np.zeros(nrank),np.zeros(nrank),np.zeros(nrank),np.zeros(nrank)
    
    mEvec,mIvec,nEvec,nIvec=np.squeeze(Rsprvecseries[ig,:,:NE,0]),np.squeeze(Rsprvecseries[ig,:,NE:,0]),np.squeeze(Lsprvecseries[ig,:,:NE,0]),np.squeeze(Lsprvecseries[ig,:,NE:,0])
    eigVforn = np.real(Bsprvseries_theo[ig,0])#1#
    # nEvec,nIvec = nEvec/eigVforn,nIvec/eigVforn
    mEvec,mIvec,nEvec,nIvec=mEvec.flatten(),mIvec.flatten(),nEvec.flatten(),nIvec.flatten()
    
    ### rescale n vector by 1/np.sqrt/(N)
    # nEvec,nIvec=nEvec/np.sqrt(N),nIvec/np.sqrt(N)#nEvec,nIvec#
    scale_std=3.0
    for irank in range(nrank):
        meanmEtotal,stdmEtotal = np.mean(mEvec),np.std(mEvec)
        varmE = mEvec - meanmEtotal
        idxwhere = np.where(np.abs(varmE)>scale_std*stdmEtotal)
        mEvec[idxwhere]=meanmEtotal
        meanmE[irank]=np.mean(mEvec)
    
        # pruning I
        meanmItotal,stdmItotal = np.mean(mIvec),np.std(mIvec)
        varmI = mIvec - meanmItotal
        idxwhere = np.where(np.abs(varmI)>scale_std*stdmItotal)
        mIvec[idxwhere]=meanmItotal
        meanmI[irank]=np.mean(mIvec)
        
        # n vector
        # pruning E
        meannEtotal,stdnEtotal = np.mean(nEvec),np.std(nEvec)
        varnE = nEvec - meannEtotal
        idxwhere = np.where(np.abs(varnE)>scale_std*stdnEtotal)
        nEvec[idxwhere]=meannEtotal
        meannE[irank]=np.mean(nEvec)
    
        # pruning I
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
    
    idrands=np.random.choice(np.arange(nrank,N),size=NI,replace=False) ### trials can not be too small, otherwise sample index would be out of bounds
    ax.scatter(nIvec[idrands],mIvec[idrands],s=1.0,c='b',alpha=0.25)#alphaval)#cm[1],alpha=alphaval)
    ax.scatter(nEvec[idrands],mEvec[idrands],s=1.0,c='r',alpha=0.25)#alphaval)#cm[0],alpha=alphaval)
    
    ax.plot([meannE[0],meannE[0]+1*dirvecE[0,0]],[meanmE[0],meanmE[0]+dirvecE[1,0]],color='gray',linestyle='--',linewidth=1.5)
    ax.plot([meannE[0],meannE[0]+1*dirvecE[0,1]],[meanmE[0],meanmE[0]+dirvecE[1,1]],color='gray',linestyle='--',linewidth=1.5)
    ax.plot([meannI[0],meannI[0]+1*dirvecI[0,0]],[meanmI[0],meanmI[0]+dirvecI[1,0]],color='gray',linestyle='--',linewidth=1.5)
    ax.plot([meannI[0],meannI[0]+1*dirvecI[0,1]],[meanmI[0],meanmI[0]+dirvecI[1,1]],color='gray',linestyle='--',linewidth=1.5)
    
    # ax.plot([meannE[0],meannE[0]+np.sqrt(N)*dirvecE[0,0]],[meanmE[0],meanmE[0]+np.sqrt(N)*dirvecE[1,0]],color='gray',linestyle='--',linewidth=1.5)
    # ax.plot([meannE[0],meannE[0]+np.sqrt(N)*dirvecE[0,1]],[meanmE[0],meanmE[0]+np.sqrt(N)*dirvecE[1,1]],color='gray',linestyle='--',linewidth=1.5)
    # ax.plot([meannI[0],meannI[0]+np.sqrt(N)*dirvecI[0,0]],[meanmI[0],meanmI[0]+np.sqrt(N)*dirvecI[1,0]],color=edgv,linestyle='--',linewidth=1.5)
    # ax.plot([meannI[0],meannI[0]+np.sqrt(N)*dirvecI[0,1]],[meanmI[0],meanmI[0]+np.sqrt(N)*dirvecI[1,1]],color=edgv,linestyle='--',linewidth=1.5)
    confidence_ellipse(np.real(nEvec),np.real(mEvec),ax,edgecolor=edgv)
    confidence_ellipse(np.real(nIvec),np.real(mIvec),ax,edgecolor=edgv)
    
    
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # ax.set_aspect('equal')
    
    # ax.set_xlabel(r'$n_i/\sqrt{N}$')
    # ax.set_ylabel(r'$m_i$')

    ### C. Gaussian mixture model
    from scipy import stats
    np.random.seed(41)
    xticks =  [-50,0,20]#np.linspace(-15,5,5)
    xlims = [-50,20]#[-20,6]
    
    yticks = [0,2]#np.linspace(-5,5,3)
    ylims = [-0.5,2]#[-5,5]
    # nvbins = np.linspace(-25,10,30)
    nvbins = np.linspace(-50,20,30)
    mvbins = np.linspace(-0.5,2,10)

    nEkde = stats.gaussian_kde(np.real(nEvec))
    nIkde = stats.gaussian_kde(np.real(nIvec))
    mEkde = stats.gaussian_kde(np.real(mEvec))
    mIkde = stats.gaussian_kde(np.real(mIvec))
    # xx = np.linspace(-25, 10, 100)
    xx = np.linspace(-50, 20, 100)
    fign, axn = plt.subplots(figsize=(5,1))
    axn.hist(np.real(nEvec), density=True, bins=nvbins, facecolor='tab:red', alpha=0.3)
    axn.plot(xx, nEkde(xx),c='tab:red')
    axn.hist(np.real(nIvec), density=True, bins=nvbins, facecolor='tab:blue',alpha=0.3)
    axn.plot(xx, nIkde(xx),c='tab:blue')
    axn.set_xlim(xlims)

    yy  = np.linspace(-0.5,2,50)
    figm, axm = plt.subplots(figsize=(3,1))
    axm.hist(np.real(mEvec), density=True, bins=mvbins,facecolor='tab:red', alpha=0.3)
    axm.plot(yy, mEkde(yy),c='tab:red')

    axm.hist(np.real(mIvec), density=True, bins=mvbins,facecolor='tab:blue' ,alpha=0.3)
    axm.plot(yy, mIkde(yy),c='tab:blue')
    axm.set_xlim(ylims)

    ### adding comparison between --> fluctuations in connectivity lead to fluctuations in dynamic
    igEIsample=[15,20]
    yticks = np.linspace(-3.0, 3.0, 3)
    ylims = [-3.0, 3.0]#[1.1, 2.2]
    xticks = np.linspace(-1.5, 1.5, 3)
    xlims = [-1.5, 1.5]
    import matplotlib.cm as cm

    figE,axE=plt.subplots(figsize=(2,4))#,sharex=True,sharey=True,tight_layout=True)
    figI,axI=plt.subplots(figsize=(2,4))#,sharex=True,sharey=True,tight_layout=True)
    iktrial =2
    for iii, idxgEI in enumerate(igEIsample):
        iktrial=3
        idrandsE     = np.random.choice(np.arange(0,NE),size=100,replace=False)
        idrandsI     = np.random.choice(np.arange(NE,N),size=100,replace=False)
        xactfull_E   = np.squeeze(xfpseries_Sparse[idxgEI, iktrial, idrandsE, -1])
        xactfull_I   = np.squeeze(xfpseries_Sparse[idxgEI, iktrial, idrandsI, -1])
        deltaxfull_E = xactfull_E - np.mean(xactfull_E)
        deltaxfull_I = xactfull_I - np.mean(xactfull_I)

        if (np.mean(xactfull_E)<0):
            deltaxfull_E=-deltaxfull_E
        if(np.mean(xactfull_I)<0):
            deltaxfull_I=-deltaxfull_I

        deltam_E = Restvecseries[idxgEI,iktrial,idrandsE,0]-np.mean(Restvecseries[idxgEI,iktrial,idrandsE,0])
        deltam_I = Restvecseries[idxgEI,iktrial,idrandsI,0]-np.mean(Restvecseries[idxgEI,iktrial,idrandsI,0])#1.0#xAm
        axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c=cm.Reds((etaseries[idxgEI]-etaseries[0])/(etaseries[-1]-etaseries[0])),alpha=0.75)
        axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c=cm.Blues((etaseries[idxgEI]-etaseries[0])/(etaseries[-1]-etaseries[0])),alpha=0.75)
        # axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c='tab:red',alpha=0.75)
        # axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c='tab:blue',alpha=0.75)
        ### predicted
        deltaxpred_E = kappa_theo_ss[idxgEI,0]*np.array(xlims)
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


    for iktrial in range(26,29):#    iktrial =-1
        figE,axE=plt.subplots(figsize=(2,2))#,sharex=True,sharey=True,tight_layout=True)
        figI,axI=plt.subplots(figsize=(2,2))#,sharex=True,sharey=True,tight_layout=True)
        for iii, idxgEI in enumerate(igEIsample):
            idrandsE     = np.random.choice(np.arange(0,NE),size=100,replace=False)
            idrandsI     = np.random.choice(np.arange(NE,N),size=100,replace=False)
            xactfull_E   = np.squeeze(xfpseries_Sparse[idxgEI, iktrial, idrandsE, -1])
            xactfull_I   = np.squeeze(xfpseries_Sparse[idxgEI, iktrial, idrandsI, -1])
            deltaxfull_E = xactfull_E - np.mean(xactfull_E)
            deltaxfull_I = xactfull_I - np.mean(xactfull_I)

            if (np.mean(xactfull_E)<0):
                deltaxfull_E=-deltaxfull_E
            if(np.mean(xactfull_I)<0):
                deltaxfull_I=-deltaxfull_I

            deltam_E = Restvecseries[idxgEI,iktrial,idrandsE,0]-np.mean(Restvecseries[idxgEI,iktrial,idrandsE,0])
            deltam_I = Restvecseries[idxgEI,iktrial,idrandsI,0]-np.mean(Restvecseries[idxgEI,iktrial,idrandsI,0])#1.0#xAm
            axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c=cm.Reds((etaseries[idxgEI]-etaseries[0])/(etaseries[-1]-etaseries[0])),alpha=0.25)
            axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c=cm.Blues((etaseries[idxgEI]-etaseries[0])/(etaseries[-1]-etaseries[0])),alpha=0.25)
            # axE.scatter(deltam_E,deltaxfull_E,s=10,marker='o',c='tab:red',alpha=0.75)
            # axI.scatter(deltam_I,deltaxfull_I,s=10,marker='o',c='tab:blue',alpha=0.75)
            ### predicted
            deltaxpred_E = kappa_theo_ss[idxgEI,2]*np.array(xlims)
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


# if(RERUN==1):
#     lst = [kappaintersect_Sparse, kappaintersect_Full, kappaintersect_R1,
#            xfpseries_Full[:,:,:,-1], xfpseries_R1[:,:,:,-1], xfpseries_Sparse[:,:,:,-1],
#            Restvecseries[:,:,:,0], Rsprvecseries[:,:,:,0],
#            Lestvecseries[:,:,:,0], Lsprvecseries[:,:,:,0],
#            Bsprvseries[:,:,0],
#            sigrcov, siglcov,
#            Sigmn_num, Sigmniid_num,
#            etassrecord, etauserecord]
#     stg = ["kappaintersect_Sparse, kappaintersect_Full, kappaintersect_R1,"
#            "xfpseries_Full, xfpseries_R1, xfpseries_Sparse,"
#            "Restvecseries, Rsprvecseries,"
#            "Lestvecseries, Lsprvecseries,"
#            "Bsprvseries,"
#            "sigrcov, siglcov,"
#            "Sigmn_num, Sigmniid_num,"
#            "etassrecord, etauserecord"]
#     data = list_to_dict(lst=lst, string=stg)
#     data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/dyns_spr_PositiveTF_reciprocal_kappa_1.npz"
#     np.savez(data_name, **data)

