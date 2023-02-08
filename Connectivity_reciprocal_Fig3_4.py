# -*- coding: utf-8 -*-
"""
@author: Yuxiu Shao
Statistics of the network connectivity;
Full-rank Gaussian Network and Rank-one Mixture of Gaussian Approximation;
Random component has reciprocal motifs
"""


''' HELP FUNCTIONS '''
import numpy as np
import matplotlib.pylab as plt

from numpy import linalg as la

# from sympy import *
import scipy.stats as stats

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
# fig.savefig('myimage.svg', format='svg', dpi=1200)


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

#### network params
Nt      = np.array([1200,300])# 4 vs 1# ([750,750]) # 750
NE,NI   = Nt[0],Nt[1]
N       = NE+NI
Nparams = np.array([NE,NI])
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#### @YX modify 2508 ---- eta with 1 decimal
nrank,ntrial,neta,ngavg=1,30,11,11#10,10

''' ## Three \bar{J} cases '''
### connectivity setting -- original
JI,JE,a,b      = 1.2,2.0,0.0,0.0#1.4,2.0,0.0,0.0 # HETE 2 JE 1.4 TWO OUTLIER#1.3,2.0,0.0,0.0# HETE 1, HETE 2 JE 1.3 ONE OUTLIER#1.2,2.0,0.0,0.0#
# ### negative lambda0
# JI,JE,a,b    = 2.0,1.2,0.0,0.0 #
# ### Dynamics setting -- new check (for reciprocal )
# JI,JE,a,b    =  0.6,1.5,0.0,0.0 
# ### suppl. conn. dyn. homogeneous eta homogeneou g
# JI,JE,a,b    = 0.60,2.0,0.0,0.0

RERUN = 1 ### 0 rerun, 1 reloading data saved
if RERUN ==0 :
    # ### RERUN = 0, reloading data saved
    # data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/conns_reciprocal_4VS1.npz"
    # data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/conns_reciprocal_etaII_4VS1.npz"
    data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/conns_reciprocal_etaEIII_JI14JE20_4VS1.npz"
    # data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/conns_reciprocal_JIdominated_4VS1.npz"
    data = np.load(data_name)


JEE,JIE,JEI,JII=JE+a,JE-a,JI-b,JI+b

### mean connectivity matrix
Am,Jsv=generate_meanmat_eig(Nparams,JEE,JIE,JEI,JII)
xAm,yAm=np.ones((N,1)),np.ones((N,1))
yAm[:NE,0],yAm[NE:,0] = yAm[:NE,0]*N/NE*JE,-yAm[NE:,0]*N/NI*JI
eigvAm = np.zeros(N)
eigvAm[0]=JE-JI

'''Network Setting for Iterating Following'''
### reciprocal correlations being positive/negative
signeta=np.ones(3)
# signeta[2]*=(-1)
# signeta[1]*=(-1)

# ### homogeneou eta homogeneou g suppl. conn. dyn 
# xee,xei,xie,xii=1.0,1.0,1.0,1.0 #
## heterogeneous g
xee,xei,xie,xii=1.0,0.5,0.2,0.8
coeffeta=np.array([1.0,1.0,1.0])#0.1])

### scaled standard deviation
gaverage=0.3
#### YX 02DEC -- THREE OUTLIERS
# gaverage=0.3#1.0#

### recorded variables 
Reigvecseries,Leigvecseries = np.zeros((ngavg,ntrial,N)),np.zeros((ngavg,ntrial,N))
Beigvseries = np.zeros((ngavg,ntrial,N),dtype=complex)
#### reconstruction based on PTT
ReigvecTseries,LeigvecTseries = np.zeros((ngavg,ntrial,N)),np.zeros((ngavg,ntrial,N))
Zrandommat   = np.zeros((neta,ntrial,N,N)) # random matrix of individual trials
#### statistic properties
armu,sigrcov = np.zeros((ngavg,ntrial,2)),np.zeros((ngavg,ntrial,2)) # 2 for E and I
almu,siglcov = np.zeros((ngavg,ntrial,2)),np.zeros((ngavg,ntrial,2))
siglr = np.zeros((ngavg,ntrial,2))

''' Iterative Processing '''
etaseries = np.linspace(0.0,1.0,neta) # 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
if (RERUN==0):
    xAm,yAm = data['xAm'],data['yAm']
    ReigvecTseries,LeigvecTseries = data['ReigvecTseries'],data['LeigvecTseries']
    Reigvecseries,Leigvecseries   = data['Reigvecseries'],data['Leigvecseries']
    Beigvseries = data['Beigvseries']
if (RERUN==1):
    for iktrial in range(ntrial):
        #### random components (individual trials)
        Xsym  = iidGaussian([0,gaverage/np.sqrt(N)],[N,N])
        XsymT = Xsym.copy().T
        X0    = Xsym.copy()  # always make mistake here
        J0    = Am.copy()+X0.copy()
        #### eigenvalues and eigenvectors of the connectivity with iid random connections
        eigvJ0,leig0,reig0,x0,y0 = decompNormalization(J0,xAm,yAm,xAm,yAm,nparams=Nparams,sort=0,nrank=1)
        for idxeta,eta in enumerate(etaseries):
            etaset=eta*np.ones(3)
            for icoeff in range(3):
              etaset[icoeff]*=coeffeta[icoeff]
            Xinit = Xsym.copy()
            ''' heterogeneous ETA method 1 '''
            ## EE ##
            asqr=(1-np.sqrt(1-etaset[0]**2))/2.0 ## when eta = 0, asqr = 0, aamp = 0, XT-0, X-1
            aamp=np.sqrt(asqr)
            Xinit[:NE,:NE]=signeta[0]*aamp*XsymT[:NE,:NE].copy()+np.sqrt(1-aamp**2)*Xsym[:NE,:NE].copy()
            ## EI IE##
            asqr=(1-np.sqrt(1-etaset[1]**2))/2.0
            aamp=np.sqrt(asqr)
            Xinit[NE:,:NE]=signeta[1]*aamp*XsymT[NE:,:NE].copy()+np.sqrt(1-aamp**2)*Xsym[NE:,:NE].copy()
            Xinit[:NE,NE:]=signeta[1]*aamp*XsymT[:NE,NE:].copy()+np.sqrt(1-aamp**2)*Xsym[:NE,NE:].copy()
            ## II ##
            asqr=(1-np.sqrt(1-etaset[2]**2))/2.0
            aamp=np.sqrt(asqr)
            Xinit[NE:,NE:]=signeta[2]*aamp*XsymT[NE:,NE:].copy()+np.sqrt(1-aamp**2)*Xsym[NE:,NE:].copy()
    
            X=Xinit.copy()
            # incase heterogeneous
            X[:NE,:NE]*=(xee)
            X[:NE,NE:]*=(xei)
            X[NE:,:NE]*=(xie)
            X[NE:,NE:]*=(xii)
    
            # overall
            J  = X.copy()+Am.copy()
            JT = J.copy().T
            ### properties
            #### eigenvalues and eigenvectors of the connectivity with reciprocal motifs/obtained using eigendecomposition
            eigvJ,leigvec,reigvec,xnorm0,ynorm0=decompNormalization(J,x0,y0,xAm,yAm,nparams=Nparams,sort=1,nrank=1)
            # #### (perturbed from the mean connectivity)
            # eigvJ,leigvec,reigvec,xnorm0,ynorm0=decompNormalization(J,xAm,yAm,xAm,yAm,nparams=Nparams,sort=1,nrank=1)
    
            #### recording the results obtained using eigendecomposition
            Beigvseries[idxeta,iktrial,:]=eigvJ.copy()
            Reigvecseries[idxeta,iktrial,:],Leigvecseries[idxeta,iktrial,:]=xnorm0[:,0].copy(),ynorm0[:,0].copy()
            
            #### properties of the elements on the connectivity eigenvectors
            axrmu,aylmu,sigxr,sigyl,sigcov = numerical_stats(xnorm0,ynorm0,xAm,yAm,eigvJ,nrank,2,ppercent=np.array([NE/N,NI/N]))
            armu[idxeta,iktrial,:],almu[idxeta,iktrial,:]       = axrmu[:,0],aylmu[:,0]
            sigrcov[idxeta,iktrial,:],siglcov[idxeta,iktrial,:] = sigxr[:,0],sigyl[:,0]
            siglr[idxeta,iktrial,:] = sigcov[:,0,0]
    
            #### reconstruction based on perturbation theory
            xnorm0,ynorm0=np.reshape(xnorm0,(N,1)),np.reshape(ynorm0,(N,1))
            xnormt,ynormt = np.zeros_like(xnorm0),np.zeros_like(ynorm0)

            eigvnorm = np.real(Beigvseries[idxeta,iktrial,0])#eigvAm[0]#
            xnormt,ynormt = X@xAm.copy()/eigvnorm,yAm.copy().T@X/eigvnorm
            
            ynormt = ynormt.T
            xnormt,ynormt = xnormt+xAm,ynormt+yAm 
            # #### ---- Normalized to be N*eigv ------Not necessary -----
            # projection = np.reshape(xnormt,(1,N))@np.reshape(ynormt,(N,1))
            # ynormt     = ynormt/projection*Beigvseries[idxeta,iktrial,0]*N
            # #### ---------------------------------------------------------------
            # _,_,xnormt,ynormt=Normalization(xnormt,ynormt,xAm,yAm,Nparams,sort=0,nrank=1)
            ReigvecTseries[idxeta,iktrial,:],LeigvecTseries[idxeta,iktrial,:] = xnormt[:,0],ynormt[:,0]

            print('recording...')
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
    #### A. eigenvalue spectrum
    yticks = np.linspace(-2.50,2.5,2)
    ylims  = [-2.5,2.5]
    xticks = np.linspace(-0.,1.,2)
    xlims  = [-0.0,1.0]
    cm='bgc'
    envelopeRe = np.zeros(neta)
    envelopeIm = np.zeros(neta)
    coeff      = coeffeta.copy()
    nAm=np.reshape(yAm.copy(),(N,1))
    mAm=np.reshape(xAm.copy(),(N,1))
    nAm,mAm=np.real(nAm),np.real(mAm)
    lambda_theo2       = np.zeros((neta,3),dtype=complex) # theoretical computed eigenvalues
    lambda_theo2_delta = np.zeros(neta)
    lambda_theo4       = np.zeros((neta,3),dtype=complex) # cut off at order 4
    lambda_general     = np.zeros((neta,4),dtype=complex) # First round revision -- S6 Appendix
    lambda_num         = np.transpose(np.squeeze(Beigvseries[:,:,:2]),(2,1,0))
    cutoff = 2

    for idxeta,eta in enumerate(etaseries):
        #### @YX 0409 notice -- might be cell-type specific  reciprocal conn.
        etaE,etaB,etaI = eta*coeff[0],eta*coeff[1],eta*coeff[2]
        etaE,etaB,etaI = etaE*signeta[0],etaB*signeta[1],etaI*signeta[2]
        etaset  = np.array([etaE,etaB,etaI])
        # gee,gei,gie,gii 
        gmat    = np.array([xee,xei,xie,xii])*gaverage
        
        ### for S6 Appendix lambda_general (without truncation)
        tauset_series = {}
        tau_set = np.zeros(14)
        # # ~~~~~ reciprocal motisf ~~~~~~~~~~
        tau_set[10:]  = etaset[0], etaset[1], etaset[1], etaset[2]
        
        tdiv, tcon, tchn, trec = tau_set[:2],np.reshape(tau_set[2:6],(2,2)), np.reshape(tau_set[6:10],(2,2)),np.reshape(tau_set[10:],(2,2))
        
        tauset_series = {'tdiv':tdiv,
                        'tcon':tcon,
                        'tchn':tchn,
                        'trec':trec
                        }
        
        Jparams = np.array([JE,JI])
        lambda_theo2[idxeta,:],lambda_theo2_delta[idxeta]=CubicLambda(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams,)
        #### SORT THE ROOTS @YX 07DEC
        realroots = np.real(np.squeeze(lambda_theo2[idxeta,:]))
        sidx      = np.argsort(realroots)
        roots_uno = lambda_theo2[idxeta,:].copy()
        for i in range(len(sidx)):
            lambda_theo2[idxeta,i] = roots_uno[sidx[2-i]]#EXC-DOMINATE
            # lambda_theo2[idxeta,i] = roots_uno[sidx[i]]#INHIBIT-DOMINATE#

        ##### @YX ---- SORT AND THEN FIND ENVELOPEre and imag
        realsort = np.sort(np.real(Beigvseries[idxeta,:,:]),axis=1)
        envelopeRe[idxeta]=np.mean(realsort[:,-2])#[:,-3])
        imagsort = np.sort(np.imag(Beigvseries[idxeta,:,:]),axis=1)
        envelopeIm[idxeta]=np.mean(imagsort[:,-3])#[:,-4])#~~~~~~ perhaps .... small -- large
        # # ###~~~~~~~~~ check extra ~~~~~~~~~~~
        # iddtest = 1
        # lambdatest = lambda_theo2[idxeta,iddtest]
        # rhslambda=nAm.T@mAm
        # rhslambda=rhslambda+nAm.T@Zrandommat[idxeta,iktrial,:,:]@Zrandommat[idxeta,iktrial,:,:]@mAm/(lambdatest**2)
        # print('second order: ',rhslambda/N)
        # rhslambda=rhslambda+nAm.T@Zrandommat[idxeta,iktrial,:,:]@Zrandommat[idxeta,iktrial,:,:]@Zrandommat[idxeta,iktrial,:,:]@Zrandommat[idxeta,iktrial,:,:]@mAm/lambdatest**4
        # rhslambda=rhslambda/N 
        # print('left ',lambdatest,'; 4-th order right ',rhslambda)

        #### Note: please carefully set the initial values, otherwise 'fsolve' doesn't work well!
        # INIT = [-0.05,0.05]
        # INIT = [-gaverage*np.sqrt(etaE),0.] ### Fig3 1
        INIT = [-0.05,0.3] ### Fig3 J (the last net)
        realp,imagp = fsolve(CubicLambda4, INIT,
                        args=(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams))
        lambda_theo4[idxeta,1]=complex(realp,imagp)
        realg,imagg = fsolve(cal_outliers_general_complex,INIT,args=(tauset_series, [], Jparams, Nparams, gmat, eigvAm),xtol=10e-9,maxfev=1000)   
        lambda_general[idxeta,1] = complex(realg, imagg)
        
        # INIT = [-0.05,-0.05]
        # INIT = [gaverage*np.sqrt(etaE),-0.] ### Fig3.1
        INIT = [-0.05,-0.3] ### Fig3.J
        realp,imagp = fsolve(CubicLambda4, INIT,
                        args=(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams))
        lambda_theo4[idxeta,2]=complex(realp,imagp)
        realg,imagg = fsolve(cal_outliers_general_complex,INIT,args=(tauset_series, [], Jparams, Nparams, gmat, eigvAm),xtol=10e-9,maxfev=1000)   
        lambda_general[idxeta,2] = complex(realg, imagg)

        INIT = [eigvAm[0],0]
        realp,imagp = fsolve(CubicLambda4, INIT,
                        args=(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams))
        lambda_theo4[idxeta,0]=complex(realp,imagp)
        realg,imagg = fsolve(cal_outliers_general_complex,INIT,args=(tauset_series, [], Jparams, Nparams, gmat, eigvAm),xtol=10e-9,maxfev=1000)   
        lambda_general[idxeta,0] = complex(realg, imagg)
        
        # INIT = [-0.3,0]#
        # INIT = [eigvAm[0]+np.sqrt(eigvAm[0]**2+4*gaverage**2*etaE),0] ### HOMOGENEOUS
        # realg,imagg = fsolve(cal_outliers_general_complex,INIT,args=(tauset_series, [], Jparams, Nparams, gmat, eigvAm))#,xtol=10e-9,maxfev=1000)   
        # lambda_general[idxeta,3] = complex(realg, imagg)

        ### ~~~~~~~~~ 4th-order ~~~~~~~~~~~~~~~
        realroots=np.real(np.squeeze(lambda_theo4[idxeta,:]))
        sidx = np.argsort(realroots)
        roots_uno = lambda_theo4[idxeta,:].copy()
        for i in range(len(sidx)):
            lambda_theo4[idxeta,i]=roots_uno[sidx[2-i]]

    #### figure Eigenvalue spectrums
    idxgavg,idxtrial=0,2#9,3 # 6,3,0
    idxgavgsample=np.array([0,4,8])#homo and 1hete and 2hete JI 1.3#np.array([0,5,9])#2hete JI 1.4 2conjugate#
    idxgavgsample=np.array([0,4,8])#4VS1
    figtspt,axtspt=plt.subplots(len(idxgavgsample),1,figsize=(3,8),sharex=True,sharey=True,tight_layout=True)
    cm='bgc'
    shiftlen=0.0#0.2#
    theta = np.linspace(0, 2 * np.pi, 200)
    for idxc,idxgavg in enumerate(idxgavgsample):
        # figtspt,axtspt=plt.subplots(figsize=(5,3))
        print('index:',idxc,idxgavg)
        idrands=np.random.choice(np.arange(nrank,N),size=N-nrank,replace=False)
        axtspt[idxc].scatter(np.real(Beigvseries[idxgavg,idxtrial,idrands]),np.imag(Beigvseries[idxgavg,idxtrial,idrands]),s=5,c=cm[idxc],alpha=0.25) # >>>>>>>>>>>>>>
        axtspt[idxc].scatter(np.real(Beigvseries[idxgavg,:,0]),np.imag(Beigvseries[idxgavg,:,0])-shiftlen*idxc,s=20,c=cm[idxc],alpha=0.25)#marker='^',alpha=0.5) # >>>>>>>>>>>>
        axtspt[idxc].set_aspect('equal')
        axtspt[idxc].scatter(np.real(lambda_theo2[idxgavg,0]),-shiftlen*idxc,s=80,c='',marker='o',edgecolor='purple')#'black')#cm[idxc]) # 

        axtspt[idxc].scatter(np.real(lambda_theo2[idxgavg,1]),np.imag(lambda_theo2[idxgavg,1]),s=80,c='',marker='o',edgecolor='red')#'=cm[idxc]) # 
        axtspt[idxc].scatter(np.real(lambda_theo2[idxgavg,2]),np.imag(lambda_theo2[idxgavg,2]),s=80,c='',marker='o',edgecolor='orange')#'=cm[idxc]) #

        # ### ~~~~~~~~~~~~ lambda_theo4 ~~~~~~~~~~~
        # axtspt[idxc].scatter(np.real(lambda_theo4[idxgavg,0]),np.imag(lambda_theo4[idxgavg,0]),s=80,c='',marker='^',edgecolor='brown')#'black')#cm[idxc]) # 
        # axtspt[idxc].scatter(np.real(lambda_theo4[idxgavg,1]),np.imag(lambda_theo4[idxgavg,1]),s=80,c='',marker='^',edgecolor='red')#'=cm[idxc]) # 
        # axtspt[idxc].scatter(np.real(lambda_theo4[idxgavg,2]),np.imag(lambda_theo4[idxgavg,2]),s=80,c='',marker='^',edgecolor='orange')#'=cm[idxc]) #
        
        ### ~~~~~~~~~~~~ lambda_theo_general ~~~~~~~~~~~
        axtspt[idxc].scatter(np.real(lambda_general[idxgavg,0]),np.imag(lambda_general[idxgavg,0]),s=80,c='',marker='^',edgecolor='brown')
        axtspt[idxc].scatter(np.real(lambda_general[idxgavg,1]),np.imag(lambda_general[idxgavg,1]),s=80,c='',marker='^',edgecolor='red')
        axtspt[idxc].scatter(np.real(lambda_general[idxgavg,2]),np.imag(lambda_general[idxgavg,2]),s=80,c='',marker='^',edgecolor='orange')
      
        axtspt[idxc].scatter(np.real(lambda_general[idxgavg,3]),np.imag(lambda_general[idxgavg,3]),s=80,c='',marker='^',edgecolor='pink')

        ####  conjugated eigenvalue outliers
        # if(idxc>0):
        #     for idxtrial in range(ntrial):
        #         imagsort = np.argsort(np.imag(Beigvseries[idxgavg,idxtrial,:]))
        #         mostIm   = Beigvseries[idxgavg,idxtrial,imagsort[0]]#(imagsort[:3])
        #         axtspt[idxc].scatter(np.real(mostIm),np.imag(mostIm),s=5,c=cm[idxc],alpha=0.25)
        #         mostIm   = Beigvseries[idxgavg,idxtrial,imagsort[-1]]#(imagsort[:3])
        #         axtspt[idxc].scatter(np.real(mostIm),np.imag(mostIm),s=5,c=cm[idxc],alpha=0.25)
        
        # if(idxc>0):
        #     for idxtrial in range(ntrial):
        #         diffimg  = np.abs(np.imag(Beigvseries[idxgavg,idxtrial,2:]))-np.abs(np.imag(Beigvseries[idxgavg,idxtrial,1:-1]))
        #         idximp   = np.where(np.abs(diffimg)>0.4)[0]
        #         print('>>>>> len:',len(idximp))
        #         if(len(idximp)>0):
        #             conjugates = Beigvseries[idxgavg,idxtrial,1+idximp[0]:5+idximp[0]]
        #             axtspt[idxc].scatter(np.real(conjugates),np.imag(conjugates),s=5,c=cm[idxc],alpha=0.25)


        axtspt[idxc].spines['right'].set_color('none')
        axtspt[idxc].spines['top'].set_color('none')
        # X axis location
        axtspt[idxc].xaxis.set_ticks_position('bottom')
        axtspt[idxc].spines['bottom'].set_position(('data', 0))

        #### effective radius 
        aee,aei,aie,aii=xee,xei,xie,xii
        ahomo=gaverage
        xee_,xei_,xie_,xii_=ahomo*aee/np.sqrt(N),ahomo*aei/np.sqrt(N),ahomo*aie/np.sqrt(N),ahomo*aii/np.sqrt(N)
        gmat = np.array([[NE*xee_**2,NI*xei_**2],[NE*xie_**2,NI*xii_**2]])
        gaverage_=0
        for i in range(2):
            for j in range(2):
                gaverage_+=gmat[i,j]/2 # ntype=2
        gaverage_=np.sqrt(gaverage_)
        eigvgm,eigvecgm = la.eig(gmat) 
        r_g2 = np.max(eigvgm)
        r_g  = np.sqrt(r_g2)
        # r_g = (1+etaseries[idxgavg])*r_g

        # ### ~~~~~~~~~~~~~~~~~~ added for heterogeneous ~~~~~~~~~~~~~
        # envelopeT=max(envelopeRe[idxgavg],envelopeIm[idxgavg])
        # r_g = envelopeT
        xr = r_g*(1+etaseries[idxgavg])*np.cos(theta)
        yr = r_g*(1-etaseries[idxgavg])*np.sin(theta)
        axtspt[idxc].plot(xr, yr, color="black", linewidth=1.0) # >>>>>


        #### some figure setting ....
        ### homo ~~~~~
        xticks = np.linspace(-0.5,0.5,3)
        xlims = [-0.6,1.2]

        yticks = np.linspace(-0.3,0.3,2)
        ylims = [-0.5,0.5]

        # ### heta eta 2 JE 1.4 2conjugate ~~~~~
        # xticks = np.linspace(-0.5,0.5,3)
        # xlims = [-0.5,1.0]
        
        # yticks = np.linspace(-0.5,0.5,2)
        # ylims = [-0.6,0.6]

        # ### homo eta hete g ~~~~~
        # xticks = np.linspace(-0.5,0.5,3)
        # xlims = [-0.5,1.0]
        
        # yticks = np.linspace(-0.3,0.3,2)
        # ylims = [-0.3,0.3]

        # ### HOMOETA-INHIBIT ~~~~~
        # xticks = np.linspace(-0.5,0.5,3)
        # xlims = [-1.0,0.6]
        
        # yticks = np.linspace(-0.3,0.3,2)
        # ylims = [-0.4,0.4]

        axtspt[idxc].set_xlim(xlims)
        axtspt[idxc].set_ylim(ylims)
        axtspt[idxc].set_xticks(xticks)
        axtspt[idxc].set_yticks(yticks)
        axtspt[idxc].set_aspect('equal')

    ############################################################################
    cm = ['purple','red','orange']
    ## ------- ORIGINAL ONE ---------
    fig,ax=plt.subplots(2,1,figsize=(4,4),sharex=True, tight_layout=True)

    ### figure settings
    # yticks = np.linspace(-0.0,1.0,2)
    # ylims  = [-0.2,1.0]
    # ~~~~~~ homo 
    yticks = np.linspace(-0.0,1.0,2)
    ylims  = [-0.2,1.2]
    # ## ~~~~~~ HOMO-INHIBIT
    # yticks = np.linspace(-1,0.0,2)
    # ylims  = [-1.0,0.2]
    # ## ~~~~~~ HOMO-INHIBIT (4VS1)
    # yticks = np.linspace(-1,0.0,2)
    # ylims  = [-1.2,0.2]

    xticks = np.linspace(0.,1.0,2)
    xlims  = [0.0,1.0]

    for i in range(3):
        ax[0].plot(etaseries,lambda_theo2[:,i].real,color=cm[i],linewidth=1.0,label='theo')
    mean_lambda_num = np.mean(lambda_num[:2,:,:].real,axis=1)
    std_lambda_num  = np.std(lambda_num[:2,:,:].real,axis=1)
    ax[0].fill_between(etaseries,mean_lambda_num[0,:]+std_lambda_num[0,:],mean_lambda_num[0,:]-std_lambda_num[0,:],alpha=0.3,facecolor=cm[0])


    #### envelope
    envelopeTotal=np.zeros_like(envelopeRe)
    for ienv in range(len(envelopeRe)):
        envelopeTotal[ienv]=max(envelopeRe[ienv],envelopeIm[ienv])

    ax[0].fill_between(etaseries,-(envelopeRe),envelopeRe, color='gray',alpha=0.2)
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].set_xticks(xticks)
    ax[0].set_yticks(yticks)
    
    ## ~~~ homo ~~~~~~~~~~
    yticks = np.linspace(-0.5,0.5,2)
    ylims  = [-0.5,0.5]

    for i in range(3):
        ax[1].plot(etaseries,np.imag(lambda_theo2[:,i]),color=cm[i],linewidth=1.,label='theo')
    ax[1].fill_between(etaseries,-(envelopeIm),envelopeIm, color='gray',alpha=0.2)
    mean_lambda_num = np.mean(lambda_num[:2,:,:].imag,axis=1)
    std_lambda_num  = np.std(lambda_num[:2,:,:].imag,axis=1)
    ax[1].fill_between(etaseries,mean_lambda_num[0,:]+std_lambda_num[0,:],mean_lambda_num[0,:]-std_lambda_num[0,:],alpha=0.3,facecolor=cm[0])

    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)
    ax[1].set_xticks(xticks)
    ax[1].set_yticks(yticks)

    #### B. elements on the rank-one connectivity eigenvectors
    idxtrial = 9#16#
    idxeta   = 9# 
    alphaval = 0.25
    edgv ='black'
    cm   ='br'
    '''loading vector changing'''
    meanmE,meanmI,meannE,meannI=np.zeros(nrank),np.zeros(nrank),np.zeros(nrank),np.zeros(nrank)
    ### --------------------------------------------------------
    mEvec,mIvec,nEvec,nIvec=np.squeeze(Reigvecseries[idxeta,:,:NE]),np.squeeze(Reigvecseries[idxeta,:,NE:]),np.squeeze(Leigvecseries[idxeta,:,:NE]),np.squeeze(Leigvecseries[idxeta,:,NE:])
    #### reconstructed eigenvectors 
    # mEvec,mIvec,nEvec,nIvec=np.squeeze(ReigvecTseries[idxeta,:,:NE]),np.squeeze(ReigvecTseries[idxeta,:,NE:]),np.squeeze(LeigvecTseries[idxeta,:,:NE]),np.squeeze(LeigvecTseries[idxeta,:,NE:])
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
    for i in range(2):
        for j in range(2):
            dirvecE[i,j]*=(1*np.sqrt(N))
            dirvecI[i,j]*=(1*np.sqrt(N))

    ### @YX 2709 ORIGINAL
    fig,ax=plt.subplots(figsize=(5,3))

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

    ax.set_aspect('equal')
    # ### ~~~~~~~ hete ~~~~~~~~
    # xticks = [-8,0,10]
    # xlims  = [-8,10]
    # yticks = [0,3]
    # ylims  = [-1,3]

    ### ~~~~~~~ hete2 (4VS1) ~~~~~~~~
    xticks = [-8,0,6]
    xlims  = [-8,6]
    yticks = [0,3]
    ylims  = [-1,3]
    ### ~~~~~~ homo ~~~~~
    # ### --- org -----
    # xticks = [-6,0,8]
    # xlims  = [-6,8]
    # yticks = [0,2]
    # ylims  = [-1,2]

    # ### --- 4VS1 ---
    # xticks = [-10,0,8]
    # xlims  = [-10,8]
    # yticks = [0,3]
    # ylims  = [-1,3]

    # ### ~~~~~~ INHIBIT DOMINATE ~~~~~
    # xticks = [-8,0,8]
    # xlims  = [-8,8]
    # yticks = [0,3]
    # ylims  = [-1,3]

    # ### ---INH 4VS1 ---
    # xticks = [-15,0,8]
    # xlims  = [-15,8]
    # yticks = [0,3]
    # ylims  = [-1,3]


    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    ax.set_title(r'$\eta=$'+str(etaseries[idxeta]))


    ax.scatter(nIvec[:NI],mIvec[:NI],s=5.0,c='tab:blue',alpha=alphaval)
    ax.scatter(nEvec[:NE],mEvec[:NE],s=5.0,c='tab:red',alpha=alphaval)


    ### C. Gaussian mixture model
    from scipy import stats
    np.random.seed(41)
    # ### ORIGINAL
    # nvbins = np.linspace(-10,13,30)
    ### ORIGINAL hete2(4VS1)
    nvbins = np.linspace(-10,10,30)
    # ### 4VS1 INH
    # nvbins = np.linspace(-15,13,30)
    mvbins = np.linspace(-1,3,30)

    nEkde = stats.gaussian_kde(np.real(nEvec))
    nIkde = stats.gaussian_kde(np.real(nIvec))
    mEkde = stats.gaussian_kde(np.real(mEvec))
    mIkde = stats.gaussian_kde(np.real(mIvec))
    xx = np.linspace(-10, 13, 50)
    fign, axn = plt.subplots(figsize=(5,1))
    axn.hist(np.real(nEvec), density=True, bins=nvbins, facecolor='tab:red', alpha=0.3)
    axn.plot(xx, nEkde(xx),c='tab:red')
    axn.hist(np.real(nIvec), density=True, bins=nvbins, facecolor='tab:blue',alpha=0.3)
    axn.plot(xx, nIkde(xx),c='tab:blue')
    axn.set_xlim(xlims)

    yy  = np.linspace(-1,3,50)
    figm, axm = plt.subplots(figsize=(3,1))
    axm.hist(np.real(mEvec), density=True, bins=mvbins,facecolor='tab:red', alpha=0.3)
    axm.plot(yy, mEkde(yy),c='tab:red')

    axm.hist(np.real(mIvec), density=True, bins=mvbins,facecolor='tab:blue' ,alpha=0.3)
    axm.plot(yy, mIkde(yy),c='tab:blue')
    axm.set_xlim(ylims)



    ### C.  preditions of individual connectivity fluctuations
    xtickms = [-2,3]
    xlimms  = [-2,3]
    ytickms = [-2,3]
    ylimms  = [-2,3]

    xticks = [-12,12]
    xlims  = [-12,12]
    yticks = [-12,12]
    ylims  = [-12,12]
    '''# CHOOSE ONE TRIAL'''
    axisshift= 1.0
    x0,y0=xAm.copy(),yAm.copy()
    # idxtrial,idxtrial_=4,0 #  ### @YX DALE'S LAW
    idxtrial,idxtrial_ = 8,3 # ### @YX original
    fig,ax=plt.subplots(2,2,figsize=(4,4))
    idxtrial=0
    # ### ~~~~~~~~~ this one 0.5, 0.9
    # idxgsamples=np.array([4,8])
    ### ~~~~~~ now this one 0.5, 1.0
    idxgsamples=np.array([8,10])

    for i in range(len(idxgsamples)):
        ax[0,i].plot(xticks,yticks,color='darkred',linestyle='--')
        ax[1,i].plot(xticks,yticks,color='darkred',linestyle='--')

        idrandsE=np.random.choice(np.arange(0,NE),size=100,replace=False)
        idrandsI=np.random.choice(np.arange(NE,N),size=100,replace=False)

        ax[0,i].scatter(np.real(Reigvecseries[idxgsamples[i],idxtrial,idrandsE]),np.real(ReigvecTseries[idxgsamples[i],idxtrial,idrandsE]),s=2,c='red',alpha=0.5)
        ax[1,i].scatter(np.real(Leigvecseries[idxgsamples[i],idxtrial,idrandsE]),np.real(LeigvecTseries[idxgsamples[i],idxtrial,idrandsE]),s=2,c='red',alpha=0.5)
        
        ax[0,i].scatter(np.real(Reigvecseries[idxgsamples[i],idxtrial,idrandsI]),np.real(ReigvecTseries[idxgsamples[i],idxtrial,idrandsI]),s=2,c='tab:blue',alpha=0.5)
        ax[1,i].scatter(np.real(Leigvecseries[idxgsamples[i],idxtrial,idrandsI]),np.real(LeigvecTseries[idxgsamples[i],idxtrial,idrandsI]),s=2,c='tab:blue',alpha=0.5)

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


    ### >>> D theoretical lambda and sigmanm
    DeltaLR,DeltaLR_num = np.zeros((neta,3),dtype=complex),np.zeros((neta,ntrial,3),dtype=complex)
    DeltaLR_eigv = np.zeros((neta,ntrial),dtype=complex)

    Jinit = np.zeros((ntrial,N,N),dtype=complex)
    ## Jinit same for identical trial # realization
    Leigvecinit,Reigvecinit = np.zeros((ntrial,N),dtype=complex),np.zeros((ntrial,N),dtype=complex)

    for idxeta,eta in enumerate (etaseries):
        etaE,etaB,etaI=eta*coeff[0],eta*coeff[1],eta*coeff[2]
        etaE,etaB,etaI = etaE*signeta[0],etaB*signeta[1],etaI*signeta[2]
        etaset = np.array([etaE,etaB,etaI])
        # gee,gei,gie,gii 
        gmat = np.array([xee,xei,xie,xii])*gaverage
        Jparams = np.array([JE,JI])

        eigtemp=np.real(lambda_theo2[idxeta,:])
        #### numerical results
        # eigtemp = np.mean(np.real(Beigvseries[idxeta,:,0]))*np.ones(2)
        #### lambda0
        # eigtemp = eigvAm[0]
        ### >>>>>>>> USING EIGENVALUE OUTLIER
        DeltaLR[idxeta,:] = sigmaOverlap(Jparams,nAm,mAm,eigtemp,gmat,etaset,Nparams,) 
          
        # ### >>>>>>>> USING EIGENVALUE0
        # DeltaLR[idxeta,:] = sigmaOverlap(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams,)  
        # # #### @YX 03DEC  USING DEFINITION
        # DeltaLR[idxeta,2]=np.squeeze(yAm.T@Zrandommat[idxeta,0,:,:]@Zrandommat[idxeta,0,:,:]@xAm/eigvAm[0]**2)/N
        # DeltaLR[idxeta,0]=np.squeeze(yAm.T@Zrandommat[idxeta,0,:,:NE]@Zrandommat[idxeta,0,:NE,:]@xAm/eigvAm[0]**2)/NE
        # DeltaLR[idxeta,1]=np.squeeze(yAm.T@Zrandommat[idxeta,0,:,NE:]@Zrandommat[idxeta,0,NE:,:]@xAm/eigvAm[0]**2)/NI

        # DeltaLR[idxeta,:] = sigmaOverlap(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams,)
        for itrial in range(ntrial):
            eigvaluem = Beigvseries[idxeta,itrial,0]

            # numerical value
            changeL_num = np.reshape(np.squeeze(Leigvecseries[idxeta,itrial,:].copy()),(N,1))
            changeL_num[:NE,0]-=np.mean(changeL_num[:NE,0])
            changeL_num[NE:,0]-=np.mean(changeL_num[NE:,0])
            changeR_num = np.reshape(np.squeeze(Reigvecseries[idxeta,itrial,:].copy()),(N,1))
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
    for i in range(2):#(3):
        meanvec, stdvec = np.mean(np.squeeze(DeltaLR_num[:,:,i]),axis=1),np.std(np.squeeze(DeltaLR_num[:,:,i]),axis=1)
        ax.plot(etaseries,DeltaLR[:,i],c=cm[i],linewidth=1.5)
        # ax.plot(etaseries,meanvec,c=cm[i],linewidth=1.5)
        ax.fill_between(etaseries,meanvec+stdvec,meanvec-stdvec,alpha=0.3,facecolor=cm[i])

    # ### ~~~~ heter ~~~~~~~~~~ 
    # yticks = np.linspace(0.0,0.4,2)
    # ylims = [0.0,0.4]
    # ### ~~~~ homo ~~~~~~~~~~ 
    # yticks = np.linspace(0.0,0.15,2)
    # ylims = [-0.04,0.18]
    # ## ~~~~ hete2 ~~~~~~~~~~ 
    # yticks = [-0.15,0,0.45]#np.linspace(-0.15,0.45,2)
    # ylims = [-0.15,0.45]

    # ## ~~~~ heteg (4VS1) ~~~~~~~~~~ 
    # yticks = [-0.1,0,0.25]#np.linspace(-0.15,0.45,2)
    # ylims = [-0.1,0.25]

    # ### ~~~~ hete2 JE1.4 2CONJUGATE~~~~~~~~~~ 
    # yticks = [-0.15,0,0.5]#np.linspace(-0.15,0.45,2)
    # ylims = [-0.15,0.5]

    ### ~~~~ HOMO -- INHIBIT ~~~~~~~~~~ 
    yticks = [-0.15,0]#np.linspace(-0.15,0.45,2)
    ylims  = [-0.15,0.0]


    xticks = np.linspace(0.0,1.0,2)
    xlims  = [-0.,1.]
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

    #### negative feedback 
    gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='Blues_r', aspect='auto',
                          extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    #### positive feedback
    # gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='Blues', aspect='auto',
    #                       extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.plot(lambda_theo2[:,0]-eigvAm[0],DeltaLR[:,2],c='black',linewidth=2.0,label='2 cut off')

    # ### ~~~~~ homo~~~~~~~~+ hete g homo eta
    # yticks = np.linspace(0.0,0.15,2)
    # ylims = [0.0,0.15]
    # xticks = np.linspace(0.0,0.15,2)
    # xlims = [0.0,0.15]
    # ### ~~~~~ hete~~~~~~~~
    # yticks = np.linspace(0.0,0.25,2)
    # ylims = [0.0,0.25]
    # xticks = np.linspace(0.0,0.25,2)
    # xlims = [0.0,0.25]


    # ### ~~~~~ hete2 JE1.4 2conjugate~~~~~~~~
    # yticks = np.linspace(0.0,0.3,2)
    # ylims = [0.0,0.3]
    # xticks = np.linspace(0.0,0.3,2)
    # xlims = [0.0,0.3]

    # ### ~~~~~ heteg ~~~~~~~~
    # yticks = np.linspace(0.0,0.2,2)
    # ylims = [0.0,0.2]
    # xticks = np.linspace(0.0,0.2,2)
    # xlims = [0.0,0.2]

    # ### ~~~~~ HOMO- INHIBIT ~~~~~~~~
    # yticks = np.linspace(-0.15,0.,2)
    # ylims = [-0.15,0.]
    # xticks = np.linspace(-0.15,0.,2)
    # xlims = [-0.15,0.]

    ## ~~~~~ HOMO- INHIBIT (4VS1)~~~~~~~~
    yticks = np.linspace(-0.16,0.,2)
    ylims = [-0.16,0.]
    xticks = np.linspace(-0.16,0.,2)
    xlims = [-0.16,0.]

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta \lambda$')
    ax.set_ylabel(r'$\sigma_{nm}$')
    
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


lst = [yAm, xAm,
        Beigvseries, Reigvecseries, Leigvecseries,
        ReigvecTseries, LeigvecTseries]
stg = ["yAm, xAm,"
        "Beigvseries, Reigvecseries, Leigvecseries, "
        "ReigvecTseries, LeigvecTseries"]
# data = list_to_dict(lst=lst, string=stg)
# data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/conns_reciprocal_etaEIII_JI14JE20_4VS1.npz"
# if (RERUN==1):
#     np.savez(data_name, **data)
