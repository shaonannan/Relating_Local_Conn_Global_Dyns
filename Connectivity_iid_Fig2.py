# -*- coding: utf-8 -*-
"""
@author: Yuxiu Shao
Statistics of the network connectivity
Full-rank Gaussian Network and Rank-one Mixture of Gaussian Approximation
Connectivity, Dynamics
"""

''' HELP FUNCTIONS '''
import numpy as np
import matplotlib.pylab as plt

from numpy import linalg as la
# import seaborn as sb


# from sympy import *
from scipy import stats

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
# fig.savefig('myimage.svg', format='svg', dpi=1200)

## reload the data saved
RERUN = 1 ## 0 reloading data; 1 reruning
if RERUN == 0:
    data_name = "your-folder-for-saving-date/conns_independent.npz" ### 
    # data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/conns_independent_4VS1.npz"
    data      = np.load(data_name)

np.random.seed(2021)
# generate mean matrix
plt.close('all')
Nt      = np.array([1200,300])#([750,750])
NE,NI   = Nt[0],Nt[1]
N       = NE+NI
Nparams = np.array([NE,NI])
### @YX 2709 original
nrank,ntrial,neta,ngavg = 1,30,30,10
## \bar{J} JI, JE for mean conn, a,b for rank-two
JI,JE,a,b = 0.60,2.0,0.0,0.0
JEE,JIE,JEI,JII=JE+a,JE-a,JI-b,JI+b
''' 
Am -- J(g0=0), l0(g0=0), r0(g0=0), S=R1(B1-lambda0In-1)^(-1)L1.T 
'''
Am,Jsv   = generate_meanmat_eig(Nparams,JEE,JIE,JEI,JII)
xAm,yAm  = np.ones((N,1)),np.ones((N,1))
yAm[:NE,0],yAm[NE:,0] = yAm[:NE,0]*N/NE*JE,-yAm[NE:,0]*N/NI*JI
eigvAm    = np.zeros(N)
eigvAm[0] = JE-JI

''' Network Setting for Iterating Following '''
signeta    =np.ones(3)
# signeta[1]*=(-1)
## heterogeneous random gain
''' >>>>>>>>>>>>>> Case 1 and Case 2 >>>>>>>>>>>>>>>>>>>>>>> '''
xee,xei,xie,xii=1.0,1.0,1.0,1.0 ## homogeneous random gain g_kl CASE 1
# xee,xei,xie,xii=1.0,0.5,0.2,0.8 ## hetegeneous random gain

### recording variables 
# numerical using eigendecomposition
Reigvecseries,Leigvecseries,Beigvseries=np.zeros((ngavg,ntrial,N),dtype=complex),np.zeros((ngavg,ntrial,N),dtype=complex),np.zeros((ngavg,ntrial,N),dtype=complex)
# analytical using perturbation theory
ReigvecTseries,LeigvecTseries,Beigvseries=np.zeros((ngavg,ntrial,N),dtype=complex),np.zeros((ngavg,ntrial,N),dtype=complex),np.zeros((ngavg,ntrial,N),dtype=complex)
## corresponding statistics \mu and \sigma \cov
armu,sigrcov = np.zeros((ngavg,ntrial,2)),np.zeros((ngavg,ntrial,2)) # 2 for E and I
almu,siglcov = np.zeros((ngavg,ntrial,2)),np.zeros((ngavg,ntrial,2))
siglr = np.zeros((ngavg,ntrial,2))

if(RERUN==0):
    Reigvecseries,Leigvecseries=data['Reigvecseries'],data['Leigvecseries']
    ReigvecTseries,LeigvecTseries=data['ReigvecTseries'],data['LeigvecTseries']
    Beigvseries = data['Beigvseries']

''' Iterative Processing '''
gaverageseries=np.linspace(0.1,1.0,ngavg)
if RERUN ==1:
    for iktrial in range(ntrial):
        # >>> generate the (base/ref) random matrix for each trial
        Xsym  = iidGaussian([0,1.0/np.sqrt(N)],[N,N])
        Xinit = Xsym.copy()
        # >>>>> Xsym and XsymT keep the same while changing the random gain gaverage
        for idxeta,gaverage in enumerate(gaverageseries):
            #  >>> random gain continuously increases   
            # ''' heterogeneity '''
            X=Xinit.copy()
            X[:NE,:NE]*=(xee*gaverage)
            X[:NE,NE:]*=(xei*gaverage)
            X[NE:,:NE]*=(xie*gaverage)
            X[NE:,NE:]*=(xii*gaverage)

            # overall
            J  = X.copy()+Am.copy()
            JT = J.copy().T
            '''   statistics >>>>>>>>>>> Full connectivity   '''
            # properties
            eigvJ,leigvec,reigvec,xnorm0,ynorm0=decompNormalization(J,xAm,yAm,xAm,yAm,nparams=Nparams,sort=1,nrank=1)
            ''' m n with eigenvalues in  '''
            ''' low rank properties m,n,eigenvalues '''
            Beigvseries[idxeta,iktrial,:]=eigvJ.copy()
            Reigvecseries[idxeta,iktrial,:],Leigvecseries[idxeta,iktrial,:]=xnorm0[:,0].copy(),ynorm0[:,0].copy()
            axrmu,aylmu,sigxr,sigyl,sigcov = numerical_stats(xnorm0,ynorm0,xAm,yAm,eigvJ,nrank,2,ppercent=np.array([NE/N,NI/N]))
            armu[idxeta,iktrial,:],almu[idxeta,iktrial,:]      = axrmu[:,0],aylmu[:,0]
            sigrcov[idxeta,iktrial,:],siglcov[idxeta,iktrial,:]= sigxr[:,0],sigyl[:,0]
            siglr[idxeta,iktrial,:] = sigcov[:,0,0]


            ### reconstruct eigenvectors using perturbation theory
            xnorm0,ynorm0 = np.reshape(xnorm0,(N,1)),np.reshape(ynorm0,(N,1))
            xnormt,ynormt = np.zeros_like(xnorm0),np.zeros_like(ynorm0)
            ### ~~~~~~~~~ first order perturbation ~~~~~~~~~
            xnormt,ynormt = (np.eye(N)+X/eigvAm[0])@xAm.copy(),yAm.copy().T@(np.eye(N)+X/eigvAm[0])
            ynormt = ynormt.T
            ReigvecTseries[idxeta,iktrial,:],LeigvecTseries[idxeta,iktrial,:] = xnormt[:,0],ynormt[:,0]

def analysis():
    ### A. eigenvalue spectrum
    idxgavg,idxtrial = 0,0# 0,3#9,3 # 6,3,0
    idxgavgsample    = np.array([9,6,3])
    figtspt,axtspt   = plt.subplots(figsize=(5,3))
    cm=['b','g','c']
    shiftlen=0.2#0.06 # for visualization
    idrands=np.random.choice(np.arange(nrank,N),size=600,replace=False) # randomly sample 600 complex eigenvalues (within bulk)
    for idxc,idxgavg in enumerate(idxgavgsample):
        axtspt.scatter(np.real(Beigvseries[idxgavg,idxtrial,idrands]),np.imag(Beigvseries[idxgavg,idxtrial,idrands]),s=10,c=cm[idxc],alpha=0.25) 
        axtspt.scatter(np.real(Beigvseries[idxgavg,:,0]),np.imag(Beigvseries[idxgavg,:,0])-(len(idxgavgsample)-1)*shiftlen+shiftlen*idxc,s=20,c=cm[idxc],alpha=0.25) 
        axtspt.set_aspect('equal')

        axtspt.scatter(eigvAm[0],-(len(idxgavgsample)-1)*shiftlen+shiftlen*idxc,s=80,c='',marker='o',edgecolor='black') # 
        axtspt.spines['right'].set_color('none')
        axtspt.spines['top'].set_color('none')
        axtspt.xaxis.set_ticks_position('bottom')
        axtspt.spines['bottom'].set_position(('data', 0))

        ### radius of the eigenvalue bulk (effective)
        aee,aei,aie,aii = xee,xei,xie,xii
        eta   = 0 # i.i.d.
        theta = np.linspace(0, 2 * np.pi, 200)
        # first do not multiply at
        ahomo=gaverageseries[idxgavg]
        xee_,xei_,xie_,xii_=ahomo*aee/np.sqrt(N),ahomo*aei/np.sqrt(N),ahomo*aie/np.sqrt(N),ahomo*aii/np.sqrt(N)
        gmat     = np.array([[NE*xee_**2,NI*xei_**2],[NE*xie_**2,NI*xii_**2]])
        eigvgm,eigvecgm=la.eig(gmat) 
        r_g2= np.max(eigvgm)
        r_g = np.sqrt(r_g2)
        eta=0
        longaxis,shortaxis=(1+eta)*r_g ,(1-eta)*r_g 
        xr = longaxis*np.cos(theta)
        yr = shortaxis*np.sin(theta)
        axtspt.plot(xr, yr, color="gray", linewidth=0.5,linestyle='--',label=r'ellipse') # radius

    #### @YX 2709 original one ----
    xticks = np.linspace(-1,1,3)
    xlims = [-2.0,2.0]
    yticks = np.linspace(-1.0,1.0,2)
    ylims = [-1.5,1.5]

    axtspt.set_xlim(xlims)
    axtspt.set_ylim(ylims)
    axtspt.set_xticks(xticks)
    axtspt.set_yticks(yticks)
    axtspt.set_aspect('equal')       

    ### B. statistics of the eigenvalue and entries on the rank-one vectors
    ng    = len(gaverageseries)
    ige   = ng
    sjump = 2
    ## calculating the statistics of perturbation lambda
    gee,gei,gie,gii=xee*gaverageseries,xei*gaverageseries,xie*gaverageseries,xii*gaverageseries
    std_Lambda_theo = np.sqrt(JE**2*(gee**2+NI/NE*gei**2)/N+JI**2*(NE/NI*gie**2+gii**2)/N)/eigvAm[0]
    mean_lambda = np.mean(Beigvseries[:,:,0].real,axis=1)
    std_lambda  = np.std(Beigvseries[:,:,0].real,axis=1)
    figE,axE    = plt.subplots(figsize=(4,4))
    stdlambda   = np.std(Beigvseries[:,:,0],axis=1)
    cm='rbg'
    axE.plot(gaverageseries[:ige],mean_lambda,color='tab:red',linestyle='-',linewidth=1.5)
    axE.fill_between(gaverageseries[:ige],mean_lambda+std_lambda,mean_lambda-std_lambda, alpha=0.3,facecolor='tab:red')

    axE.plot(gaverageseries[:ige],eigvAm[0]*np.ones(ige),color='black',linestyle='--',linewidth=1.5)
    axE.plot(gaverageseries[:ige],eigvAm[0]*np.ones(ige)+std_Lambda_theo[:ige],color='gray',linestyle='--',linewidth=1.5,label='numerical')
    axE.plot(gaverageseries[:ige],eigvAm[0]*np.ones(ige)-std_Lambda_theo[:ige],color='gray',linestyle='--',linewidth=1.5,label='numerical')
    axE.set_title(r'$\sigma_{\lambda}$')
    axE.legend()

    '''
    plt and ax
    '''
    ### @YX ORIGINAL ----
    ylims=[0.0,0.05]
    xlims=[0.1,1.]
    yticks = np.linspace(0.0,0.05,2)
    xticks = np.linspace(0.1,1.0,2)

    axE.set_xlim(xlims)
    axE.set_xticks(xticks)
    axE.set_ylim([1.3,1.5])
    axE.set_yticks([1.3,1.4,1.5])

    '''
    right eigenvector
    '''

    ## calculating the statistics of perturbation lambda
    std_Reig_theo=np.zeros((ngavg,2))

    std_Reig_theo[:,0],std_Reig_theo[:,1] = np.sqrt((NE/N*(xee*gaverageseries)**2+NI/N*(xei*gaverageseries)**2)/eigvAm[0]**2),np.sqrt((NE/N*(xie*gaverageseries)**2+NI/N*(xii*gaverageseries)**2)/eigvAm[0]**2)

    figR0,axR0=plt.subplots(2,1,figsize=(4,4))
    cm='rbg'

    mean_sig_Reig_num,std_sig_Reig_num = np.zeros((ng,2)),np.zeros((ng,2))
    mean_sig_Reig_num = np.mean(sigrcov[:,:,:2],axis=1)
    std_sig_Reig_num  = np.std(sigrcov[:,:,:2],axis=1)

    # >>>>>>> Theoretical solutions
    axR0[0].plot(gaverageseries[:ige],std_Reig_theo[:ige,0]**2,color='tab:red',linewidth=1.5,label='analytical')
    axR0[1].plot(gaverageseries[:ige],std_Reig_theo[:ige,1]**2,color='tab:blue',linewidth=1.5,label='analytical')
    axR0[0].set_title(r'$\sigma_{m}^E$')
    axR0[1].set_title(r'$\sigma_{m}^I$')

    # >>>>> numerical solutions
    axR0[0].plot(gaverageseries[:ige],mean_sig_Reig_num[:,0],linestyle='--',color='tab:red',lw=1.5)
    axR0[1].plot(gaverageseries[:ige],mean_sig_Reig_num[:,1],linestyle='--',color='tab:blue',lw=1.5)
    # >>>>>>> Theoretical solutions
    axR0[0].set_title(r'$\sigma_{m}^E$')
    axR0[1].set_title(r'$\sigma_{m}^I$')
    axR0[0].legend()
    axR0[1].legend()

    for i in range(2):
        axR0[i].set_xlim(xlims)
        axR0[i].set_xticks(xticks)
    '''
    left eigenvector
    '''
    ## calculating the statistics of perturbation lambda
    std_Leig_theo=np.zeros((ngavg,2))
    std_Leig_theo[:,0],std_Leig_theo[:,1] = np.sqrt((N/NE*(JE*xee)**2+N/NI*(JI*xie)**2)/eigvAm[0]**2)*gaverageseries,np.sqrt((N/NE*(JE*xei)**2+N/NI*(JI*xii)**2)/eigvAm[0]**2)*gaverageseries

    figL0,axL0=plt.subplots(2,1,figsize=(4,4))

    mean_sig_Leig_num,std_sig_Leig_num = np.zeros((ng,2)),np.zeros((ng,2))
    mean_sig_Leig_num= np.mean(siglcov[:,:,:2],axis=1)
    std_sig_Leig_num= np.std(siglcov[:,:,:2],axis=1)
    # >>>>>>> Theoretical solutions
    axL0[0].plot(gaverageseries[:ige],np.real(std_Leig_theo[:ige,0])**2,color='tab:red',linewidth=1.5,label='analytical')
    axL0[1].plot(gaverageseries[:ige],np.real(std_Leig_theo[:ige,1])**2,color='tab:blue',linewidth=1.5,label='analytical')
    axL0[0].set_title(r'$\sigma_{n}^E$')
    axL0[1].set_title(r'$\sigma_{n}^I$')
    for i in range(2):
        axL0[i].set_xlim(xlims)
        axL0[i].set_xticks(xticks)
    # >>>>>> numerical solutions
    axL0[0].plot(gaverageseries[:ige],mean_sig_Leig_num[:,0],linestyle='--',color='tab:red',lw=1.5)
    axL0[1].plot(gaverageseries[:ige],mean_sig_Leig_num[:,1],linestyle='--',color='tab:blue',lw=1.5)
    axL0[0].legend()
    axL0[1].legend()

    figRL,axRL=plt.subplots(2,1,figsize=(4,4),sharex=True,tight_layout=True)
    axRL[0].plot(gaverageseries[:ige],np.real(std_Leig_theo[:ige,0])**2,color='tab:red',linewidth=1.5,label='analytical')
    axRL[0].plot(gaverageseries[:ige],np.real(std_Leig_theo[:ige,1])**2,color='tab:blue',linewidth=1.5,label='analytical')
    axRL[0].set_title(r'$\sigma_{n}^p$')
    axRL[0].set_xlim(xlims)
    axRL[0].set_xticks(xticks)

    axRL[1].plot(gaverageseries[:ige],np.real(std_Reig_theo[:ige,0])**2,color='tab:red',linewidth=1.5,label='analytical')
    axRL[1].plot(gaverageseries[:ige],np.real(std_Reig_theo[:ige,1])**2,color='tab:blue',linewidth=1.5,label='analytical')
    axRL[1].set_title(r'$\sigma_{m}^p$')
    axRL[1].set_xlim(xlims)
    axRL[1].set_xticks(xticks)

    axRL[0].plot(gaverageseries[:ige],mean_sig_Leig_num[:,0],linestyle='--',color='tab:red',lw=1.5)
    axRL[0].plot(gaverageseries[:ige],mean_sig_Leig_num[:,1],linestyle='--',color='tab:blue',lw=1.5)

    axRL[1].plot(gaverageseries[:ige],mean_sig_Reig_num[:,0],linestyle='--',color='tab:red',lw=1.5)
    axRL[1].plot(gaverageseries[:ige],mean_sig_Reig_num[:,1],linestyle='--',color='tab:blue',lw=1.5)

    axRL[0].legend()
    axRL[1].legend()

    ### C. loading patterns of eigenvectors
    idxtrial=9#16#
    idxeta=9#4# # 9 FOR COMPARISON
    alphaval=0.25
    edgv='black'
    cm='br'
    ntrial = np.shape(Reigvecseries)[1]
    '''loading vector changing'''
    meanmE,meanmI,meannE,meannI=np.zeros(nrank),np.zeros(nrank),np.zeros(nrank),np.zeros(nrank)
    #### change dimensionality 
    mEvec,mIvec,nEvec,nIvec=np.squeeze(Reigvecseries[idxeta,:,:NE]),np.squeeze(Reigvecseries[idxeta,:,NE:]),np.squeeze(Leigvecseries[idxeta,:,:NE]),np.squeeze(Leigvecseries[idxeta,:,NE:])

    #>>>>>>> check reconstructed result
    # mEvec,mIvec,nEvec,nIvec=np.squeeze(ReigvecTseries[idxeta,:,:NE]),np.squeeze(ReigvecTseries[idxeta,:,NE:]),np.squeeze(LeigvecTseries[idxeta,:,:NE]),np.squeeze(LeigvecTseries[idxeta,:,NE:])

    # >>>>>> results obtained using eigendecomposition
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

    xticks = [-10,-5,0,10]
    xlims  = [-10,13]
    yticks = [0,5]
    ylims  = [-2,5]
    
    # xticks = np.linspace(-20,20,2)
    # xlims = [-20,20]
    # yticks = np.linspace(-10,10,2)
    # ylims = [-10,10]

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_aspect('equal')

    # idrandE=np.random.choice(np.arange(0,len(nEvec)),size=NE,replace=False)
    # idrandI=np.random.choice(np.arange(0,len(nIvec)),size=NI,replace=False)
    # ax.scatter(nIvec[idrandI],mIvec[idrandI],s=5.0,c='blue',alpha=alphaval)#cm[1],alpha=alphaval)
    # ax.scatter(nEvec[idrandE],mEvec[idrandE],s=5.0,c='red',alpha=alphaval)#cm[0],alpha=alphaval)
    ax.scatter(nIvec[:NI],mIvec[:NI],s=5.0,c='tab:blue',alpha=alphaval)
    ax.scatter(nEvec[:NE],mEvec[:NE],s=5.0,c='tab:red',alpha=alphaval)

    ### B1. CALCULATE THE HISTOGRAM -- TEST THE ENTRIES ON EIGENVECTORS ARE GAUSSIAN DISTRIBUTED
    np.random.seed(41)
    nvbins = np.linspace(-10,15,50)#np.linspace(-10,10,100)#
    mvbins =  np.linspace(-2,5,20)#np.linspace(-2,2,)

    nEkde = stats.gaussian_kde(np.real(nEvec))
    nIkde = stats.gaussian_kde(np.real(nIvec))
    mEkde = stats.gaussian_kde(np.real(mEvec))
    mIkde = stats.gaussian_kde(np.real(mIvec))
    xx = np.linspace(-10, 15, 100)#np.linspace(-10, 10, 1000)
    fign, axn = plt.subplots(figsize=(5,1))
    axn.hist(np.real(nEvec), density=True, bins=nvbins, facecolor='tab:red', alpha=0.3)
    axn.plot(xx, nEkde(xx),c='r')
    axn.hist(np.real(nIvec), density=True, bins=nvbins, facecolor='tab:blue',alpha=0.3)
    axn.plot(xx, nIkde(xx),c='b')
    axn.set_xlim(xlims)

    yy  =  np.linspace(-2,5,80)#np.linspace(-2,2,20)
    figm, axm = plt.subplots(figsize=(3,1))
    axm.hist(np.real(mEvec), density=True, bins=mvbins,facecolor='tab:red', alpha=0.3)
    axm.plot(yy, mEkde(yy),c='r')

    axm.hist(np.real(mIvec), density=True, bins=mvbins,facecolor='tab:blue' ,alpha=0.3)
    axm.plot(yy, mIkde(yy),c='b')
    axm.set_xlim(ylims)


    ### C.  COMPARE INDIVIDUAL ENTRIES ON THE EIGENVECTORS (RECONSTRUCT V.S. EIGENDECOMPOSITION)
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
    # ### ~~~~~~~~~ this one 0.5, 0.9
    # idxgsamples=np.array([4,8])
    ### ~~~~~~ now this one 0.5, 1.0
    idxgsamples=np.array([4,9])
    gsamples=gaverageseries[idxgsamples]

    for i in range(len(idxgsamples)):
        ax[0,i].plot(xticks,yticks,color='darkred',linestyle='--')
        ax[1,i].plot(xticks,yticks,color='darkred',linestyle='--')
        idrandsE=np.random.choice(np.arange(0,NE),size=100,replace=False)
        idrandsI=np.random.choice(np.arange(NE,N),size=100,replace=False)

        ax[0,i].scatter(np.real(Reigvecseries[idxgsamples[i],idxtrial,idrandsE]),np.real(ReigvecTseries[idxgsamples[i],idxtrial,idrandsE]),s=2,c='tab:red',alpha=0.5)
        ax[1,i].scatter(np.real(Leigvecseries[idxgsamples[i],idxtrial,idrandsE]),np.real(LeigvecTseries[idxgsamples[i],idxtrial,idrandsE]),s=2,c='tab:red',alpha=0.5)
        
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
    # print(np.shape(rprime_the),np.shape(rprime_num))
    # ax.scatter(np.real(rprime_the),rprime_num,c='gray',alpha=0.5)
    # ax.set_xlim([-0.05,0.05])
    # ax.set_ylim([-0.05,0.05])
    '''
    # # CHOOSE TWO TRIALS
    # '''
    # fig,ax=plt.subplots(2,2,figsize=(4,4))
    # gsamples=gaverageseries[idxgsamples]
    # for i in range(len(idxgsamples)):
    #     ax[0,i].plot(xticks,yticks,color='darkred',linestyle='--')
    #     ax[1,i].plot(xticks,yticks,color='darkred',linestyle='--')
    #     #### @YX modify 2508 -- redundancy
    #     #### @YX modify 2508 -- from Reigvecseries[i,...] to Reigvecseries[idxgsamples[i]...]
    #     # ax[0,i].scatter(np.real(Reigvecseries[idxgsamples[i],idxtrial_,:]),np.real(ReigvecTseries[idxgsamples[i],idxtrial,:])+axisshift,s=5,c='gray',alpha=0.5)
    #     # ax[1,i].scatter(np.real(Leigvecseries[idxgsamples[i],idxtrial_,:]),np.real(LeigvecTseries[idxgsamples[i],idxtrial,:]),s=5,c='gray',alpha=0.5)
        
    #     ax[0,i].scatter(np.real(Reigvecseries[idxgsamples[i],idxtrial_,:NE]),np.real(ReigvecTseries[idxgsamples[i],idxtrial,:NE]),s=2,c='red',alpha=0.5)
    #     ax[1,i].scatter(np.real(Leigvecseries[idxgsamples[i],idxtrial_,:NE]),np.real(LeigvecTseries[idxgsamples[i],idxtrial,:NE]),s=2,c='red',alpha=0.5)
        
    #     ax[0,i].scatter(np.real(Reigvecseries[idxgsamples[i],idxtrial_,NE:]),np.real(ReigvecTseries[idxgsamples[i],idxtrial,NE:]),s=2,c='blue',alpha=0.5)
    #     ax[1,i].scatter(np.real(Leigvecseries[idxgsamples[i],idxtrial_,NE:]),np.real(LeigvecTseries[idxgsamples[i],idxtrial,NE:]),s=2,c='blue',alpha=0.5)
    
    #     # ax[0,i].scatter(np.real(rperturb_the[:,i]),rperturb_num[:,i]+axisshift,s=5,c='gray',alpha=0.5)
    #     # ax[1,i].scatter(np.real(lperturb_the[:,i]),lperturb_num[:,i],s=5,c='gray',alpha=0.5)
    #     ax[0,i].set_xticks([])
    #     ax[0,i].set_yticks([])
    #     ax[1,i].set_xticks([])
    #     ax[1,i].set_yticks([])



    # for i in range(2):
    #     ax[1,i].set_xlim(xlims)
    #     ax[1,i].set_ylim(ylims)
    # ax[1,0].set_yticks(yticks)

    # for i in range(2):
    #     ax[0,i].set_xlim(xlimms)
    #     ax[0,i].set_ylim(ylimms)
    # ax[0,0].set_yticks(ytickms)
    # for i in range(2):
    #     ax[1,i].set_xticks(xticks)
    # # print(np.shape(rprime_the),np.shape(rprime_num))
    # # ax.scatter(np.real(rprime_the),rprime_num,c='gray',alpha=0.5)
    # # ax.set_xlim([-0.05,0.05])
    # # ax.set_ylim([-0.05,0.05])
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
        ReigvecTseries, LeigvecTseries, siglcov]
stg = ["yAm, xAm,"
        "Beigvseries, Reigvecseries, Leigvecseries, "
        "ReigvecTseries, LeigvecTseries, siglcov"]
data = list_to_dict(lst=lst, string=stg)
# data_name = "/Users/yuxiushao/Public/DataML/EI_LOWRANK/conns_independent_4VS1.npz"
# if (RERUN==1):
#     np.savez(data_name, **data)




