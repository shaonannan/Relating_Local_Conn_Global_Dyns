import numpy as np
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
import matplotlib.patches as mpatches
from functools import partial
import random

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# from sympy import *
from scipy.linalg import schur, eigvals
from scipy.special import comb, perm

       
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

shiftx = 1.5

def generate_meanmat_eig(Nparams,JEE,JIE,JEI,JII):
    # mean value 
    # first use rank-1 structure
    NE,NI=Nparams[0],Nparams[1]
    N=NE+NI
    at = Nparams/N   
    Am=np.zeros((N,N))
    Am[:NE,:NE],Am[:NE,NE:]=JEE/NE,-JEI/NI
    Am[NE:,:NE],Am[NE:,NE:]=JIE/NE,-JII/NI

    Jsv=np.zeros((2,2))
    Jsv[0,0],Jsv[0,1],Jsv[1,0],Jsv[1,1]=JEE,JEI,JIE,JII
    return (Am,Jsv)#,ua,svam,va)
    
def gmatamplitude_eig(gavgfix,typenum):
    Amplit = gavgfix*typenum
    numsample = typenum**2
    Amplitg= np.zeros(numsample)
    idxc=0
    while (1):
      if idxc>=numsample:
        Amplitg[numsample-1]=1.0-np.sum(Amplitg[:numsample-1])
        break
      p=np.random.random(1)
      Amplitg[idxc]=np.minimum(p,1.0-np.sum(Amplitg))
      if np.sum(Amplitg)>1.0:
        continue 
      elif np.sum(Amplitg)==1.0:
        break
      else:
        idxc +=1
        # Amplitg[idxc]=np.min(p,1.0-np.sum(Amplitg))
    # Amplitg=0
    Amplitg*=Amplit
    Amplitg=np.sqrt(Amplitg)
    return Amplitg

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    print(np.shape(cov))
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D()         .rotate_deg(45)         .scale(scale_x, scale_y)         .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def gradtanh(xorg):
    gradx = xorg.copy()
    nneuron,nknum=np.shape(gradx)[0],np.shape(gradx)[1]
    for i in range(nneuron):
        for j in range(nknum):
            gradx[i,j]=4/(np.exp(-xorg[i,j])+np.exp(xorg[i,j]))**2
    return gradx

import math
import cmath
def ThreeSquare(x):
    if x.imag == 0:
        # real
        m = x.real
        ans = -math.pow(-m,1/3) if m<0 else math.pow(m,1/3)
    else:
        # imag
        ans = x**(1/3)
    return ans

def RoundAns(x,num):
    if x.imag == 0:
        m = x.real
        ans = round(m,num)
    else:
        m = round(x.real,num)
        n = round(x.imag,num)
        ans = complex(m,n)
    return ans
    
def Cubic(args):
    a,b,c,d = args
    p = c/a-b**2/(3*a**2)
    q = d/a+2*b**3/(27*a**3)-b*c/(3*a**2)
    w = complex(-0.5,(3**0.5)/2)
    ww = complex(-0.5,-(3**0.5)/2)
    A = cmath.sqrt((q/2)**2+(p/3)**3)
    B = ThreeSquare(-q/2+A)
    C = ThreeSquare(-q/2-A)
    y1 = B+C
    y2 = w*B+ww*C
    y3 = ww*B+w*C
    D = b/(3*a)
    roots=[RoundAns(y1-D,6),RoundAns(y2-D,6),RoundAns(y3-D,6)]
    return roots


# %%
def CrossRadiusLambda(x,EPSP,kparams,nparams,Npercent):
    kE,kI,Cs=kparams[0],kparams[1],kparams[2]
    
    NE,NI=nparams[0],nparams[1]
    N = NE+NI
    IPSP = EPSP*x
    ## Radius
    gE2,gI2 = EPSP**2*(1-kE*Cs/NE)*kE*Cs*N/NE,IPSP**2*(1-kI*Cs/NI)*kI*Cs*N/NI
    alphaNEI = Npercent.copy()
    gMmat = np.zeros((2,2))
    gMmat[:,0],gMmat[:,1]=gE2*alphaNEI[0],gI2*alphaNEI[1]
    # gMmat = np.array([[gE2*alphaNEI[0],gI2*alphaNEI[1]],[gE2*alphaNEI[0],gI2*alphaNEI[1]]])
    eigvgm,eigvecgm=la.eig(gMmat) 
    r_g2=np.max(eigvgm)
    r_g = np.sqrt(r_g2)
    ## Lambda 
    lambda_Gm = np.abs((EPSP*kE*Cs)*(1-x*kI/kE))

    return (r_g-lambda_Gm)

def theo_radius(epsp,kparams,nparams,gei):
    cgei = gei.copy()
    rho = kparams[0]*kparams[2]/nparams[0]
    kE,kI = kparams[0]*kparams[2],kparams[1]*kparams[2]
    NE,NI = nparams[0],nparams[1]
    N = NE+NI
    Radius2 = EPSP**2*kI*(1-rho)*(cgei**2+4)

    return np.sqrt(Radius2)

def criticcRad(x,EPSP,KE,KI,A,):
    result = EPSP**2*KI*(1-A)*(x**2+4)-EPSP**2*KE**2*(1-x/(KE/KI))**2
    return result
def criticcX(x,EPSP,KE,KI,A,):
    result = EPSP*KE*(1-x/4.0)-1.0
    return result


# %%
### dynamics
def iidGaussian(stats,shapem):
	mu,sig = stats[0],stats[1]
	nx,ny = shapem[0],shapem[1]
	return np.random.normal(mu,sig,(nx,ny))
def odeIntegral(x,t,J,I=0):
	x = np.squeeze(x)
	x = np.reshape(x,(len(x),1))
	# print('size:',np.shape(x),np.shape(J@np.tanh(x)))
	dxdt = -x+J@np.tanh(x)
	return np.squeeze(dxdt)
def odesimulation(t,xinit,Jpt,I):
	return scipy.integrate.odeint(partial(odeIntegral,J=Jpt,I=I),xinit,t)

def odeIntegralP(x,t,J,I=0):
	x = np.squeeze(x)
	x = np.reshape(x,(len(x),1))
	# print('size:',np.shape(x),np.shape(J@np.tanh(x)))
	dxdt = -x+J@(1.0+np.tanh(x-shiftx))
	return np.squeeze(dxdt)
def odesimulationP(t,xinit,Jpt,I):
	return scipy.integrate.odeint(partial(odeIntegralP,J=Jpt,I=I),xinit,t)

## for single value 
def transferf(xorg):
	return np.tanh(xorg)
def transferdf(xorg):
    return np.log(np.cosh(xorg))
    # return 4/(np.exp(-xorg)+np.exp(xorg))**2
	# return 1/np.cosh(xorg)**2

def decompunperturbedEI(Junperturb,nrank=1):
	eigvAm,eigvecrAm=la.eig(Junperturb)
	meigvecAm=eigvecrAm.copy()
	neigvecAm=la.inv(eigvecrAm.copy())
	neigvecAm=neigvecAm.T
	for i in range(nrank):
		normalizecoe=(np.sum(neigvecAm[:,i]*meigvecAm[:,i]))
		neigvecAm[:,i]*=(eigvAm[i]/normalizecoe)
	for i in range(nrank,N):
		normalizecoe=(np.sum(neigvecAm[:,i]*meigvecAm[:,i]))
		neigvecAm[:,i]/=normalizecoe
	return meigvecAm,neigvecAm,eigvAm	

def decompNormalization(Jconn,refxvec,refyvec,xsign,ysign,nparams,sort=0,nrank=1):
	'''
	### modified version (July 2021):
		(previous) refxvec,refyvec --> used for normalization
					xsign, ysign --> used for unifying pos/neg signs
		(new) considering the changes in outlier affects normalization
				using m(r) n(lambda l) rather than r and l 
	'''
	if(len(nparams)==2):
		NE,NI= nparams[0],nparams[1]
	else:
		NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
		NE=NE1+NE2
	N = NE+NI
	eigvJ,eigrvecJ=la.eig(Jconn)
	inveigrvecJ=la.inv(eigrvecJ)
	meig = np.squeeze(eigrvecJ[:,:].copy())
	neig = np.squeeze(inveigrvecJ[:,:].copy()) # inverse
	neig=neig.copy().T
	leig, reig = np.zeros((N,N)),np.zeros((N,N))
	'''    Sort Eigenvalue in ascending     '''
	if (sort ==1):
		eigenvalue_Rsort = np.squeeze(-np.abs(np.real(eigvJ.copy())))#np.squeeze(-(np.real(eigvJ.copy())))#
		idxsort          = np.argsort(eigenvalue_Rsort)
		eigenvalue_sort  = np.squeeze(eigvJ.copy())
		eigvJ= eigvJ[idxsort]  #### >>>>>>>>>>>>>> resorting >>>>>>>>>
		# print("eigv sort:",eigvJ[:2])
		## >>> for eigendecomposition 
		reig = meig[:,idxsort]
		leig = neig[:,idxsort]
		normval=np.sum(reig*leig.copy(),axis=0)
		normval=np.repeat(np.reshape(normval,(1,N)),N,axis=0)
		leig=leig.copy()/normval.copy()   # error
	elif (sort==0):	
		reig=meig.copy()
		normval=np.sum(reig*neig.copy(),axis=0)
		normval=np.repeat(np.reshape(normval,(1,N)),N,axis=0)
		leig=neig.copy()/normval.copy()
	## ------------- adding July (normalization on m&n)|| reference should be on m & n
	## for reference
	for i in range(nrank):
		leig[:,i]*=eigvJ[i]
	for j in range(nrank,N):
		leig[:,j]*=eigvJ[j]
	leig,reig = leig*np.sqrt(N),reig*np.sqrt(N)
	tildex,tildey=np.reshape(reig[:,0].copy(),(N,1)),np.reshape(leig[:,0].copy(),(N,1))
	## make sure the E in y is positive, for negative change signs of x and y
	if (np.mean(tildex[:NE,0])*xsign[0,0])<0:
		tildex*=(-1)
		tildey*=(-1)
	x,y=np.reshape(tildex,(N,1)),np.reshape(tildey,(N,1))
	x=np.sqrt(np.squeeze(tildey.T@refxvec)/np.squeeze(refyvec.T@tildex))*tildex
	y=np.sqrt(np.squeeze(tildex.T@refyvec)/np.squeeze(refxvec.T@tildey))*tildey
	return (eigvJ,leig,reig,x,y)

def decompUnnormalization(Jconn,refxvec,refyvec,xsign,ysign,nparams,sort=0,nrank=1):
	if(len(nparams)==2):
    		NE,NI= nparams[0],nparams[1]
	else:
		NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
		NE=NE1+NE2
	N = NE+NI
	eigvJ,eigrvecJ=la.eig(Jconn)
	inveigrvecJ=la.inv(eigrvecJ)
	meig = np.squeeze(eigrvecJ[:,:].copy())
	neig = np.squeeze(inveigrvecJ[:,:].copy()) # inverse
	neig=neig.copy().T
	leig, reig = np.zeros((N,N)),np.zeros((N,N))
	'''    Sort Eigenvalue in ascending     '''
	if (sort ==1):
		eigenvalue_Rsort = np.squeeze(-np.abs(np.real(eigvJ.copy())))
		idxsort=np.argsort(eigenvalue_Rsort)
		eigenvalue_sort = np.squeeze(eigvJ.copy())
		eigvJ= eigvJ[idxsort]  #### >>>>>>>>>>>>>> resorting >>>>>>>>>
		## >>> for eigendecomposition 
		reig = meig[:,idxsort]
		leig = neig[:,idxsort]
		normval=np.sum(reig*leig.copy(),axis=0)
		normval=np.repeat(np.reshape(normval,(1,N)),N,axis=0)
		leig=leig.copy()/normval.copy()   # error
	elif (sort==0):	
		reig=meig.copy()
		normval=np.sum(reig*neig.copy(),axis=0)
		normval=np.repeat(np.reshape(normval,(1,N)),N,axis=0)
		leig=neig.copy()/normval.copy()
	## ------------- adding July (normalization on m&n)|| reference should be on m & n   
	for i in range(nrank):
		leig[:,i]*=eigvJ[i]
	for j in range(nrank,N):
		leig[:,j]*=eigvJ[j]
	leig,reig = leig*np.sqrt(N),reig*np.sqrt(N)
	tildex,tildey=np.reshape(reig[:,0].copy(),(N,1)),np.reshape(leig[:,0].copy(),(N,1))
	## make sure the E in y is positive, for negative change signs of x and y
	if (np.mean(tildex[:NE,0])*xsign[0,0])<0:
		tildex*=(-1)
		tildey*=(-1)
	x,y=np.reshape(tildex,(N,1)),np.reshape(tildey,(N,1))
	return (eigvJ,leig,reig,x,y)

def Normalization(xvec,yvec,refxvec,refyvec,nparams,sort=0,nrank=1):
	'''
	Lestvec  = np.reshape(LestvecT.T,(N,1))
	Restvec  = np.reshape(Restvec.copy()/np.linalg.norm(Restvec.copy()),(N,1)) ## move in
	_,_,xnormt,ynormt=Normalization(Restvec.copy(),Lestvec.copy(),xAm.copy(),yAm.copy(),nparams=Nparams,sort=0,nrank=1)
	Restvecseries[ig,iktrial,:,0],Lestvecseries[ig,iktrial,:,0] =xnormt[:,0]*np.sqrt(N),ynormt[:,0]*np.sqrt(N)
	'''
	if(len(nparams)==2):
    		NE,NI= nparams[0],nparams[1]
	else:
		NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
		NE=NE1+NE2
	N = NE+NI
	leig, reig = np.zeros((N,N)),np.zeros((N,N))
	#### m1 (ERROR)
	# normreig=np.linalg.norm(xvec.copy())
	# reig = xvec.copy()/normreig
	# leig = yvec.copy()*normreig
	#### m2
	normreig=np.sqrt(np.sum(xvec.copy()**2))/np.sqrt(N)
	reig = xvec.copy()/normreig
	leig = yvec.copy()*normreig
	#### m3
	# normreig=np.mean(xvec.copy())
	# reig = xvec.copy()/normreig
	# leig = yvec.copy()*normreig
	# print("right normalized:",np.mean(reig[:,0]))
	## for reference
	tildex,tildey=np.reshape(reig[:,0].copy(),(N,1)),np.reshape(leig[:,0].copy(),(N,1))
	if (np.mean(tildex[:NE,0])*refxvec[0,0])<0:
		tildex*=(-1)
		tildey*=(-1)
	x,y=np.reshape(tildex,(N,1)),np.reshape(tildey,(N,1))
	x=np.sqrt(np.squeeze(tildey.T@refxvec)/np.squeeze(refyvec.T@tildex))*tildex
	y=np.sqrt(np.squeeze(tildex.T@refyvec)/np.squeeze(refxvec.T@tildey))*tildey
	return (leig,reig,x,y)
    

def numerical_stats(xrvec,ylvec,xrref,ylref,eigv,nrank,npop,ppercent):## ppercent might be useful
	nneuron = np.shape(xrvec)[0]
	nnpop   = np.zeros(npop)
	for i in range(npop):
		nnpop[i] = int(ppercent[i]*nneuron)
	axrmu,aylmu  = np.zeros((npop,nrank)),np.zeros((npop,nrank))
	sigxr,sigyl  = np.zeros((npop,nrank)),np.zeros((npop,nrank))
	sigcov       = np.zeros((npop,nrank,nrank))

	for irank in range(nrank):
		for ipop in range(npop):
			nns,nne = np.sum(nnpop[:ipop]),np.sum(nnpop[:ipop])+nnpop[ipop]
			nns = nns.astype(np.int32)
			nne = nne.astype(np.int32)
			axrmu[ipop,irank],aylmu[ipop,irank] = np.mean(xrvec[nns:nne,irank]),np.mean(ylvec[nns:nne,irank])
			sigxr[ipop,irank],sigyl[ipop,irank] = np.std(xrvec[nns:nne,irank])**2,np.std(ylvec[nns:nne,irank])**2

			# axrmu[ipop,irank],aylmu[ipop,irank] = np.mean(xrref[nns:nne,irank]),np.mean(ylref[nns:nne,irank])
			# sigxr[ipop,irank],sigyl[ipop,irank] = np.mean((xrvec[nns:nne,irank]-xrref[nns:nne,irank])**2),np.mean((ylvec[nns:nne,irank]-ylref[nns:nne,irank])**2)

	for ipop in range(npop):
		neuronpop = int(nnpop[ipop])
		nns,nne = np.sum(nnpop[:ipop]),np.sum(nnpop[:ipop])+nnpop[ipop]
		nns = nns.astype(np.int32)
		nne = nne.astype(np.int32)
		noisexr,noiseyl = np.zeros((neuronpop,nrank)),np.zeros((neuronpop,nrank))
		for irank in range(nrank):
			noisexr[:,irank],noiseyl[:,irank] = xrvec[nns:nne,irank],ylvec[nns:nne,irank]
        ##### @YX 07DEC -- AN ERROR
			noisexr[:,irank]-=axrmu[ipop,irank]## error axrmu[irank]
			noiseyl[:,irank]-=aylmu[ipop,irank]
		sigcov[ipop,:,:] = noiseyl.T@noisexr/neuronpop

	return(axrmu,aylmu,sigxr,sigyl,sigcov)


# %%
### Statistics of Gaussian random matrix 
### Gaussian Pparameters
gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(300)
gauss_points = gauss_points*np.sqrt(2)

def Phi(mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)
def PhiP(mu, delta0):
    integrand = (1.0+np.tanh(mu+np.sqrt(delta0)*gauss_points-shiftx))
    return gaussian_norm * np.dot (integrand,gauss_weights)

def derPhi(mu,delta0):
    # integrand = np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    #### @YX 0109 MODIFY THE FUNCTION -- SO THAT WE CAN GET THE SMOOTH RESULTS
    integrand = 2/(np.cosh(2*(mu+np.sqrt(delta0)*gauss_points))+1.0)
    return gaussian_norm * np.dot (integrand,gauss_weights)
def derPhiP(mu,delta0):
    # integrand = np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    #### @YX 0109 MODIFY THE FUNCTION -- SO THAT WE CAN GET THE SMOOTH RESULTS
    # integrand = 1/(np.cosh(shiftx-(mu+np.sqrt(delta0)*gauss_points))**2)
    integrand = 2/(1 + np.cosh(2*shiftx - 2*(mu+np.sqrt(delta0)*gauss_points)))
    # 1/(cosh^2(a - x))
    return gaussian_norm * np.dot (integrand,gauss_weights)

def innerdeuxPhi(mu,delta0):
    # integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    # return gaussian_norm * np.dot(integrand,gauss_weights)
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)
def innerdeuxPhiP(mu,delta0):
    # integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    # return gaussian_norm * np.dot(integrand,gauss_weights)
    integrand = 1+np.tanh(mu+np.sqrt(delta0)*gauss_points-shiftx)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

def innerdeuxderPhi(mu,delta0):
    # integrand = np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    integrand = 2/(np.cosh(2*(mu+np.sqrt(delta0)*gauss_points))+1.0)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)
def innerdeuxderPhiP(mu,delta0):
    # integrand = np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    # integrand = 1/(np.cosh(shiftx-(mu+np.sqrt(delta0)*gauss_points))**2)
    integrand = 2/(1 + np.cosh(2*shiftx - 2*(mu+np.sqrt(delta0)*gauss_points)))
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

### no correlation
## solve the dynamics using the consistency of kappa
def iidperturbation(x,JE,JI,gmat,nparams):
    # JE,JI,g0 = args
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    # print('gset:',gee,gei,gie,gii)
    if(len(nparams)==2):
        NE,NI= nparams[0],nparams[1]
    else:
        NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
        NE=NE1+NE2
    N = NE+NI
    muphi,delta0phiE,delta0phiI = x, x**2*(gee**2*NE/N+gei**2*NI/N)/(JE-JI)**2, x**2*(gie**2*NE/N+gii**2*NI/N)/(JE-JI)**2
    delta_kappa = -x+(JE*Phi(muphi,delta0phiE)-JI*Phi(muphi,delta0phiI))
    # delta_kappa = -x+(JE-JI)*(Phi(muphi,delta0phiE)+Phi(muphi,delta0phiI))/2.0 ## ERROR!!!!!
    return delta_kappa
def iidperturbationP(x,JE,JI,gmat,nparams):
    # JE,JI,g0 = args
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    # print('gset:',gee,gei,gie,gii)
    if(len(nparams)==2):
        NE,NI= nparams[0],nparams[1]
    else:
        NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
        NE=NE1+NE2
    N = NE+NI
    muphi,delta0phiE,delta0phiI = x, x**2*(gee**2*NE/N+gei**2*NI/N)/(JE-JI)**2, x**2*(gie**2*NE/N+gii**2*NI/N)/(JE-JI)**2
    delta_kappa = -x+(JE*PhiP(muphi,delta0phiE)-JI*PhiP(muphi,delta0phiI))
    # delta_kappa = -x+(JE-JI)*(Phi(muphi,delta0phiE)+Phi(muphi,delta0phiI))/2.0 ## ERROR!!!!!
    return delta_kappa

## solve the dynamics using the consistency of mu and sigma    
def iidfull_mudelta_consistency(x,JE,JI,gmat,nparams):
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    muxE,muxI     = x[0],x[1]
    sigx2E,sigx2I = x[2],x[3]
    if(len(nparams)==2):
        NE,NI= nparams[0],nparams[1]
    else:
        NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
        NE=NE1+NE2
    N = NE+NI
    inner2meanE,inner2meanI = innerdeuxPhi(muxE,sigx2E),innerdeuxPhi(muxI,sigx2I)
    PhimeanE,PhimeanI = Phi(muxE,sigx2E),Phi(muxI,sigx2I)
    consistency = [x[2]-(gee**2*inner2meanE*NE/N+gei**2*inner2meanI*NI/N),x[0]-(JE*PhimeanE-JI*PhimeanI),x[3]-(gie**2*inner2meanE*NE/N+gii**2*inner2meanI*NI/N),x[1]-(JE*PhimeanE-JI*PhimeanI)]
    return consistency
def iidfull_mudelta_consistencyP(x,JE,JI,gmat,nparams):
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    muxE,muxI     = x[0],x[1]
    sigx2E,sigx2I = x[2],x[3]
    if(len(nparams)==2):
        NE,NI= nparams[0],nparams[1]
    else:
        NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
        NE=NE1+NE2
    N = NE+NI
    inner2meanE,inner2meanI = innerdeuxPhiP(muxE,sigx2E),innerdeuxPhiP(muxI,sigx2I)
    PhimeanE,PhimeanI = PhiP(muxE,sigx2E),PhiP(muxI,sigx2I)
    consistency = [x[2]-(gee**2*inner2meanE*NE/N+gei**2*inner2meanI*NI/N),x[0]-(JE*PhimeanE-JI*PhimeanI),x[3]-(gie**2*inner2meanE*NE/N+gii**2*inner2meanI*NI/N),x[1]-(JE*PhimeanE-JI*PhimeanI)]
    return consistency

def iidR1_mudelta_consistency(x,JE,JI,gmat,nparams):
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    muxE,muxI     = x[0],x[1]
    sigx2E,sigx2I = x[2],x[3]
    if(len(nparams)==2):
        NE,NI= nparams[0],nparams[1]
    else:
        NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
        NE=NE1+NE2
    N = NE+NI
    PhimeanE,PhimeanI = Phi(muxE,sigx2E),Phi(muxI,sigx2I)
    consistency = [x[2]-(gee**2*NE/N+gei**2*NI/N)/(JE-JI)**2*(JE*PhimeanE-JI*PhimeanI)**2,x[0]-(JE*PhimeanE-JI*PhimeanI),x[3]-(gie**2*NE/N+gii**2*NI/N)/(JE-JI)**2*(JE*PhimeanE-JI*PhimeanI)**2,x[1]-(JE*PhimeanE-JI*PhimeanI)]
    return consistency

def iidR1_mudelta_consistencyP(x,JE,JI,gmat,nparams):
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    muxE,muxI     = x[0],x[1]
    sigx2E,sigx2I = x[2],x[3]
    if(len(nparams)==2):
        NE,NI= nparams[0],nparams[1]
    else:
        NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
        NE=NE1+NE2
    N = NE+NI
    PhimeanE,PhimeanI = PhiP(muxE,sigx2E),PhiP(muxI,sigx2I)
    consistency = [x[2]-(gee**2*NE/N+gei**2*NI/N)/(JE-JI)**2*(JE*PhimeanE-JI*PhimeanI)**2,x[0]-(JE*PhimeanE-JI*PhimeanI),x[3]-(gie**2*NE/N+gii**2*NI/N)/(JE-JI)**2*(JE*PhimeanE-JI*PhimeanI)**2,x[1]-(JE*PhimeanE-JI*PhimeanI)]
    return consistency

## solve the dynamics using the consistency of kappa (sym)
def symperturbation(x,JE,JI,gmat,nparams,etaset,eigvuse):
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    NE,NI = nparams[0],nparams[1]
    N = NE+NI
    npercent = nparams/N
    #### @YX modify 24_08_2021
    # sigmam2E,sigmam2I = gee**2*npercent[0]/(JE-JI)**2+gei**2*npercent[1]/(JE-JI)**2,gie**2*npercent[0]/(JE-JI)**2+gii**2*npercent[1]/(JE-JI)**2
    # delta0phiE,delta0phiI = x**2*sigmam2E,x**2*sigmam2I
    # muphiE,muphiI = x,x
    # delta_kappa = -x+JE*Phi(muphiE,delta0phiE)-JI*Phi(muphiI,delta0phiI) 
    # #### CORRECT
    # #### change etaset[1]--etaset[3/4] 12/01/22
    # sigmaE_term = (gee**2*JE*etaset[0]-gie*gei*JI*etaset[3])/(JE-JI)**2*derPhi(muphiE,delta0phiE)*x*npercent[0]
    # sigmaI_term = (gei*gie*JE*etaset[3]-gii**2*JI*etaset[5])/(JE-JI)**2*derPhi(muphiI,delta0phiI)*x*npercent[1]
    # delta_kappa = delta_kappa+(sigmaE_term+sigmaI_term)
    print(">>>>>>>>>>>>>>>>>>>>",eigvuse)
    sigmam2E,sigmam2I = gee**2*npercent[0]/eigvuse**2+gei**2*npercent[1]/eigvuse**2,gie**2*npercent[0]/eigvuse**2+gii**2*npercent[1]/eigvuse**2

    delta0phiE,delta0phiI = x**2*sigmam2E,x**2*sigmam2I
    muphiE,muphiI = x,x
    delta_kappa = -x+JE*Phi(muphiE,delta0phiE)-JI*Phi(muphiI,delta0phiI) 
    #### CORRECT
    #### change etaset[1]--etaset[3/4] 12/01/22
    sigmaE_term = (gee**2*JE*etaset[0]-gie*gei*JI*etaset[3])/eigvuse**2*derPhi(muphiE,delta0phiE)*x*npercent[0]
    sigmaI_term = (gei*gie*JE*etaset[3]-gii**2*JI*etaset[5])/eigvuse**2*derPhi(muphiI,delta0phiI)*x*npercent[1]
    delta_kappa = delta_kappa+(sigmaE_term+sigmaI_term)
    return delta_kappa
def symperturbationP(x,JE,JI,gmat,nparams,etaset,eigvuse):
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    NE,NI = nparams[0],nparams[1]
    N = NE+NI
    npercent = nparams/N
    eigvAm = JE-JI
    ## MIGHT CHANGE?
    sigmam2E,sigmam2I = gee**2*npercent[0]/eigvuse**2+gei**2*npercent[1]/eigvuse**2,gie**2*npercent[0]/eigvuse**2+gii**2*npercent[1]/eigvuse**2
    # ### INDIVIDUAL -- UNCHANGED
    # sigmam2E,sigmam2I = gee**2*npercent[0]/eigvAm**2+gei**2*npercent[1]/eigvAm**2,gie**2*npercent[0]/eigvAm**2+gii**2*npercent[1]/eigvAm**2

    delta0phiE,delta0phiI = x**2*sigmam2E,x**2*sigmam2I
    muphiE,muphiI = x,x
    delta_kappa = -x+JE*PhiP(muphiE,delta0phiE)-JI*PhiP(muphiI,delta0phiI) 
    sigmaE_term = (gee**2*JE*etaset[0]-gie*gei*JI*etaset[3])/eigvuse**2*derPhiP(muphiE,delta0phiE)*x*npercent[0]
    sigmaI_term = (gei*gie*JE*etaset[3]-gii**2*JI*etaset[5])/eigvuse**2*derPhiP(muphiI,delta0phiI)*x*npercent[1]
    delta_kappa = delta_kappa+(sigmaE_term+sigmaI_term)
    return delta_kappa

## solve the dynamics using the consistency of mu and sigma (sym)
def symfull_mudelta_consistency(x,JE,JI,g0):
    mux,sigx2 = x[0],x[1]
    inner2mean = innerdeuxPhi(mux,sigx2)
    Phimean = Phi(mux,sigx2)
    consistency = [x[1]-g0**2*inner2mean,x[0]-(JE-JI)*Phimean]
    return consistency

def symR1_mudelta_consistency(x,JE,JI,g0):
    mux,sigx2 = x[0],x[1]
    outer2mean = Phi(mux,sigx2)**2
    Phimean = Phi(mux,sigx2)
    consistency = [x[1]-g0**2*outer2mean,x[0]-(JE-JI)*Phimean]
    return consistency

def symsparse_kappa(x,gEI,JE,JI,EPSP,kparams,nparams,etass,lambda0):
    IPSP = gEI*EPSP
    kE,kI = kparams[0]*kparams[2],kparams[1]*kparams[2]
    if(len(nparams)==2):
        NE,NI= nparams[0],nparams[1]
    else:
        NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
        NE=NE1+NE2
    N = NE+NI
    p = kparams[0]*kparams[2]/nparams[0]

    ## sigma m2
    sigm2E,sigm2I = EPSP**2*kE*(1-p)/lambda0**2+IPSP**2*kI*(1-p)/lambda0**2,EPSP**2*kE*(1-p)/lambda0**2+IPSP**2*kI*(1-p)/lambda0**2
    sigm2E=sigm2E*(x**2)
    muE,muI = x,x
    meanphi=Phi(muE,sigm2E)
    derivphi=derPhi(muE,sigm2E)
    ZE = JE*(EPSP**2)*p*(NE*(etass[0]-p)-NI*gEI*(etass[1]-p))*derivphi*x 
    ZI = -JI*(EPSP**2)*gEI*p*(-NE*(etass[1]-p)+NI*gEI*(etass[2]-p))*derivphi*x 
    ## mean parts 
    kappa_curr = ZE/lambda0**2+ZI/lambda0**2
    kappa_curr +=(JE*meanphi-JI*meanphi)
    delta_kappa =x- kappa_curr
    return x-kappa_curr

def symsparse_kappaP(x,gEI,JE,JI,EPSP,kparams,nparams,etass,lambda0):
    IPSP = gEI*EPSP
    kE,kI = kparams[0]*kparams[2],kparams[1]*kparams[2]
    if(len(nparams)==2):
        NE,NI= nparams[0],nparams[1]
    else:
        NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
        NE=NE1+NE2
    N = NE+NI
    p = kparams[0]*kparams[2]/nparams[0]

    ## sigma m2
    sigm2E,sigm2I = EPSP**2*kE*(1-p)/lambda0**2+IPSP**2*kI*(1-p)/lambda0**2,EPSP**2*kE*(1-p)/lambda0**2+IPSP**2*kI*(1-p)/lambda0**2
    sigm2E=sigm2E*(x**2)
    muE,muI = x,x
    meanphi=PhiP(muE,sigm2E)
    derivphi=derPhiP(muE,sigm2E)
    ZE = JE*(EPSP**2)*p*(NE*(etass[0]-p)-NI*gEI*(etass[1]-p))*derivphi*x 
    ZI = -JI*(EPSP**2)*gEI*p*(-NE*(etass[1]-p)+NI*gEI*(etass[2]-p))*derivphi*x 
    ## mean parts 
    kappa_curr = ZE/lambda0**2+ZI/lambda0**2
    kappa_curr +=(JE*meanphi-JI*meanphi)
    delta_kappa =x- kappa_curr
    return x-kappa_curr


### functions compare/how variance and mean co-change under the condition with iid Random Perturbation
def iid_muCdelta_consistency(x,JE,JI,g0, sigx2):
    mux = x[0]
    Phimean = Phi(mux,sigx2)
    consistency = x[0]-(JE-JI)*Phimean
    return consistency


def randbin(M,N,P):  
    return np.random.choice([0, 1], size=(M,N), p=[P, 1-P])

#%%
def blocklikeEtasymmetry(Jmat,nparams,EPSP,IPSP):
    J,JT = Jmat.copy(),Jmat.copy().T
    if(len(nparams)==2):
        NE,NI= nparams[0],nparams[1]
    else:
        NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
        NE=NE1+NE2
    K = int(NE/NI)
    N = NE+NI
    ## ee
    AEE,AEET = np.squeeze(J[:NE,:NE])/EPSP,np.squeeze(JT[:NE,:NE])/EPSP
    etaEE = np.sum((AEE*AEET).flatten())/NE
    etaEE = etaEE/(np.sum((AEE*AEE).flatten())/NE)
    ## ei set
    etaEIset = np.zeros(K)
    for i in range(K):
        subJIE,subJEI = J[NE:,i*NI:(i+1)*NI]/EPSP,JT[NE:,i*NI:(i+1)*NI]/IPSP
        etaEIset[i]   = np.sum((subJIE*subJEI).flatten())/NI

        etaEIset[i]   = etaEIset[i]/(np.sum((subJIE*subJIE).flatten())/NI)
    ## ii
    AII,AIIT = np.squeeze(J[NE:,NE:])/IPSP,np.squeeze(JT[NE:,NE:])/IPSP
    etaII = np.sum((AII*AIIT).flatten())/NI
    etaII = etaII/(np.sum((AII*AII).flatten())/NI)
    
    return etaEE,etaEIset,etaII
def blocklikeEtasymmetry_binary(Jmat,nparams):
    J,JT = Jmat.copy(),Jmat.copy().T
    if(len(nparams)==2):
        NE,NI= nparams[0],nparams[1]
    else:
        NE1,NE2,NI = nparams[0],nparams[1],nparams[2]
        NE=NE1+NE2
    K = int(NE/NI)
    N = NE+NI
    ## ee
    AEE,AEET = np.squeeze(J[:NE,:NE]),np.squeeze(JT[:NE,:NE])
    etaEE = np.sum((AEE*AEET).flatten())/NE
    etaEE = etaEE/(np.sum((AEE*AEE).flatten())/NE)
    ## ei set
    etaEIset = np.zeros(K)
    for i in range(K):
        subJIE,subJEI = J[NE:,i*NI:(i+1)*NI],JT[NE:,i*NI:(i+1)*NI]
        etaEIset[i]   = np.sum((subJIE*subJEI).flatten())/NI

        etaEIset[i]   = etaEIset[i]/(np.sum((subJIE*subJIE).flatten())/NI)
    ## ii
    AII,AIIT = np.squeeze(J[NE:,NE:]),np.squeeze(JT[NE:,NE:])
    etaII = np.sum((AII*AIIT).flatten())/NI
    etaII = etaII/(np.sum((AII*AII).flatten())/NI)
    
    return etaEE,etaEIset,etaII


# %%
def generate_SymSpr_Mat(eta,prob,nsize):
    n1,n2=nsize[0],nsize[1]
    subProb = randbin(1,n1*n2,1-prob)
    # print("mean,var",np.mean(subProb),np.std(subProb)**2,prob*(1-prob))
    Jbsub= np.reshape(subProb,(n1,n2))
    JbsubT = Jbsub.copy().T

    idxTsym,idyTsym = np.where(JbsubT>0)
    nnonzero=len(idxTsym)
    # print('nonzero:',nnonzero)
    cutIdx = int(nnonzero*eta)
    # print("fnum:",cutIdx)
    Idxuse=random.sample(range(0,nnonzero),cutIdx)
    idxT,idyT = idxTsym[Idxuse],idyTsym[Idxuse]
    JbsubT_=np.zeros((n2,n1))
    JbsubT_[idxT,idyT]=1
    ## compensate
    idxTinv,idyTinv = np.where(JbsubT<1)
    zeros=len(idxTinv)
    cutIdxinv = (nnonzero-cutIdx)
    print("should / available snum:",cutIdxinv, zeros)
    if zeros<cutIdxinv:
        cutIdxinv = zeros#int(nnonzero*(1-eta))
    # print("snum:",cutIdxinv)
    # print("sumnum:",cutIdx+cutIdxinv,n1*n2*prob)
    Idxinvuse=random.sample(range(0,zeros),cutIdxinv)
    idxinvT,idyinvT = idxTinv[Idxinvuse],idyTinv[Idxinvuse]
    JbsubT_[idxinvT,idyinvT]=1 ## complementary 
    # print("IJ_mean,var",np.mean(np.squeeze(JbsubT_).flatten()),np.std(np.squeeze(JbsubT_).flatten())**2,prob*(1-prob))
    return Jbsub,JbsubT_ 

def generate_SymSpr_Mat_trial(Jiid,eta,prob,nsize):
    n1,n2=nsize[0],nsize[1]
    # subProb = randbin(1,n1*n2,1-prob)
    # # print("mean,var",np.mean(subProb),np.std(subProb)**2,prob*(1-prob))
    # Jbsub= np.reshape(subProb,(n1,n2))
    Jbsub = Jiid.copy()
    JbsubT = Jbsub.copy().T

    idxTsym,idyTsym = np.where(JbsubT>0)
    nnonzero=len(idxTsym)
    # print('nonzero:',nnonzero)
    cutIdx = int(nnonzero*eta)
    # print("fnum:",cutIdx)
    Idxuse=random.sample(range(0,nnonzero),cutIdx)
    idxT,idyT = idxTsym[Idxuse],idyTsym[Idxuse]
    JbsubT_=np.zeros((n2,n1))
    JbsubT_[idxT,idyT]=1
    ## compensate
    idxTinv,idyTinv = np.where(JbsubT<1)
    zeros=len(idxTinv)
    cutIdxinv = (nnonzero-cutIdx)
    print("should / available snum:",cutIdxinv, zeros)
    if zeros<cutIdxinv:
        cutIdxinv = zeros#int(nnonzero*(1-eta))
    # print("snum:",cutIdxinv)
    # print("sumnum:",cutIdx+cutIdxinv,n1*n2*prob)
    Idxinvuse=random.sample(range(0,zeros),cutIdxinv)
    idxinvT,idyinvT = idxTinv[Idxinvuse],idyTinv[Idxinvuse]
    JbsubT_[idxinvT,idyinvT]=1 ## complementary 
    # print("IJ_mean,var",np.mean(np.squeeze(JbsubT_).flatten()),np.std(np.squeeze(JbsubT_).flatten())**2,prob*(1-prob))
    return Jbsub,JbsubT_ 

def generate_SymSpr_MatEI(eta,prob,nsize):
    n1,n2=nsize[0],nsize[1]
    probie,probei=prob[0],prob[1]
    subProb = randbin(1,n1*n2,1-probie)
    Jbsub   = np.reshape(subProb,(n1,n2)) #subIE
    JbsubT  = Jbsub.copy().T #subEI

    idxTsym,idyTsym = np.where(JbsubT>0)
    nnonzero=len(idxTsym)
    cutIdx = int(nnonzero*eta)
    Idxuse=random.sample(range(0,nnonzero),cutIdx)
    idxT,idyT = idxTsym[Idxuse],idyTsym[Idxuse]
    JbsubT_=np.zeros((n2,n1))
    JbsubT_[idxT,idyT]=1
    ## compensate
    idxTinv,idyTinv = np.where(JbsubT<1)
    zeros=len(idxTinv)
    cutIdxinv = (nnonzero-cutIdx)
    if zeros<cutIdxinv:
        cutIdxinv = zeros#int(nnonzero*(1-eta))
    Idxinvuse=random.sample(range(0,zeros),cutIdxinv)
    idxinvT,idyinvT = idxTinv[Idxinvuse],idyTinv[Idxinvuse]
    JbsubT_[idxinvT,idyinvT]=1 ## complementary 
    return Jbsub,JbsubT_  

def generate_SymSpr_MatEI_trial(Jiid,eta,prob,nsize):
    n1,n2=nsize[0],nsize[1]
    probie,probei=prob[0],prob[1]
    # subProb = randbin(1,n1*n2,1-probie)
    # Jbsub   = np.reshape(subProb,(n1,n2)) #subIE
    Jbsub = Jiid.copy()
    JbsubT  = Jbsub.copy().T #subEI

    idxTsym,idyTsym = np.where(JbsubT>0)
    nnonzero=len(idxTsym)
    cutIdx = int(nnonzero*eta)
    Idxuse=random.sample(range(0,nnonzero),cutIdx)
    idxT,idyT = idxTsym[Idxuse],idyTsym[Idxuse]
    JbsubT_=np.zeros((n2,n1))
    JbsubT_[idxT,idyT]=1
    ## compensate
    idxTinv,idyTinv = np.where(JbsubT<1)
    zeros=len(idxTinv)
    cutIdxinv = (nnonzero-cutIdx)
    if zeros<cutIdxinv:
        cutIdxinv = zeros#int(nnonzero*(1-eta))
    Idxinvuse=random.sample(range(0,zeros),cutIdxinv)
    idxinvT,idyinvT = idxTinv[Idxinvuse],idyTinv[Idxinvuse]
    JbsubT_[idxinvT,idyinvT]=1 ## complementary 
    return Jbsub,JbsubT_  