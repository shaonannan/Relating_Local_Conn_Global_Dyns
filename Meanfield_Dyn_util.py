'''
Mean-field Dyn
@YX 1210 Edit
'''
import numpy as np
import matplotlib.pylab as plt
import matplotlib
from numpy import linalg as la
from scipy import linalg as scpla
import scipy
# import seaborn as sb
from cmath import *
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


### Statistics of Gaussian random matrix 
### Gaussian Pparameters
gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(300)
gauss_points = gauss_points*np.sqrt(2)

def Phi(mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def derPhi(mu,delta0):
    # integrand = np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    #### @YX 0109 MODIFY THE FUNCTION -- SO THAT WE CAN GET THE SMOOTH RESULTS
    integrand = 2/(np.cosh(2*(mu+np.sqrt(delta0)*gauss_points))+1.0)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def innerdeuxPhi(mu,delta0):
    # integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    # return gaussian_norm * np.dot(integrand,gauss_weights)
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

def innerdeuxderPhi(mu,delta0):
    # integrand = np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    integrand = 2/(np.cosh(2*(mu+np.sqrt(delta0)*gauss_points))+1.0)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

### no correlation
## solve the dynamics using the consistency of kappa
def iidperturbation(x,JE,JI,gmat,nparams):
	gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
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
## @YX 3009 ADD---following, calculate the variance 
def iid_theoDyn_Stats(kappa,JE,JI,gmat,nparams):
    # JE,JI,g0 = args
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    # print('gset:',gee,gei,gie,gii)
    NE,NI = nparams[0],nparams[1]
    N = NE+NI
    muphi,delta0phiE,delta0phiI = kappa, kappa**2*(gee**2*NE/N+gei**2*NI/N)/(JE-JI)**2, kappa**2*(gie**2*NE/N+gii**2*NI/N)/(JE-JI)**2
    return muphi,muphi,delta0phiE,delta0phiI
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