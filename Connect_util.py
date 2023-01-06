'''
Connectivity
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

# generate gmat, same gavg
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

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

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

''' theoretical \lambda '''
def CubicLambda(jparams,nAm,mAm,eigvAm,gmat,etaset,nparams,): ## etaset includes coeffset,signset,
    NE,NI = nparams[0],nparams[1]
    N=NE+NI
    JE,JI = jparams[0],jparams[1]
    bxxt = np.eye(N)
    etaee,etaei,etaii = etaset[0],etaset[1],etaset[2]
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    #### @YX 0409 -- correct -- validate expression
    #### sigmaE 
    se = JE*gee**2*etaee-JI*gie*gei*etaei
    si = JE*gei*gie*etaei-JI*gii**2*etaii
    ppercent = nparams/N
    b2 = se*ppercent[0]+si*ppercent[1]
    coeff2 = [1,-eigvAm[0],0,-b2]  ## lambda_0 -- eigvAm
    ''' Cubic Equation '''
    roots = Cubic(coeff2)
    return (roots,b2/eigvAm[0]**2)

''' theoretical \lambda '''
def CubicLambda4(x,jparams,nAm,mAm,eigvAm,gmat,etaset,nparams,): ## etaset includes coeffset,signset,
    NE,NI = nparams[0],nparams[1]
    N=NE+NI
    JE,JI = jparams[0],jparams[1]
    bxxt = np.eye(N)
    etaee,etaei,etaii = etaset[0],etaset[1],etaset[2]
    gee,gei,gie,gii   = gmat[0],gmat[1],gmat[2],gmat[3]
    #### sigmaE 
    se = JE*gee**2*etaee-JI*gie*gei*etaei
    si = JE*gei*gie*etaei-JI*gii**2*etaii
    ppercent = nparams/N
    b2  = se*ppercent[0]+si*ppercent[1]
    RL2,IL2 = (x[0]**2-x[1]**2),(x[0]*x[1]*2) 
    L2 = RL2**2+IL2**2
    ### ~~~~~~~~ b4 ~~~~~~~~~~~~~~~~~~~~
    b4 = JE*(ppercent[0]*gee**2*etaee+ppercent[1]*gei*gie*etaei)**2-JI*(ppercent[0]*gei*gie*etaei+ppercent[1]*gii**2*etaii)**2
    RL4,IL4 = (x[0]**4+x[1]**4-6*x[0]**2*x[1]**2),(4*x[0]**3*x[1]-4*x[0]*x[1]**3)
    L4 = RL4**2+IL4**2

    realp=x[0]-eigvAm[0].real - b2*RL2/L2 -b4*RL4/L4 
    imagp=x[1]-eigvAm[0].imag + b2*IL2/L2 +b4*IL4/L4
    
    return [realp,imagp]

''' theoretical \lambda '''
def sigmaOverlap(jparams,nAm,mAm,eigvAm,gmat,etaset,nparams,): ## etaset includes coeffset,signset,
    NE,NI = nparams[0],nparams[1]
    N=NE+NI
    JE,JI = jparams[0],jparams[1]
    etaee,etaei,etaii = etaset[0],etaset[1],etaset[2]
    gee,gei,gie,gii = gmat[0],gmat[1],gmat[2],gmat[3]
    #### sigmaE 
    se = JE*gee**2*etaee-JI*gie*gei*etaei
    si = JE*gei*gie*etaei-JI*gii**2*etaii
    ppercent = nparams/N
    stt = se*ppercent[0]+si*ppercent[1]
    se,si,stt = se/eigvAm[0]**2,si/eigvAm[0]**2,stt/eigvAm[0]**2

    return (se,si,stt)

def TransitionEta(x,Jparams,nAm,mAm,eigvAm,gmat,signeta,Nparams,):
    etaset = x*np.ones(3)
    for i in range(3):
        etaset[i]*=signeta[i]
    root,_ = CubicLambda(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams,)
    return 1-root[0]
def TransitionOver(x,targetv,Jparams,nAm,mAm,eigvAm,gmat,signeta,Nparams,):
    etaset = x*np.ones(3)
    for i in range(3):
        etaset[i]*=signeta[i]
    root = sigmaOverlap(Jparams,nAm,mAm,eigvAm,gmat,etaset,Nparams,)
    #### @YX 3008 NOTICE -- we should use root[2] -- 0-E 1-I 2-overall
    return targetv-root[2]
def TransitionEta_spr(x, Jparams, nAm, mAm, eigvAm, gmat, coeffs, aprob, Nparams,):
    etaset = x*np.ones(3)
    etasetgau=np.zeros_like(etaset)
    for i in range(len(etaset)):
        etasetgau[i]=(etaset[i]*coeffs[i]-aprob)/(1-aprob)
    etasetgau[1]*=(-1)
    root, rt_ = CubicLambda(Jparams, nAm, mAm, eigvAm, gmat, etasetgau, Nparams,)
    return 1-root[0]#rt_-eigvAm[0]#


def TransitionOver_spr(x, targetv, Jparams, nAm, mAm, eigvAm, gmat, coeffs, aprob, Nparams,):
    etaset = x*np.ones(3)
    etasetgau = np.zeros_like(etaset)
    for i in range(len(etaset)):
        etasetgau[i]=(etaset[i]*coeffs[i]-aprob)/(1-aprob)
    etasetgau[1]*=(-1)
    root = sigmaOverlap(Jparams, nAm, mAm, eigvAm, gmat, etasetgau, Nparams,)
    # @YX 3008 NOTICE -- we should use root[2] -- 0-E 1-I 2-overall
    return targetv-root[2]

### suppl. finding the bifurcation point
def find_bifurcation(x,gmat,signeta,jparams,eigvAm,nparams,target):
    etaset = np.ones(3)*x
    for i in range(3):
        etaset[i]=etaset[i]*signeta[i]
    lambda_theo, _ = CubicLambda(jparams, 0, 0, eigvAm, gmat, etaset, nparams,)
    return np.real(lambda_theo[0])-target