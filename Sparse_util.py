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
