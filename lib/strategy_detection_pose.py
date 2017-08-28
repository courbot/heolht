"""

This module gather the main package functions one can use elsewhere, *e.g.* in the demo.


:author: Jean-Baptiste Courbot - jean-baptiste.courbot@univ-lyon1.fr
:date: november 02, 2015
"""


import numpy as np 
import scipy.stats as st
import scipy.ndimage.morphology as morph

import sys 
sys.path.insert(0,'../lib')
import initial_detection_pose as idp
import extended_detection_pose as edp



def detection_strategy(params):
    """ Complete Detection Strategy.
    
    The method relies on:

    #. an initial bright source unsupervised detection. 
    #. a faint source detection accounting for similarity with the initial detection. 

    :param ndarray Y: Hyperspectral, multiple observation datacube. Shape: (spatial,spatial,spectral,observation).    
    :param int P: number of observation
    :param bool diag: set if the estimated covariance matrix sould be constrained to be diagonal
    :param float pfa_bright: target false alarm for the method
    :param float FWHM: Full Width at Half Maximum for the spatial FSF, in pixels.
    :param int taille_f: size of the FSF window, in pixels.
    :param float beta: parameter for the FSF description.
    :param ndarray centre: spatial location of the source of interest.
    
    
    
    """
    
    # The initial detection may be provided
    if hasattr(params,'X_init')==False:    
        X_init, val_init = detection_initiale_pose(params)
        
        params.X_init = X_init
        
    else:
        X_init = params.X_init
        val_init=0
        
    taille_f = params.taille_f
    X_init_dilat = morph.binary_dilation(X_init,structure=np.ones((taille_f,taille_f)))
    params.X_sig = X_init_dilat
    
    X_ext_1, val_ext_1, ests = detection_etendue_pose(params)    
    

    
    return X_ext_1, val_ext_1, ests, X_init, val_init


def detection_initiale_pose(params):
    """ Initial Bright Source Detection.
    
    The detection is made through a sparsity-constrained GLR test, accounting 
    for the spatial neighborhood through the use of the spatial Field Spread 
    Function (FSF).
    
    See Paris, S.; Mary, D.; Ferrari, A., "Detection Tests Using Sparse Models,
    With Application to Hyperspectral Data," in *Signal Processing, IEEE 
    Transactions on* , vol.61, no.6, pp.1481-1494, March 15, 2013
    doi: 10.1109/TSP.2013.2238533
    

    :param ndarray Y: Hyperspectral, multiple observation datacube. Shape: (spatial,spatial,spectral,observation).    
    :param int P: number of observation
    :param bool diag: set if the estimated covariance matrix sould be constrained to be diagonal
    :param float pfa_bright: target false alarm for the method
    :param float FWHM: Full Width at Half Maximum for the spatial FSF, in pixels.
    :param int taille_f: size of the FSF window, in pixels.
    :param float beta: parameter for the FSF description.
    :param ndarray centre: spatial location of the source of interest.

    :returns: **X_init** *(bool image)* - binary initial detection map
    :returns: **val_init** *(float image)* - continuous initial detection map (test statistic map)
    """
    # Some useful parameters:
    S = params.Y.shape[0]

    marge = int(params.taille_f/2)
    
    # GLR with sparsity constraints:
    #x,val_init = idp.GLRT_1s_pose(Y,FWHM=FWHM,taille_f=taille_f, beta=beta, pfa = pfa_bright, P=P,ksi = 5, approx=diag )
    x,val_init = idp.GLRT_1s_pose(params)#params.Y, params.P, params.diag, params.pfa_bright, params.FWHM, params.taille_f, params.beta, ksi = 5 )
    
    
    # threshold rough evaluation:
    mean_est = np.median(val_init[~np.isnan(val_init)])
    std_est = np.std(val_init[~np.isnan(val_init)])
    
    ksi_init = st.norm.isf(params.pfa_bright, loc=mean_est,scale = std_est)
    
    # casting the results in the good shape.
    val_init_new = np.zeros(shape=(S,S)) ; val_init_new[marge:S-marge,marge:S-marge] = val_init ; val_init = val_init_new
    
    X_init = np.zeros_like(val_init) +np.nan

    X_init[~np.isnan(val_init)] = val_init[~np.isnan(val_init)] > ksi_init

    # Removal of pixels non-connex to the central ones.
    center = params.centre
    seed = np.zeros_like(X_init) ; 
    seed[center[0], center[1]] = 1; 
    seed[center[0]+1, center[1]] = 1; seed[center[0]-1, center[1]] = 1; 
    seed[center[0], center[1]+1] = 1; seed[center[0], center[1]-1] = 1; 
        
    X_init_new = morph.binary_propagation(seed, mask = X_init); X_init = X_init_new
    
    return X_init, val_init

def detection_etendue_pose(params):
    """ Extended Faint Source Detection.
    
    The detection is made through a sparsity-constrained GLR test, accounting 
    for the spatial neighborhood through the use of the spatial Field Spread 
    Function (FSF).
    
    The threshold :math:`\\xi` is set and provide a target maximum probability of false alarm. 
    
    See Courbot, J.-B.; Mazet, V.; Monfrini,E.; Collet, C, *Detection of Faint 
    Extended Sources in Hyperspectral Data and Application to HDF-S MUSE 
    Observation*, ICASSP 2016, *Submitted*.

    :param ndarray Y: Hyperspectral, multiple observation datacube. Shape: (spatial,spatial,spectral,observation).    
    :param bool_ndarray X_init: Initial detection map for the reference estimation.    
    :param bool_ndarray X_sig: map on which the covariance matrix is estimated (may be identical to X_init).
    :param int P: number of observation.
    :param bool diag: set if the estimated covariance matrix sould be constrained to be diagonal.
    :param float pfa_faint: target maximum false alarm for the method.
    :param float FWHM: Full Width at Half Maximum for the spatial FSF, in pixels.
    :param int taille_f: size of the FSF window, in pixels.
    :param float beta: parameter for the FSF description.
    

    :returns: **X_ext** *(bool image)* - binary extended detection map
    :returns: **val_ext** *(float image)* - continuous extended detection map (test statistic map)
    """
    
    # Some useful parameters:
    S = params.S
    marge = params.marge

    # Threshold corresponding to the maximum false alarm probability
    if params.confident == False:
        df =  (params.X_init==1).sum()
        
    elif params.confident==True:
        df =  (params.X_init==1).sum() * params.W#
        
    params.ksi_ext = st.chi2.isf(params.pfa_faint, df = df) 
        
    
    x,val_ext,ests = edp.GLR_as_pose(params)

    # casting the results in the good shape.
    val_ext_new = np.zeros(shape=(S,S))+np.nan ; val_ext_new[marge:S-marge,marge:S-marge] = val_ext ; val_ext = val_ext_new
    
    # actual thresholding
    X_ext = np.zeros_like(val_ext)

    X_ext[~np.isnan(val_ext)] = val_ext[~np.isnan(val_ext)] > params.ksi_ext
    
    return X_ext, val_ext, ests
    
