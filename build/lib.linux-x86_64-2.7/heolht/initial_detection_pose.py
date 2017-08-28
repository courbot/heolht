"""
This module contains the sources for the bright unsupervised detection.

The detection is made through a sparsity-constrained GLR test, accounting 
for the spatial neighborhood through the use of the spatial Field Spread 
Function (FSF).

See Paris, S.; Mary, D.; Ferrari, A., "Detection Tests Using Sparse Models,
With Application to Hyperspectral Data," in * Signal Processing, IEEE 
Transactions on * , vol.61, no.6, pp.1481-1494, March 15, 2013
doi: 10.1109/TSP.2013.2238533

:author: Jean-Baptiste Courbot - jean-baptiste.courbot@univ-lyon1.fr
:date: november 06 2015

"""

import numpy as np 
import sys 
import scipy.linalg as la

sys.path.insert(0,'../lib')
import detection_preprocessing as nbp


def whitening(params):
    """
    Standard spectral whitening.
    
    :param ndarray Y: Hyperspectral, multiple observation datacube. Shape: (spatial,spatial,spectral,observation).    
    :param int S: spatial size (Y is supposed isotropic, *i.e.* square).
    :param int W: number of spectral band.
    :param int P: number of observation
    """
    S = params.S
    W = params.W 
    P = params.P
    
    Y = params.Y

    if hasattr(params,'Y_sig'):
        Y_sig = params.Y_sig
    else:
        if Y.ndim==4:
            Y_sig = params.Y.mean(axis=3)
        else:
            Y_sig = params.Y
    
    if P != 1 : 
    # When there are multiple observations, the whitening is done by observation.
        Y_tout_snr_blanc=np.zeros(shape=Y.shape)
        
        Y_src = np.mean(Y[:,:,:,:],axis=3)
        
        # Covariance estimation on the averaged datacube.
        liste_vec = np.reshape(Y_src,(S*S,Y_src.shape[2]))
           
        liste_vec_sig = np.reshape(Y_sig,(S*S,Y_sig.shape[2]))
        Sig_init = np.cov(liste_vec_sig[~np.isnan(liste_vec_sig).any(1)],rowvar=0)
        Sig_inv_dem = la.inv(la.sqrtm(Sig_init))        
        
        
        for p in range(P):
            Y_pose = Y[:,:,:,p]   
            
            # reshaping the datacube into an array.
            liste_vec = np.reshape(Y_pose,(S*S,Y_pose.shape[2]))
      
            # Actual whitening:
            liste_blanc = np.dot(liste_vec,Sig_inv_dem)

            # Reshaping the whitened array into a datacube.
            Y_tout_snr_blanc[:,:,:,p]  =  np.reshape(liste_blanc,(S,S,W))
            
        Y_src = np.reshape(Y_tout_snr_blanc[:,:,:,:],(S,S,W*P))
    else: 
        # One observation. The process is the same as above.
        if Y.ndim == 4:
            Y_src = np.mean(Y[:,:,:,:],axis=3)
        else :
            Y_src = Y
        
        liste_vec = np.reshape(Y_src,(S*S,Y_src.shape[2]))
        #liste_vec = liste_vec[~np.isnan(liste_vec).all(1)]   
        liste_vec_sig = np.reshape(Y_sig,(S*S,Y_sig.shape[2]))
        Sig_init = np.cov(liste_vec_sig[~np.isnan(liste_vec_sig).all(1)] ,rowvar=0)
        Sig_inv_dem = la.inv(la.sqrtm(Sig_init))
        
        liste_blanc = np.dot(liste_vec,Sig_inv_dem)
        Y_src = np.reshape(liste_blanc,(S,S,W))
        
    return Y_src

def GLRT_1s_pose(params):#Y, P, diag, pfa_bright, FWHM, taille_f, beta, ksi ):   
    
    """
    GLR test with 1-sparsity constraint.
    
    :param ndarray Y: Hyperspectral, multiple observation datacube. Shape: (spatial,spatial,spectral,observation).    
    :param int P: number of observation
    :param bool diag: set if the estimated covariance matrix sould be constrained to be diagonal
    :param float pfa_bright: target false alarm for the method
    :param float FWHM: Full Width at Half Maximum for the spatial FSF, in pixels.
    :param int taille_f: size of the FSF window, in pixels.
    :param float beta: parameter for the FSF description.
    :param float ksi: test threshold.
    
    :returns: **X** *(bool image)* - binary extended detection map
    :returns: **T** *(float image)* - continuous extended detection map (test statistic map)
    """
    #params.Y, params.P, params.diag, params.pfa_bright, params.FWHM, params.taille_f, params.beta, ksi = 5
    # Useful parameters
    S = params.S
    W = params.W 
    P = params.P
   
   
    # Dictionnary
    D = params.D#nbp.gen_dic(W, P=P)
    # Field Spread Function
    F = params.F#nbp.Moffat(taille_f, FWHM,beta)
    

    
    # 1) Data whitening
    Y_src = whitening(params) #ignoring nans
    
    # 2) Data reshaping for spatial, observation features.
    Y_3d, D_3d = nbp.replique_3d_pose(Y_src, F,D, P = P)    #keeping nans
        
    # 3) Actual detection
        
    # Observation reshaping (in a 2D array).    
    X = Y_3d.reshape((Y_3d.shape[0]**2, Y_3d.shape[2]))
    #X = X[~np.isnan(X).all(1)]    # removing nans
    
    
    nb_lambda = W 
    pas = params.taille_f**2
    
    
    if params.diag==False:
        # The covariance matrix is assumed block-diagonal.
        # Therefore, to avoid manipulating large array the calculus are done by
        # blocks.
        prod = 0
        sum_numer = 0
        sum_denom = 0

        for pl in range(P* nb_lambda):
            # Block beginning and end.
            deb = pl * pas      
            fin = (pl+1) * pas  
        
            # Arrays to be manipulated at this step.
            Y_pl = X[:,deb:fin] 
            D_pl = D_3d[deb:fin,:] 

            # Covariance matrix for the current block, and its inverse.            
            Sigma_pl = np.cov(Y_pl, rowvar=0)           
            Sigma_pl_inv = la.inv(Sigma_pl)
            
            Sig_inv_D = np.dot(Sigma_pl_inv,D_pl)
            
            # Numerators and denominators for the current block.
            numer_courant = (np.dot(Y_pl,Sig_inv_D))
            denom_courant =  np.diag((np.dot(D_pl.T,Sig_inv_D)) )           

            # Values are added to the total numerator, denominator values.
            sum_numer += numer_courant    
            sum_denom += denom_courant

        prod = (sum_numer**2)/sum_denom
        
    else:
        # The covariance matrix is supposed to be the identity matrix.
        numer = np.dot(X,D_3d)
        
        denom = np.diag(np.dot(D_3d.T,D_3d))
         
        prod = numer**2 / denom[np.newaxis,:]#(np.dot(X,D_3d))**2
         
    # The GLR is based on a search for the max value. This is done by:
    val = np.max(prod,axis=1)  
    val[np.isnan(X).all(1)] = np.nan
    # Decision, actual thresholding: 
    dec = np.zeros_like(val)
    dec[~np.isnan(val)] = val[~np.isnan(val)] > 1#ksi
    
    # Reshaping into original dimensions.
    marge = params.marge
    T = val.reshape((S-2*marge,S-2*marge)) 
    X = dec.reshape((S-2*marge,S-2*marge))    
    
    return X, T



