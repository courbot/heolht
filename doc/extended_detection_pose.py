
"""
Extended Source detection module.

The detection is made through a sparsity-constrained GLR test, accounting 
for the spatial neighborhood through the use of the spatial Field Spread 
Function (FSF).

The threshold :math:`\\xi` is set and provide a target maximum probability of false alarm. 

See Courbot, J.-B.; Mazet, V.; Monfrini,E.; Collet, C, *Detection of Faint 
Extended Sources in Hyperspectral Data and Application to HDF-S MUSE 
Observation*, ICASSP 2016, *Submitted*.


:author: Jean-Baptiste Courbot - jean-baptiste.courbot@univ-lyon1.fr
:date: november 16, 2015

"""



import numpy as np 

import numpy.ma as ma
import sys 
import scipy.linalg as la
import itertools

sys.path.insert(0,'../HEOLTH_1.3')

import detection_preprocessing as dp




def estimer_xb(liste_vec, liste_msk1d,D_3d_unip):
    """ 
    Maximum Likelihood estimations for the provided vectors, under a 
    1-sparse assumption. This function will be used in the ML estimation 
    add-on.
    
    :param ndarray liste_vec: hyperspectral datacube arranged to form an array. Shape: (spatial*spatial, spectral).
    :param ndarray liste_msk1d: masked region to process, arranged as an array. Shape: (spatial*spatial).
    :param ndarray D_3d_unip: dictionary, considering spatial (FSF-weighted-) neighborhood, evaluated on 1 observation.
    """
    
    S = np.sqrt(liste_vec.shape[0]) # cette grandeur est sans marge, donc varie selon taille_f

    Prod = np.dot(liste_vec, D_3d_unip)         # (2304, 200)

    Ind = np.argmax(Prod,axis=1)                # (2304,)

    Valmax = np.amax(Prod, axis=1)              # (2304,)

     # On boucle sur chacun des points de la region de detection initiale.
    range1 = np.arange(S*S)*(liste_msk1d==1)
    nb_test = (liste_msk1d==1).sum()

    X_b_tout = np.zeros(shape=(nb_test,liste_vec.shape[1]) ) 
    i = 0
    for s in itertools.ifilter(None,range1):

        X_b_tout[i,:] = D_3d_unip[:, Ind[s]] * Valmax[s]
        i += 1   
        
    return X_b_tout,Ind
    
def whitening_masked(Y,X_sig,F,S,W,P):
    """ 
    Spectral whitening while ignoring a masked region.
    
    :param ndarray Y: Hyperspectral, multiple observation datacube. Shape: (spatial,spatial,spectral,observation).    
    :param ndarray X_sig: region to ignore in the whitening. Shape: (spatial,spatial).
    :param ndarray F: FSF to consider in the 3D repliation process.
    :param int S: spatial size (Y is supposed isotropic, *i.e.* square).
    :param int W: number of spectral band.
    :param int P: number of observation
    
    """
    
    
    
    if P != 1 : 
    # When there are multiple observations, the whitening is done by observation.
        Y_tout_snr_blanc=np.zeros(shape=Y.shape)

        for p in range(P):
            Y_pose = Y[:,:,:,p]   

            # reshaping the datacube into an array.
            liste_vec = np.reshape(Y_pose,(S*S,Y_pose.shape[2]))
            liste_msk1d = np.reshape(X_sig, S*S)    
            liste_msk =  np.tile(liste_msk1d[:,np.newaxis],(1,W))
            
            liste_ext = np.ma.masked_array(liste_vec, np.invert(liste_msk==0))
            liste_ext = ma.compress_rowcols(liste_ext,axis=0)   
            
            # Covariance estimation on the averaged datacube.
            Sig_init = np.cov(liste_ext,rowvar=0)
            Sig_inv_dem = la.inv(la.sqrtm(Sig_init))
            
            # Actual whitening:
            liste_blanc = np.dot(liste_vec,Sig_inv_dem)
            
            # Reshaping the whitened array into a datacube.
            Y_tout_snr_blanc[:,:,:,p]  =  np.reshape(liste_blanc,(S,S,W))
            
        Y_src = np.reshape(Y_tout_snr_blanc[:,:,:,:],(S,S,W*P))
        
        
        # Now we process data averaged over multiple observations, which will 
        # used altogether with the multiple observation data.
        Y_src_unip = np.mean(Y[:,:,:,:],axis=3)
        
        liste_vec = np.reshape(Y_src_unip,(S*S,Y_src_unip.shape[2]))
        liste_msk1d = np.reshape(X_sig, S*S)    
        liste_msk =  np.tile(liste_msk1d[:,np.newaxis],(1,W))
        
        liste_ext = np.ma.masked_array(liste_vec, np.invert(liste_msk==0))
        liste_ext = ma.compress_rowcols(liste_ext,axis=0)            
        
        Sig_init = np.cov(liste_ext,rowvar=0)
        Sig_inv_dem = la.inv(la.sqrtm(Sig_init))
        liste_blanc = np.dot(liste_vec,Sig_inv_dem)
        Y_src_unip = np.reshape(liste_blanc,(S,S,W))        

        D_unip = dp.gen_dic(W, P=1)
        Y_3d_unip, D_3d_unip = dp.replique_3d_pose(Y_src_unip, F,D_unip, P = 1)
        D_3d_unip = D_3d_unip[:Y_3d_unip.shape[2],:] 
        
        liste_vec_unip = Y_3d_unip.reshape((Y_3d_unip.shape[0]**2, Y_3d_unip.shape[2]))
        
    else: 
        # One observation. The process is the same as above.
        Y_src = np.mean(Y[:,:,:,:],axis=3)
        
        liste_vec = np.reshape(Y_src,(S*S,Y_src.shape[2]))
        liste_msk1d = np.reshape(X_sig, S*S)    
        liste_msk =  np.tile(liste_msk1d[:,np.newaxis],(1,W))
        
        liste_ext = np.ma.masked_array(liste_vec, np.invert(liste_msk==0))
        liste_ext = ma.compress_rowcols(liste_ext,axis=0)            
        
        Sig_init = np.cov(liste_ext,rowvar=0)
        Sig_inv_dem = la.inv(la.sqrtm(Sig_init))
        liste_blanc = np.dot(liste_vec,Sig_inv_dem)
        Y_src = np.reshape(liste_blanc,(S,S,W))
        
        liste_vec_unip = 0    
        D_3d_unip = 0
        
    return Y_src, liste_vec_unip, D_3d_unip


def GLR_as_pose(Y, X_init, X_sig, P, diag, pfa_faint, FWHM, taille_f, beta, ksi ):   
    """ 
    GLR test with similarity/sparsity constraint.
    
    :param ndarray Y: Hyperspectral, multiple observation datacube. Shape: (spatial,spatial,spectral,observation).    
    :param ndarray X_init: Initial detection map to consider for similarity.
    :param ndarray X_sig: map to consider for covariance estimation. May be identical to X_init.
    :param int P: number of observation.
    :param bool diag: set if the estimated covariance matrix sould be constrained to be diagonal.
    :param float pfa_faint: target false alarm for the method.
    :param float FWHM: Full Width at Half Maximum for the spatial FSF, in pixels.
    :param int taille_f: size of the FSF window, in pixels.
    :param float beta: parameter for the FSF description.
    :param float ksi: test threshold.
    
    
    :returns: **X** *(bool image)* - binary extended detection map.
    :returns: **T** *(float image)* - continuous extended detection map (test statistic map).
    :returns: **ind_2d** *(int image)* - indices of the best-fitted atoms from the dictionary.
    """
    
    # Dimensions
    S = Y.shape[0]
    W = Y.shape[2]
    
    # Dictionnary
    D = dp.gen_dic(W, P=P)
    # Field Spread Function
    F = dp.Moffat(taille_f, FWHM,beta)
    
    marge = int(taille_f/2)
    
    # 1) Data whitening while ignoring a masked region:
    Y_src, liste_vec_unip, D_3d_unip = whitening_masked(Y,X_sig,F,S,W,P)
    
    # 2) Data reshaping for spatial, observation features.
    Y_3d, D_3d = dp.replique_3d_pose(Y_src, F,D, P = P)    

    # 3) Actual detection

    # reshaping into a 2D array of appropriate size
    liste_vec = Y_3d.reshape((Y_3d.shape[0]**2, Y_3d.shape[2]))
    liste_msk1d = np.reshape(X_init[marge:S-marge,marge:S-marge], (S-2*marge)**2)  

    # MLE estimates from the initially detected region
    if P ==1:
        X_b_tout,Ind = estimer_xb(liste_vec, liste_msk1d,D_3d)
    else:
        # The estimates are on the observation-averaged datacube.
        X_b_tout,Ind = estimer_xb(liste_vec_unip, liste_msk1d,D_3d_unip)
        
        # Estimates have to be reshaped:
        X_b_tout_new = np.zeros(shape=(X_b_tout.shape[0], P * X_b_tout.shape[1]))
        
        for p in range(P):
            X_b_tout_new[:,p::P] = X_b_tout
            
        X_b_tout = X_b_tout_new
        
        
    # Indices of MLE 1-soarse estimtes in the dictionary. 
    taille_ind = int(np.sqrt(Ind.shape[0]))
    ind_2d = np.reshape(Ind,(taille_ind,taille_ind)) 
    ind_2d_new = np.zeros(shape=(S,S)); ind_2d_new[marge:S-marge,marge:S-marge] = ind_2d; ind_2d = ind_2d_new



    # The covariance matrix is assumed (at least) block-diagonal.
    # Therefore, to avoid manipulating large array the calculus are done by
    # blocks.
    nb_lambda = W 
    pas = taille_f**2
    contrib = 0

    sum_denom = 0
    sum_numer = 0
    for pl in range(P* nb_lambda):
            # Block beginning and end.
            deb = pl * pas      
            fin = (pl+1) * pas  
            
            # Arrays to be manipulated at this step.
            Y_pl = liste_vec[:,deb:fin] 
            X_pl = X_b_tout[:,deb:fin] 

            if diag == True :
                # Here we assume the covariance matrix to be diagonal
                
                # Numerators and denominators for the current block.
                numer_courant = np.dot(Y_pl,  X_pl.T )

                denom_courant =  np.diag(np.dot(X_pl,  X_pl.T))
                
                
                # Values are added to the total numerator, denominator values.
                sum_numer += numer_courant
                sum_denom += denom_courant
                
            else :
                # Here we assume the covariance matrix is block diagonal.
                   
                # Covariance matrix for the current block, and its inverse.
                Sigma_pl = np.cov(Y_pl, rowvar=0)  
                Sigma_pl_inv = la.inv(Sigma_pl)
                
                Sig_inv_X = np.dot(Sigma_pl_inv,X_pl.T)
                
                # Numerators and denominators for the current block.
                numer_courant = np.dot(Y_pl,  Sig_inv_X )
                denom_courant = np.diag(np.dot(X_pl,  Sig_inv_X))
                
                # Values are added to the total numerator, denominator values.
                sum_numer += numer_courant
                sum_denom += denom_courant         
            
    
    # Total test statistic
    contrib = 0.5 * sum_numer**2 / sum_denom    
    
    # Here we sum over the bright initial spectra set.
    val = contrib.sum(axis = 1) 
    
    # Actual thresholding, decision:
    dec = val > ksi
    
    # Reshaping into original dimensions.
    marge = int(taille_f/2)
    T = val.reshape((S-2*marge,S-2*marge))
    X = dec.reshape((S-2*marge,S-2*marge))    
    
    return X, T, ind_2d

