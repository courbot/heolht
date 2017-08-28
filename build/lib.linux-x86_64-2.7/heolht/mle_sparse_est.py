"""
MLE estimation from a datacube, assuming the signals are 1-sparse.

:author: Jean-Baptiste Courbot - jean-baptiste.courbot@univ-lyon1.fr
:date: november 18 2015
"""
import numpy as np 
import sys 

sys.path.insert(0,'../lib')

import detection_preprocessing as dp



def mle_sparse_estimate(liste_vec, D_3d_unip):
    """ 
    Maximum Likelihood estimations for the provided vectors, under a 
    1-sparse assumption. This function will be used in the ML estimation 
    add-on.
    
    :param ndarray liste_vec: hyperspectral datacube arranged to form an array. Shape: (spatial*spatial, spectral).
    :param ndarray D_3d_unip: dictionary, considering spatial (FSF-weighted-) neighborhood, evaluated on 1 observation.
    """


    Prod = np.dot(liste_vec, D_3d_unip)      

    Ind = np.argmax(Prod,axis=1)            

    Valmax = np.amax(Prod, axis=1)       

    ind_est = Ind
    weight_est = Valmax
    
    return ind_est,weight_est
    
def get_sparse_estimate(params):
    """ Calculate the best-fit dictionnary indices.
    
    :param ndarray Y: Hyperspectral, multiple observation datacube. Shape: (spatial,spatial,spectral,observation). 
    :param ndarray F: Field Spread Function, windowed in a square.
    :param ndarray D: Dictionary to use for MLE fits.
    """
    if params.Y.ndim ==4:
        Y = params.Y.mean(axis=3)
    else:
        Y = params.Y

    # Dictionnary
    D = params.D
    # Field Spread Function
    F = params.F

    Y_3d, D_3d = dp.replique_3d_pose(Y, F,D, P=1) 
    
    liste_vec = Y_3d.reshape(Y_3d.shape[0]**2, Y_3d.shape[2])
    
    ind_est,weight_est = mle_sparse_estimate(liste_vec,D_3d)
    
    #%%
    
    im_ind_est = np.reshape(ind_est,(Y_3d.shape[0], Y_3d.shape[1])).astype(float)
    im_weight_est = np.reshape(weight_est,(Y_3d.shape[0], Y_3d.shape[1])).astype(float)
    
    im_ind_est[np.isnan(Y_3d).all(2)] = np.nan
    im_weight_est[np.isnan(Y_3d).all(2)] = np.nan

    return im_ind_est,im_weight_est
   
def get_moments(im_ind_est,im_weight_est, D,W,pas,pas_spectral):
    """
    Calculate the 0,1,2 moments from indices arrays
    
    :param ndarray im_ind_est: estimated indices array. Shape:(spatial,spatial).
    :param ndarray im_weight_est: corresponding weights. Shape: (spatial,spatial).
    :param ndarray D: Dictionary used for MLE fits.
    :param int W: spectra length.
    :param float pas: width sampling used for the dictionnary creation.
    :param float pas_spectral: Angstrom/bandwidth.
    """
    s_new = im_ind_est.shape[0]
    flux = np.ones_like(im_ind_est)
    for i in range(s_new):
        for j in range(s_new):
            if ~np.isnan(im_ind_est[i,j]):
                flux[i,j] = D[:,im_ind_est[i,j]].sum() * im_weight_est[i,j]
            else:
                flux[i,j] = np.nan
    
    
    width = np.floor((im_ind_est/W))*pas * pas_spectral * 2 * np.sqrt(2*np.log(2))
    width[np.isnan(flux)] = np.nan    
    
    position = (im_ind_est%W)
    position[np.isnan(flux)] = np.nan

    return flux,position,width
