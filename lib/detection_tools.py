
"""
Module containing various tools, to load/create simulations and to measure 
tests performances.

:author: Jean-Baptiste Courbot - jean-baptiste.courbot@univ-lyon1.fr
:date: november 16, 2015


"""
import numpy as np 
from  scipy.misc import imresize
from scipy.signal import convolve2d
import sys 
sys.path.insert(0,'../HEOLTH_1.3')

import detection_preprocessing as dp

def get_simu(fileloc,lambda_max = 30):
    """ 
    Load a simulation and spectrally truncate it
    
    :param str fileloc: Path to file.
    :param int lambda_max: Maximal wawelenght to cut the datacube.   
    
    """
    
    # Get the data
    simu=np.load(fileloc)
    
    # Various parts of simulations.
    source_gal = simu['gal']
    source_halo = simu['halo'] 
    bruit_pose = (simu['cube_bruit']); 
    
    # Truncation within (0,lambda_max)
    bruit_pose = bruit_pose[:,:,:lambda_max,:]
    source_gal = source_gal[:,:,:lambda_max]
    source_halo = source_halo[:,:,:lambda_max]
    
    #Generationg binary ground truth:
    gal = simu['Im_gal']    ; gal = gal > gal.mean() ;
    halo = simu['Im_halo']  ; halo = halo > halo.mean() ;
    
    # Ensuring that the simulations are set to zero outside of the binary ground truth:
    source_gal = source_gal * gal[:,:,np.newaxis]
    source_halo = source_halo * halo[:,:,np.newaxis]
    
    return simu, gal, source_gal, halo, source_halo, bruit_pose 



def gen_signaux_snr(objet_principal, source_gal,source_halo, bruit_pose, SNR, simu,convolve=0):
    """ 
    Generates simulated noisy datacubes containing a "galaxy" and a "halo" according to a SNR range.
    
    :param str objet_principal: ['halo'|'galaxy'] Indicates with respect to what object the SNR is evaluated.
    :param ndarray source_gal:  Noiseless datacube containing only the "galaxy". Shape: (spatial, spatial, spectral).
    :param ndarray source_halo:  Noiseless datacube containing only the "halo". Shape: (spatial, spatial, spectral).
    :param ndarray bruit_pose: Noise, over multiple observations, to be added. Shape: (spatial, spatial, spectral, observation).
    :param float SNR: Signal-to-Noise Ratio range to use.
    :param struct simu: Simulation data coming from np.load(...). Could be used to retrieve parameters 2,3,4.
    :param bool convolve: Indicates if the source data should be convolved or not.
    
    
    """
    
    # Useful informations
    S = source_gal.shape[0]
    W = source_gal.shape[2]
    P = bruit_pose.shape[3]
    c = int(S/2)
    FWHM = 0.66*1/0.2
    beta=2.6
    
    # FSF generation (if concolving is considered).
    F = dp.Moffat(11, FWHM,beta)    
    
    if objet_principal == 'gal':
        #Galaxy resampling to have it spread over a large datacube region (twice bigger).
        
        xa = np.arange(S) ; ya = np.arange(S); za = np.arange(W)
        Xa,Ya,Za = np.meshgrid(xa,ya,za)
        
        xb = np.arange(2*S) ; yb = np.arange(2*S); zb = np.arange(W)
        Xb,Yb,Zb = np.meshgrid(xb,yb,zb)
        
        source_gal = simu['gal']
        
        src = (source_gal)
        
        src_interp = np.zeros(shape=(2*S,2*S,W))
        for w in range(W):
            src_interp[:,:,w] =imresize(src[:,:,w],(2*S,2*S))
        
            
        src2 = src_interp[25:75,25:75,:]
        
        source_gal = src2
        gal = simu['Im_gal'] ;   gal = imresize(gal,(2*S,2*S)); gal = gal > gal.mean() ; gal = gal[25:75,25:75]
        
        source_gal = source_gal * gal[:,:,np.newaxis]
        

        s = source_gal[c,c,:]
    
    else:
        gal = simu['Im_gal'] ; gal = gal > gal.mean()
        s = source_halo[c,c,:]
        
    # Estimated covariance matrix.    
    sigma = np.cov(np.reshape(bruit_pose.mean(axis=3),(S*S,W)),rowvar=0)
    
    # Observed SNR
    snr_init = 10.*np.log10((np.linalg.norm(s)**2)/np.trace(sigma))
    
    # Convolving, if needed.
    if convolve !=0:
        for w in range(W):
            source_halo[:,:,w] = convolve2d(source_halo[:,:,w],F,mode='same')
            source_gal[:,:,w] = convolve2d(source_gal[:,:,w],F,mode='same')
        
    # All datacube at all observations, at all SNR.
    Y_tout_snr = np.zeros(shape=(S,S,W,P,SNR.size))
    
    for i in range(SNR.size):
        # Differential SNR calculus:
        snr_cible = SNR[i]
        snr_diff = snr_cible-snr_init
    
        # Differential SNR, in dB:
        snr_diff_dec = np.sqrt(10.**(snr_diff/10.))
        
        # Actual mixing of galaxy, halo, noise.
        Y_tout_snr[:,:,:,:,i] = snr_diff_dec*(source_halo[:,:,:,np.newaxis] + 0.1*source_gal[:,:,:,np.newaxis]) + bruit_pose
        
    return Y_tout_snr,snr_diff_dec

  
def qualif_test(carte, verite, nb_elt):
    """
    Performance measure on an unmasked region.
    
    :param ndarray carte: Continuous statistic map
    :param bool,ndarray verite: True map.
    :param int  nb_elt: Number ot thresholding to make.
    
    :returns: **PFA** (*float*) - Estimated probability of false alarm.
    :returns: **PDET** (*float*) - Estimated probability of detection.
    :returns: **Ksi** (*float*) - Corresponding threshold.
    """
    debut = carte.min()
    
    fin = carte.max()
    
    pas = float((fin-debut))/float(nb_elt)
    
    PFA = np.zeros(shape=(nb_elt))
    PD = np.zeros(shape=(nb_elt))
    Ksi = np.zeros(shape=(nb_elt))
    
    for e in range(nb_elt):
        Ksi[e] = debut + e * pas
        
        carte_seuil = carte > Ksi[e]
        
        PFA[e] = np.mean((carte_seuil!=0)*(verite==0))/np.mean((verite==0))
        PD[e] = np.mean((carte_seuil!=0)*(verite==1))/np.mean((verite==1))
    

    return PFA,PD,Ksi
    
    
    
def qualif_test_halo(carte,init, verite, nb_elt):
    """
    Performance measure on a masked region.
    
    :param ndarray carte: Continuous statistic map
    :param ndarray init: Region to ignore in evaluation.
    :param bool,ndarray verite: True map.
    :param int  nb_elt: Number ot thresholding to make.
    
    :returns: **PFA** (*float*) - Estimated probability of false alarm.
    :returns: **PDET** (*float*) - Estimated probability of detection.
    :returns: **Ksi** (*float*) - Corresponding threshold.
    """

    
    debut = carte.min()
    fin = (carte*(init==0)).max()
    
    pas = float((fin-debut))/float(nb_elt)

    PFA = np.zeros(shape=(nb_elt))
    PD = np.zeros(shape=(nb_elt))
    Ksi = np.zeros(shape=(nb_elt))
    

    
    for e in range(nb_elt):
        Ksi[e] = debut + e * pas
        
        carte_seuil = ((carte > Ksi[e]) + (init==1))>0
        
        PFA[e] = np.ma.mean((carte_seuil!=0)*(verite==0)*(init==0))/np.ma.mean((verite==0)*(init==0))
        PD[e] = np.ma.mean((carte_seuil!=0)*(verite==1)*(init==0))/np.ma.mean((verite==1)*(init==0))
    

    return PFA,PD,Ksi    
