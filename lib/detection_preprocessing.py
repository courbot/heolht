"""
This module contains tools for the pre-processing otf the hyperspectral data before 
the detection tests.

:author: Jean-Baptiste Courbot - jean-baptiste.courbot@univ-lyon1.fr
:date: november 23, 2015
"""
import numpy as np 
from scipy.stats import norm

def gen_dic(W,pas=0.25, nb_ech=10, asym=0, P=1):
    """ 
    Spectral Gaussian line dictionnary generation with eventual assymettry.
    
    :param int W: Number of spectral band.
    :param float pas: Location step for the lines.
    :param int nb_ech: Number of lines positions.
    :param bool asym: Option to set the lines asymetry.
    
    :returns: **D** (*ndarray*) - Normalized dictionnary.
    """

    D = np.eye(W,nb_ech*W)
    
    sig = pas
    
    # sampling:
    ran = np.arange(W)
    for e in range(nb_ech):
        for col in range(W):       
            D[:,e*W+col] = norm.pdf(ran, col,(e+1)*sig)
            if asym == 1:
                D[:,e*W+col] = np.convolve(D[:,e*W+col],[-1, 1, 1],'same')
        # Normalization : the norm of one 'standard' columns is 1.          
        norm_ech = np.sqrt(np.sum(D[:,e*W+int(W/2)]**2))
        D[:,e*W:(e+1)*W] /= norm_ech
    
    # Dictionary replication, if multiple pose are considered:
    D = np.tile(D, (P,1))     

      
    return D

def Moffat(dim, FWHM,beta):
    """
    Moffat function windowed in a square. 
    :param int dim:      Square size
    :param float FWHM:     Full Width at Half Maximum of the function
    :param float beta:     Parameter of the Moffat function.
    
    :returns: **Moff** (*ndarray*) - Moffat function values.
    """
    demidim = np.floor(dim/2)
    X = np.tile(np.arange(-demidim,1+demidim)[:,np.newaxis],(1,dim)) ; Y = np.tile(np.arange(-demidim,1+demidim)[np.newaxis,:],(dim,1)) ; 
    R2 = X.astype(float)**2+Y.astype(float)**2


    alpha = FWHM/(2*np.sqrt(2**(1/beta)-1))
    Moff = (1 + R2/alpha**2)**(-beta) 
    
    return Moff

def replique_3d_pose(cube, F,D,P=1,interlace=False):  
    """
    Replication of an hyperspectral cube and dictionnary along a local neighborhood weighed by a given FSF.
    
    :param ndarray cube:     Hyperspectral cube to transform.
    :param ndarray F:        FSF to be used (e.g., windowed Moffat).
    :param ndarray D:        Dictionnary to replicate with the same process. 
    
    
    """
    S = cube.shape[1]
    W = cube.shape[2]
    
    taille_f = F.size
    largeur_f = F.shape[0]
    ordre = np.reshape(np.arange(0,taille_f), F.shape)
    marge = int(np.floor(float(largeur_f)/2))
    S_new = (S-2*marge)

    
    # vecteurs en liste
    liste_vec = np.zeros(shape=((S_new)*(S_new),taille_f*W))        
    
    D_3d = np.zeros(shape=(taille_f*D.shape[0],D.shape[1]))    
    
   
    for a in range(largeur_f):
        decal_x =  np.arange(-marge, largeur_f-marge)[a]
        for b in range(largeur_f):
            decal_y =  np.arange(-marge, largeur_f-marge)[b]
            
            # Position dans la liste des vecteurs
            pos = ordre[a,b]


            liste_vec[:,(pos)::(taille_f)] = np.reshape(cube[marge+decal_x:S-marge+decal_x,marge+decal_y:S-marge+decal_y,:], (S_new*S_new, W))#liste_vec_tout[pos,:,:]
            D_3d[pos::taille_f,:] = D * F[a,b]


    Y_3d = np.reshape(liste_vec,(S_new,S_new,taille_f*W))
           
    
    return Y_3d, D_3d
