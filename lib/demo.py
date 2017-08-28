"""
Package demonstration.



:author: Jean-Baptiste Courbot - jean-baptiste.courbot@univ-lyon1.fr
:date: november 18, 2015
"""


import numpy as np 
import matplotlib.pyplot as plt

import sys 
import numpy.ma as ma
import scipy.stats as st
import scipy.signal as si


sys.path.insert(0,'../lib')

import detection_tools as dt
import detection_preprocessing as dp
import strategy_detection_pose as sdp
import mle_sparse_est as mse
import parameters

if __name__=='__main__':
    plt.close('all')
    # Dimensions 
    S = 50
    W = 30
    P = 54
    centre = np.array([int(S/2), int(S/2)])
    
    # FSF parameters:
    FWHM = 0.66*1/0.2
    beta=2.6
    taille_f = 1
    marge = int(taille_f/2)
    
    #Physical parameters
    lambda_0 = 4750    # see Bacon et al, 2015
    lambda_lya = 1216
    pas_spectral = 1.25 # Angstrom / spectral band
    
    #Dictionary parameters
    beta=2.6
    pas = 0.15
    nb_ech_dic = 50


    D = dp.gen_dic(W, pas = pas, nb_ech=nb_ech_dic,asym=0)
    F = dp.Moffat(taille_f, FWHM,beta)
    
    
    
    SNR = np.array([-7])
    
    
    F_range = np.array([1,3,5,7,9,11])
#    taille_f = F_range[-1]
    marge = int(taille_f/2)
    
    test_simu = 1
    
    
    pfa_bright = 0.001
    pfa_faint = 0.0001    
    
    if test_simu ==1:
        simu, gal, source_gal, halo, source_halo,bp = dt.get_simu('../data/simu_H-3.npz',lambda_max = 30)
        bruit_pose = st.norm.rvs(loc=0,scale=1,size=bp.shape)
        objet_principal = 'halo'
        Y_tout_snr,snr_diff_dec = dt.gen_signaux_snr(objet_principal, source_gal,source_halo, bruit_pose, SNR, simu)
        Y  = Y_tout_snr[:,:,:,:,0]  
        # let us insert Nan :
        #Y[5,5,:,:] = np.nan 
        lambda_0 = 16
        Y_src = Y.mean(axis=3)
        
        #dat= np.load("../data/empty_cubes.npz")
        empty_cube=snr_diff_dec*st.norm.rvs(loc=0,scale=1,size=(S,S,W))#np.swapaxes(dat['empty_cube'],0,2)
                

    else:
        dat= np.load('../data/id43.npz')
        Y_src = dat['Y_src']
        # Soustraction de mediane
        ss_cube_medfilt = si.medfilt(Y_src,(1,1,301))
        Y = Y_src - ss_cube_medfilt
        Y_src = Y[:,:,135:165]
        
        
        Y = np.tile(Y[:,:,:,np.newaxis],(1,1,1,2))
        Y = Y[:,:,135:165,:]
        lambda_0 = 150
       
        empty_cube=st.norm.rvs(loc=0,scale=1,size=(S,S,W))
        
        
    # to change if the object center is not the subcube center :
    centre=np.array([int(S/2),int(S/2)])
    params = parameters.Params(Y,
                               centre,
                               pfa_bright=pfa_bright,
                               pfa_faint=pfa_faint,
                               taille_f = 11,
                               FWHM=FWHM)
                               
    params.Y_sig =  empty_cube
    params.confident=False
    
    
    Xe1,ve1,vem1,Xi,vi = sdp.detection_strategy(params) 
    
    #%%
    reg_init = ma.masked_array(Y_src, np.tile((Xi==0)[:,:,np.newaxis], (1,1,Y_src.shape[2])))
    sp_init = ma.mean(ma.mean(reg_init, axis=0), axis=0)

    reg_ext = ma.masked_array(Y_src, np.tile(((Xe1-Xi)==0)[:,:,np.newaxis], (1,1,Y_src.shape[2])))
    sp_ext = ma.mean(ma.mean(reg_ext, axis=0), axis=0)
    
    ma_ext = (Xe1 - Xi)
    reg_reste = ma.masked_array(Y_src, np.tile(ma_ext[:,:,np.newaxis], (1,1,Y_src.shape[2])))
    sp_reste = ma.mean(ma.mean(reg_reste, axis=0), axis=0)
    
    #%%    
#    im_ind_est,im_weight_est = mse.get_sparse_estimate(params) # pb ici !!
#
#    flux,position,largeur = mse.get_moments(im_ind_est,im_weight_est,params.D,params.W,params.pas_dic,pas_spectral)
    # normalizing flux ?
    #flux *= Y[centre[0]+marge,centre[1]+marge,:].sum()/flux[centre[0],centre[1]]    

    # Adjusting positions ?
    #position = lambda_0+(coords_lya[2,obj]-W/2 + position )*pas_spectral
    
        #%%
    #plt.close('all')    
    PFA = np.array([0.1,0.01,0.001, 0.0001])
    ksi = st.chi2.isf(PFA, df=Xi.sum())
    detec = Xe1
    detec = detec[marge:S-marge,marge:S-marge]
    
    nb_li = 3
    nb_col = 3
    plt.figure(figsize=(5*nb_col, 5*nb_li))
    
    
    plt.subplot(nb_li, nb_col,1) ;
    plt.imshow(Y_src.mean(axis=2).T, cmap=plt.cm.gray_r,origin='lower', interpolation='nearest',vmin=0)
    if test_simu==1:
        plt.contour(gal.T,1,linestyles='-',colors='#cc0000',linewidths=1.5);
        plt.contour(halo.T,1,linestyles='-',colors='#1d829e',linewidths=3)
        
    plt.title('White image and ground truth')
    
    plt.subplot(nb_li, nb_col,2) ;
    plt.imshow(Y_src[:,:,lambda_0-3:lambda_0+3].mean(axis=2).T, cmap=plt.cm.gray_r,origin='lower', interpolation='nearest',vmin=0)
    plt.contour(Xi.T,1,linestyles='-',colors='#cc0000',linewidths=1.5);
    plt.contour(Xe1.T,1,linestyles='-',colors='#1d829e',linewidths=3)
    plt.title('Narrow-band Image, Detection Maps')
    
    
    plt.subplot(nb_li,nb_col,3)
    plt.imshow(ve1.T, origin='lower',vmin=0,cmap=plt.cm.jet,interpolation='nearest')
    plt.colorbar()

    plt.contour(ve1.T, ksi)    

    
    #%%
    plt.subplot(nb_li,nb_col,(nb_col+1,nb_col+2))
    sp_range = np.arange(0,126,1.25)
    
    plt.plot(sp_reste,'-',color='#cccccc',linewidth=1,label='Outer region')
    plt.plot(sp_init,':',color='#cc0000',linewidth=2.5,label='Bright region')
    plt.plot(sp_ext,'-',color='#1d829e',linewidth=1.5, label='Faint spectra')
    plt.legend(loc='upper right',title='Integrated spectra on:')
    plt.grid()
    plt.title('Individual Spectra')
    plt.ylabel('Intensity')
    plt.xlabel('Bandwidth (1.25 Angstrom) ')
    plt.xlim((0,sp_init.size))
    
    
#
#    plt.subplot(nb_li,nb_col,7)
#    plt.imshow(flux.T, cmap=plt.cm.gray_r,origin='lower', interpolation='nearest',vmin=0)
#    plt.xlabel('q (pixel)'); plt.ylabel('p (pixel)')
#    plt.colorbar(fraction=0.046, pad=0.04)
#    plt.title('Flux estime')
#    plt.contour(detec.T, 1,linewidths=2,colors='w')
#    
#    plt.subplot(nb_li,nb_col,8)
#    plt.imshow(position.T, cmap=plt.cm.Spectral,origin='lower', interpolation='nearest',vmin=0)
#    plt.xlabel('q (pixel)'); plt.ylabel('p (pixel)')
#    plt.colorbar(fraction=0.046, pad=0.04)
#    plt.title('Position estimee')
#
#    plt.contour(detec.T, 1,linewidths=2,colors='w')
#    
#    
#    plt.subplot(nb_li,nb_col,9)
#    plt.imshow(largeur.T, cmap=plt.cm.coolwarm,origin='lower', interpolation='nearest',vmin=0)
#    plt.xlabel('q (pixel)'); plt.ylabel('p (pixel)')
#    plt.colorbar(fraction=0.046, pad=0.04)
#    plt.title('FWHM estimee')
#
#    plt.contour(detec.T, 1,linewidths=2,colors='w')
    
    plt.tight_layout()
    
#%% Display results of step 1 alone

#plt.figure(figsize=(10,5))
#
#
reg_init2 = ma.masked_array(source_gal, np.tile((gal==0)[:,:,np.newaxis], (1,1,Y_src.shape[2])))
sp_init2 = ma.mean(ma.mean(reg_init2, axis=0), axis=0)
sp_gal = sp_init2-sp_init2[0]
#
#plt.subplot(1,2,1)
#plt.imshow(Y_src[:,:,lambda_0-3:lambda_0+3].mean(axis=2).T, cmap=plt.cm.gray,origin='lower', interpolation='nearest')
#plt.contour(gal.T,1,linestyles='-',colors='b',linewidths=1.5);
#plt.contour(Xi.T,1,linestyles='-',colors='#cc0000',linewidths=1.5);
#
#
#plt.subplot(1,2,2)
#
#plt.plot(sp_init2/snr_diff_dec,':',color='b',linewidth=2.5,label='Bright region')
#plt.plot(sp_init,':',color='#cc0000',linewidth=2.5,label='Bright region')
#
#from matplotlib2tikz import save as tikz_save
#
#tikz_save('/home/miv/courbot/Dropbox/Manuscrit/II/II-3/figures/test_init_ex.tex')
#
##%%

#plt.figure(figsize=(10,5))
#
#
reg_init2 = ma.masked_array(source_halo, np.tile((halo==0)[:,:,np.newaxis], (1,1,Y_src.shape[2])))
sp_init2 = ma.mean(ma.mean(reg_init2, axis=0), axis=0)
sp_halo = sp_init2-sp_init2[0]
#
#plt.subplot(1,2,1)
#plt.imshow(Y_src[:,:,lambda_0-3:lambda_0+3].mean(axis=2).T, cmap=plt.cm.gray,origin='lower', interpolation='nearest')
#plt.contour(halo.T,1,linestyles='-',colors='b',linewidths=1.5);
#plt.contour(ve1.T>40,1,linestyles='-',colors='#1d829e',linewidths=1.5);
#
#
#plt.subplot(1,2,2)
#
#plt.plot(sp_init2*snr_diff_dec,':',color='b',linewidth=2.5,label='Bright region')
#plt.plot(sp_ext,':',color='#1d829e',linewidth=2.5,label='Bright region')

#from matplotlib2tikz import save as tikz_save

#tikz_save('/home/miv/courbot/Dropbox/Manuscrit/II/II-3/figures/test_ext_ex.tex')

#%%
nb_li = 1
nb_col = 4
plt.figure(figsize=(5*nb_col, 5*nb_li))


plt.subplot(nb_li, nb_col,1) ;
plt.imshow(Y_src.mean(axis=2).T, cmap=plt.cm.gray_r,origin='lower', interpolation='nearest',vmin=0)
if test_simu==1:
    plt.contour(gal.T,1,linestyles='-',colors='#cc0000',linewidths=1.5);
    plt.contour(halo.T,1,linestyles='-',colors='#1d829e',linewidths=3)
    
plt.title('White image and ground truth')

plt.subplot(nb_li, nb_col,3) ;
plt.imshow(Y_src[:,:,lambda_0-3:lambda_0+3].mean(axis=2).T, cmap=plt.cm.gray_r,origin='lower', interpolation='nearest',vmin=0)
plt.contour(Xi.T,1,linestyles='-',colors='#cc0000',linewidths=1.5);
plt.contour(Xe1.T,1,linestyles='-',colors='#1d829e',linewidths=3)
plt.title('Narrow-band Image, Detection Maps')

#
#plt.subplot(nb_li,nb_col,3)
#plt.imshow(ve1.T, origin='lower',vmin=0,cmap=plt.cm.jet,interpolation='nearest')
#plt.colorbar()
#
#plt.contour(ve1.T, ksi)    



plt.subplot(nb_li,nb_col,4)
sp_range = np.arange(0,126,1.25)

plt.plot(sp_reste,'-',color='#cccccc',linewidth=1,label='Outer region')
plt.plot(sp_init,':',color='#cc0000',linewidth=2.5,label='Bright region')
plt.plot(sp_ext,'-',color='#1d829e',linewidth=1.5, label='Faint spectra')
plt.legend(loc='upper right',title='Integrated spectra on:')
plt.grid()
plt.title('Individual Spectra')
plt.ylabel('Intensity')
plt.xlabel('Bandwidth (1.25 Angstrom) ')
plt.xlim((0,sp_init.size))