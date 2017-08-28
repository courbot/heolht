
Detection Demonstration
***********************

See the notebook *demo_detection* to edit the code to your convenience.

:author: Jean-Baptiste Courbot - jean-baptiste.courbot@univ-lyon1.fr
:date: november 18, 2015

**Imports**

.. code:: python

    import numpy as np 
    import matplotlib.pyplot as plt
    import sys 
    import numpy.ma as ma
    import scipy.stats as st
    import scipy.signal as si
    
    %matplotlib inline
    
    sys.path.insert(0,'../HEOLTH_1.3')
    
    import detection_tools as dt
    import detection_preprocessing as dp
    import strategy_detection_pose as sdp
    import mle_sparse_est as mse

**Various Parameters**

.. code:: python

    # Dimensions 
    S = 50
    W = 30
    P = 54
    centre = np.array([int(S/2), int(S/2)])
    
    # Various MUSE parameters
    lambda_0 = 4750    # see Bacon et al, 2015
    lambda_lya = 1216
    pas_spectral = 1.25 # Angstrom / spectral band
    
    # Dictionary generation
    pas = 0.15
    nb_ech_dic = 50
    
    D = dp.gen_dic(W, pas = pas, nb_ech=nb_ech_dic,asym=0)
    
    # FSF generation
    taille_f = 11
    marge = int(taille_f/2)
    FWHM = 0.66*1/0.2
    beta=2.6
    
    F = dp.Moffat(taille_f, FWHM,beta)
    
    # SNR to use (if the data is simulated)
    SNR = np.array([0])
    
    # False alarm to use for the bright and faint source detection, respectively.
    pfa_bright = 0.0001
    pfa_faint = 0.0001 

.. code:: python

    plt.figure(figsize=(8,4));
    plt.subplot(2,2,(1,3))
    plt.plot(D[:,10::240], linewidth=2.5) ; plt.grid(); plt.xlim((0,W-1)); plt.ylim((0,1)); plt.tight_layout()
    plt.title('Dictionary sample at $\\lambda=10$')
    
    profil_f = np.append(np.append(F[5,0],F[5,:].T),F[5,10])
    
    plt.subplot(2,2,4); 
    plt.plot(np.arange(-6,7,1),profil_f, 'k',linewidth=2.5,drawstyle='steps-mid'); plt.xlim((-5.5,5.5));plt.grid()
    plt.title('FSF profile')
    
    plt.subplot(2,2,2);
    plt.imshow(F[:6,:], interpolation='nearest', extent=(-5.5,5.5,-0.5,5.5),aspect='auto',cmap=plt.cm.Blues);  
    plt.title('Half FSF')
    
    plt.tight_layout()



.. image:: output_5_0.png


**Data Loading**

.. code:: python

    # IMPORTANT !
    # Here you set which of the data is used hereafter. 
    #Two subcubes are available : a simulated ones (test_simu=1) and a MUSE one (test_simu=0).
    
    test_simu=1
    
    
    if test_simu ==1:
        objet_principal = 'halo'
        simu, gal, source_gal, halo, source_halo,bruit_pose = dt.get_simu('./data/simu_H-3.npz',lambda_max = 30)
        bruit_pose = st.norm.rvs(loc=0,scale=1,size=bruit_pose.shape)
        Y_tout_snr,gal2 = dt.gen_signaux_snr(objet_principal, source_gal,source_halo, bruit_pose, SNR, simu)
        Y  = Y_tout_snr[:,:,:,:,0]  
        lambda_0 = 16
        Y_src = Y.mean(axis=3)
    else:
        dat= np.load('./data/id43.npz')
        Y_src = dat['Y_src']
        # Median filtering
        ss_cube_medfilt = si.medfilt(Y_src,(1,1,301))
        Y = Y_src - ss_cube_medfilt
        Y = np.tile(Y[:,:,:,np.newaxis],(1,1,1,2))
        Y = Y[:,:,135:165,:]
        lambda_0 = 150
    
        

**Actual Detection**

.. code:: python

    Xe1,ve1,vem1,Xi,vi = sdp.detection_strategy(Y, P=1, diag=True, pfa_bright=pfa_bright, pfa_faint=pfa_faint, FWHM=FWHM, taille_f=taille_f, beta=beta) 

**Extracting the spectra of interest**

.. code:: python

    reg_init = ma.masked_array(Y_src, np.tile(np.invert(Xi)[:,:,np.newaxis], (1,1,Y_src.shape[2])))
    sp_init = ma.mean(ma.mean(reg_init, axis=0), axis=0)
    
    reg_ext = ma.masked_array(Y_src, np.tile(np.invert(Xe1)[:,:,np.newaxis], (1,1,Y_src.shape[2])))
    sp_ext = ma.mean(ma.mean(reg_ext, axis=0), axis=0)
    
    ma_ext = (Xe1 + Xi)
    reg_reste = ma.masked_array(Y_src, np.tile(ma_ext[:,:,np.newaxis], (1,1,Y_src.shape[2])))
    sp_reste = ma.mean(ma.mean(reg_reste, axis=0), axis=0)

**Maximum Likelihood Best Fits**

.. code:: python

    im_ind_est,im_weight_est = mse.get_sparse_estimate(Y_src,F,D)
    flux,position,largeur = mse.get_moments(im_ind_est,im_weight_est,D,W,pas,pas_spectral)

**Display**

.. code:: python

    plt.close('all')
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
    plt.imshow(ve1.T, origin='lower',vmin=0,cmap=plt.cm.Blues,interpolation='nearest')
    plt.title('Detection Statistic')
    
    
    PFA = np.array([0.01,0.001, 0.0001])
    ksi = st.chi2.isf(PFA, df=Xi.sum())
    plt.contour(ve1.T, ksi)    
    
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
    plt.xlim((0,sp_init.size-1))
    
    
    
    plt.subplot(nb_li,nb_col,7)
    plt.imshow(flux.T, cmap=plt.cm.gray_r,origin='lower', interpolation='nearest',vmin=0)
    plt.xlabel('q (pixel)'); plt.ylabel('p (pixel)')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Estimated Flux')
    plt.contour(detec.T, 1,linewidths=2,colors='w')
    
    plt.subplot(nb_li,nb_col,8)
    plt.imshow(position.T, cmap=plt.cm.Spectral,origin='lower', interpolation='nearest',vmin=0)
    plt.xlabel('q (pixel)'); plt.ylabel('p (pixel)')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Estimated Spectral Position')
    
    plt.contour(detec.T, 1,linewidths=2,colors='w')
    
    
    plt.subplot(nb_li,nb_col,9)
    plt.imshow(largeur.T, cmap=plt.cm.coolwarm,origin='lower', interpolation='nearest',vmin=0)
    plt.xlabel('q (pixel)'); plt.ylabel('p (pixel)')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Estimated FWHM')
    
    plt.contour(detec.T, 1,linewidths=2,colors='w')
    
    plt.tight_layout()



.. image:: output_15_0.png

