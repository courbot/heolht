# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:49:27 2016

@author: courbot
"""
import detection_preprocessing as dp

class Params():
    
    def __init__(self,
                 Y, 
                 centre,
                 P = 1,
                 diag = True,
                 pfa_bright = 0.001,
                 pfa_faint = 0.001,
                 FWHM = 0.66*1/0.2,
                 taille_f = 11,
                 beta = 2.6,
                 confident = 0,
                 pas_dic = 0.15,
                 nb_ech_dic = 50
                 ):


        self.Y = Y
        self.centre = centre
        self.P = P
        self.diag = diag
        self.pfa_bright=pfa_bright
        self.pfa_faint = pfa_faint
        self.FWHM = FWHM
        self.taille_f = taille_f
        self.beta = beta
        self.confident = confident
        self.pas_dic = pas_dic
        self.nb_ech_dic = nb_ech_dic        
        
        
        self.marge = int(taille_f/2)
        self.S = Y.shape[0]
        self.W = Y.shape[2]
        self.F = dp.Moffat(taille_f, FWHM,beta)
        self.D = dp.gen_dic(self.W, pas = pas_dic, nb_ech=nb_ech_dic,asym=0)
        
        

   
        

        
        
