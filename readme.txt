HEOLTH v1.3

Documentation : ./doc/_build/html/index.html
Catalogues : 	./results/HDFS_1.24_sample (1.3)/2015-11-Lya-ext-HDFS-sample-1.24.pdf
		./results/HDFS_1.24_full (1.3)/2015-11-Lya-ext-HDFS-full-1.24.pdf


===================================
Contenu des objets :
===================================
	mpdaf.images :
		MUSE_WHITE	Image blanche sur le sous-cube
		MUSE_NB		Image Narrow-Band dans la largeur fournie
		DET_STAT	Carte de statistique de détection
		DET_BIN_ALL	Carte binaire de détection : toute la détection
		DET_BIN_GAL				     détection initiale (partie brillante)
		DET_BIN_HAL				     détection périphérique
		
		FLUX		Flux dans DET_BIN_ALL
		POSITION	Position de la raie dans DET_BIN_ALL
		FWHM		FWHM de la raie dans DET_BIN_ALL
	NB. ces trois dernières valeurs sont indicatives, et sont estimées à partir d'une mise en correspondance de gaussienne spectrale dans un voisinage local.
	Dans certains cas (NaN, etc) les estimations sont clairement faussées.

	mpdaf.spectrum :	Spectres moyen sur les données réelles :
		CENTER			De la détection initiale.
		PERIPH_ADJ		De la détection périphérique adjacente à l'objet central.
		PERIPH_NOADJ		Des autres régions de détection périphérique.
		EXT			Du reste du sous-cube.
	mdaf.keys :		
		FWHM_PIX		FWHM de la FSF à la longueur d'onde de l'objet.
		IM_SIZE			Taille de l'image.
		IM_MARGIN		Marge dans les images, qui sont masquées/à zéro dans cette marge.

===================================
Paramètres actuels :
===================================
- FSF fenêtrée à 11 px
- PFA cible halo = 10e-4
- PFA cible galaxie = 10e-2


===================================
Mises à jour :
===================================
1.3 (2015-11-23) - améliorations du code
	- Paramètres : inchangés
	- Ajustements :
		- tous les paramètres sont stockés dans une classe unique. Une instance est à définir pour lancer le code.
		- une option "confident" est ajoutée, pour ne préserver que la détection la plus sûre. Cela a pour effet de retirer la plupart des 'blobs' observés sur les objets MUSE.
		- une option peremt de choisir un autre cube pour l'évaluation de la matrice de covariance.
	- Démonstrations :
		- Essais sur des cubes MUSE "vides", reportés dans le notebook 'test_astro.ipynb'.
===================
1.3 (2015-11-23) - ajustements, simplifications du code
	- Paramètres : inchangés
	- Ajustements :
		- ajout des paramètres de SNR estime, de spectre exterieur isole.
	- Catalogue pdf :
		- Visualisations des moments (flux, position, fwhm) sur le sous-cube complet plutôt que sur la seule région de détection.
	- Documentation complète du code.
	- Démonstrations :
		- Détection sur des données simulées/réelles.
		- Génération de catalogue de détection et d'objets source (MPDAF) associés.
===================
1.2 (2015-08-26) - correction et améliorations du code
	- Paramètres : inchangés
	- Correction : 
		- rectification de l'utilisation de la FSF,
		- FWHM de la FSF exprimée en fonction de la longueur d'onde (cf. [Bacon et al, 2015, The MUSE 3D view of the Hubble Deep Field South]),
		- soustraction du continu : filtrage spectral médian par fenêtre glissante, avec une fenêtre de taille 301 centrée sur la longueur d'onde de l'objet (cf. L.Wisotzki et al, in prep., Extended Lyman α haloes around individual high-redshift galaxies revealed with MUSE),
	- Catalogue pdf :
		- ajout de la PSF et de sa FWHM à l'échelle du cube.
	- Ajustements :
		- ajout des paramètres de taille, marge utiles pour les images, FWHM de la PSF.
===================
1.1 (2015-07-30) - corrections de code et ajustements de format
	- Paramètres : 
		- FSF fenêtrée à 11 px
		- PFA cible halo = 10e-4
		- PFA cible galaxie = 10e-2
	- Corrections :
		- Implémentation de la formulation exacte du test, évitant deux opérations de racine carrée matricielles. Cela a divise les temps de calcul par 2 à FSF fixée.
		- Correction des blanchiments préalable aux tests.
	- Ajustements :	
		- PFA cible halo passée de 10e-3 à 10e-4.
		- Largeur du fenêtrage de FSF passé de 7 à 11px
		- Harmonisation des champs : DET_BIN_ALL, DET_BIN_GAL, DET_BIN_HAL pour les détections complètes.
		- Correction du passage seuil/PFA cible.
		- Séparation des spectres dans la détection externe en connexes et non connexes.
		- Conversion de l'image "largeur" (correspondant aux écarts-type de gaussiennes) en FWHM
===================
1.0 (2015-07-21) - version initiale
	- Paramètres : 
		- FSF fenêtrée à 7 px
		- PFA cible halo = 10e-3
		- PFA cible galaxie = 10e-2
===================================
	
