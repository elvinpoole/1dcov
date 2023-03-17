import numpy as np 
import treecorr 
import healpy as hp 
import pylab as plt 
import fitsio as fio

theta_file  = '../theory/galaxy_xi/theta.txt'
wtheta_file = '../theory/galaxy_xi/bin_3_3.txt'
theta  = np.loadtxt(theta_file)
wtheta = np.loadtxt(wtheta_file)

mock0 = fio.read(f'../data/y6_ln_mocks/y6_maglim_con_v2_nside512_nosys_mock0.fits.gz')

#sp_file = '/Users/jackelvinpoole/osu_laptop/DES/sysmaps/y3/sof_depth/y3a2_gold_2_2_1_sof_nside4096_nest_i_depth.fits.gz'
sp_file = '../data/sysmaps/y6_AIRMASS_WMEAN_i_4096_RING.fits.gz'
sp_hp = hp.read_map(sp_file)

#degrade the SP map and get the pixel centers
nside = 512
sp_hp = hp.ud_grade(sp_hp,nside)

#apply mask from mock0
sp_hp_masked = np.ones(hp.nside2npix(nside))*hp.UNSEEN
sp_hp_masked[mock0['HPIX']] = sp_hp[mock0['HPIX']]

#get the ra dec of the pixels
select = (sp_hp_masked != hp.UNSEEN)
sp  = sp_hp[select]
pix = np.arange(len(sp_hp))[select]
ra_pix,dec_pix = hp.pix2ang(nside, pix, lonlat=True)

#get the ngal for each mock
ngal = []
for imock in range(100):
	mock = fio.read(f'../data/y6_ln_mocks/y6_maglim_con_v2_nside512_nosys_mock{imock}.fits.gz')
	ngal.append(mock['bin3']) 
ngal = np.array(ngal)

nbins = 10
#equal area bins
sp_edges_eq = np.percentile(sp,np.linspace(0,100,nbins+1)) 
#equal size bins (ignoring first and last 0.1%)
sp_edges_reg = np.linspace(np.percentile(sp,0.1),np.percentile(sp,99.9),nbins+1) 

labels=['eq','reg']
for iedges, sp_edges in enumerate([sp_edges_eq, sp_edges_reg]):
	label = labels[iedges]
	sp_low  = sp_edges[:-1]
	sp_high = sp_edges[1:]
	sp_cen = (sp_low+sp_high)/2.


	#create catalog object for the pixel centers of each patch
	cats = {}
	for i in range(nbins):
		select_sp = (sp >= sp_low[i])*(sp < sp_high[i])
		ra_patch = ra_pix[select_sp]
		dec_patch = dec_pix[select_sp]
		cats[i] = treecorr.Catalog(ra=ra_patch,dec=dec_patch,ra_units='degrees',dec_units='degrees')

	#compute pair counts for each combination of catalog objects
	npairs = {}
	for i in range(nbins):
		for j in range(nbins):
			if i>j:
				continue

			#should match the theta binning
			nn = treecorr.NNCorrelation(max_sep=250.,min_sep=2.5,nbins=20,sep_units='arcmin' )

			nn.process(cats[i],cats[j])
			npairs[i,j] = nn.npairs
			npairs[j,i] = nn.npairs


	#galaxy data
	#lets say we have 1000000 galaxies in one tomo bin (~maglim y3)
	nobjects = np.sum(mock0['bin3'])
	npix_sp = np.array([cats[i].nobj for i in range(nbins)])
	n_sp = nobjects*npix_sp/sum(npix_sp) #estimated number of objects per sp bins
	nbar = nobjects/len(sp)


	cov_sn  = np.identity(nbins) #shot noise only
	cov_all = np.identity(nbins) #shot noise + sample variance terms
	for i in range(nbins):
		for j in range(nbins):
			sum_nw = np.sum(npairs[i,j]*wtheta*(nbar**2.) )
			if i == j:
				cov_sn[i,j] = n_sp[i]
				cov_all[i,j] = n_sp[i] + sum_nw
			else:
				cov_sn[i,j] = 0.0
				cov_all[i,j] = sum_nw


	#make same covariance for the LN mocks
	#First compute N(SP) for each mock
	n_sp_mocks = []
	for i in range(nbins):
		select_sp = (sp >= sp_low[i])*(sp < sp_high[i])
		n_sp_mocks.append(np.array([np.sum(ngal[imock][select_sp]) for imock in range(100)]))
	n_sp_mocks = np.array(n_sp_mocks)
	n_sp_mocks = n_sp_mocks.T
	cov_mocks = np.cov(n_sp_mocks,rowvar=False)

	#get error on mock cov using JK? does this work?
	jk_covs = [np.cov(np.vstack((n_sp_mocks[:5*ijk],n_sp_mocks[5*(ijk+1):])),rowvar=False) for ijk in range(20)]
	diags_ms = np.array([(np.sqrt(jk_covs[ijk].diagonal())-np.sqrt(cov_mocks.diagonal())) for ijk in range(20)])
	diag_err = np.sqrt(np.sum(diags_ms**2.,axis=0)*(20-1)/20)

	#plt.plot(sp_cen, np.sqrt(cov_mocks.diagonal()),  ls='-', marker='.', label='mocks')
	plt.errorbar(sp_cen, np.sqrt(cov_mocks.diagonal()), diag_err, fmt='.', ls='-', label='mocks', color='g')
	plt.plot(sp_cen, np.sqrt(cov_all.diagonal()), ls='-', lw=2, marker='.', label='shot noise',color='b')
	plt.plot(sp_cen, np.sqrt(cov_sn.diagonal()),  ls='-', lw=2., marker='.', label='shot noise + SV',color='orange')
	
	plt.legend()
	plt.axhline(0,color='k',ls='--')
	plt.xlabel('SOF depth i-band Y3', fontsize=15)
	plt.ylabel(r'$\sigma_{N^{(SP)}}$', fontsize=15)
	plt.savefig('diagonal_{0}.png'.format(label))
	plt.close()

	max_val = np.max([cov_sn,cov_all,cov_mocks])
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
	im = axes[0].imshow(cov_sn,vmin=0,vmax=max_val)
	im = axes[1].imshow(cov_all,vmin=0,vmax=max_val)
	im = axes[2].imshow(cov_mocks,vmin=0,vmax=max_val)
	fig.colorbar(im, ax=axes.ravel().tolist())
	plt.savefig('cov_{0}.png'.format(label))
	plt.close()




