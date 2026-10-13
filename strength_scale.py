from astropy.table import Table
import matplotlib.pyplot as pl
pl.ion()
import numpy as np
from glob import glob
from astrodendro import Dendrogram
from astropy.io import fits 
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


from utils import *

table_file = "allstats.ecsv"
pix2pc = 0.25


stats_table = Table.read(table_file, format="ascii.ecsv")
if not set(flat_columns).issubset(set(stats_table.colnames)):
    raise RuntimeError("allstats.ecsv is not in the expected flat format")

stats_table = stats_table[flat_columns]
clouds = np.unique(stats_table["cloud"])

lines=["13CO10","CS21","C18O10"]
colindex = pl.cm.jet(np.linspace(0, 1, len(lines)))
colindex=['b','g','m']

pl.clf()
# test
clouds=['N113','A439',"N159W"]


firstplot=False

if firstplot:
    for icloud,cloud in enumerate(clouds):
        linefiles = glob("../products_1pc/"+cloud+"_*4asec_grid_K.fits")
        linefiles = np.array([x for x in linefiles if "cont" not in x])
        stats = table_to_stats_for_cloud(stats_table, cloud)
    
        dendrofile = "dendro/"+cloud+"_13CO10_dendrogram.hdf5"
        dend=Dendrogram.load_from(dendrofile)
        
        for iline, line in enumerate(lines):
            pl.clf()
            vox2flux = 0.25**2 * stats[line]['chanwid_kms'] 
    
            size = 2* np.sqrt(stats[line]["npix_2d"]/np.pi) * pix2pc

            y = stats[line]['moments'][0]
            y = stats[line]['pk_from_mom'][:,0]
            y = stats[line]['fitparms'][:,0]/stats[line]['fiterrs'][:,0]

            pl.plot(size, y, '.', label=line, color=colindex[iline])

            for d in dend.leaves:
                struc=d
                myplot,=pl.plot(size[d.idx], y[d.idx], 'o', mfc="none")
                while struc.parent is not None:
                    i0=struc.idx
                    i1=struc.parent.idx
                    pl.plot(size[[i0,i1]], y[[i0,i1]], '-', color=myplot.get_color(),alpha=0.2)
                    struc = struc.parent

            pl.xlabel("size [pc]")
            pl.xlim(0,25)
            pl.title(cloud+" "+line)
            pl.ylabel("peak")
            pl.yscale("log")
            pl.legend(prop={'size': 8})



if True:
    for icloud,cloud in enumerate(clouds):
        linefiles = glob("../products_1pc/"+cloud+"_*4asec_grid_K.fits")
        linefiles = np.array([x for x in linefiles if "cont" not in x])
        stats = table_to_stats_for_cloud(stats_table, cloud)
    
        dendrofile = "dendro/"+cloud+"_13CO10_dendrogram.hdf5"
        dend=Dendrogram.load_from(dendrofile)
    
        for iline, line in enumerate(lines):
            vox2flux = 0.25**2 * stats[line]['chanwid_kms'] 
    
            size = 2* np.sqrt(stats[line]["npix_2d"]/np.pi) * pix2pc
            deltaflux = np.zeros_like(size)
            
            z=np.where([line in x for x in linefiles])[0]
            linedata=fits.getdata(linefiles[z[0]])
            
            #deltaflux = np.array([dend[i].values(subtree=False).sum() * vox2flux for i in range(len(dend))])
            deltaflux = np.array([linedata[dend[i].indices(subtree=False)].sum() * vox2flux for i in range(len(dend))])
    
            u=np.argsort(size)
            if icloud==0:
                myplot,=pl.plot(size[u], np.cumsum(deltaflux[u])/np.sum(deltaflux[u]), label=line, color=colindex[iline]) #label=cloud+" "+line)
            else:
                pl.plot(size[u], np.cumsum(deltaflux[u])/np.sum(deltaflux[u]), color=colindex[iline]) # label=cloud
    
    
    pl.xlabel("size [pc]")
    pl.xlim(0,25)
    pl.ylabel("normalized cumulative flux")
    pl.legend(prop={'size': 8})



linestack = np.zeros((len(lines),len(clouds),30))

pl.clf()
regsize= np.arange(0, 30)
for icloud,cloud in enumerate(clouds):
    dendrofile = "dendro/"+cloud+"_13CO10_dendrogram.hdf5"
    dend=Dendrogram.load_from(dendrofile)
    linefiles = glob("../products_1pc/"+cloud+"_*4asec_grid_K.fits")
    linefiles = np.array([x for x in linefiles if "cont" not in x])

    for iline,line in enumerate(lines):

        stats = table_to_stats_for_cloud(stats_table, cloud)
        vox2flux = 0.25**2 * stats[line]['chanwid_kms'] 

        size = 2* np.sqrt(stats[line]["npix_2d"]/np.pi) * pix2pc
        z=np.where([line in x for x in linefiles])[0]
        linedata=fits.getdata(linefiles[z[0]])
        
        #deltaflux = np.array([dend[i].values(subtree=False).sum() * vox2flux for i in range(len(dend))])
        deltaflux = np.array([linedata[dend[i].indices(subtree=False)].sum() * vox2flux for i in range(len(dend))])

        u=np.argsort(size)

        cumulative_flux = np.cumsum(deltaflux[u])/np.sum(deltaflux[u])
        # unique_size, size_counts = np.unique(size[u], return_counts=True)
        # cumulative_at_size = cumulative_flux[np.cumsum(size_counts) - 1]
        # spline = CubicSpline(unique_size, cumulative_at_size)
        # regridded = spline(np.clip(regsize, unique_size[0], unique_size[-1]))
        regridded = np.interp(regsize, size[u],cumulative_flux)
        #smooth_regrid = savgol_filter(regridded, 5, 2)
        #smooth_regrid=regridded
        #smooth_regrid = np.convolve(regridded, np.hanning(5)/np.sum(np.hanning(5)), mode='same')

        #smooth_diff = np.diff(regridded)
        #smooth_diff = np.convolve(np.diff(regridded), np.hanning(3)/np.sum(np.hanning(3)), mode='same')
        smooth_diff = savgol_filter(np.diff(regridded), 5, 2)

        if icloud==0:
            myplot,=pl.plot(regsize[:-1], smooth_diff, '.', color=colindex[iline],alpha=0.1*(3-iline))
            #myplot,=pl.plot(size[u][:-1], np.diff(cumulative_flux), '.', label=line,color=colindex[iline],alpha=0.2)
        else:
            pl.plot(regsize[:-1], smooth_diff, '.', color=myplot.get_color(),alpha=0.1*(3-iline))
            #pl.plot(size[u][:-1], np.diff(cumulative_flux), '.', color=myplot.get_color(),alpha=0.2)

        smooth_diff[smooth_diff<0] = np.nan
        linestack[iline, icloud, :-1] = smooth_diff

for iline, line in enumerate(lines):
    mean_diff = np.nanmean(linestack[iline, :, :], axis=0)
    pl.plot(regsize, mean_diff, '-', label=f"{line} mean",color=colindex[iline])
   

pl.xlabel("size [pc]")
pl.xlim(0,25)
pl.ylim(0,.25)
pl.ylabel("fraction of flux")
pl.legend(prop={'size': 8})

pl.savefig("strength_scale.png")



print("Finished plotting vs size scale for all clouds.")