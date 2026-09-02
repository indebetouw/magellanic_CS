"""Extract dendrogram structure statistics from spectral-line FITS cubes.

The script pairs 13CO(1-0) dendrogram files with available line cubes, measures
moments and Gaussian-fit peak properties for each structure, and stores the
results in the flat ``allstats.ecsv`` table.
"""

from astrodendro import Dendrogram
from glob import glob
from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
import os
pl.ion()

key_line = '13CO10'
debug = False

table_file = "allstats.ecsv"
plot_dir = "process_dendro_plots"
os.makedirs(plot_dir, exist_ok=True)
flat_columns = [
    "cloud",
    "line",
    "linefile",
    "cuberms",
    "chanwid_kms",
    "pixperbm",
    "dendro_idx",
    "npix_2d",
    "pk_1darea",
    "pk_2darea",
    "pk_from_mom_1darea",
    "pk_from_mom_2darea",
    "pk_from_fit_1darea",
    "pk_from_fit_2darea",
    "mom0",
    "mom1",
    "mom2",
    "fit_amp",
    "fit_b",
    "fit_v0",
    "fit_vsig",
    "delta_amp",
    "delta_v0",
    "delta_vsig",
]
flat_dtypes = [
    "U64",
    "U64",
    "U256",
    float,
    float,
    float,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


def stats_to_table(cloud, stats):
    rows = []
    chanwid_kms = float(stats[key_line]["chanwid_kms"])
    pixperbm = float(stats[key_line]["pixperbm"])
    for line, stat in stats.items():
        nstruct = stat["pk"].shape[0]
        cuberms = float(stat.get("linerms", np.nan))
        fiterrs = stat.get("fiterrs", np.full((nstruct, 3), np.nan))
        for i in range(nstruct):
            row = {
                "cloud": cloud,
                "line": line,
                "linefile": stat.get("linefile", ""),
                "cuberms": cuberms,
                "chanwid_kms": chanwid_kms,
                "pixperbm": pixperbm,
                "dendro_idx": i,
                "npix_2d": int(stat["npix_2d"][i]),
                "pk_1darea": float(stat["pk"][i, 0]),
                "pk_2darea": float(stat["pk"][i, 1]),
                "pk_from_mom_1darea": float(stat["pk_from_mom"][i, 0]),
                "pk_from_mom_2darea": float(stat["pk_from_mom"][i, 1]),
                "pk_from_fit_1darea": float(stat["pk_from_fit"][i, 0]),
                "pk_from_fit_2darea": float(stat["pk_from_fit"][i, 1]),
                "mom0": float(stat["moments"][0, i]),
                "mom1": float(stat["moments"][1, i]),
                "mom2": float(stat["moments"][2, i]),
                "fit_amp": float(stat["fitparms"][i, 0]),
                "fit_b": np.nan,
                "fit_v0": float(stat["fitparms"][i, 1]),
                "fit_vsig": float(stat["fitparms"][i, 2]),
                "delta_amp": float(fiterrs[i, 0]),
                "delta_v0": float(fiterrs[i, 1]),
                "delta_vsig": float(fiterrs[i, 2]),
            }
            rows.append(row)

    return Table(
        rows=rows,
        names=flat_columns,
        dtype=flat_dtypes,
    )


def table_to_stats_for_cloud(tab, cloud):
    cloud_rows = tab[tab["cloud"] == cloud]
    stats = {}
    if len(cloud_rows) == 0:
        return stats

    for line in np.unique(cloud_rows["line"]):
        line_rows = cloud_rows[cloud_rows["line"] == line]
        nstruct = int(np.max(line_rows["dendro_idx"])) + 1

        pk = np.zeros((nstruct, 2))
        pk_from_mom = np.zeros((nstruct, 2))
        pk_from_fit = np.zeros((nstruct, 2))
        moments = np.zeros((3, nstruct))
        npix_2d = np.zeros(nstruct, dtype=int)
        fitparms = np.zeros((nstruct, 3))
        fiterrs = np.full((nstruct, 3), np.nan)

        for row in line_rows:
            i = int(row["dendro_idx"])
            pk[i, 0] = float(row["pk_1darea"])
            pk[i, 1] = float(row["pk_2darea"])
            pk_from_mom[i, 0] = float(row["pk_from_mom_1darea"])
            pk_from_mom[i, 1] = float(row["pk_from_mom_2darea"])
            pk_from_fit[i, 0] = float(row["pk_from_fit_1darea"])
            pk_from_fit[i, 1] = float(row["pk_from_fit_2darea"])
            moments[0, i] = float(row["mom0"])
            moments[1, i] = float(row["mom1"])
            moments[2, i] = float(row["mom2"])
            npix_2d[i] = int(row["npix_2d"])
            fitparms[i, 0] = float(row["fit_amp"])
            fitparms[i, 1] = float(row["fit_v0"])
            fitparms[i, 2] = float(row["fit_vsig"])
            fiterrs[i, 0] = float(row["delta_amp"])
            fiterrs[i, 1] = float(row["delta_v0"])
            fiterrs[i, 2] = float(row["delta_vsig"])

        stats[str(line)] = {
            "line": str(line),
            "linefile": str(line_rows["linefile"][0]),
            "linerms": float(line_rows["cuberms"][0]),
            "chanwid_kms": float(line_rows["chanwid_kms"][0]),
            "pixperbm": float(line_rows["pixperbm"][0]),
            "pk": pk,
            "npix_2d": npix_2d,
            "pk_from_mom": pk_from_mom,
            "pk_from_fit": pk_from_fit,
            "moments": moments,
            "fitparms": fitparms,
            "fiterrs": fiterrs,
        }

    return stats


if os.path.exists(table_file):
    loaded_table = Table.read(table_file, format="ascii.ecsv")
    # This is a backfill section - if starting from scratch these columns will get populated
    # below in the main anaysis loop
    if not {"chanwid_kms", "pixperbm", "npix_2d"}.issubset(loaded_table.colnames):
        channel_data_by_cloud = {}
        for cloud in np.unique(loaded_table["cloud"]):
            cloud = str(cloud)
            key_rows = loaded_table[
                (loaded_table["cloud"] == cloud)
                & (loaded_table["line"] == key_line)
            ]
            if len(key_rows) == 0:
                raise RuntimeError(f"Missing {key_line} rows for cloud {cloud}")
            key_header = fits.getheader(str(key_rows["linefile"][0]))
            chanwid_kms = (
                abs(key_header["CDELT3"])
                / key_header["RESTFRQ"]
                * 299792.458
            )
            pixel_area_arcsec2 = abs(key_header["CDELT1"] * key_header["CDELT2"]) * 3600**2
            pixperbm = 4**2 * np.pi / (4 * np.log(2) * pixel_area_arcsec2)
            dendrogram = Dendrogram.load_from(f"{cloud}_{key_line}_dendrogram.hdf5")
            channel_data_by_cloud[cloud] = (
                chanwid_kms,
                pixperbm,
                np.array([np.sum(structure.get_mask().sum(axis=0) > 0) for structure in dendrogram]),
            )
        loaded_table["chanwid_kms"] = [channel_data_by_cloud[str(cloud)][0] for cloud in loaded_table["cloud"]]
        loaded_table["pixperbm"] = [channel_data_by_cloud[str(cloud)][1] for cloud in loaded_table["cloud"]]
        loaded_table["npix_2d"] = [channel_data_by_cloud[str(cloud)][2][int(index)] for cloud, index in zip(loaded_table["cloud"], loaded_table["dendro_idx"])]
    if set(flat_columns).issubset(set(loaded_table.colnames)):
        stats_table = loaded_table[flat_columns]
        processed_clouds = set(str(c) for c in stats_table["cloud"])
    else:
        print("Existing allstats.ecsv is not flat format; starting a new flat table.")
        stats_table = Table(names=flat_columns, dtype=flat_dtypes)
        processed_clouds = set()
else:
    stats_table = Table(names=flat_columns, dtype=flat_dtypes)
    processed_clouds = set()

cloud_linefiles = {}
all_lines = []
seen_lines = set()
prescreen_errors = []

dendrofiles = glob("dendro/*"+key_line+"*hdf5")

for dendrofile in dendrofiles:
    cloud = dendrofile.split("/")[1].split("_")[0]
    linefiles = glob("../products_1pc/"+cloud+"_*4asec_grid_K.fits")
    linefiles = np.array([x for x in linefiles if "cont" not in x])
    cloud_linefiles[cloud] = linefiles

    key_linefiles = [x for x in linefiles if x.split("/")[-1].split("_")[1] == key_line]
    if len(key_linefiles) == 0:
        print(f"ERROR: key line file not found for {dendrofile}")
        prescreen_errors.append(dendrofile)
    else:
        try:
            dcheck = Dendrogram.load_from(dendrofile)
            dshape = np.shape(dcheck.data)
            keyshape = fits.getdata(key_linefiles[0]).shape
            if dshape != keyshape:
                print(f"ERROR: shape mismatch for {dendrofile}: dendrogram {dshape} vs key line {keyshape}")
                prescreen_errors.append(dendrofile)
        except Exception as err:
            print(f"ERROR: pre-screen failed for {dendrofile}: {err}")
            prescreen_errors.append(dendrofile)

    for linefile in linefiles:
        line = linefile.split("/")[-1].split("_")[1]
        if line not in seen_lines:
            seen_lines.add(line)
            all_lines.append(line)

if len(prescreen_errors) > 0:
    unique_errors = sorted(set(prescreen_errors))
    print("Pre-screen failed for dendrogram files:")
    for dendrofile in unique_errors:
        print("  ", dendrofile)
    raise RuntimeError("Aborting because dendrogram and key-line cube shapes do not match.")

if key_line in all_lines:
    all_lines = [key_line] + [line for line in all_lines if line != key_line]

line_colors = {}
if len(all_lines) > 0:
    color_map = pl.cm.get_cmap("tab20", max(len(all_lines), 1))
    for i, line in enumerate(all_lines):
        line_colors[line] = color_map(i)

for dendrofile in dendrofiles:
    cloud = dendrofile.split("/")[1].split("_")[0]
    cloud_is_new = cloud not in processed_clouds
    if cloud_is_new:
        print("Processing dendrogram file:", dendrofile)
        d=Dendrogram.load_from("dendro/"+cloud+"_13CO10_dendrogram.hdf5")

        linefiles = cloud_linefiles.get(cloud, np.array([]))
        lines=np.array([x.split("/")[-1].split("_")[1] for x in linefiles])
        # put key line first
        if key_line in lines:
            z = np.where(lines==key_line)[0][0]
            linefiles=np.concatenate([[linefiles[z]],np.delete(linefiles,z)])

        stats={}
        key_header = fits.getheader(linefiles[0])
        chanwid_kms = abs(key_header["CDELT3"]) / key_header["RESTFRQ"] * 299792.458
        pixel_area_arcsec2 = abs(key_header["CDELT1"] * key_header["CDELT2"]) * 3600**2
        pixperbm = 4**2 * np.pi / (4 * np.log(2) * pixel_area_arcsec2)
        for linefile in linefiles:
            #if 'C18O10' in linefile: debug=True

            stat={}
            stat['linefile']=linefile
            stat['line']=linefile.split("/")[-1].split("_")[1]
            stat['chanwid_kms'] = chanwid_kms
            stat['pixperbm'] = pixperbm

            linedata=fits.getdata(linefile)
            linermsspec = np.nanstd(linedata, axis=(1, 2))
            valid_channels = np.flatnonzero(
                np.any(np.isfinite(linedata), axis=(1, 2))
            )
            edge_channels = np.unique(
                np.concatenate([valid_channels[:5], valid_channels[-5:]])
            )
            linerms = np.nanmedian(linermsspec[edge_channels])

            pk = np.zeros([len(d),2]) # true peak of angle-summed spectrum, divided by area at that channel, and divided by 2d mask area
            pk_from_mom = np.zeros([len(d),2])
            pk_from_fit = np.zeros([len(d),2])
            moments = np.zeros([3,len(d)])
            npix_2d = np.zeros(len(d), dtype=int)
            fitparms = np.zeros([len(d),3])
            fiterrs = np.full([len(d),3], np.nan)
            for x in d:
                mask=x.get_mask()
                # linedata*mask puts zeros where mask is False, so nansum works here, nanmean doesn't
                spectrum=np.nansum(linedata*mask, axis=(1, 2))

                # collect area spectrum, and create 2D mask
                mask2d = mask.sum(axis=0) > 0
                npix_2d[x.idx] = np.sum(mask2d)
                areaspectrum = np.zeros(linedata.shape[0])
                for i in range(linedata.shape[0]):
                    areaspectrum[i] = np.sum(mask[i])


                pkindex = np.nanargmax(spectrum)
                pk[x.idx,0]=np.nanmax(spectrum)/areaspectrum[pkindex]
                pk[x.idx,1]=np.nanmax(spectrum)/np.sum(mask2d)

                mom0=np.nansum(linedata*mask) # K*km/s*pix2
                moments[0,x.idx]=mom0
                moments[1,x.idx]=np.nansum(np.arange(linedata.shape[0])*spectrum)/mom0 # K*km/s*pix2 weighted mean
                moments[2,x.idx]=np.sqrt(np.nansum((np.arange(linedata.shape[0])-moments[1,x.idx])**2*spectrum)/mom0) # K*km/s*pix2 weighted std

                if np.isfinite(moments[2,x.idx]):
                    # divide by area of nearest channel to the mom1
                    pk_from_mom[x.idx,0] = moments[0,x.idx]/moments[2,x.idx]/np.sqrt(2*np.pi)/areaspectrum[int(np.round(moments[1,x.idx]))]
                    # divide by the total 2D mask area
                    pk_from_mom[x.idx,1] = moments[0,x.idx]/moments[2,x.idx]/np.sqrt(2*np.pi)/np.sum(mask2d)

                if stat['line']==key_line:
                    momentsk=moments
                else:
                    momentsk=stats[key_line]['moments']
                # now get a wider range of indices over which we fit a Gaussian, over +/- 3 sigma
                vindices = np.arange(
                    np.max([np.floor(momentsk[1,x.idx]-3*momentsk[2,x.idx]),0]),
                    np.min([np.ceil(momentsk[1,x.idx]+3*momentsk[2,x.idx]),linedata.shape[0]-1]),
                    dtype=int)

                # gaussfit a,b,v0,vsig    [ y=a*exp(-0.5*((v-v0)/vsig)**2)+b ]
                z=np.where(np.isfinite(spectrum[vindices]))[0]
                vindices=vindices[z]

                # TODO would be better to seed the amplitude with the cube rms
                try:
                    popt, pcov = curve_fit(
                        lambda x, *p: p[0] * np.exp(-0.5*((x-p[1])/p[2])**2),
                        vindices,spectrum[vindices],
                        bounds=[(np.min([np.nanmin(spectrum[vindices]),-1e-3]),
                                 momentsk[1,x.idx]-momentsk[2,x.idx],  # center can move by 1 sigma
                                 0.5*momentsk[2,x.idx]), # width can vary from half to twice the initial guess
                                (np.max([np.nanmax(spectrum[vindices]),1e-3]),
                                 momentsk[1,x.idx]+momentsk[2,x.idx],
                                 2*momentsk[2,x.idx])],
                        p0=[1e-4,momentsk[1,x.idx],momentsk[2,x.idx]])

                    fitparms[x.idx]=popt
                    fiterrs[x.idx]=np.sqrt(np.clip(np.diag(pcov), 0, np.inf))

                    # divide by area of nearest channel to the mom1
                    pk_from_fit[x.idx,0] = popt[0]/areaspectrum[int(np.round(popt[1]))]
                    # divide by the total 2D mask area
                    pk_from_fit[x.idx,1] = popt[0]/np.sum(mask2d)
                except Exception as err:
                    print(f"curve_fit failed for cloud={cloud} line={stat['line']} idx={x.idx}: {err}")
                    fitparms[x.idx] = np.nan
                    fiterrs[x.idx] = np.nan
                    pk_from_fit[x.idx] = np.nan
                    continue

                stat['pk_from_mom'] = pk_from_mom
                stat['pk_from_fit'] = pk_from_fit
                # check some of these for C18O ; maybe put them out in a bank of plots

                if debug:
                    pl.clf()
                    pl.plot(spectrum,'*')
                    pl.plot(vindices,popt[0] * np.exp(-0.5*((vindices-popt[1])/popt[2])**2))
                    pl.title(f"Peak spectrum for {stat['line']} at dendrogram index {x.idx}")
                    pl.xlim(np.max([np.min(vindices),0]), np.min([np.max(vindices),linedata.shape[0]-1]))
                    pl.show()



            stat['linerms'] = linerms
            stat['pk']=pk
            stat['moments']=moments
            stat['npix_2d']=npix_2d
            stat['fitparms']=fitparms
            stat['fiterrs']=fiterrs
            stats[stat['line']]=stat
            print(stat['line']+" done")
    else:
        print("Skipping already processed cloud (using saved rows for plotting):", cloud)
        stats = table_to_stats_for_cloud(stats_table, cloud)

    if len(stats) == 0:
        print("No stats available for cloud:", cloud)
        continue

    lines = np.array(list(stats.keys()))
    if key_line in lines:
        z = np.where(lines == key_line)[0][0]
        lines = np.concatenate([[lines[z]], np.delete(lines, z)])


    # diagnostic - do the different peak finding methods agree?
    pl.gcf().set_size_inches(10, 8)
    pl.clf()
    for line in lines:
        if line not in stats:
            continue
        pl.subplot(221)
        myplot, = pl.plot(stats[line]['pk_from_mom'][:,0],stats[line]['pk'][:,0],'.', label=line, color=line_colors.get(line))
        pl.subplot(222)
        pl.plot(stats[line]['pk_from_mom'][:,0],stats[line]['pk_from_fit'][:,0],'.',color=myplot.get_color())
        pl.subplot(223)
        pl.plot(stats[line]['pk_from_mom'][:,0],stats[line]['pk_from_mom'][:,1],'.', color=myplot.get_color())
        pl.subplot(224)
        pl.plot(stats[line]['pk'][:,0],stats[line]['pk'][:,1],'.',color=myplot.get_color())
    pl.subplot(221)
    pl.xlabel('Mom0 / Mom2 / sqrt(2*pi)')
    pl.ylabel('Peak')
    pl.legend(prop={'size': 8})
    pl.xscale('log')
    pl.yscale('log')
    pl.plot(pl.xlim(),pl.xlim(),'k',alpha=0.3)
    
    pl.subplot(222)
    pl.xlabel('Mom0 / Mom2 / sqrt(2*pi)')
    pl.ylabel('Peak from fit')
    pl.xscale('log')
    pl.yscale('log')
    pl.plot(pl.xlim(),pl.xlim(),'k',alpha=0.3)
    pl.ylim(pl.xlim())
    
    pl.subplot(223)
    pl.xlabel('mom peak w/1d area')
    pl.ylabel('mom peak w/2d area')
    pl.xscale('log')
    pl.yscale('log')
    pl.plot(pl.xlim(),pl.xlim(),'k',alpha=0.3)
    
    pl.subplot(224)
    pl.xlabel('Peak w/1d area')
    pl.ylabel('Peak w/2d area')
    pl.xscale('log')
    pl.yscale('log')
    pl.plot(pl.xlim(),pl.xlim(),'k',alpha=0.3)
    
    pl.subplots_adjust(top=0.95,right=0.95,hspace=0.25,wspace=0.25)
    pl.savefig(os.path.join(plot_dir, cloud+"_peak_vs_mom0_over_mom2_log.png"))
    
    for k in range(4):
        pl.subplot(2,2,1+k)
        pl.xscale("linear") 
        pl.yscale("linear") 
        pl.xlim(-1,2)
        pl.ylim(-1,2)
    pl.savefig(os.path.join(plot_dir, cloud+"_peak_vs_mom0_over_mom2.png"))


    if cloud_is_new:
        cloud_table = stats_to_table(cloud, stats)
        if len(cloud_table) > 0:
            if len(stats_table) == 0:
                stats_table = cloud_table
            else:
                stats_table = vstack([stats_table, cloud_table], metadata_conflicts="silent")
        processed_clouds.add(cloud)

# at the end of processing all dendrogram files, save allstats to a flat ECSV table
stats_table.write(table_file, format="ascii.ecsv", overwrite=True)


