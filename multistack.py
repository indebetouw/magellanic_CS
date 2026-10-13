from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as pl
from glob import glob
pl.ion()

# this stacks spectra of faint lines in bins of peak intensity of a baseline line

baseline = '13CO10'
faintlines = ['C18O10','CS21']
faintlines = ['C18O10']
cloud = 'GMC104'
nbin=4
evenbinsize = False # put same # pixels in each bin, approximately
velshift = True
median_spectra = False
nsigma = 2 # cutoff below which we ignore values in the base line

if median_spectra:
    stack_function = np.nanmedian
    stack_name = "Median"
else:
    stack_function = np.nanmean
    stack_name = "Mean"

ddir = "../products_1pc/"
base_files = glob(f"{ddir}/{cloud}_{baseline}_*.fits")

if len(base_files) == 0:
    raise FileNotFoundError(f"No base files found for {cloud} {baseline}")
elif len(base_files) > 1:
    raise FileExistsError(f"Multiple base files found for {cloud} {baseline}")

faint_hdus = []
for faintline in faintlines:
    faint_files = glob(f"{ddir}/{cloud}_{faintline}_*.fits")
    if len(faint_files) == 0:
        raise FileNotFoundError(f"No faint files found for {cloud} {faintline}")
    elif len(faint_files) > 1:
        raise FileExistsError(f"Multiple faint files found for {cloud} {faintline}")
    faint_hdus.append(fits.open(faint_files[0])[0])

base_hdu = fits.open(base_files[0])[0]

peak_base = np.nanmax(base_hdu.data,axis=0)

def velocity_axis(cube_hdu):
    channel = np.arange(cube_hdu.data.shape[0])
    frequency = (
        cube_hdu.header["CRVAL3"]
        + (channel + 1 - cube_hdu.header["CRPIX3"]) * cube_hdu.header["CDELT3"]
    )
    restfreq = cube_hdu.header["RESTFRQ"]
    return (restfreq - frequency) / restfreq * 299792.458


def hanning_smooth(spectrum):
    # classic radio-astronomy 3-tap Hanning smooth: weights 0.25, 0.5, 0.25
    window = np.hanning(5)[1:-1]
    window = window / window.sum()
    padded = np.pad(spectrum, 1, mode="edge")
    return np.convolve(padded, window, mode="valid")


def hanning_smooth_wide(spectrum):
    # wider 5-tap Hanning smooth for heavier smoothing
    window = np.hanning(7)[1:-1]
    window = window / window.sum()
    padded = np.pad(spectrum, 2, mode="edge")
    return np.convolve(padded, window, mode="valid")

finite_peak_values = peak_base[np.isfinite(peak_base)]
clipped_peak_values = sigma_clip(finite_peak_values, sigma=3, maxiters=3).compressed()
if len(clipped_peak_values) == 0:
    raise ValueError("peak_base has no finite unmasked pixels for RMS calculation")
peak_rms = np.sqrt(np.mean(clipped_peak_values**2))
peak_threshold = nsigma * peak_rms
print(f"peak_threshold={peak_threshold}")

finite_peak = np.isfinite(peak_base) & (peak_base >= peak_threshold)
peak_values = np.sort(peak_base[finite_peak])
npix = len(peak_values)

if npix == 0:
    raise ValueError("peak_base has no finite pixels above the clipped RMS threshold to bin")
if nbin < 1:
    raise ValueError("nbin must be at least 1")
if evenbinsize and npix < nbin:
    raise ValueError("nbin cannot exceed the number of finite peak_base pixels")

peak_bin = np.full(peak_base.shape, -1, dtype=int)
if evenbinsize:
    bin_size = npix // nbin
    extra_faint_pixels = npix % nbin
    bin_ends = extra_faint_pixels + bin_size * np.arange(1, nbin + 1)
    intensity_bins = np.concatenate(([peak_values[0]], peak_values[bin_ends - 1]))

    ranked_bin = np.empty(npix, dtype=int)
    sorted_index = np.argsort(peak_base[finite_peak])
    bin_starts = np.concatenate(([0], bin_ends[:-1]))
    for ibin, (start, end) in enumerate(zip(bin_starts, bin_ends)):
        ranked_bin[sorted_index[start:end]] = ibin
    peak_bin[finite_peak] = ranked_bin
else:
    intensity_bins = np.linspace(peak_threshold, np.max(finite_peak_values), nbin + 1)
    peak_bin[finite_peak] = np.searchsorted(
        intensity_bins[1:-1],
        peak_base[finite_peak],
        side="right",
    )

bin_counts = np.bincount(peak_bin[finite_peak], minlength=nbin)
valid_bins = np.where(bin_counts > 0)[0]
if len(valid_bins) == 0:
    raise ValueError("No populated intensity bins to plot")

mean_peak_intensity = np.full(nbin, np.nan)

for ibin in valid_bins:
    bin_mask = peak_bin == ibin
    mean_peak_intensity[ibin] = np.nanmean(peak_base[bin_mask])

line_colors = pl.cm.jet(np.linspace(0, 1, nbin))
base_velocity = velocity_axis(base_hdu)
base_peak_velocity = np.full(peak_base.shape, np.nan)
base_peak_velocity[finite_peak] = base_velocity[
    np.nanargmax(base_hdu.data[:, finite_peak], axis=0)
]

pl.rcParams["axes.labelsize"] = 8
pl.rcParams["xtick.labelsize"] = 7
pl.rcParams["ytick.labelsize"] = 7

fig, axes = pl.subplots(2, 2, sharex=False)
spectra_axes = {baseline: axes[0, 0], faintlines[0]: axes[1, 0]}
lines_info = [(baseline, base_hdu)] + list(zip(faintlines, faint_hdus))
stacked_peak_intensity = {line: np.full(nbin, np.nan) for line, _ in lines_info}

for panel_index, (line, cube_hdu) in enumerate(lines_info):
    velocity = velocity_axis(cube_hdu)
    if velshift:
        velocity_plot = velocity - np.nanmedian(base_peak_velocity[finite_peak])
    else:
        velocity_plot = velocity

    stacked_spectra = np.full((nbin, cube_hdu.data.shape[0]), np.nan)
    for ibin in valid_bins:
        bin_mask = peak_bin == ibin
        if velshift:
            shifted_spectra = np.full((cube_hdu.data.shape[0], np.sum(bin_mask)), np.nan)
            for pixel_index, (spectrum, peak_velocity) in enumerate(zip(
                cube_hdu.data[:, bin_mask].T,
                base_peak_velocity[bin_mask],
            )):
                finite_spectrum = np.isfinite(spectrum)
                if np.sum(finite_spectrum) < 2:
                    continue
                shifted_velocity = velocity - peak_velocity
                velocity_order = np.argsort(shifted_velocity[finite_spectrum])
                shifted_spectra[:, pixel_index] = np.interp(
                    velocity_plot,
                    shifted_velocity[finite_spectrum][velocity_order],
                    spectrum[finite_spectrum][velocity_order],
                    left=np.nan,
                    right=np.nan,
                )
            stacked_spectra[ibin] = stack_function(shifted_spectra, axis=1)
        else:
            stacked_spectra[ibin] = stack_function(cube_hdu.data[:, bin_mask], axis=1)

    axis = spectra_axes.get(line)
    for ibin in valid_bins[::-1]:
        if np.nanmax(stacked_spectra[ibin]) < 1:
            stacked_spectra[ibin] = hanning_smooth_wide(stacked_spectra[ibin])
        stacked_peak_intensity[line][ibin] = np.nanmax(stacked_spectra[ibin])
        if axis is not None:
            axis.plot(
                velocity_plot,
                stacked_spectra[ibin],
                color=line_colors[ibin],
                label=f"{mean_peak_intensity[ibin]:.2f} K",
            )
    if axis is not None:
        axis.set_ylabel(f"{stack_name} {line} (K)")
        if panel_index == 0:
            if velshift:
                axis.set_title(f"{cloud} shifted and stacked spectra")
            else:
                axis.set_title(f"{cloud} stacked spectra")
            axis.legend(title="Mean peak",prop={"size": 8})
bottom_xlabel = "Velocity offset (km/s)" if velshift else "Velocity (km/s)"
axes[1, 0].set_xlabel(bottom_xlabel)
#pl.yscale("log")
#pl.ylim(peak_rms/5, None)

for faintline in faintlines:
    axes[1, 1].plot(
        stacked_peak_intensity[baseline][valid_bins],
        stacked_peak_intensity[faintline][valid_bins],
        "o-",
        label=faintline,
    )
axes[1, 1].set_xlabel(f"{stack_name} {baseline} peak (K)")
axes[1, 1].set_ylabel(f"{stack_name} faint line peak (K)")
axes[1, 1].legend(fontsize=8)


celestial_wcs = WCS(base_hdu.header).celestial
axes[0, 1].remove()
velocity_map_axis = pl.subplot(2, 2, 2, projection=celestial_wcs)
velocity_map_axis.imshow(base_peak_velocity, origin="lower", cmap="jet")
velocity_map_axis.set_xlabel("RA")
velocity_map_axis.set_ylabel("Dec")
pl.colorbar(velocity_map_axis.imshow(base_peak_velocity, origin="lower", cmap="jet"), ax=velocity_map_axis,label="km/s")



filename = f"{cloud}_{stack_name.lower()}_spectra"
if median_spectra:
    filename+= "_median"
else:
    filename+= "_mean"
if evenbinsize:
    filename += "_evenpixperbin"
if velshift:
    pl.savefig(f"{filename}_velshift.png")
else:
    pl.savefig(f"{filename}.png")

print("Peak intensities for each bin:")
for ibin in valid_bins:
    print(f"Bin {ibin}: {mean_peak_intensity[ibin]:.2f} K")


print("Finished saving spectra plots.")