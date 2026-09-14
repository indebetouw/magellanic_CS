from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np
import matplotlib.pyplot as pl
from glob import glob
pl.ion()


baseline = '13CO10'
faintline = 'C18O10'
cloud = 'N159W'
nbin=3
evenbinsize = True
velshift = False
median_spectra = False
nsigma = 2

if median_spectra:
    stack_function = np.nanmedian
    stack_name = "Median"
else:
    stack_function = np.nanmean
    stack_name = "Mean"

ddir = "../products_1pc/"
base_files = glob(f"{ddir}/{cloud}_{baseline}_*.fits")
faint_files = glob(f"{ddir}/{cloud}_{faintline}_*.fits")

if len(base_files) == 0:
    raise FileNotFoundError(f"No base files found for {cloud} {baseline}")
elif len(base_files) > 1:
    raise FileExistsError(f"Multiple base files found for {cloud} {baseline}")

if len(faint_files) == 0:
    raise FileNotFoundError(f"No faint files found for {cloud} {faintline}")
elif len(faint_files) > 1:
    raise FileExistsError(f"Multiple faint files found for {cloud} {faintline}")

base_hdu = fits.open(base_files[0])[0]
faint_hdu = fits.open(faint_files[0])[0]

peak_base = np.nanmax(base_hdu.data,axis=0)

def velocity_axis(cube_hdu):
    channel = np.arange(cube_hdu.data.shape[0])
    frequency = (
        cube_hdu.header["CRVAL3"]
        + (channel + 1 - cube_hdu.header["CRPIX3"]) * cube_hdu.header["CDELT3"]
    )
    restfreq = cube_hdu.header["RESTFRQ"]
    return (restfreq - frequency) / restfreq * 299792.458

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

fig, axes = pl.subplots(2, 1, sharex=False)
for panel_index, (axis, (line, cube_hdu)) in enumerate(zip(
    axes,
    [(baseline, base_hdu), (faintline, faint_hdu)],
)):
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

    for ibin in valid_bins[::-1]:
        axis.plot(
            velocity_plot,
            stacked_spectra[ibin],
            color=line_colors[ibin],
            label=f"{mean_peak_intensity[ibin]:.2f} K",
        )
    axis.set_ylabel(f"{stack_name} {line} intensity (K)")
    if panel_index == 0:
        axis.legend(title="Mean peak",prop={"size": 8})
if velshift:
    axes[-1].set_xlabel("Velocity offset (km/s)")
else:
    axes[-1].set_xlabel("Velocity (km/s)")
#pl.yscale("log")
#pl.ylim(peak_rms/5, None)
filename = f"{cloud}_{stack_name.lower()}_spectra"
if median_spectra:
    filename+= "_median"
else:
    filename+= "_mean"
if velshift:
    pl.savefig(f"{filename}_velshift.png")
else:
    pl.savefig(f"{filename}.png")

print("Peak intensities for each bin:")
for ibin in valid_bins:
    print(f"Bin {ibin}: {mean_peak_intensity[ibin]:.2f} K")


print("Finished saving spectra plots.")