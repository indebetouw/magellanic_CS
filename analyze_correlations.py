"""Plot voxel-by-voxel distributions for two spectral-line cubes in one cloud."""

from argparse import ArgumentParser
from glob import glob
from pathlib import Path

from astropy.io import fits
from matplotlib.colors import LogNorm
import matplotlib.pyplot as pl
import numpy as np


PRODUCT_DIRECTORY = "../products_1pc"
DEFAULT_CLOUD = "A439"
key_line = "13CO10"
other_line = "C18O10"
OUTPUT_DIRECTORY = "analyze_correlation_plots"
MINIMUM_THRESHOLD_VOXELS = 1000
LOW_BRIGHTNESS_THRESHOLD = 0.1
LOW_BRIGHTNESS_BIN_COUNT = 15
HIGH_BRIGHTNESS_VOXELS_PER_BIN = 3000


def find_cube(cloud, line):
    pattern = f"{PRODUCT_DIRECTORY}/{cloud}_*{line}*4asec_grid_K.fits"
    paths = glob(pattern)
    if len(paths) != 1:
        raise RuntimeError(f"Expected one cube matching {pattern}, found {paths}")
    return paths[0]


def find_clouds():
    pattern = f"{PRODUCT_DIRECTORY}/*_{key_line}_*4asec_grid_K.fits"
    return sorted({Path(path).name.split("_", 1)[0] for path in glob(pattern)})


def cumulative_means(key_line_values, other_line_values, bins):
    positive = key_line_values > 0
    key_line_values = key_line_values[positive]
    other_line_values = other_line_values[positive]
    if len(key_line_values) == 0:
        raise RuntimeError(f"No positive {key_line} voxel values found")

    order = np.argsort(key_line_values)
    key_line_sorted = key_line_values[order]
    other_line_sorted = other_line_values[order]
    maximum_threshold = key_line_sorted[-min(MINIMUM_THRESHOLD_VOXELS, len(key_line_sorted))]
    thresholds = np.linspace(np.min(key_line_sorted), maximum_threshold, bins)
    key_line_sums = np.concatenate(([0.0], np.cumsum(key_line_sorted[::-1])))
    other_line_sums = np.concatenate(([0.0], np.cumsum(other_line_sorted[::-1])))
    first_indices = np.searchsorted(key_line_sorted, thresholds, side="left")
    counts = len(key_line_sorted) - first_indices
    mean_key_line = key_line_sums[counts] / counts
    mean_other_line = other_line_sums[counts] / counts
    return mean_key_line, mean_other_line, counts, thresholds


def differential_means(key_line_values, other_line_values, bins):
    positive = key_line_values > 0
    key_line_values = key_line_values[positive]
    other_line_values = other_line_values[positive]
    if len(key_line_values) == 0:
        raise RuntimeError(f"No positive {key_line} voxel values found")

    order = np.argsort(key_line_values)
    key_line_sorted = key_line_values[order]
    other_line_sorted = other_line_values[order]
    low_brightness_end = np.searchsorted(
        key_line_sorted,
        LOW_BRIGHTNESS_THRESHOLD,
        side="left",
    )
    high_brightness_count = len(key_line_sorted) - low_brightness_end
    full_high_brightness_bins = high_brightness_count // HIGH_BRIGHTNESS_VOXELS_PER_BIN
    high_brightness_start = len(key_line_sorted) - (
        full_high_brightness_bins * HIGH_BRIGHTNESS_VOXELS_PER_BIN
    )
    low_brightness_edges = np.linspace(
        0,
        high_brightness_start,
        LOW_BRIGHTNESS_BIN_COUNT + 1,
        dtype=int,
    )
    high_brightness_edges = np.arange(
        high_brightness_start,
        len(key_line_sorted) + 1,
        HIGH_BRIGHTNESS_VOXELS_PER_BIN,
        dtype=int,
    )
    bin_edges = np.unique(
        np.concatenate((low_brightness_edges, high_brightness_edges[1:]))
    )
    first_indices = bin_edges[:-1]
    last_indices = bin_edges[1:]
    counts = last_indices - first_indices
    key_line_sums = np.concatenate(([0.0], np.cumsum(key_line_sorted)))
    other_line_sums = np.concatenate(([0.0], np.cumsum(other_line_sorted)))
    mean_key_line = (key_line_sums[last_indices] - key_line_sums[first_indices]) / counts
    mean_other_line = (other_line_sums[last_indices] - other_line_sums[first_indices]) / counts
    thresholds = key_line_sorted[first_indices]
    return mean_key_line, mean_other_line, counts, thresholds


def noise_per_selected_voxel(other_line_rms, counts, pixels_per_beam):
    return other_line_rms / np.sqrt(counts / pixels_per_beam)


def shifted_spectra(spectra, center_channels):
    """Shift spectra to a shared center channel without wrapping edge values."""
    channel_count, spectrum_count = spectra.shape
    centered_spectra = np.full((channel_count, spectrum_count), np.nan)
    center_channel = channel_count // 2
    source_channels = np.arange(channel_count)
    for spectrum_index, center in enumerate(center_channels):
        destination_channels = source_channels - center + center_channel
        valid = (destination_channels >= 0) & (destination_channels < channel_count)
        centered_spectra[destination_channels[valid], spectrum_index] = spectra[
            source_channels[valid], spectrum_index
        ]
    return centered_spectra


def weighted_spectrum_mean(spectra, weights):
    finite = np.isfinite(spectra)
    weighted_values = np.where(finite, spectra * weights[np.newaxis, :], 0.0)
    weight_sums = np.sum(finite * weights[np.newaxis, :], axis=1)
    return np.divide(
        np.sum(weighted_values, axis=1),
        weight_sums,
        out=np.full(spectra.shape[0], np.nan),
        where=weight_sums > 0,
    )


def spectrum_mean(spectra):
    finite = np.isfinite(spectra)
    counts = np.sum(finite, axis=1)
    return np.divide(
        np.nansum(spectra, axis=1),
        counts,
        out=np.full(spectra.shape[0], np.nan),
        where=counts > 0,
    )


def plot_aligned_spectra(
    cloud,
    key_line_data,
    other_line_data,
    highest_bin_mask,
    output,
):
    selected_pixels = np.any(highest_bin_mask, axis=0)
    key_line_spectra = key_line_data[:, selected_pixels]
    other_line_spectra = other_line_data[:, selected_pixels]
    channels = np.arange(key_line_data.shape[0])
    moment_zero = np.nansum(key_line_spectra, axis=0)
    moment_one = np.divide(
        np.nansum(channels[:, np.newaxis] * key_line_spectra, axis=0),
        moment_zero,
        out=np.full(key_line_spectra.shape[1], np.nan),
        where=moment_zero > 0,
    )
    peak_weights = np.nanmax(key_line_spectra, axis=0)
    valid_spectra = np.isfinite(moment_one) & np.isfinite(peak_weights) & (peak_weights > 0)
    if not np.any(valid_spectra):
        raise RuntimeError(f"No valid {key_line} spectra in the highest threshold bin")

    centered_key_line = shifted_spectra(
        key_line_spectra[:, valid_spectra],
        np.rint(moment_one[valid_spectra]).astype(int),
    )
    centered_other_line = shifted_spectra(
        other_line_spectra[:, valid_spectra],
        np.rint(moment_one[valid_spectra]).astype(int),
    )
    weights = peak_weights[valid_spectra]
    centered_channels = channels - (key_line_data.shape[0] // 2)

    figure, axis = pl.subplots(figsize=(6, 5), constrained_layout=True)
    axis.plot(centered_channels, spectrum_mean(centered_key_line), label=f"{key_line} unweighted")
    axis.plot(
        centered_channels,
        weighted_spectrum_mean(centered_key_line, weights),
        label=f"{key_line} peak-weighted",
    )
    axis.plot(centered_channels, spectrum_mean(centered_other_line), label=f"{other_line} unweighted")
    axis.plot(
        centered_channels,
        weighted_spectrum_mean(centered_other_line, weights),
        label=f"{other_line} peak-weighted",
    )
    axis.set_xlabel(f"Channel offset from {key_line} moment 1")
    axis.set_ylabel("Mean brightness temperature (K)")
    axis.set_title(f"{cloud} highest-threshold pixel spectra (N = {np.sum(valid_spectra):,})")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=200)
    pl.close(figure)
    print(f"Saved {output} from {np.sum(valid_spectra):,} centered pixel spectra")


def plot_cumulative_means(
    cloud,
    key_line_values,
    other_line_values,
    shuffled_other_line_values,
    highest_bin_image,
    other_line_rms,
    pixels_per_beam,
    bins,
    output,
):
    mean_key_line, mean_other_line, counts, _ = cumulative_means(
        key_line_values,
        other_line_values,
        bins,
    )

    figure, axis = pl.subplots(figsize=(6, 5), constrained_layout=True)
    axis.plot(mean_key_line, mean_other_line, ".-", markersize=4, label=f"Aligned {other_line}")
    for index in range(len(mean_key_line) - 5, len(mean_key_line)):
        axis.annotate(
            f"{counts[index]:,}",
            (mean_key_line[index], mean_other_line[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    shuffled_other_line_means = []
    for shuffled_values in shuffled_other_line_values:
        shuffled_mean_key_line, shuffled_mean_other_line, _, _ = cumulative_means(
            key_line_values,
            shuffled_values,
            bins,
        )
        shuffled_other_line_means.append(shuffled_mean_other_line)
        axis.plot(
            shuffled_mean_key_line,
            shuffled_mean_other_line,
            ".-",
            markersize=4,
            color="0.6",
            alpha=0.7,
        )
    mean_shuffled_other_line = np.mean(shuffled_other_line_means, axis=0)
    axis.plot(
        mean_key_line,
        mean_shuffled_other_line,
        "ko",
        markersize=4,
        label=f"Mean shuffled {other_line} ({len(shuffled_other_line_values)} realizations)",
    )
    axis.plot(
        mean_key_line,
        noise_per_selected_voxel(other_line_rms, counts, pixels_per_beam),
        "k--",
        label=rf"{other_line} RMS / $\sqrt{{N_{{\rm voxels}}/N_{{\rm pix/beam}}}}$",
    )
    axis.set_xlabel(f"Mean {key_line} for voxels above threshold (K)")
    axis.set_ylabel(f"Mean {other_line} for the same voxels (K)")
    axis.set_title(f"{cloud} cumulative above-threshold voxel means")
    axis.grid(alpha=0.25)
    #axis.set_xscale("log")
    #axis.set_yscale("log")
    axis.legend(loc="center left")
    inset_axis = axis.inset_axes([0.03, 0.64, 0.30, 0.30])
    inset_axis.imshow(highest_bin_image, origin="lower", cmap="magma")
    inset_axis.set_title("Highest-threshold voxels", fontsize=8)
    inset_axis.set_xticks([])
    inset_axis.set_yticks([])
    figure.savefig(output, dpi=200)
    pl.close(figure)
    print(f"Saved {output} using {bins} linearly spaced {key_line} thresholds")


def plot_differential_means(
    cloud,
    key_line_values,
    other_line_values,
    shuffled_other_line_values,
    highest_differential_bin_image,
    other_line_rms,
    pixels_per_beam,
    bins,
    output,
):
    mean_key_line, mean_other_line, counts, _ = differential_means(
        key_line_values,
        other_line_values,
        bins,
    )

    figure, axis = pl.subplots(figsize=(6, 5), constrained_layout=True)
    axis.plot(mean_key_line, mean_other_line, ".-", markersize=4, label=f"Aligned {other_line}")
    for index in range(len(mean_key_line) - 5, len(mean_key_line)):
        axis.annotate(
            f"{counts[index]:,}",
            (mean_key_line[index], mean_other_line[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    shuffled_other_line_means = []
    for shuffled_values in shuffled_other_line_values:
        shuffled_mean_key_line, shuffled_mean_other_line, _, _ = differential_means(
            key_line_values,
            shuffled_values,
            bins,
        )
        shuffled_other_line_means.append(shuffled_mean_other_line)
        axis.plot(
            shuffled_mean_key_line,
            shuffled_mean_other_line,
            ".-",
            markersize=4,
            color="0.6",
            alpha=0.7,
        )
    mean_shuffled_other_line = np.mean(shuffled_other_line_means, axis=0)
    axis.plot(
        mean_key_line,
        mean_shuffled_other_line,
        "ko",
        markersize=4,
        label=f"Mean shuffled {other_line} ({len(shuffled_other_line_values)} realizations)",
    )
    axis.plot(
        mean_key_line,
        noise_per_selected_voxel(other_line_rms, counts, pixels_per_beam),
        "k--",
        label=rf"{other_line} RMS / $\sqrt{{N_{{\rm voxels}}/N_{{\rm pix/beam}}}}$",
    )
    axis.set_xlabel(f"Mean {key_line} in threshold interval (K)")
    axis.set_ylabel(f"Mean {other_line} for the same voxels (K)")
    axis.set_title(f"{cloud} differential threshold voxel means")
    axis.grid(alpha=0.25)
    axis.legend(loc="center left")
    inset_axis = axis.inset_axes([0.03, 0.64, 0.30, 0.30])
    inset_axis.imshow(highest_differential_bin_image, origin="lower", cmap="magma")
    inset_axis.set_title("Brightest-threshold interval", fontsize=8)
    inset_axis.set_xticks([])
    inset_axis.set_yticks([])
    figure.savefig(output, dpi=200)
    pl.close(figure)
    print(
        f"Saved {output} using {LOW_BRIGHTNESS_BIN_COUNT} linear intervals below "
        f"approximately {LOW_BRIGHTNESS_THRESHOLD} K and "
        f"{HIGH_BRIGHTNESS_VOXELS_PER_BIN}-voxel intervals above"
    )


def analyze_cloud(cloud, args, output_directory):
    output = output_directory / f"{cloud}_{key_line}_vs_{other_line}_pixel_distribution.png"
    cumulative_output = output_directory / f"{cloud}_{key_line}_vs_{other_line}_cumulative_means.png"
    differential_output = output_directory / f"{cloud}_{key_line}_vs_{other_line}_differential_means.png"
    spectra_output = output_directory / f"{cloud}_{key_line}_vs_{other_line}_highest_bin_spectra.png"

    key_line_file = find_cube(cloud, key_line)
    other_line_file = find_cube(cloud, other_line)
    key_line_data = fits.getdata(key_line_file, memmap=True)
    other_line_data = fits.getdata(other_line_file, memmap=True)
    other_line_header = fits.getheader(other_line_file)

    if key_line_data.shape != other_line_data.shape:
        raise RuntimeError(
            f"Cube shape mismatch: {key_line_data.shape} versus {other_line_data.shape}"
        )

    other_line_noise_channels = np.concatenate((other_line_data[:5], other_line_data[-5:]))
    other_line_rms = np.sqrt(np.nanmean(other_line_noise_channels**2))
    pixels_per_beam = (
        other_line_header["BMAJ"]
        * other_line_header["BMIN"]
        / other_line_header["CDELT2"] ** 2
        * np.pi
        / (4 * np.log(2))
    )

    other_line_valid_spatial_pixels = np.all(np.isfinite(other_line_data), axis=0)
    good = np.isfinite(key_line_data) & other_line_valid_spatial_pixels
    key_line_values = key_line_data[good]
    other_line_values = other_line_data[good]
    if len(key_line_values) == 0:
        raise RuntimeError(f"No finite {key_line}/{other_line} voxel pairs found")

    _, _, _, thresholds = cumulative_means(key_line_values, other_line_values, args.threshold_bins)
    highest_bin_mask = good & (key_line_data >= thresholds[-1])
    highest_bin_image = np.sum(highest_bin_mask, axis=0)
    plot_aligned_spectra(
        cloud,
        key_line_data,
        other_line_data,
        highest_bin_mask,
        spectra_output,
    )
    _, _, _, differential_thresholds = differential_means(
        key_line_values,
        other_line_values,
        args.threshold_bins,
    )
    highest_differential_bin_mask = good & (key_line_data >= differential_thresholds[-1])
    highest_differential_bin_image = np.sum(highest_differential_bin_mask, axis=0)

    random_generator = np.random.default_rng(args.shuffle_seed)
    shuffled_other_line_values = []
    for realization in range(args.shuffle_realizations):
        shuffled_other_line = other_line_data[
            random_generator.permutation(other_line_data.shape[0])
        ]
        shuffled_values = shuffled_other_line[good]
        if not np.all(np.isfinite(shuffled_values)):
            raise RuntimeError(
                f"Shuffled {other_line} realization {realization + 1} has non-finite selected pixels"
            )
        shuffled_other_line_values.append(shuffled_values)

    figure, axis = pl.subplots(figsize=(6, 5), constrained_layout=True)
    histogram = axis.hist2d(
        key_line_values,
        other_line_values,
        bins=args.bins,
        norm=LogNorm(),
        cmap="viridis",
    )
    colorbar = figure.colorbar(histogram[3], ax=axis)
    colorbar.set_label("Number of voxels")
    axis.set_xlabel(f"{key_line} brightness temperature (K)")
    axis.set_ylabel(f"{other_line} brightness temperature (K)")
    axis.set_title(f"{cloud} voxel distribution (N = {len(key_line_values):,})")
    figure.savefig(output, dpi=200)
    pl.close(figure)
    print(f"Saved {output} from {len(key_line_values):,} finite voxel pairs")
    plot_cumulative_means(
        cloud,
        key_line_values,
        other_line_values,
        shuffled_other_line_values,
        highest_bin_image,
        other_line_rms,
        pixels_per_beam,
        args.threshold_bins,
        cumulative_output,
    )
    plot_differential_means(
        cloud,
        key_line_values,
        other_line_values,
        shuffled_other_line_values,
        highest_differential_bin_image,
        other_line_rms,
        pixels_per_beam,
        args.threshold_bins,
        differential_output,
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloud",
        action="append",
        help="Cloud identifier to analyze; repeat to select multiple clouds",
    )
    parser.add_argument("--bins", type=int, default=300, help="Number of bins per axis")
    parser.add_argument(
        "--threshold-bins",
        type=int,
        default=40,
        help=f"Number of linearly spaced {key_line} thresholds for cumulative means",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=12345,
        help=f"Random seed for the {other_line} channel permutation",
    )
    parser.add_argument(
        "--shuffle-realizations",
        type=int,
        default=100,
        help=f"Number of shuffled {other_line} channel realizations",
    )
    args = parser.parse_args()
    output_directory = Path(OUTPUT_DIRECTORY)
    output_directory.mkdir(exist_ok=True)
    clouds = args.cloud or find_clouds()
    if not clouds:
        raise RuntimeError(f"No {key_line} cubes found in {PRODUCT_DIRECTORY}")
    for cloud in clouds:
        analyze_cloud(cloud, args, output_directory)


if __name__ == "__main__":
    main()