"""Plot voxel-by-voxel 13CO(1-0) and C18O(1-0) distributions for one cloud."""

from argparse import ArgumentParser
from glob import glob

from astropy.io import fits
from matplotlib.colors import LogNorm
import matplotlib.pyplot as pl
import numpy as np


PRODUCT_DIRECTORY = "../products_1pc"
DEFAULT_CLOUD = "A439"
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


def cumulative_means(co13_values, c18o_values, bins):
    positive = co13_values > 0
    co13_values = co13_values[positive]
    c18o_values = c18o_values[positive]
    if len(co13_values) == 0:
        raise RuntimeError("No positive 13CO voxel values found")

    order = np.argsort(co13_values)
    co13_sorted = co13_values[order]
    c18o_sorted = c18o_values[order]
    maximum_threshold = co13_sorted[-min(MINIMUM_THRESHOLD_VOXELS, len(co13_sorted))]
    thresholds = np.linspace(np.min(co13_sorted), maximum_threshold, bins)
    co13_sums = np.concatenate(([0.0], np.cumsum(co13_sorted[::-1])))
    c18o_sums = np.concatenate(([0.0], np.cumsum(c18o_sorted[::-1])))
    first_indices = np.searchsorted(co13_sorted, thresholds, side="left")
    counts = len(co13_sorted) - first_indices
    mean_co13 = co13_sums[counts] / counts
    mean_c18o = c18o_sums[counts] / counts
    return mean_co13, mean_c18o, counts, thresholds


def differential_means(co13_values, c18o_values, bins):
    positive = co13_values > 0
    co13_values = co13_values[positive]
    c18o_values = c18o_values[positive]
    if len(co13_values) == 0:
        raise RuntimeError("No positive 13CO voxel values found")

    order = np.argsort(co13_values)
    co13_sorted = co13_values[order]
    c18o_sorted = c18o_values[order]
    low_brightness_end = np.searchsorted(
        co13_sorted,
        LOW_BRIGHTNESS_THRESHOLD,
        side="left",
    )
    high_brightness_count = len(co13_sorted) - low_brightness_end
    full_high_brightness_bins = high_brightness_count // HIGH_BRIGHTNESS_VOXELS_PER_BIN
    high_brightness_start = len(co13_sorted) - (
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
        len(co13_sorted) + 1,
        HIGH_BRIGHTNESS_VOXELS_PER_BIN,
        dtype=int,
    )
    bin_edges = np.unique(
        np.concatenate((low_brightness_edges, high_brightness_edges[1:]))
    )
    first_indices = bin_edges[:-1]
    last_indices = bin_edges[1:]
    counts = last_indices - first_indices
    co13_sums = np.concatenate(([0.0], np.cumsum(co13_sorted)))
    c18o_sums = np.concatenate(([0.0], np.cumsum(c18o_sorted)))
    mean_co13 = (co13_sums[last_indices] - co13_sums[first_indices]) / counts
    mean_c18o = (c18o_sums[last_indices] - c18o_sums[first_indices]) / counts
    thresholds = co13_sorted[first_indices]
    return mean_co13, mean_c18o, counts, thresholds


def noise_per_selected_voxel(c18o_rms, counts, pixels_per_beam):
    return c18o_rms / np.sqrt(counts / pixels_per_beam)


def plot_cumulative_means(
    cloud,
    co13_values,
    c18o_values,
    shuffled_c18o_values,
    highest_bin_image,
    c18o_rms,
    pixels_per_beam,
    bins,
    output,
):
    mean_co13, mean_c18o, counts, _ = cumulative_means(
        co13_values,
        c18o_values,
        bins,
    )

    figure, axis = pl.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot(mean_co13, mean_c18o, ".-", markersize=4, label="Aligned C$^{18}$O")
    for index in range(len(mean_co13) - 5, len(mean_co13)):
        axis.annotate(
            f"{counts[index]:,}",
            (mean_co13[index], mean_c18o[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    shuffled_c18o_means = []
    for shuffled_values in shuffled_c18o_values:
        shuffled_mean_co13, shuffled_mean_c18o, _, _ = cumulative_means(
            co13_values,
            shuffled_values,
            bins,
        )
        shuffled_c18o_means.append(shuffled_mean_c18o)
        axis.plot(
            shuffled_mean_co13,
            shuffled_mean_c18o,
            ".-",
            markersize=4,
            color="0.6",
            alpha=0.7,
        )
    mean_shuffled_c18o = np.mean(shuffled_c18o_means, axis=0)
    axis.plot(
        mean_co13,
        mean_shuffled_c18o,
        "ko",
        markersize=4,
        label=f"Mean shuffled C$^{{18}}$O ({len(shuffled_c18o_values)} realizations)",
    )
    axis.plot(
        mean_co13,
        noise_per_selected_voxel(c18o_rms, counts, pixels_per_beam),
        "k--",
        label=r"C$^{18}$O RMS / $\sqrt{N_{\rm voxels}/N_{\rm pix/beam}}$",
    )
    axis.set_xlabel(r"Mean $^{13}$CO(1-0) for voxels above threshold (K)")
    axis.set_ylabel(r"Mean C$^{18}$O(1-0) for the same voxels (K)")
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
    print(f"Saved {output} using {bins} linearly spaced 13CO thresholds")


def plot_differential_means(
    cloud,
    co13_values,
    c18o_values,
    shuffled_c18o_values,
    highest_differential_bin_image,
    c18o_rms,
    pixels_per_beam,
    bins,
    output,
):
    mean_co13, mean_c18o, counts, _ = differential_means(
        co13_values,
        c18o_values,
        bins,
    )

    figure, axis = pl.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot(mean_co13, mean_c18o, ".-", markersize=4, label="Aligned C$^{18}$O")
    for index in range(len(mean_co13) - 5, len(mean_co13)):
        axis.annotate(
            f"{counts[index]:,}",
            (mean_co13[index], mean_c18o[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    shuffled_c18o_means = []
    for shuffled_values in shuffled_c18o_values:
        shuffled_mean_co13, shuffled_mean_c18o, _, _ = differential_means(
            co13_values,
            shuffled_values,
            bins,
        )
        shuffled_c18o_means.append(shuffled_mean_c18o)
        axis.plot(
            shuffled_mean_co13,
            shuffled_mean_c18o,
            ".-",
            markersize=4,
            color="0.6",
            alpha=0.7,
        )
    mean_shuffled_c18o = np.mean(shuffled_c18o_means, axis=0)
    axis.plot(
        mean_co13,
        mean_shuffled_c18o,
        "ko",
        markersize=4,
        label=f"Mean shuffled C$^{{18}}$O ({len(shuffled_c18o_values)} realizations)",
    )
    axis.plot(
        mean_co13,
        noise_per_selected_voxel(c18o_rms, counts, pixels_per_beam),
        "k--",
        label=r"C$^{18}$O RMS / $\sqrt{N_{\rm voxels}/N_{\rm pix/beam}}$",
    )
    axis.set_xlabel(r"Mean $^{13}$CO(1-0) in threshold interval (K)")
    axis.set_ylabel(r"Mean C$^{18}$O(1-0) for the same voxels (K)")
    axis.set_title(f"{cloud} differential threshold voxel means")
    axis.grid(alpha=0.25)
    axis.legend(loc="center left")
    inset_axis = axis.inset_axes([0.03, 0.64, 0.30, 0.30])
    inset_axis.imshow(highest_differential_bin_image, origin="lower", cmap="magma")
    inset_axis.set_title("Brightest-threshold interval", fontsize=8)
    inset_axis.set_xticks([])
    inset_axis.set_yticks([])
    figure.savefig(output, dpi=200)
    print(
        f"Saved {output} using {LOW_BRIGHTNESS_BIN_COUNT} linear intervals below "
        f"approximately {LOW_BRIGHTNESS_THRESHOLD} K and "
        f"{HIGH_BRIGHTNESS_VOXELS_PER_BIN}-voxel intervals above"
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", default=DEFAULT_CLOUD, help="Cloud identifier")
    parser.add_argument("--bins", type=int, default=300, help="Number of bins per axis")
    parser.add_argument("--output", help="Output PNG filename")
    parser.add_argument(
        "--threshold-bins",
        type=int,
        default=40,
        help="Number of linearly spaced 13CO thresholds for cumulative means",
    )
    parser.add_argument(
        "--cumulative-output",
        help="Output PNG filename for cumulative means",
    )
    parser.add_argument(
        "--differential-output",
        help="Output PNG filename for differential means",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=12345,
        help="Random seed for the C18O channel permutation",
    )
    parser.add_argument(
        "--shuffle-realizations",
        type=int,
        default=100,
        help="Number of shuffled C18O channel realizations",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = f"{args.cloud}_13CO10_vs_C18O10_pixel_distribution.png"
    if args.cumulative_output is None:
        args.cumulative_output = f"{args.cloud}_13CO10_vs_C18O10_cumulative_means.png"
    if args.differential_output is None:
        args.differential_output = f"{args.cloud}_13CO10_vs_C18O10_differential_means.png"

    co13_file = find_cube(args.cloud, "13CO10")
    c18o_file = find_cube(args.cloud, "C18O10")
    co13 = fits.getdata(co13_file, memmap=True)
    c18o = fits.getdata(c18o_file, memmap=True)
    c18o_header = fits.getheader(c18o_file)

    if co13.shape != c18o.shape:
        raise RuntimeError(f"Cube shape mismatch: {co13.shape} versus {c18o.shape}")

    c18o_noise_channels = np.concatenate((c18o[:5], c18o[-5:]))
    c18o_rms = np.sqrt(np.nanmean(c18o_noise_channels**2))
    pixels_per_beam = (
        c18o_header["BMAJ"]
        * c18o_header["BMIN"]
        / c18o_header["CDELT2"] ** 2
        * np.pi
        / (4 * np.log(2))
    )

    c18o_valid_spatial_pixels = np.all(np.isfinite(c18o), axis=0)
    good = np.isfinite(co13) & c18o_valid_spatial_pixels
    co13_values = co13[good]
    c18o_values = c18o[good]
    if len(co13_values) == 0:
        raise RuntimeError("No finite 13CO/C18O voxel pairs found")

    _, _, _, thresholds = cumulative_means(co13_values, c18o_values, args.threshold_bins)
    highest_bin_mask = good & (co13 >= thresholds[-1])
    highest_bin_image = np.sum(highest_bin_mask, axis=0)
    _, _, _, differential_thresholds = differential_means(
        co13_values,
        c18o_values,
        args.threshold_bins,
    )
    highest_differential_bin_mask = good & (co13 >= differential_thresholds[-1])
    highest_differential_bin_image = np.sum(highest_differential_bin_mask, axis=0)

    random_generator = np.random.default_rng(args.shuffle_seed)
    shuffled_c18o_values = []
    for realization in range(args.shuffle_realizations):
        shuffled_c18o = c18o[random_generator.permutation(c18o.shape[0])]
        shuffled_values = shuffled_c18o[good]
        if not np.all(np.isfinite(shuffled_values)):
            raise RuntimeError(
                f"Shuffled C18O realization {realization + 1} has non-finite selected pixels"
            )
        shuffled_c18o_values.append(shuffled_values)

    figure, axis = pl.subplots(figsize=(7, 6), constrained_layout=True)
    histogram = axis.hist2d(
        co13_values,
        c18o_values,
        bins=args.bins,
        norm=LogNorm(),
        cmap="viridis",
    )
    colorbar = figure.colorbar(histogram[3], ax=axis)
    colorbar.set_label("Number of voxels")
    axis.set_xlabel(r"$^{13}$CO(1-0) brightness temperature (K)")
    axis.set_ylabel(r"C$^{18}$O(1-0) brightness temperature (K)")
    axis.set_title(f"{args.cloud} voxel distribution (N = {len(co13_values):,})")
    figure.savefig(args.output, dpi=200)
    print(f"Saved {args.output} from {len(co13_values):,} finite voxel pairs")
    plot_cumulative_means(
        args.cloud,
        co13_values,
        c18o_values,
        shuffled_c18o_values,
        highest_bin_image,
        c18o_rms,
        pixels_per_beam,
        args.threshold_bins,
        args.cumulative_output,
    )
    plot_differential_means(
        args.cloud,
        co13_values,
        c18o_values,
        shuffled_c18o_values,
        highest_differential_bin_image,
        c18o_rms,
        pixels_per_beam,
        args.threshold_bins,
        args.differential_output,
    )


if __name__ == "__main__":
    main()