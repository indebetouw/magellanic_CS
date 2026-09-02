"""Analyze and visualize spectral-line statistics derived from dendrograms.

The script reads the flat ``allstats.ecsv`` table, creates per-cloud diagnostic
plots, fits log-space line relations, and writes the fit summary to
``linefit_logxy.ecsv``.
"""

from astropy.io import fits
from astropy.table import Table
import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np
import os

pl.ion()

key_line = "13CO10"
corr_line = "CS21"
select_good_method = "error"
snr_threshold = 3
max_key_peak = 100
table_file = "allstats.ecsv"
regression_file = f"linefit_logxy_{select_good_method}.ecsv"
plot_dir = "analyze_dendro_plots"
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

reg_columns = [
    "cloud",
    "line",
    "fit_type",
    "npts",
    "slope",
    "offset",
    "delta_slope",
    "delta_offset",
]

reg_dtypes = [
    "U64",
    "U64",
    "U16",
    int,
    float,
    float,
    float,
    float,
]


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


def build_line_colors(tab):
    all_lines = {str(line) for line in tab["line"]}
    cloud_lines = {
        str(cloud): {str(line) for line in tab[tab["cloud"] == cloud]["line"]}
        for cloud in np.unique(tab["cloud"])
    }
    key_line_means = {}
    for cloud, lines in cloud_lines.items():
        if "12CO10" not in lines:
            continue
        peaks = np.array(
            tab[(tab["cloud"] == cloud) & (tab["line"] == "12CO10")]["pk_from_mom_1darea"],
            dtype=float,
        )
        valid_peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
        key_line_means[cloud] = np.mean(valid_peaks) if len(valid_peaks) else -np.inf
    reference_cloud = max(
        cloud_lines,
        key=lambda cloud: (len(cloud_lines[cloud]), key_line_means.get(cloud, -np.inf)),
    )
    reference_rows = tab[tab["cloud"] == reference_cloud]
    mean_peaks = {}
    for line in cloud_lines[reference_cloud]:
        peaks = np.array(
            reference_rows[reference_rows["line"] == line]["pk_from_mom_1darea"],
            dtype=float,
        )
        valid_peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
        mean_peaks[line] = np.mean(valid_peaks) if len(valid_peaks) else -np.inf

    ranked_lines = sorted(mean_peaks, key=mean_peaks.get, reverse=True)
    all_lines = ranked_lines + sorted(all_lines - set(ranked_lines))

    line_colors = {}
    if all_lines:
        color_map = mpl.colormaps.get_cmap("tab20").resampled(max(len(all_lines), 1))
        for i, line in enumerate(all_lines):
            line_colors[line] = color_map(i)
    return all_lines, line_colors


def main():
    if select_good_method not in {"peak_threshold", "error"}:
        raise ValueError(
            "select_good_method must be 'peak_threshold' or 'error'"
        )

    if not os.path.exists(table_file):
        raise FileNotFoundError(f"Missing input table: {table_file}")

    stats_table = Table.read(table_file, format="ascii.ecsv")
    if not set(flat_columns).issubset(set(stats_table.colnames)):
        raise RuntimeError("allstats.ecsv is not in the expected flat format")

    stats_table = stats_table[flat_columns]
    regression_table = Table(names=reg_columns, dtype=reg_dtypes)
    all_lines, line_colors = build_line_colors(stats_table)

    clouds = []
    seen_clouds = set()
    for cloud in stats_table["cloud"]:
        cloud = str(cloud)
        if cloud not in seen_clouds:
            seen_clouds.add(cloud)
            clouds.append(cloud)

    cloud_correlations = {}

    for cloud in clouds:
        stats = table_to_stats_for_cloud(stats_table, cloud)
        if len(stats) == 0:
            continue

        lines = np.array([line for line in all_lines if line in stats])

        pl.gcf().set_size_inches(10, 8)
        pl.clf()
        for line in lines:
            if line not in stats:
                continue
            pl.subplot(221)
            myplot, = pl.plot(stats[line]["pk_from_mom"][:, 0], stats[line]["pk"][:, 0], ".", label=line, color=line_colors.get(line))
            pl.subplot(222)
            pl.plot(stats[line]["pk_from_mom"][:, 0], stats[line]["pk_from_fit"][:, 0], ".", color=myplot.get_color())
            pl.subplot(223)
            pl.plot(stats[line]["pk_from_mom"][:, 0], stats[line]["pk_from_mom"][:, 1], ".", color=myplot.get_color())
            pl.subplot(224)
            pl.plot(stats[line]["pk"][:, 0], stats[line]["pk"][:, 1], ".", color=myplot.get_color())
        pl.subplot(221)
        pl.xlabel("Mom0 / Mom2 / sqrt(2*pi)")
        pl.ylabel("Peak")
        pl.legend(prop={"size": 8})
        pl.xscale("log")
        pl.yscale("log")
        pl.plot(pl.xlim(), pl.xlim(), "k", alpha=0.3)

        pl.subplot(222)
        pl.xlabel("Mom0 / Mom2 / sqrt(2*pi)")
        pl.ylabel("Peak from fit")
        pl.xscale("log")
        pl.yscale("log")
        pl.plot(pl.xlim(), pl.xlim(), "k", alpha=0.3)
        pl.ylim(pl.xlim())

        pl.subplot(223)
        pl.xlabel("mom peak w/1d area")
        pl.ylabel("mom peak w/2d area")
        pl.xscale("log")
        pl.yscale("log")
        pl.plot(pl.xlim(), pl.xlim(), "k", alpha=0.3)

        pl.subplot(224)
        pl.xlabel("Peak w/1d area")
        pl.ylabel("Peak w/2d area")
        pl.xscale("log")
        pl.yscale("log")
        pl.plot(pl.xlim(), pl.xlim(), "k", alpha=0.3)

        pl.subplots_adjust(top=0.95, right=0.95, hspace=0.25, wspace=0.25)
        pl.savefig(os.path.join(plot_dir, cloud + "_peak_vs_mom0_over_mom2_log.png"))

        for k in range(4):
            pl.subplot(2, 2, 1 + k)
            pl.xscale("linear")
            pl.yscale("linear")
            pl.xlim(-1, 2)
            pl.ylim(-1, 2)
        pl.savefig(os.path.join(plot_dir, cloud + "_peak_vs_mom0_over_mom2.png"))

        if corr_line in stats:
            key_values = np.array(stats[corr_line]["pk_from_mom"][:, 0], dtype=float)
            for line in lines:
                if line == corr_line or line not in stats:
                    continue
                other_values = np.array(stats[line]["pk_from_mom"][:, 1], dtype=float)
                good = np.isfinite(key_values) & np.isfinite(other_values)
                if np.sum(good) > 2:
                    corr = float(np.corrcoef(key_values[good], other_values[good])[0, 1])
                    cloud_correlations.setdefault(line, []).append(corr)

        pl.gcf().set_size_inches(6, 5)
        pl.clf()
        momentsk = stats[key_line]["moments"]
        all_xvals = []
        all_yvals = []
        for line in lines:
            if line == key_line or line not in stats:
                continue
            xvals = stats[key_line]["pk_from_mom"][:, 0]
            yvals = stats[line]["pk_from_mom"][:, 1]
            finite_positive = (
                np.isfinite(xvals)
                & np.isfinite(yvals)
                & (xvals > 0)
                & (xvals <= max_key_peak)
                & (yvals > 0)
            )
            all_xvals.append(xvals[finite_positive])
            all_yvals.append(yvals[finite_positive])

        pl.plot(np.concatenate(all_xvals), np.concatenate(all_yvals), alpha=0)
        pl.xscale("log")
        pl.yscale("log")
        xx = pl.xlim()
        yy = pl.ylim()
        pl.xlim(xx)
        pl.ylim(yy)
        for line in lines:
            if line not in stats:
                continue
            if line != key_line:
                xvals = stats[key_line]["pk_from_mom"][:, 0]
                yvals = stats[line]["pk_from_mom"][:, 1]
                channel_fwhm = momentsk[2, :] / stats[key_line]["chanwid_kms"]
                beam_count = stats[key_line]["npix_2d"] / stats[key_line]["pixperbm"]
                xerrs = stats[key_line]["linerms"] / np.sqrt(np.clip(2.35 * channel_fwhm * beam_count, 1e-12, np.inf))
                yerrs = stats[line]["linerms"] / np.sqrt(np.clip(2.35 * channel_fwhm * beam_count, 1e-12, np.inf))
                peak_threshold = stats[line]["linerms"] / np.sqrt(np.clip(2.35 * np.nanmedian(momentsk[2, :]), 1e-12, np.inf))

                finite_positive = (
                    np.isfinite(xvals)
                    & np.isfinite(yvals)
                    & (xvals > 0)
                    & (xvals <= max_key_peak)
                    & (yvals > 0)
                )
                npts = int(np.sum(finite_positive))
                if select_good_method == "peak_threshold":
                    high = finite_positive & (yvals > peak_threshold)
                    high_label_prefix = ">rms"
                else:
                    high = finite_positive & np.isfinite(yerrs) & (yvals / yerrs > snr_threshold)
                    high_label_prefix = f">{snr_threshold:g}sigma"
                slope = np.nan
                offset = np.nan
                delta_slope = np.nan
                delta_offset = np.nan
                if npts >= 3:
                    logx = np.log10(xvals[finite_positive])
                    logy = np.log10(yvals[finite_positive])
                    coeff, cov = np.polyfit(logx, logy, 1, cov=True)
                    slope = float(coeff[0])
                    offset = float(coeff[1])
                    delta_slope = float(np.sqrt(np.clip(cov[0, 0], 0, np.inf)))
                    delta_offset = float(np.sqrt(np.clip(cov[1, 1], 0, np.inf)))

                if np.isfinite(slope) and np.isfinite(delta_slope):
                    label = f"{line} ({slope:.1f}±{delta_slope:.1f})"
                else:
                    label = f"{line} (nan)"

                plot_color = line_colors.get(line)
                pl.plot(xvals[high], yvals[high], '.', label=label, color=plot_color)
                pl.plot(
                    xvals[finite_positive & ~high],
                    yvals[finite_positive & ~high],
                    'o',
                    markerfacecolor='none',
                    color=plot_color,
                )
                y_bottom = pl.ylim()[0]
                lower_yerrs = np.where(yvals - yerrs < y_bottom, 0, yerrs)
                pl.errorbar(
                    xvals[finite_positive],
                    yvals[finite_positive],
                    xerr=xerrs[finite_positive],
                    yerr=np.array([lower_yerrs, yerrs])[:, finite_positive],
                    fmt='none',
                    color=plot_color,
                    alpha=0.2,
                    capsize=2,
                )
                if select_good_method == "peak_threshold":
                    pl.fill_between(pl.xlim(), np.ones(2) * pl.ylim()[0], np.ones(2) * peak_threshold, alpha=0.1, color=plot_color)

                if np.isfinite(slope) and np.isfinite(offset):
                    xmin = np.nanmin(xvals[finite_positive])
                    xmax = np.nanmax(xvals[finite_positive])
                    if np.isfinite(xmin) and np.isfinite(xmax) and (xmin > 0) and (xmax > xmin):
                        xfit = np.logspace(np.log10(xmin), np.log10(xmax), 200)
                        yfit = (10.0 ** offset) * xfit ** slope
                        fit_linestyle = '--'
                        if np.isfinite(delta_slope) and (delta_slope > np.abs(slope)):
                            fit_linestyle = ':'
                        pl.plot(xfit, yfit, fit_linestyle, color=plot_color, alpha=0.9)

                nhigh = int(np.sum(high))
                if nhigh > (npts / 2) and nhigh >= 3:
                    logx_high = np.log10(xvals[high])
                    logy_high = np.log10(yvals[high])
                    coeff_high, cov_high = np.polyfit(logx_high, logy_high, 1, cov=True)
                    slope_high = float(coeff_high[0])
                    offset_high = float(coeff_high[1])
                    delta_slope_high = float(np.sqrt(np.clip(cov_high[0, 0], 0, np.inf)))
                    delta_offset_high = float(np.sqrt(np.clip(cov_high[1, 1], 0, np.inf)))

                    high_label = f"{line} ({high_label_prefix} {slope_high:.1f}±{delta_slope_high:.1f})"
                    xfit_high = np.logspace(np.log10(np.nanmin(xvals[high])), np.log10(np.nanmax(xvals[high])), 200)
                    yfit_high = (10.0 ** offset_high) * xfit_high ** slope_high
                    pl.plot(xfit_high, yfit_high, '-', color=plot_color, alpha=0.9, label=high_label)

                regression_table.add_row((cloud, str(line), "peak", npts, slope, offset, delta_slope, delta_offset))

        pl.legend(prop={'size': 8})
        pl.xlabel(f'{key_line} pk from mom')
        pl.ylabel('Other lines pk from mom')
        pl.yscale('log')
        pl.xlim(xx)
        pl.savefig(os.path.join("./", cloud + f"_pk_different_lines_{select_good_method}.png"))

        pl.clf()
        xx = None
        for line in lines:
            if line not in stats or line == key_line:
                continue

            xvals = stats[key_line]["pk_from_mom"][:, 0]
            yvals = stats[line]["pk_from_mom"][:, 1]
            beam_count = stats[key_line]["npix_2d"] / stats[key_line]["pixperbm"]
            yerrs = stats[line]["linerms"] / np.sqrt(
                np.clip(
                    2.35
                    * momentsk[2, :]
                    / stats[key_line]["chanwid_kms"]
                    * beam_count,
                    1e-12,
                    np.inf,
                )
            )
            signal_to_noise = yvals / yerrs
            good = (
                np.isfinite(xvals)
                & np.isfinite(signal_to_noise)
                & (xvals > 0)
                & (xvals <= max_key_peak)
                & (signal_to_noise > 0)
            )
            pl.plot(
                xvals[good],
                signal_to_noise[good],
                ".",
                label=line,
                color=line_colors.get(line),
            )
            if xx is None and np.any(good):
                pl.xscale("log")
                xx = pl.xlim()

        if xx is not None:
            pl.xlim(xx)
        pl.xlabel(f"{key_line} pk from mom")
        pl.ylabel("Other lines signal-to-noise")
        pl.yscale("log")
        pl.legend(prop={"size": 8})
        pl.savefig(os.path.join(plot_dir, cloud + "_pk_vs_snr.png"))

        pl.clf()
        xx = None
        for line in lines:
            if line not in stats:
                continue
            if line != key_line:
                xvals = stats[key_line]["moments"][0, :]
                yvals = stats[line]["moments"][0, :]

                good = (
                    np.isfinite(xvals)
                    & np.isfinite(yvals)
                    & (xvals > 0)
                    & (xvals <= max_key_peak)
                    & (yvals > 0)
                )
                npts = int(np.sum(good))
                slope = np.nan
                offset = np.nan
                delta_slope = np.nan
                delta_offset = np.nan
                if npts >= 3:
                    logx = np.log10(xvals[good])
                    logy = np.log10(yvals[good])
                    coeff, cov = np.polyfit(logx, logy, 1, cov=True)
                    slope = float(coeff[0])
                    offset = float(coeff[1])
                    delta_slope = float(np.sqrt(np.clip(cov[0, 0], 0, np.inf)))
                    delta_offset = float(np.sqrt(np.clip(cov[1, 1], 0, np.inf)))

                if np.isfinite(slope) and np.isfinite(delta_slope):
                    label = f"{line} ({slope:.1f}±{delta_slope:.1f})"
                else:
                    label = f"{line} (nan)"

                myplot, = pl.plot(xvals, yvals, '.', label=label, color=line_colors.get(line))
                if not xx:
                    pl.xscale('log')
                    xx = pl.xlim()

                if np.isfinite(slope) and np.isfinite(offset):
                    xmin = np.nanmin(xvals[good])
                    xmax = np.nanmax(xvals[good])
                    if np.isfinite(xmin) and np.isfinite(xmax) and (xmin > 0) and (xmax > xmin):
                        xfit = np.logspace(np.log10(xmin), np.log10(xmax), 200)
                        yfit = (10.0 ** offset) * xfit ** slope
                        fit_linestyle = '-'
                        if np.isfinite(delta_slope) and (delta_slope > np.abs(slope)):
                            fit_linestyle = ':'
                        pl.plot(xfit, yfit, fit_linestyle, color=myplot.get_color(), alpha=0.9)

                regression_table.add_row((cloud, str(line), "mom0", npts, slope, offset, delta_slope, delta_offset))

        pl.legend(prop={'size': 8})
        pl.xlabel(f'{key_line} mom0')
        pl.ylabel('Other lines mom0')
        pl.yscale('log')
        pl.xlim(xx)
        pl.savefig(os.path.join(plot_dir, cloud + f"_mom0_different_lines_{select_good_method}.png"))

        pl.gcf().set_size_inches(5, 4)

    if len(cloud_correlations) > 0:
        pl.clf()
        pl.gcf().set_size_inches(8, 6)
        bins = np.linspace(-1, 1, 21)
        hist_data = []
        for line, coeffs in cloud_correlations.items():
            coeffs = np.array(coeffs, dtype=float)
            coeffs = coeffs[np.isfinite(coeffs)]
            if len(coeffs) == 0:
                continue
            mean_corr = float(np.mean(coeffs))
            hist_data.append((line, coeffs, mean_corr))

        hist_data = sorted(hist_data, key=lambda item: item[2], reverse=True)
        for line, coeffs, mean_corr in hist_data:
            label = f"{line} ({mean_corr:.2f})"
            color = line_colors.get(line)
            pl.hist(coeffs, bins=bins, histtype="stepfilled", alpha=0.2, linewidth=1.2, label=label, color=color)
            pl.hist(coeffs, bins=bins, histtype="step", linewidth=1.6, color=color)
        pl.xlabel("Correlation coefficient")
        pl.ylabel("Number of clouds")
        pl.title(f"Cloud-wise correlation with {corr_line}")
        pl.grid(alpha=0.2)
        pl.legend(prop={"size": 8}, loc="best")
        pl.tight_layout()
        pl.savefig(os.path.join("./", f"{corr_line}_correlation_histograms_{select_good_method}.png"))

    regression_table.write(regression_file, format="ascii.ecsv", overwrite=True)


if __name__ == "__main__":
    main()
