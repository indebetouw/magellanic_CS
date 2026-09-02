from astropy.table import Table
import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np
import os

pl.ion()

regression_file = "linefit_logxy_error.ecsv"
line_output_file = "peak_intercept_vs_slope_by_line.png"
cloud_output_file = "peak_intercept_vs_slope_by_cloud.png"


def build_line_colors(tab):
    lines = []
    seen = set()
    for line in tab["line"]:
        line = str(line)
        if line not in seen:
            seen.add(line)
            lines.append(line)

    colors = {}
    if len(lines) > 0:
        color_map = mpl.colormaps.get_cmap("tab20").resampled(max(len(lines), 1))
        for index, line in enumerate(lines):
            colors[line] = color_map(index)
    return lines, colors


def build_cloud_colors(tab):
    clouds = []
    seen = set()
    for cloud in tab["cloud"]:
        cloud = str(cloud)
        if cloud not in seen:
            seen.add(cloud)
            clouds.append(cloud)

    colors = {}
    if len(clouds) > 0:
        color_map = mpl.colormaps.get_cmap("tab20").resampled(max(len(clouds), 1))
        for index, cloud in enumerate(clouds):
            colors[cloud] = color_map(index)
    return clouds, colors


def plot_intercept_vs_slope(reg, group_key, colors, output_file, title):
    pl.clf()
    pl.gcf().set_size_inches(7, 6)

    groups = []
    seen = set()
    for value in reg[group_key]:
        value = str(value)
        if value not in seen:
            seen.add(value)
            groups.append(value)

    if group_key == "line":
        def group_sort_key(group):
            group_rows = reg[reg[group_key] == group]
            if len(group_rows) == 0:
                return np.inf
            slope = np.array(group_rows["slope"], dtype=float)
            offset = np.array(group_rows["offset"], dtype=float)
            good = np.isfinite(slope) & np.isfinite(offset)
            if np.sum(good) == 0:
                return np.inf
            return float(np.mean(offset[good]))

        groups = sorted(groups, key=group_sort_key, reverse=True)

    for group in groups:
        group_rows = reg[reg[group_key] == group]
        if len(group_rows) == 0:
            continue
        slope = np.array(group_rows["slope"], dtype=float)
        offset = np.array(group_rows["offset"], dtype=float)
        dslope = np.array(group_rows["delta_slope"], dtype=float)
        ok = np.isfinite(slope) & np.isfinite(offset) 
        good = np.isfinite(slope) & np.isfinite(offset) & (dslope < 0.2)
        if np.sum(ok) == 0:
            continue
        show_summary = group_key == "line" and len(group_rows) >= 3
        meh = np.isfinite(slope) & np.isfinite(offset) & (dslope >= 0.2)
        pl.plot(
            slope[good],
            offset[good],
            ".",
            color=colors.get(group),
            label=group if (group_key != "line" or not show_summary) else None,
        )
        pl.plot(
            slope[meh],
            offset[meh],
            "o", mfc="none", markersize=6,
            color=colors.get(group),
            label=None,
        )

        if show_summary:
            slope_good = slope[good]
            offset_good = offset[good]
            mean_slope = float(np.mean(slope_good))
            mean_offset = float(np.mean(offset_good))
            rms_slope = float(np.sqrt(np.mean((slope_good - mean_slope) ** 2)))
            rms_offset = float(np.sqrt(np.mean((offset_good - mean_offset) ** 2)))
            pl.errorbar(
                mean_slope,
                mean_offset,
                xerr=rms_slope,
                yerr=rms_offset,
                fmt="o",
                color=colors.get(group),
                markersize=9,
                capsize=3,
                elinewidth=1.4,
                markeredgewidth=1.2,
                label=group,
            )

    pl.xlabel("Slope")
    pl.ylabel("Intercept")
    pl.title(title)
    pl.grid(alpha=0.2)
    pl.xlim(left=-0.5)
    pl.ylim(bottom=-3.)
    pl.legend(prop={"size": 8}, loc="best")
    pl.tight_layout()
    pl.savefig(output_file)


def print_cs21_slope_correlations(reg):
    cs21_rows = reg[reg["line"] == "CS21"]
    if len(cs21_rows) == 0:
        print("No CS21 peak-fit rows found; skipping slope correlations.")
        return

    cs21_by_cloud = {
        str(row["cloud"]): float(row["slope"])
        for row in cs21_rows
        if np.isfinite(row["slope"])
    }

    lines = []
    seen = set()
    for line in reg["line"]:
        line = str(line)
        if line not in seen and line != "CS21":
            seen.add(line)
            lines.append(line)

    print("CS21 slope correlations by line (paired by cloud):")
    results = []
    for line in lines:
        line_rows = reg[reg["line"] == line]
        paired_cs21 = []
        paired_line = []
        for row in line_rows:
            cloud = str(row["cloud"])
            slope = float(row["slope"])
            if not np.isfinite(slope):
                continue
            if cloud not in cs21_by_cloud:
                continue
            paired_cs21.append(cs21_by_cloud[cloud])
            paired_line.append(slope)

        paired_cs21 = np.array(paired_cs21, dtype=float)
        paired_line = np.array(paired_line, dtype=float)
        if len(paired_cs21) <= 2:
            continue

        corr = float(np.corrcoef(paired_cs21, paired_line)[0, 1])
        results.append((corr, line, len(paired_cs21)))

    if len(results) == 0:
        print("  No lines with more than 2 paired clouds.")
        return

    for corr, line, npts in sorted(results, key=lambda item: item[0], reverse=True):
        print(f"  {line}: n={npts} r={corr:.6f}")


def main():
    if not os.path.exists(regression_file):
        raise FileNotFoundError(f"Missing regression table: {regression_file}")

    reg = Table.read(regression_file, format="ascii.ecsv")
    required = {"cloud", "line", "fit_type", "slope", "offset"}
    if not required.issubset(set(reg.colnames)):
        raise RuntimeError("linefit_logxy.ecsv is not in the expected format")

    reg = reg[reg["fit_type"] == "peak"]
    if len(reg) == 0:
        raise RuntimeError("No peak fits were found in the regression table")

    lines, line_colors = build_line_colors(reg)
    clouds, cloud_colors = build_cloud_colors(reg)

    print_cs21_slope_correlations(reg)

    plot_intercept_vs_slope(
        reg,
        "line",
        line_colors,
        line_output_file,
        "Peak-fit intercept vs slope colored by line",
    )
    plot_intercept_vs_slope(
        reg,
        "cloud",
        cloud_colors,
        cloud_output_file,
        "Peak-fit intercept vs slope colored by cloud",
    )


if __name__ == "__main__":
    main()
