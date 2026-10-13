import numpy as np

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
