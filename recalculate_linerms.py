"""Recalculate cube RMS values in allstats.ecsv without rerunning dendrogram analysis."""

import argparse
from pathlib import Path

from astropy.io import fits
from astropy.table import Table
import numpy as np


def edge_channel_rms(linefile):
    """Return the median RMS from the first and last finite cube channels."""
    cube = fits.getdata(linefile)
    channel_rms = np.nanstd(cube, axis=(1, 2))
    valid_channels = np.flatnonzero(np.any(np.isfinite(cube), axis=(1, 2)))
    if len(valid_channels) == 0:
        raise ValueError(f"Cube contains no finite channels: {linefile}")
    edge_channels = np.unique(
        np.concatenate([valid_channels[:5], valid_channels[-5:]])
    )
    return float(np.nanmedian(channel_rms[edge_channels]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_file", nargs="?", default="allstats.ecsv")
    args = parser.parse_args()

    table_path = Path(args.table_file)
    table = Table.read(table_path, format="ascii.ecsv")
    if not {"linefile", "cuberms"}.issubset(table.colnames):
        raise RuntimeError("Input table must contain linefile and cuberms columns")

    rms_by_file = {}
    for linefile in dict.fromkeys(map(str, table["linefile"])):
        rms_by_file[linefile] = edge_channel_rms(linefile)
        print(f"{linefile}: {rms_by_file[linefile]:.6g}")

    table["cuberms"] = [rms_by_file[str(linefile)] for linefile in table["linefile"]]
    table.write(table_path, format="ascii.ecsv", overwrite=True)
    print(f"Updated cuberms for {len(table)} rows in {table_path}")


if __name__ == "__main__":
    main()