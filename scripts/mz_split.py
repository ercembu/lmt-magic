"""Split a MageZero training file (LabeledStateWriter's HDF5 layout) into a head and a tail by state.

    mz_split.py <in.hdf5> <head.hdf5> <tail.hdf5> [tail fraction, default 0.1]

The runner writes each finished game's states as one block, so cutting at a state index
splits by game (only the game on the cut is shared). mz_loop.py validates on the tail of
both seats' files: holding out a seat instead -- the same games seen from the other side,
with the opposite labels -- let the network memorize which cards won each game and scored
worse than predicting the mean.
"""
import sys

import h5py
import numpy as np

# offsets array -> the per-item arrays it indexes
GROUPS = {
    "/offsets": ("/indices", "/values"),
    "/edge_offsets": ("/edge_child", "/edge_parent", "/edge_label"),
    "/policy_offsets": ("/policy_node", "/policy_visits"),
}


def _write(f, src, lo, hi):
    """States [lo, hi) of the open file src into the open file f."""
    f.create_dataset("/row", data=src["/row"][lo:hi])
    for off_key, item_keys in GROUPS.items():
        off = src[off_key][lo:hi + 1].astype(np.int64)
        a, b = int(off[0]), int(off[-1])
        f.create_dataset(off_key, data=off - a)
        for k in item_keys:
            f.create_dataset(k, data=src[k][a:b])


def split(path, head_path, tail_path, tail=0.1):
    """Returns (head states, tail states)."""
    with h5py.File(path, "r") as src:
        n = src["/row"].shape[0]
        cut = n - max(1, int(round(n * tail))) if n > 1 else n
        with h5py.File(head_path, "w") as f:
            _write(f, src, 0, cut)
        with h5py.File(tail_path, "w") as f:
            _write(f, src, cut, n)
    return cut, n - cut


if __name__ == "__main__":
    print(split(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.1))
