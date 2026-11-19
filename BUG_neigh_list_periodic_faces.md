# Bug: `neigh_list` misses sticker pairs that straddle a periodic box face

**Package:** MC_exchange 1.3.0, `src/MC_exchange/neigh_list.py` (`neigh_list`, l. 170–237)
**Found:** 2026-11-04, while replaying every MC call of the 3B tc40 runs (95 %, shift + swap, 48 MPI ranks)
**Effect:** 3.2–3.4 % of all sticker pairs within `Rc = 1.12` are never listed, so no exchange can
happen across the periodic faces of the box. The loss is the same for every f, which is why it went
unnoticed. Exchange sites per call are 7.1–7.2 % lower than with a correct list.

## Cause
Each rank keeps its own atoms plus ghosts within `Rc` of its subdomain, including ghosts taken
*across the periodic box face* (`x >= box_hi - Rc` for the rank at `x_lo = 0`, `x < box_lo + Rc` for the
rank at `x_hi = box_hi`). All atoms are then put into cells with

```python
x_idx = int((data['x'] - x_lo) / cell_size) % num_cells[0]      # l. 181 (same for y, z)
```

using their in-box coordinate. A periodic ghost at `x ≈ 35.5` seen by the rank whose subdomain is
`[0, 9.03)` gets `int(35.5/1.12) % 9 = 31 % 9 = 4`, i.e. a cell in the middle of the subdomain, instead of
cell `-1` next to the face. The neighbour search (`±1` cells, l. 198–207) therefore never compares it
with the atoms at `x ≈ 0`. The rank on the other side misplaces its periodic ghosts in the same way,
so the pair is lost on both ranks. The PBC distance check (l. 224) cannot recover pairs that are never
compared.

A second, latent problem: `num_cells = ceil(subdomain_length / Rc)` makes the last cell narrower than `Rc`.
In a dimension with only one rank (the subdomain is the whole periodic box, so the modulo wrap is
needed), two atoms in cells `0` and `n-2` can then be closer than `Rc` without being in adjacent cells.

## Evidence
`PEHMA_reports/scripts/tauc/candidates.py` rebuilds every call from the fast trajectories and the
bond-event tables. Emulating the 48-rank decomposition (`MPI.Compute_dims(48, 3) = [4, 4, 3]`) and the
cell assignment above reproduces the logged `n_sites`/`n_options` to 0.1 % (40 runs × 251 calls);
finding all pairs gives 7 % more sites. Lost pairs: f = 3: 3.24 %, f = 4: 3.37 %, f = 5: 3.41 %,
f = 6: 3.36 %. Rate ratios between f, and all conclusions of the tc40 and τ_c reports, are unchanged.

## Suggested fix
> Superseded: this approach was not used. See **Resolution** below.

Shift periodic ghosts by one box length before assigning cells, and give ghosts their own cell layer
instead of wrapping inside a subdomain:

```python
L = box_dims[1::2] - box_dims[::2]                       # box lengths
for d, (lo, hi) in enumerate(subdomain_dims):
    x = pos[:, d]
    if grid[d] > 1:                                      # several ranks along d: no wrap inside a rank
        x = np.where(x >= lo - Rc + L[d], x - L[d], x)   # periodic ghost below lo (rank at the low face)
        x = np.where(x <  hi + Rc - L[d], x + L[d], x)   # periodic ghost above hi (rank at the high face)
        idx[:, d] = np.floor((x - lo) / Rc).astype(int)  # -1 ... n, no modulo
    else:                                                # one rank owns the whole periodic dimension
        n = max(1, int(L[d] // Rc))                      # floor: every cell >= Rc
        idx[:, d] = np.floor((x - lo) / (L[d] / n)).astype(int) % n
```

The neighbour-cell lookup should wrap only in dimensions with `grid[d] == 1`.

## Test to add (`tests/`)
- Two stickers at `x = 0.1` and `x = box_hi − 0.3` (other coordinates equal) must be neighbours for
  1, 2, 8 and 48 ranks.
- For a random configuration, the merged neighbour list must give the same pair set as
  `scipy.spatial.cKDTree(pos, boxsize=L).query_pairs(Rc)` for every rank count.

## Resolution (1.3.1, 2026-11-19)
Confirmed by an independent reproduction (3000 random points, L = 36.12, Rc = 1.12): pairs missing at
1 / 2 / 4 / 8 ranks were 1.79 / 2.51 / 3.23 / 3.41 %. The narrow-last-cell problem was not latent. It was active in
every dimension that had a single rank, including all serial runs. A third bug turned up during the check:
`_calculate_subdomain_boundaries` used `box_hi / grid` and so assumed `box_lo = 0`, and a centred box lost 86 %
of pairs. The tc40 runs are unaffected because their boxes start at 0.

Fix: the per-rank cell list is replaced by `scipy.spatial.cKDTree(pos, boxsize=L)` on rank 0. Coordinates
are first wrapped into `[lo, hi)`. Real runs do have stickers outside the box when MC is called (3–18 per call in a
short test). Each atom's entry goes to the rank whose `MPI.Compute_dims` subdomain contains it. Spatially compact
chunks matter: chunks by atom id (i.e. by chain) caused 4.4 % cross-rank conflict rejections at 8–16 ranks in a
high-T stress test, where spatial chunks cause 0 %. A failure on rank 0 is now raised on every rank instead of
hanging the scatter. The subdomain helpers are removed. On the 95 % configuration in `tests/datas`, the new list
contains every old pair with bit-identical distances, has 2.3 % (1 rank) to 4.4 % (8 ranks) more pairs, does not depend
on the rank count, and is built in 0.014 s instead of 0.15–1.05 s. `tests/test_neigh_list.py` covers face, edge and corner pairs,
a brute-force comparison (including shifted boxes), atoms outside the box, the inclusive cut-off and
non-periodic dimensions. It passes on 1, 2, 4 and 8 ranks.
