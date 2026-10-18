# Release plan — MC_exchange 1.2.0: acceptance statistics

Status: **implemented on branch `feature/acceptance-stats` (§1, §3, §4, §5 steps 1–6)**. §2 was dropped.
Still to do: §5 steps 7–8 (build, tag, release, cluster install) and the §6 changes in PEHMA.
Revised 2026-10-17 after a review against the code, and on 2026-10-18 against the PEHMA Stage 3B
scripts (see §6 and §7 for what changed and why).

## Why
Stage 3B of the PEHMA vitrimer project now records every accepted exchange (lifetimes, back reactions,
reaction counts) by reconstructing moves from the `bonds_to_delete` / `bonds_to_create` dicts. What it
**cannot** get from the current return values is how many moves were *attempted* and *why* the rest
failed. Without that there is no acceptance rate, so we cannot tell whether a slow exchange rate is
caused by geometry (few sticker neighbourhoods, overstretched candidate bonds), energetics (Metropolis
rejections) or the MPI bookkeeping (cross-rank conflicts).

## Scope
1. Add attempt/rejection counters to `complementary_bond_exchange` (`src/MC_exchange/exchange.py`).
2. Optionally return the accepted moves themselves (`return_moves=True`).
3. Fix the `gather_bonds` bugs (`src/MC_exchange/data.py:84–87` and `:108`).
4. Tests (including un-ignoring `tests/`), version bump, release.

The change is **purely additive**. Existing keys (`n_shifts`, `n_swaps`, `total`), argument defaults and
the shape of the returned tuple stay the same, so every existing caller (3A, 3B, tests) keeps working
on either version.

## 1. New `exchange_stats` keys

All counters are global, summed over ranks, and identical on every rank after the call.

| Key | Counted where (current line numbers) | Meaning |
|---|---|---|
| `n_sites` | after `if len(potential_exchanges) == 0: continue` (~l. 891) | Neighbourhoods that offered ≥ 1 potential exchange. Each site proposes exactly one trial move, so this is also the number of proposals. |
| `n_options` | same place, `+= len(potential_exchanges)` | Total candidate moves enumerated before `random.choice`. Shows how crowded the neighbourhoods are. |
| `n_proposed_shift` / `n_proposed_swap` | after `trial_move = random.choice(...)` (~l. 894), by `len(trial_move['old_bonds'])` | Proposals split by mechanism. |
| `n_proposed_overstretched` | after `delta_E = fene_new - fene_old` (~l. 925), `if not np.isfinite(delta_E)` | Proposals whose new bond(s) would be ≥ R0, so the raw FENE term is `np.inf` and the move is always rejected. These are geometric rejections, also included in `n_metropolis_rejected_*`. |
| `n_metropolis_rejected_shift` / `_swap` | `if not accept` after the Metropolis test (~l. 928–934) | Rejected on energy (including the overstretched ones above). |
| `n_accepted_local_shift` / `_swap` | `if accept:` (~l. 940) | Accepted on the rank, before cross-rank conflict resolution. |
| `n_conflict_rejected_shift` / `_swap` | root rank, the `if conflict: continue` branch (~l. 978) | Accepted locally, then dropped because another rank's move used an overlapping atom. |
| `n_skipped_quarantined` | the `if atom_main in already_exchanged_atoms: continue` branch (~l. 802) | Sites skipped because their central sticker had already reacted this call (optional, diagnostic). |
| `sum_dE_proposed`, `sum_dE_accepted` | at the Metropolis test | Sums of **finite** ΔE only, so the caller can form ⟨ΔE⟩ = `sum_dE_proposed / (n_sites − n_proposed_overstretched)` and `sum_dE_accepted / (n_accepted_local_shift + n_accepted_local_swap)`. |

**Why the finite-only rule matters:** `calculate_raw_fene_potential` returns `np.inf` for `distance >= R0`.
The free sticker and the bound partner are each within the neighbour cutoff of `atom_main` (1.12 in
`src/example/loop.py`), so they can be up to ~2.24 apart, beyond R0 = 1.5. One such proposal makes a
plain sum `inf` for the whole call (confirmed on a synthetic system: `sum_dE_proposed = inf` in every
run).

**Corrected during implementation:** an *existing* bond can be ≥ R0. LAMMPS `bond_style fene` only warns
("FENE bond too long") between ~0.95·R0 and 2·R0 and errors beyond that. Such a move has ΔE = −∞ (or NaN
if the new bond is also ≥ R0) and is accepted (or rejected). So `n_proposed_overstretched` counts only
ΔE = +∞, and both sums skip every non-finite ΔE. The check uses `math.isfinite` (≈17 ns) instead of
`np.isfinite` (≈290 ns on a scalar), which halved the measured overhead.

Identities that must hold (and are asserted in tests):

```
n_sites                    = n_proposed_shift + n_proposed_swap
n_proposed_X               = n_metropolis_rejected_X + n_accepted_local_X          (X = shift, swap)
n_accepted_local_X         = n_conflict_rejected_X + n_X                            (n_shifts / n_swaps)
n_proposed_overstretched  <= n_metropolis_rejected_shift + n_metropolis_rejected_swap
```

The conflict counters are counted independently on root (not derived as `n_accepted_local_X − n_X`),
so the third identity is a real check of the conflict loop and not true by definition.

Acceptance ratios the caller can then form:
- Metropolis acceptance: `n_accepted_local_X / n_proposed_X`
- Conflict loss: `n_conflict_rejected_X / n_accepted_local_X`
- Overall: `n_X / n_proposed_X`
- Geometric loss: `n_proposed_overstretched / n_sites`

### Implementation sketch

Counters are **plain Python ints/floats** inside the loop. Benchmarked on a synthetic 2000–8000 sticker
system, `numpy` element increments (`arr[i] += 1`) cost ~3× more than list/int increments: +3.2–3.7 %
vs +1.2–1.3 % on the call. Convert to an array once, just before the collective.

```python
# top of complementary_bond_exchange, next to local_accepted_moves = []
# layout: sites, options, prop_sh, prop_sw, overstretched, metro_sh, metro_sw, acc_sh, acc_sw, quarantined
local_counts = [0] * 10
local_dE = [0.0, 0.0]                        # sum_dE_proposed (finite only), sum_dE_accepted

# ... increment in the branches listed above; kind index k = 0 if len(trial_move['old_bonds']) == 1 else 1
# if np.isfinite(delta_E): local_dE[0] += delta_E  else: local_counts[4] += 1

# after the gather of moves: ONE collective for all 12 numbers.
# float64 is exact for integer counts up to 2**53, so ints and sums can share one buffer.
send = np.array(local_counts + local_dE, dtype=np.float64)
recv = np.empty_like(send)
comm.Allreduce(send, recv, op=MPI.SUM)

# root: count conflict rejections per kind inside the existing conflict loop (conflicts = [0, 0]),
# then send them in ONE bcast together with the existing counters, replacing the two current bcasts:
#   global_shifts, global_swaps, conflicts = comm.bcast((global_shifts, global_swaps, conflicts), root=0)

# when building exchange_stats: counts back to Python int, sums to Python float
#   exchange_stats['n_sites'] = int(recv[0]), ...  exchange_stats['sum_dE_proposed'] = float(recv[10]), ...
```

Values in `exchange_stats` are plain Python `int` (counts) and `float` (sums), never numpy scalars or
float-typed counts. 3B logs them with `str(int(...))` and `f'{float(...):.10g}'`, which would tolerate
either, but the README and other callers should be able to rely on the types.

Every rank must reach the `Allreduce`, including when `return_stats=False`. Make it unconditional:
the cost is negligible, and the code path stays the same whatever the arguments. (Putting it under
`if return_stats:` is also deadlock-free, since the flag is the same on every rank, but adds a branch for
no gain.) With the merged `bcast`, the call makes the same number of collectives as today plus one
small `Allreduce`.

Also update the `return_stats` paragraph and the `Returns` section of the docstring (~l. 735–752) to list
the new keys.

## 2. Optional: `return_moves=True` — DROPPED (decision 2026-10-18)

Return the conflict-filtered list of moves, e.g.
`[{'kind': 'shift'|'swap', 'old_bonds': [...], 'new_bonds': [...]}, ...]`, built in the root loop that
already iterates `all_moves`, then broadcast. 3B could then drop its connected-component reconstruction
(`_group_moves`, `3B_vitrimer.py:91`). This is not required: the reconstruction is exact because moves are
atom-disjoint, and 3B already cross-checks it against `n_shifts`/`n_swaps` every loop
(`3B_vitrimer.py:245`). Switching would also mean rewriting `ExchangeTracker.update`, which
classifies moves by their `(len(old), len(new), n_atoms)` signature. **Recommendation: skip §2 for
1.2.0.** The only benefit is simpler code in 3B, against new API surface and an extra broadcast every loop.
Defaults to `False` if implemented.

**Return shape (fixed):** the moves go **into the stats dict** as `exchange_stats['moves']`, not into the
tuple. The tuple therefore stays a 2-tuple (`return_stats=False`) or a 3-tuple (`return_stats=True`) in
all cases, and `return_moves=True` without `return_stats=True` raises `ValueError`. The broadcast can be
merged into the same `bcast` as the counters. The cost is one extra pickled list, comparable to the
existing `bonds_to_delete` / `bonds_to_create` broadcasts. That is the only part of this release whose
communication cost is above noise.

## 3. Bug fixes in `gather_bonds` (`src/MC_exchange/data.py`)

```python
# l. 87, list filter compares the column against itself
selected_bond_data = bond_data[np.isin(bond_data['type'], bond_data)]   # current
selected_bond_data = bond_data[np.isin(bond_data['type'], bond_type)]   # fix

# l. 108, shape broadcast ignores the `root` argument (the dtype bcast at l. 101 uses it correctly)
selected_datashape = comm.bcast(selected_datashape, root=0)              # current
selected_datashape = comm.bcast(selected_datashape, root=root)           # fix

# l. 84, numpy integer types (e.g. np.int64 from an array) fall through to "all bonds"
if isinstance(bond_type, int):                                           # current
if isinstance(bond_type, (int, np.integer)):                             # fix
```

Found during implementation: the final data `Bcast` in `gather_bonds` (l. 119) and in `gather_atoms`
(l. 272) also hard-codes `root=0`, and `gather_atoms` (l. 237) has the same `int`-only check. All three
are fixed the same way. A new MPI test also exposed `gather_atoms` l. 210, where the coordinate `Gatherv`
used `rank == 0` for the receive counts. With `root != 0`, the root crashed (`TypeError`) and the other
ranks deadlocked. This is now fixed to `rank == root`.

The first and third only trigger for non-`int` `bond_type`. The second only triggers when `root != 0`.
3A/3B pass a Python int and the default root, so production runs are unaffected, but other callers can
currently get the wrong selection or a broadcast from the wrong rank.

## 4. Tests (`tests/test_exchange.py`, currently empty)

**Prerequisite:** `.gitignore` line 1 is `test*/`, so `tests/` is not tracked at all (`git ls-files` shows
no tests). It also holds ~17 MB of run output (`tests/datas` 14 MB, `tests/dumps` 2.7 MB). Replace the
rule with:

```
tests/datas/
tests/dumps/
tests/logs/
```

Then check `git status` before the first commit, so only the `.py`/small fixture files are added.

- **Identities**: build a small periodic box by hand with known A/B stickers and bonds, with free
  stickers so shifts are possible. Pass the neighbour list as a hand-built dict. Run with
  `return_stats=True` under a fixed `random.seed` (the function only uses Python's `random`, so
  `np.random.seed` is not needed), and assert the four identities above.
- **Overstretched**: place a free sticker so the new bond is > R0. Assert `n_proposed_overstretched ≥ 1`,
  that `sum_dE_proposed` is finite, and that the move is rejected.
- **Mechanism switches**: `bond_shift=False` means all `*_shift` counters are 0. `bond_swap=False`
  means all `*_swap` counters are 0. Both off means everything is 0.
- **T → 0 / T → ∞ limits**: use a small positive T (e.g. `1e-6`), **not 0**. At low T, every proposal with
  ΔE ≫ kB·T is rejected (build the fixture so positive ΔE are O(1), not tiny). At very high T,
  `n_metropolis_rejected_shift + n_metropolis_rejected_swap == n_proposed_overstretched`, not 0, because
  ΔE = +∞ is rejected at any T. (`n_proposed_overstretched` is not split by kind.) To get exactly 0,
  build geometry where all new bonds are < R0.
- **Backward compatibility**: `return_stats=False` still returns a 2-tuple, and the old keys keep
  their meaning (`n_shifts + 2*n_swaps == len(bonds_to_delete)`, the existing assertion at ~l. 1026).
  `return_moves=True` with `return_stats=False` raises `ValueError`.
- **`gather_bonds`**: needs a LAMMPS instance, so it can be skipped if `lammps` is not importable. Test a
  list of two bond types and an `np.int64` type.
- **MPI** (`mpirun -np 2/4 python -m pytest tests/test_exchange.py`): `neigh_list` gives out sites in
  dictionary-order chunks, not by spatial subdomain, and the counters legitimately depend on the rank
  count. On 1 rank, a site overlapping an accepted move is skipped as quarantined. On N ranks, the same
  move becomes a conflict rejection, and each rank's random sequence changes. So assert rank-count
  independence only for a deterministic setup: **atom-disjoint** neighbourhoods, each with exactly one
  candidate move with ΔE < 0. Separately, build two neighbourhoods that share an atom, each with exactly
  one ΔE < 0 move, and give them to different ranks. Assert `n_conflict_rejected_shift + _swap == 1`
  (with ≥ 2 ranks; the move from the lower rank wins, because `gather` keeps rank order), and that the
  identities still hold.

## 5. Release steps

1. Branch `feature/acceptance-stats`.
2. Fix `.gitignore` (§4 prerequisite).
3. Implement §1 (and §2 if wanted), plus §3.
4. Add the tests from §4 and run `pytest`, then `mpirun -np 4 python -m pytest tests/test_exchange.py`.
5. Bump the version to **1.2.0** (new feature, backward compatible) in all three places:
   - `pyproject.toml` (currently 1.1.3)
   - `src/MC_exchange/__init__.py` `__version__` (currently stale at **1.1.1**)
   - `CITATION.cff` `version:` (currently stale at 1.0.2) and `date-released:` (currently 2026-03-05)
6. Update `README.md`. It has no section on the return statistics yet, so add a short subsection under
   "Functionalities" covering `return_stats`, all keys, the identities and the ratios.
7. Build locally (`python -m build` → `dist/`). Tag `v1.2.0` and publish a GitHub Release, which
   triggers `.github/workflows/publish.yml` → PyPI.
8. On the cluster, run `pip install -U mc_exchange==1.2.0` in the `PEHMA_BERs` env (and `bond_exchange`
   if 2A uses it).
9. Apply the two 3B changes from §6 (`ACCEPTANCE_COUNT_KEYS`, `CLAUDE.md`). Run one short 3B job
   (a few loops) and check that the `NOTE` line is gone and that no `-1`/`nan`/`inf` values appear in the
   acceptance columns.

## 6. Matching change in PEHMA Stage 3B

Checked on 2026-10-18 against `Homopolymer systems/Melts/PEHMA/`: `3B_vitrimer.py`, `3B_setup.lmp`,
`3B_submit.slurm`, `CLAUDE.md`, and `data_analysis/` (no script reads the acceptance columns yet).

**What 3B does now.** `ExchangeTracker` (rank 0) writes one CSV row per loop to
`dumps/3B/{sys}_tc{tauc}_exchange_log_{seed}.csv`. The header is built from `ACCEPTANCE_COUNT_KEYS`
(11 keys, `3B_vitrimer.py:133–137`) and `ACCEPTANCE_SUM_KEYS` (2 keys, l. 138). It reads counts with
`str(int(stats.get(k, -1)))` and sums with `f'{float(stats.get(k, nan)):.10g}'` (l. 255–256), and prints a
one-time `NOTE` for missing keys (l. 249–254). It only reads the keys listed in the tuples, so extra
keys in `exchange_stats` (e.g. a `moves` list) are ignored safely. All 13 key names match §1 exactly.

**Assumptions in §1 that 3B confirms:**
- `return_stats=True` and the 3-tuple unpacking (l. 365). Unchanged by this plan.
- `Rc=1.12` for the neighbour list (l. 355). This confirms the overstretched-proposal issue under real
  production settings, not only the example script.
- Default `K=30`, `R0=1.5` passed implicitly. They match `3B_setup.lmp` (`fene_k 30.0`, `fene_r0 1.5`,
  `special_bonds fene`), which is exactly the setting the raw-FENE ΔE assumes.
- `cross_link_types 2` is reduced to a Python `int` in `main` (l. 489–491), so the `gather_bonds` list
  bug (§3) is not hit in production. `ExchangeTracker._pairs` filters by type itself anyway (l. 174–175).
  After §3 lands, that comment becomes stale, but the code stays correct.
- Production cost: `--tauc 25000` MD steps per loop (`3B_submit.slurm:51`), so the +1.2 % on the exchange
  call is negligible end to end.

**Required changes in 3B (in the PEHMA repo, after 1.2.0 is published):**
1. Add `'n_proposed_overstretched'` to `ACCEPTANCE_COUNT_KEYS`. Append it at the end of the tuple, so
   existing column positions in the CSV do not move.
2. Update the column list in `PEHMA/CLAUDE.md` (l. 70) to match.

**Observations outside this release (not blocking, recorded for the user to decide):**
- **The MC moves are not reproducible.** 3B never seeds Python's `random`; `--seed` only goes to the
  LAMMPS Langevin thermostat. Each rank's `random` is seeded from OS entropy. That makes the streams
  independent across ranks, which is correct, but runs cannot be replayed. If reproducibility is wanted,
  seed per rank, e.g. `random.seed(seed * 1000 + rank)`. Never use the same seed on all ranks, or the
  rank streams become correlated.
- **The Metropolis T is the instantaneous kinetic temperature.** It comes from `compute thermo_temp`
  (l. 362), not the thermostat set-point `--temperature`. For a large melt the fluctuations are small
  (~√(2/3N)), but passing the set-point is the textbook choice, and it makes acceptance rates compare
  cleanly between runs.

## 7. Review log (2026-10-17)

Checked against `exchange.py`, `data.py`, `neigh_list.py`, `calculations.py`, `__init__.py`,
`pyproject.toml`, `CITATION.cff`, `.gitignore`, `README.md`, `publish.yml`.

Confirmed correct: all line references, the first three identities (verified numerically:
608 = 560 + 48, 560 = 303 + 257, 48 = 28 + 20), backward compatibility, the `data.py:87` fix, the
CITATION staleness, `test_exchange.py` being empty.

Corrected in this revision:
1. `sum_dE_proposed` was always `inf` → sum finite ΔE only, add `n_proposed_overstretched`.
2. `tests/` was git-ignored → `.gitignore` fix added as a release step.
3. MPI test assumed spatial subdomains and rank-count independence → restricted to a deterministic,
   atom-disjoint setup, plus an explicit conflict test.
4. High-T limit claimed `n_metropolis_rejected → 0` → false because of ΔE = +∞; corrected. T = 0 is excluded.
5. `return_moves` tuple shape was undefined → moved into the stats dict.
6. Two more `gather_bonds` bugs (`root=0` at l. 108, `np.integer` at l. 84).
7. `__version__` in `__init__.py` (1.1.1) and `date-released` added to the version bump.
8. README has no stats section yet → "add", not "update".
9. Counters as Python ints, one merged `Allreduce`, one merged `bcast`.

Second pass (2026-10-18), against the 3B scripts (details in §6):
10. All 13 existing 3B key names match §1. Only `n_proposed_overstretched` needs to be added (at the end
    of the tuple). 3B's CSV writer only reads listed keys, so the earlier claim that `moves` "must be
    excluded from the CSV writer" was unnecessary and has been removed.
11. Production settings (Rc = 1.12, K = 30, R0 = 1.5, `special_bonds fene`, int bond type) confirm the
    overstretched-ΔE issue and that the `gather_bonds` bugs do not affect current runs.
12. §2 (`return_moves`) recommended to skip: 3B's reconstruction is exact and already cross-checked.
13. `exchange_stats` values are specified as Python `int`/`float`.
14. Recorded, outside this release: 3B does not seed `random` (not reproducible), and uses the
    instantaneous T.

Implementation (2026-10-18):
15. §2 dropped. Counters, `.gitignore`, the `data.py` fixes, the version bump (pyproject, `__init__`,
    CITATION), README and docstring are done as specified.
16. `tests/test_exchange.py` has 18 tests and `tests/test_data.py` has 8 (a tiny in-memory LAMMPS box).
    On 1 rank, 24 pass and 2 are skipped (the multi-rank tests); on 2 and 4 ranks, all 26 pass.
    `test_data.py` skips itself if LAMMPS cannot load. On this Mac it needs
    `DYLD_LIBRARY_PATH=/opt/homebrew/lib`, because the venv's `liblammps.dylib` cannot find
    `libmpi.12.dylib` on its own.
17. Mutation checks: the tests catch a missing conflict counter (7 failures), a missing finite-ΔE guard
    (6 failures), and the old `data.py` (filter failures, then a deadlock in the `root != 0` test).
18. Two extra defects were found and fixed: `gather_atoms` l. 210 (`rank == 0`), and ΔE = −∞ from an
    existing bond ≥ R0 (see §1).
19. Final overhead, actual 1.1.3 vs 1.2.0 code, same synthetic systems, median of 21: **+0.65–1.15 %**.
20. Not changed: `CITATION.cff` `commit:` still points at the 1.0.2 commit. Update it after tagging, or
    remove it.

Estimated CPU overhead (measured, synthetic 2000/8000 stickers, Rc = 1.12, 1 rank, median of 15):
**+1.2–1.3 %** of `complementary_bond_exchange` with int counters (+3.2–3.7 % with numpy counters as in
the original sketch). One extra 12-double `Allreduce` adds a few µs per call on one node, or ~10–50 µs
between nodes. End to end, with the LAMMPS MD between calls, this is well below 1 %. `return_moves` adds
one pickled list broadcast, which is the only cost above noise.
