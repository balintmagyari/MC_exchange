# Handoff: `feature/acceptance-stats` vs `main` (MC_exchange 1.2.0)

This is the context a new Claude Code session needs to continue this work. Read it fully before changing
anything. The longer design rationale and review history are in `RELEASE_PLAN_v1.2.0_acceptance_stats.md`
(committed on the branch); this file is the factual summary of what changed.

- Repo: `/Users/bmagyari/Documents/PhD🔬/Simulations/Python Packages/MC_exchange` (GitHub `balintmagyari/MC_exchange`)
- Branch: `feature/acceptance-stats`, one commit `331d846` on top of `main` (`efa5ff7`). **Not pushed, not merged, not tagged.**
- Diff: 10 files, +809 / −20 (`git diff --stat main..feature/acceptance-stats`)
- Version: 1.1.3 → **1.2.0** (new feature, backward compatible)

## Standing instructions from the user

1. **Never modify anything outside this repository.** In particular, make no changes in the PEHMA simulation
   project (`/Users/bmagyari/Documents/PhD🔬/Simulations/Homopolymer systems/Melts/PEHMA`). You may read it.
2. `return_moves` (plan §2) was **dropped by the user's decision**. Do not implement it.
3. Work on the feature branch and commit only after double-checking (run the tests on 1, 2 and 4 ranks).
   Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Why the change was made

The downstream PEHMA Stage 3B script (`3B_vitrimer.py`, written by another agent) logs a CSV row per
exchange loop. It wants acceptance rates: how many moves were *attempted* and *why* the rest failed
(geometry, Metropolis energetics, or MPI cross-rank conflicts). Version 1.1.3 only returned `n_shifts`,
`n_swaps` and `total`. 3B was already prepared for the new keys: it reads them with `stats.get(key, -1)`
for counts and `stats.get(key, nan)` for sums, and prints a one-time NOTE when they are missing.

## File-by-file changes

### `src/MC_exchange/exchange.py`: `complementary_bond_exchange` (def at l. 666)

Only this function changed. `perform_bond_swap` and `evaluate_bond_exchange` are untouched; note that
`evaluate_bond_exchange` contains look-alike code, so a non-unique string match there is a trap.

- `import math` added (l. 4).
- **Local counters** initialised next to `local_accepted_moves = []` (~l. 801–811). They are plain Python
  ints and floats, deliberately not numpy: element updates on a numpy array were ~3× slower in a benchmark.
  - `n_sites`, `n_options`, `n_overstretched`, `n_skipped_quarantined`
  - `sum_dE_proposed`, `sum_dE_accepted`
  - two-element lists indexed by `kind` (0 = shift, 1 = swap): `n_proposed`, `n_metropolis_rejected`,
    `n_accepted_local`
- **Increments:**
  - l. 836: `n_skipped_quarantined += 1` in the `if atom_main in already_exchanged_atoms: continue` branch.
  - l. 928: `n_sites += 1` and `n_options += len(potential_exchanges)`, right after the
    `if len(potential_exchanges) == 0: continue`. After `random.choice`, `kind` is computed from
    `len(trial_move['old_bonds'])` and `n_proposed[kind] += 1`.
  - l. 970: `dE_finite = math.isfinite(delta_E)`. If finite, add to `sum_dE_proposed`. Otherwise, if
    `delta_E > 0` (i.e. +inf), increment `n_overstretched`. −inf and NaN are neither summed nor counted
    (see design note 2).
  - l. 986: `if not accept: n_metropolis_rejected[kind] += 1`.
  - l. 993: inside `if accept:`, `n_accepted_local[kind] += 1` and, if `dE_finite`,
    `sum_dE_accepted += delta_E`.
  - l. 1037, root only, in the `if conflict:` branch of the move loop:
    `n_conflict_rejected[kind] += 1`. `n_conflict_rejected = [0, 0]` is initialised on root next to
    `global_shifts`/`global_swaps`, and on the non-root branch too.
- **Communication** after the existing `bcast`s of the bond dicts:
  - The two old `bcast`s of `global_shifts` and `global_swaps` were merged into **one** `bcast` of the
    tuple `(global_shifts, global_swaps, n_conflict_rejected)`.
  - **One** `comm.Allreduce(..., op=MPI.SUM)` (l. 1088) over a float64 array of 12 values:
    `[n_sites, n_options, prop_sh, prop_sw, overstretched, metro_sh, metro_sw, acc_sh, acc_sw,
    quarantined, sum_dE_proposed, sum_dE_accepted]`. float64 is exact for integer counts up to 2**53.
  - The `Allreduce` is **unconditional** (it runs even when `return_stats=False`), so every rank always
    reaches it and it cannot deadlock.
- **`exchange_stats`** (l. 1091): the old keys are unchanged. The new keys are cast back to Python
  `int`/`float`; `n_conflict_rejected_*` are already Python ints from the bcast. The existing
  `assert global_shifts + global_swaps*2 == len(complete_bonds_to_delete)` is kept.
- **Docstring:** the `return_stats` parameter and the `Returns` section now document every key and the
  identities.

### `src/MC_exchange/data.py`: bug fixes (`gather_bonds`, `gather_atoms`)

| Line | Before | After | Effect of old bug |
|---|---|---|---|
| 84 | `isinstance(bond_type, int)` | `isinstance(bond_type, (int, np.integer))` | `np.int64` type returned *all* bonds |
| 87 | `np.isin(bond_data['type'], bond_data)` | `np.isin(bond_data['type'], bond_type)` | list of types gave a wrong (empty) selection |
| 108 | `comm.bcast(selected_datashape, root=0)` | `root=root` | wrong source rank when `root != 0` |
| 119 | `comm.Bcast([... bond ...], root=0)` | `root=root` | same |
| 210 | `3*send_counts if rank == 0 else None` (×2) | `rank == root` | `gather_atoms(root!=0)`: root raised TypeError, other ranks **deadlocked** |
| 237 | `isinstance(atom_type, int)` | `(int, np.integer)` | `np.int64` type returned all atoms |
| 272 | `comm.Bcast([... atom ...], root=0)` | `root=root` | wrong source rank |

Production callers (PEHMA 3A/3B) pass a Python `int` bond type and the default root, so their past
results are **not** affected by these bugs.

### Tests (new files; `tests/` was previously git-ignored and untracked)

`tests/test_exchange.py` (18 tests, pure Python with mpi4py, no LAMMPS):
- Helpers: `make_atoms`, `make_bonds`, `neighbours` (hand-built neighbour-list entry),
  `run` (seeds Python `random` and calls with `return_stats=True`), and `shift_site(offset, origin,
  new_bond, old_bond)`. `shift_site` builds a deterministic single-candidate shift site:
  A(offset+1)–B(offset+2) bonded, plus a free A(offset+3).
- `assert_identities`, and `random_system`: a random A/B melt with a brute-force Rc = 1.12 neighbour
  list, split into contiguous chunks per rank the way `neigh_list` does it.
- The tests:
  - `test_backward_compatible_return_shapes`, `test_stat_types`
  - `test_downhill_shift_is_accepted`, `test_swap_counters`
  - `test_overstretched_proposal_is_counted_and_sum_stays_finite`
  - `test_existing_overstretched_bond_keeps_sums_finite`
  - `test_low_and_high_temperature_limits` (T = 1e-6 and 1e9, never T = 0)
  - `test_quarantined_site_is_counted`
  - `test_identities_random_system` ×5 seeds
  - `test_mechanism_switches` ×3
  - `test_counts_independent_of_rank_count_for_disjoint_sites`
  - `test_cross_rank_conflict_is_counted` (skipped on 1 rank; ranks 0 and 1 propose conflicting moves
    and root keeps rank 0's)
- Convention: neighbour lists are given only to rank 0 (`{}` elsewhere) unless a test is about
  distribution.

`tests/test_data.py` (8 tests, needs LAMMPS):
- A module fixture builds a tiny in-memory LAMMPS box: 6 atoms of alternating types 1/2 and 3 bonds of
  types 1/2/3.
- Tests: `test_gather_bonds_type_filter` ×4, `test_gather_atoms_type_filter` ×3, and
  `test_non_zero_root` (skipped on 1 rank).
- It skips itself if LAMMPS cannot be loaded.

### Other files

- `.gitignore`: `test*/` replaced by `tests/datas/`, `tests/dumps/` and `tests/logs/` (~17 MB of run
  output stays ignored).
- `pyproject.toml` 1.1.3 → 1.2.0; `src/MC_exchange/__init__.py` `__version__` 1.1.1 → 1.2.0 (it was stale);
  `CITATION.cff` `version` 1.0.2 → 1.2.0 and `date-released` → `'2026-10-18'`.
- `README.md`: "version 1.1.0" → "1.2.0", plus a new subsection "Exchange statistics" under
  "Functionalities". It has a table of all keys, the identities, the derived ratios, and the caveat about
  existing bonds ≥ R0.
- `RELEASE_PLAN_v1.2.0_acceptance_stats.md`: the full plan, with a review log (§7) explaining every decision.

## The `exchange_stats` contract (consumed by PEHMA 3B; do not rename keys)

Counts are `int`, sums are `float`, and all are global (summed over ranks, identical on every rank).
X ∈ {shift, swap}.

| Key | Meaning |
|---|---|
| `n_shifts`, `n_swaps`, `total` | (old) successful reactions after conflict resolution |
| `n_sites` | neighbourhoods that proposed a trial move (one proposal each) |
| `n_options` | candidate moves enumerated before `random.choice` |
| `n_proposed_X` | proposals per mechanism |
| `n_proposed_overstretched` | proposals with ΔE = +∞ (a new bond ≥ R0); always rejected, also in `n_metropolis_rejected_X`; not split by kind |
| `n_metropolis_rejected_X` | rejected by the Metropolis test |
| `n_accepted_local_X` | accepted on its rank, before conflict resolution |
| `n_conflict_rejected_X` | locally accepted, then dropped on root because it shares an atom with an earlier-rank move |
| `n_skipped_quarantined` | sites whose central sticker had already reacted in this call |
| `sum_dE_proposed`, `sum_dE_accepted` | sums of **finite** ΔE only |

Identities (asserted in the tests):
```
n_sites            = n_proposed_shift + n_proposed_swap
n_proposed_X       = n_metropolis_rejected_X + n_accepted_local_X
n_accepted_local_X = n_conflict_rejected_X + n_X            (n_shifts / n_swaps)
n_proposed_overstretched <= n_metropolis_rejected_shift + n_metropolis_rejected_swap
```

## Key design decisions and non-obvious facts

1. **Why ΔE can be infinite:** `calculate_raw_fene_potential` returns `np.inf` for `distance >= R0`
   (R0 = 1.5). With the production neighbour cutoff Rc = 1.12, a free sticker and a bound partner can be up
   to ~2.24 apart, so ΔE = +∞ proposals are common. A plain sum would always be `inf`; this was confirmed
   on synthetic systems.
2. **An existing bond can be longer than R0.** LAMMPS `bond_style fene` only warns ("FENE bond too long")
   up to 2·R0 and errors beyond that. ΔE is then −∞ (the move is accepted), or NaN if the new bond is also
   too long (the move is rejected). Therefore:
   - only +∞ counts as overstretched;
   - every non-finite ΔE is excluded from both sums;
   - the README's mean-ΔE formula `sum_dE_proposed / (n_sites − n_proposed_overstretched)` is exact
     only when no existing bond is ≥ R0.
3. **`math.isfinite`, not `np.isfinite`:** about 17 ns vs 290 ns on a scalar. Switching halved the
   measured overhead.
4. **Conflict counts are counted on root, not derived as `n_accepted_local_X − n_X`.** That keeps the
   third identity a real check of the conflict loop instead of something true by definition.
5. **The counters legitimately depend on the MPI rank count.** On 1 rank, an overlapping site is skipped
   (quarantined). On N ranks the same move becomes a conflict rejection, and each rank's random stream
   differs. `neigh_list` distributes sites in dictionary-order chunks, **not** by spatial subdomain. So
   rank-count independence is only tested for atom-disjoint, single-candidate, ΔE < 0 sites.
6. `complementary_bond_exchange` uses only Python's `random`, never `np.random`; seed `random` in tests.
7. Neighbour-list semantics: for each `atom_main`, only neighbours with `id > atom_main` that are not
   quarantined join the site, and a site needs ≥ 3 stickers and ≥ 1 bonded pair to propose anything.

## Verification performed (all on the final commit)

- `pytest tests/test_data.py tests/test_exchange.py`: on 1 rank, 24 passed and 2 skipped (the
  multi-rank tests). With `mpirun -np 2` and `-np 4`, all 26 passed on every rank.
- **Mutation checks:**
  - removing the conflict-counter increment → 7 failures;
  - removing the finite-ΔE guard → 6 failures;
  - the old `main` version of `data.py` → filter failures, then a deadlock in `test_non_zero_root`.
- **Overhead,** real `main` code vs branch code, synthetic 2000/8000-sticker systems, Rc = 1.12, 1 rank,
  median of 21 runs: **+0.65–1.15 %** of the `complementary_bond_exchange` call. One extra 12-double
  `Allreduce` costs a few µs; the number of `bcast` calls is unchanged (two merged into one). End to end
  it is negligible, since 3B runs 25,000 MD steps between exchange calls.
- `python -m build` succeeds (`mc_exchange-1.2.0` sdist and wheel; built into the scratchpad, not `dist/`).

## How to run things (environment specifics)

- macOS, Python 3.14 venv `/Users/bmagyari/Python_venv/dev_venv`. MC_exchange is installed **editable**,
  so `import MC_exchange` resolves to `src/`. mpirun is Homebrew MPICH (`/opt/homebrew/bin/mpirun`).
- The venv's `liblammps.dylib` cannot find `libmpi.12.dylib` on its own. LAMMPS-backed tests need:
  ```zsh
  export DYLD_LIBRARY_PATH=/opt/homebrew/lib
  python3 -m pytest tests/test_data.py tests/test_exchange.py -q
  mpirun -np 2 python3 -m pytest tests/test_data.py tests/test_exchange.py -q -p no:cacheprovider
  mpirun -np 4 python3 -m pytest tests/test_data.py tests/test_exchange.py -q -p no:cacheprovider
  ```
  Without it, `test_data.py` skips and `test_exchange.py` still runs.
- MPI tests can hang if a collective is mismatched. Use `-o faulthandler_timeout=30` and run `mpirun` in
  the background with a time limit, then `pkill -f "mpirun -np"`.
- Do **not** run plain `pytest tests/` without paths. `tests/` also contains the user's untracked
  scripts and fixtures (`test_loop.py`, `test_calculations.py`, `test_neigh_list.py`, `__init__.py`,
  `Vitrimer_setup.lmp`, `bond_to_*.json`). They are not part of this change; leave them untracked and
  untouched unless the user says otherwise.

## Open items (not done)

1. Merge the branch into `main`, push, tag `v1.2.0`, and publish a GitHub Release. The release triggers
   `.github/workflows/publish.yml` → PyPI (OIDC trusted publishing). Only do this when the user asks.
2. `CITATION.cff` `commit:` still points at an old commit (`8c07c2b…`). Update it after tagging or remove
   it; also update `date-released` if the release date differs from 2026-10-18.
3. On the cluster: `pip install -U mc_exchange==1.2.0` in the `PEHMA_BERs` env.
4. **PEHMA changes, for the user to make (you must not edit PEHMA):**
   - append `'n_proposed_overstretched'` to the **end** of `ACCEPTANCE_COUNT_KEYS` in `3B_vitrimer.py`
     (~l. 133), so the existing CSV columns don't move;
   - update the column list in PEHMA `CLAUDE.md` (~l. 70);
   - then run a short 3B job and check there is no NOTE line and no `-1`/`nan`/`inf` in the acceptance
     columns.

   All 13 other key names already match exactly.
5. Observations in PEHMA 3B, outside this release, for the user to decide:
   - Python `random` is never seeded, so MC moves can't be reproduced. If seeding is added, seed per rank
     (e.g. `seed*1000 + rank`), never with the same seed on every rank.
   - The Metropolis T is the instantaneous `thermo_temp`, not the thermostat set-point.
