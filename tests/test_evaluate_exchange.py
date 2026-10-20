"""Regression tests for evaluate_bond_exchange: moves must be applied atomically so that
every sticker keeps at most one crosslink, and non-finite energy changes must be rejected.

Run serially with `pytest tests/test_evaluate_exchange.py` and in parallel with
`mpirun -np 4 python -m pytest tests/test_evaluate_exchange.py`. Every test is valid on any
number of ranks; the cross-rank conflict test is skipped on a single rank.
"""
import random
import warnings
from collections import Counter

import numpy as np
import pytest
from mpi4py import MPI

from MC_exchange import evaluate_bond_exchange

COMM = MPI.COMM_WORLD
BOX = np.array([0.0, 50.0, 0.0, 50.0, 0.0, 50.0])


# -------------------- Helpers --------------------

def make_atoms(spec: dict) -> np.ndarray:
    """spec: {id: (x, y, z)} -> atoms structured array sorted by id."""
    atoms = np.zeros(len(spec), dtype=[('id', 'i4'), ('type', 'i4'), ('mol', 'i4'),
                                       ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    for row, atom_id in enumerate(sorted(spec)):
        atoms[row] = (atom_id, 1, atom_id, *spec[atom_id])
    return atoms


def make_bonds(pairs: list) -> np.ndarray:
    return np.array([(2, a, b) for a, b in pairs],
                    dtype=[('type', 'i4'), ('atom 1', 'i4'), ('atom 2', 'i4')])


def neighbours(atom_main: int, others: list) -> dict:
    """Hand-built neighbour-list entry; only the keys are used by the exchange."""
    return {atom_main: {o: 1.0 for o in others}}


def run(neigh, bonds, atoms, T=1.0, seed=0, box=BOX, **kwargs):
    random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return evaluate_bond_exchange(neigh, bonds, atoms, box, T=T, **kwargs)


def apply(pairs: list, deleted: dict, created: dict) -> set:
    """Topology after the exchange, as a set of sorted bond tuples."""
    old = {tuple(sorted((int(a), int(b)))) for a, b in pairs}
    gone = {tuple(sorted(p)) for p in deleted.items()}
    new = {tuple(sorted(p)) for p in created.items()}
    assert gone <= old, 'a deleted bond does not exist'
    assert not (new & (old - gone)), 'a created bond already exists'
    return (old - gone) | new


def assert_valid_exchange(pairs: list, deleted: dict, created: dict):
    """Crosslink density and one-bond-per-sticker valency are preserved."""
    after = apply(pairs, deleted, created)
    assert len(deleted) == len(created)
    assert len(after) == len(pairs)
    valency = Counter(a for bond in after for a in bond)
    assert max(valency.values(), default=0) <= 1, f'multiply bonded stickers: {valency}'
    bonded_before = {int(a) for bond in pairs for a in bond}
    assert len(valency) == len(bonded_before), 'number of crosslinked stickers changed'


def swap_site(offset: int, origin: tuple):
    """Two stretched bonds (o+1)-(o+2) and (o+3)-(o+4); (o+1)-(o+3) and (o+2)-(o+4) are short,
    so the swap is strongly downhill and always accepted."""
    x, y, z = origin
    a, b, c, d = offset + 1, offset + 2, offset + 3, offset + 4
    spec = {a: (x, y, z), b: (x, y + 1.3, z), c: (x + 1.0, y, z), d: (x + 1.0, y + 1.3, z)}
    return spec, [(a, b), (c, d)], neighbours(a, [b, c, d])


def random_system(n_stickers=600, box=10.0, bonded_fraction=0.6, rc=1.12, seed=1):
    """Random sticker melt with a brute-force neighbour list (identical on every rank),
    split into contiguous chunks across ranks like MC_exchange.neigh_list does."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, box, size=(n_stickers, 3))
    ids = np.arange(1, n_stickers + 1)
    pairs = []
    for i in range(0, int(bonded_fraction * n_stickers) - 1, 2):
        d = rng.normal(size=3)
        pos[i + 1] = (pos[i] + 0.97 * d / np.linalg.norm(d)) % box
        pairs.append((int(ids[i]), int(ids[i + 1])))
    atoms = make_atoms({int(ids[i]): tuple(pos[i]) for i in range(n_stickers)})

    neigh = {}
    for i in range(n_stickers):
        d = pos - pos[i]
        d -= box * np.round(d / box)
        r = np.sqrt((d ** 2).sum(axis=1))
        js = np.where((r <= rc) & (ids != ids[i]))[0]
        if len(js):
            neigh[int(ids[i])] = {int(ids[j]): float(r[j]) for j in js}

    items = list(neigh.items())
    chunk = -(-len(items) // COMM.Get_size())
    rank = COMM.Get_rank()
    local = dict(items[rank * chunk:(rank + 1) * chunk])
    return local, pairs, atoms, np.array([0, box, 0, box, 0, box])


# -------------------- Bug 1: swaps must be applied atomically --------------------

def test_downhill_swap_is_applied_completely():
    spec, pairs, neigh = swap_site(0, (5.0, 5.0, 5.0))
    neigh = neigh if COMM.Get_rank() == 0 else {}
    deleted, created = run(neigh, make_bonds(pairs), make_atoms(spec), bond_shift=False)

    assert deleted == {1: 2, 3: 4}
    assert created == {1: 3, 2: 4}
    assert_valid_exchange(pairs, deleted, created)


def test_disjoint_swaps_on_all_ranks_are_all_applied():
    spec, pairs, neigh_all = {}, [], []
    for s in range(8):
        sp, pr, nb = swap_site(4 * s, (5.0 * (s + 1), 5.0, 5.0))
        spec.update(sp); pairs += pr; neigh_all.append(nb)
    rank, size = COMM.Get_rank(), COMM.Get_size()
    neigh = {k: v for i, nb in enumerate(neigh_all) if i % size == rank for k, v in nb.items()}
    deleted, created = run(neigh, make_bonds(pairs), make_atoms(spec), seed=rank, bond_shift=False)

    assert len(deleted) == len(created) == 16
    assert_valid_exchange(pairs, deleted, created)


@pytest.mark.skipif(COMM.Get_size() < 2, reason='needs at least 2 MPI ranks')
def test_cross_rank_conflict_rejects_whole_swap():
    # Rank 0 accepts a shift that breaks bond 1-2; rank 1 accepts a swap of 1-2 and 3-4.
    # Root keeps rank 0's shift and must drop the entire swap, leaving bond 3-4 untouched.
    spec, pairs, _ = swap_site(0, (5.0, 5.0, 5.0))
    spec[5] = (5.0, 6.3, 5.9)       # Free sticker 0.9 from atom 2: downhill shift 1-2 -> 2-5
    rank = COMM.Get_rank()
    neigh = {0: neighbours(1, [2, 5]), 1: neighbours(1, [2, 3, 4])}.get(rank, {})
    deleted, created = run(neigh, make_bonds(pairs), make_atoms(spec), seed=rank)

    assert deleted == {1: 2}
    assert created == {2: 5}
    assert_valid_exchange(pairs, deleted, created)


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('bond_shift, bond_swap', [(True, True), (False, True), (True, False)])
def test_random_system_preserves_crosslink_density(seed, bond_shift, bond_swap):
    neigh, pairs, atoms, box = random_system(seed=seed)
    deleted, created = run(neigh, make_bonds(pairs), atoms, T=1.0, seed=seed + COMM.Get_rank(),
                           box=box, bond_shift=bond_shift, bond_swap=bond_swap)

    assert len(deleted) > 0
    assert_valid_exchange(pairs, deleted, created)


def test_result_is_identical_on_every_rank():
    neigh, pairs, atoms, box = random_system(seed=4)
    result = run(neigh, make_bonds(pairs), atoms, seed=COMM.Get_rank(), box=box)
    assert all(r == result for r in COMM.allgather(result))


# -------------------- Bug 2: non-finite delta U must be rejected --------------------

def test_nan_swap_is_rejected():
    # Both old bonds are >= R0 (LAMMPS fene only warns), and both new configurations
    # contain a bond >= R0, so delta U = inf - inf = nan.
    spec = {1: (5.0, 5.0, 5.0), 2: (5.0, 6.6, 5.0), 3: (6.4, 5.0, 5.0), 4: (8.0, 5.0, 5.0)}
    pairs = [(1, 2), (3, 4)]
    neigh = neighbours(1, [2, 3, 4]) if COMM.Get_rank() == 0 else {}
    for seed in range(20):
        deleted, created = run(neigh, make_bonds(pairs), make_atoms(spec), T=1e-6, seed=seed,
                               bond_shift=False)
        assert deleted == {} and created == {}


def test_nan_shift_is_rejected():
    # Old bond 1-2 and both candidate bonds to free sticker 3 are >= R0
    spec = {1: (5.0, 5.0, 5.0), 2: (5.0, 6.6, 5.0), 3: (5.0, 8.2, 5.0)}
    neigh = neighbours(1, [2, 3]) if COMM.Get_rank() == 0 else {}
    for seed in range(20):
        deleted, created = run(neigh, make_bonds([(1, 2)]), make_atoms(spec), T=1e-6, seed=seed,
                               bond_swap=False)
        assert deleted == {} and created == {}


def test_overstretched_old_bond_relaxing_to_finite_is_accepted():
    # delta U = finite - inf = -inf: relaxing an overstretched bond is downhill and accepted
    spec = {1: (5.0, 5.0, 5.0), 2: (5.0, 6.6, 5.0), 3: (5.0, 7.5, 5.0)}
    neigh = neighbours(1, [2, 3]) if COMM.Get_rank() == 0 else {}
    deleted, created = run(neigh, make_bonds([(1, 2)]), make_atoms(spec), T=1e-6,
                           bond_swap=False)
    assert deleted == {1: 2} and created == {2: 3}


# -------------------- Zero temperature --------------------

def shift_site(offset: int, origin: tuple, new_bond: float = 0.9, old_bond: float = 1.3):
    """One shift-only site: (o+1)-(o+2) bonded, free (o+3) at new_bond from (o+2) and >= R0 from (o+1).

    The only candidate move is (o+1, o+2) -> (o+2, o+3), with
    delta U = FENE(new_bond) - FENE(old_bond).
    """
    x, y, z = origin
    a, b, c = offset + 1, offset + 2, offset + 3
    spec = {a: (x, y, z), b: (x, y + old_bond, z), c: (x, y + old_bond, z + new_bond)}
    return spec, [(a, b)], neighbours(a, [b, c])


def test_downhill_move_is_accepted_at_zero_temperature():
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0))
    neigh = neigh if COMM.Get_rank() == 0 else {}
    deleted, created = run(neigh, make_bonds(pairs), make_atoms(spec), T=0.0, bond_swap=False)
    assert deleted == {1: 2} and created == {2: 3}

    spec, pairs, neigh = swap_site(0, (5.0, 5.0, 5.0))
    neigh = neigh if COMM.Get_rank() == 0 else {}
    deleted, created = run(neigh, make_bonds(pairs), make_atoms(spec), T=0.0, bond_shift=False)
    assert deleted == {1: 2, 3: 4} and created == {1: 3, 2: 4}


def test_uphill_move_is_rejected_at_zero_temperature():
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0), new_bond=1.3, old_bond=0.97)
    neigh = neigh if COMM.Get_rank() == 0 else {}
    for seed in range(20):
        deleted, created = run(neigh, make_bonds(pairs), make_atoms(spec), T=0.0, seed=seed,
                               bond_swap=False)
        assert deleted == {} and created == {}


# -------------------- Exchange statistics --------------------

COUNT_KEYS = ('n_shifts', 'n_swaps', 'total', 'n_sites', 'n_options',
              'n_proposed_shift', 'n_proposed_swap', 'n_proposed_overstretched',
              'n_metropolis_rejected_shift', 'n_metropolis_rejected_swap',
              'n_accepted_local_shift', 'n_accepted_local_swap',
              'n_conflict_rejected_shift', 'n_conflict_rejected_swap',
              'n_skipped_quarantined')
SUM_KEYS = ('sum_dE_proposed', 'sum_dE_accepted')


def run_stats(neigh, bonds, atoms, T=1.0, seed=0, box=BOX, **kwargs):
    return run(neigh, bonds, atoms, T=T, seed=seed, box=box, return_stats=True, **kwargs)


def assert_identities(stats):
    assert stats['total'] == stats['n_shifts'] + stats['n_swaps']
    assert stats['n_sites'] == stats['n_proposed_shift'] + stats['n_proposed_swap']
    for kind, n_done in (('shift', 'n_shifts'), ('swap', 'n_swaps')):
        assert stats[f'n_proposed_{kind}'] == (stats[f'n_metropolis_rejected_{kind}']
                                               + stats[f'n_accepted_local_{kind}'])
        assert stats[f'n_accepted_local_{kind}'] == stats[f'n_conflict_rejected_{kind}'] + stats[n_done]
    assert stats['n_proposed_overstretched'] <= (stats['n_metropolis_rejected_shift']
                                                 + stats['n_metropolis_rejected_swap'])
    assert stats['n_options'] >= stats['n_sites']


def test_return_shapes_and_stat_types():
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0))
    neigh = neigh if COMM.Get_rank() == 0 else {}
    assert len(run(neigh, make_bonds(pairs), make_atoms(spec))) == 2

    out = run_stats(neigh, make_bonds(pairs), make_atoms(spec))
    assert len(out) == 3
    stats = out[2]
    assert set(stats) == set(COUNT_KEYS) | set(SUM_KEYS)
    assert all(type(stats[k]) is int for k in COUNT_KEYS)
    assert all(type(stats[k]) is float for k in SUM_KEYS)


def test_shift_counters():
    from MC_exchange.calculations import calculate_raw_fene_potential as fene
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0))
    neigh = neigh if COMM.Get_rank() == 0 else {}
    deleted, created, stats = run_stats(neigh, make_bonds(pairs), make_atoms(spec), bond_swap=False)

    dE = fene(0.9) - fene(1.3)
    assert deleted == {1: 2} and created == {2: 3}
    assert stats['n_sites'] == stats['n_proposed_shift'] == stats['n_accepted_local_shift'] == 1
    assert stats['n_shifts'] == stats['total'] == 1
    assert stats['n_options'] == 2          # Free sticker 3 bonding to either 1 or 2
    assert stats['sum_dE_proposed'] == pytest.approx(dE)
    assert stats['sum_dE_accepted'] == pytest.approx(dE)
    assert_identities(stats)


def test_swap_counters():
    spec, pairs, neigh = swap_site(0, (5.0, 5.0, 5.0))
    neigh = neigh if COMM.Get_rank() == 0 else {}
    _, _, stats = run_stats(neigh, make_bonds(pairs), make_atoms(spec), bond_shift=False)

    assert stats['n_sites'] == stats['n_proposed_swap'] == stats['n_accepted_local_swap'] == 1
    assert stats['n_swaps'] == stats['total'] == 1
    assert stats['n_options'] == 2          # The two possible re-pairings
    assert stats['sum_dE_accepted'] < 0
    assert_identities(stats)


def test_alpha_scales_summed_delta_u():
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0))
    neigh = neigh if COMM.Get_rank() == 0 else {}
    _, _, s1 = run_stats(neigh, make_bonds(pairs), make_atoms(spec), bond_swap=False)
    _, _, s2 = run_stats(neigh, make_bonds(pairs), make_atoms(spec), bond_swap=False, alpha=2.0)
    assert s2['sum_dE_proposed'] == pytest.approx(2.0 * s1['sum_dE_proposed'])


def test_overstretched_proposal_is_counted_and_sums_stay_finite():
    # Free sticker is >= R0 from both bonded stickers: delta U = +inf
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0), new_bond=1.6)
    neigh = neigh if COMM.Get_rank() == 0 else {}
    deleted, _, stats = run_stats(neigh, make_bonds(pairs), make_atoms(spec), T=1e6, bond_swap=False)

    assert deleted == {}
    assert stats['n_proposed_overstretched'] == stats['n_metropolis_rejected_shift'] == 1
    assert stats['sum_dE_proposed'] == stats['sum_dE_accepted'] == 0.0
    assert_identities(stats)


def test_nan_proposal_is_rejected_and_sums_stay_finite():
    spec = {1: (5.0, 5.0, 5.0), 2: (5.0, 6.6, 5.0), 3: (5.0, 8.2, 5.0)}
    neigh = neighbours(1, [2, 3]) if COMM.Get_rank() == 0 else {}
    _, _, stats = run_stats(neigh, make_bonds([(1, 2)]), make_atoms(spec), bond_swap=False)

    assert stats['n_metropolis_rejected_shift'] == 1
    assert stats['n_proposed_overstretched'] == 0
    assert stats['sum_dE_proposed'] == stats['sum_dE_accepted'] == 0.0
    assert_identities(stats)


def test_minus_inf_proposal_is_accepted_and_left_out_of_sums():
    spec = {1: (5.0, 5.0, 5.0), 2: (5.0, 6.6, 5.0), 3: (5.0, 7.5, 5.0)}
    neigh = neighbours(1, [2, 3]) if COMM.Get_rank() == 0 else {}
    _, _, stats = run_stats(neigh, make_bonds([(1, 2)]), make_atoms(spec), bond_swap=False)

    assert stats['n_shifts'] == 1
    assert stats['sum_dE_proposed'] == stats['sum_dE_accepted'] == 0.0
    assert_identities(stats)


def test_quarantined_site_is_counted():
    # Site 1 shifts bond 1-2 to 2-3; the next site is centred on the already reacted atom 2
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0))
    spec[4] = (5.0, 6.3, 4.1)
    neigh = {**neigh, 2: {3: 1.0, 4: 1.0}} if COMM.Get_rank() == 0 else {}
    _, _, stats = run_stats(neigh, make_bonds(pairs), make_atoms(spec), bond_swap=False)

    assert stats['n_shifts'] == 1
    assert stats['n_skipped_quarantined'] == 1
    assert_identities(stats)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_identities_random_system(seed):
    neigh, pairs, atoms, box = random_system(seed=seed)
    deleted, created, stats = run_stats(neigh, make_bonds(pairs), atoms, seed=seed + COMM.Get_rank(),
                                        box=box)

    assert stats['n_sites'] > 0
    assert stats['n_shifts'] + 2 * stats['n_swaps'] == len(deleted) == len(created)
    assert all(s == stats for s in COMM.allgather(stats))
    assert_identities(stats)


def test_counts_independent_of_rank_count_for_disjoint_sites():
    spec, pairs, neigh_all = {}, [], []
    for s in range(8):
        sp, pr, nb = swap_site(4 * s, (5.0 * (s + 1), 5.0, 5.0))
        spec.update(sp); pairs += pr; neigh_all.append(nb)
    rank, size = COMM.Get_rank(), COMM.Get_size()
    neigh = {k: v for i, nb in enumerate(neigh_all) if i % size == rank for k, v in nb.items()}
    _, _, stats = run_stats(neigh, make_bonds(pairs), make_atoms(spec), seed=rank, bond_shift=False)

    assert stats['n_sites'] == stats['n_swaps'] == 8
    assert stats['n_conflict_rejected_swap'] == 0
    assert_identities(stats)


@pytest.mark.skipif(COMM.Get_size() < 2, reason='needs at least 2 MPI ranks')
def test_cross_rank_conflict_is_counted():
    # Same set-up as test_cross_rank_conflict_rejects_whole_swap
    spec, pairs, _ = swap_site(0, (5.0, 5.0, 5.0))
    spec[5] = (5.0, 6.3, 5.9)
    rank = COMM.Get_rank()
    neigh = {0: neighbours(1, [2, 5]), 1: neighbours(1, [2, 3, 4])}.get(rank, {})
    _, _, stats = run_stats(neigh, make_bonds(pairs), make_atoms(spec), seed=rank)

    assert stats['n_accepted_local_shift'] == stats['n_shifts'] == 1
    assert stats['n_accepted_local_swap'] == stats['n_conflict_rejected_swap'] == 1
    assert stats['n_swaps'] == 0
    assert_identities(stats)
