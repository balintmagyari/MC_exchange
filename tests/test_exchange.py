"""Tests for the acceptance statistics of complementary_bond_exchange.

Run serially with `pytest tests/test_exchange.py` and in parallel with
`mpirun -np 4 python -m pytest tests/test_exchange.py`. Every test is valid on any
number of ranks; the cross-rank conflict test is skipped on a single rank.
"""
import random

import numpy as np
import pytest
from mpi4py import MPI

from MC_exchange import complementary_bond_exchange

COMM = MPI.COMM_WORLD
A, B = 1, 2                                     # Sticker types
BOX = np.array([0.0, 50.0, 0.0, 50.0, 0.0, 50.0])

COUNT_KEYS = ('n_sites', 'n_options', 'n_proposed_shift', 'n_proposed_swap',
              'n_proposed_overstretched',
              'n_metropolis_rejected_shift', 'n_metropolis_rejected_swap',
              'n_accepted_local_shift', 'n_accepted_local_swap',
              'n_conflict_rejected_shift', 'n_conflict_rejected_swap',
              'n_skipped_quarantined')
SUM_KEYS = ('sum_dE_proposed', 'sum_dE_accepted')


# -------------------- Helpers --------------------

def make_atoms(spec: dict) -> np.ndarray:
    """spec: {id: (type, (x, y, z))} -> atoms structured array sorted by id."""
    atoms = np.zeros(len(spec), dtype=[('id', 'i4'), ('type', 'i4'), ('mol', 'i4'),
                                       ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    for row, atom_id in enumerate(sorted(spec)):
        atype, (x, y, z) = spec[atom_id]
        atoms[row] = (atom_id, atype, 1, x, y, z)
    return atoms


def make_bonds(pairs: list) -> np.ndarray:
    return np.array([(1, a, b) for a, b in pairs],
                    dtype=[('type', 'i4'), ('atom 1', 'i4'), ('atom 2', 'i4')])


def neighbours(atom_main: int, others: list) -> dict:
    """Hand-built neighbour-list entry; distances are not used by the exchange."""
    return {atom_main: {o: 1.0 for o in others}}


def run(neigh, bonds, atoms, T=1.0, seed=0, **kwargs):
    random.seed(seed)
    return complementary_bond_exchange(neigh, bonds, atoms, BOX, T=T,
                                       sticker_types_A=A, sticker_types_B=B,
                                       return_stats=True, **kwargs)


def shift_site(offset: int, origin: tuple, new_bond: float = 0.9, old_bond: float = 1.3):
    """One shift-only site: A(offset+1)-B(offset+2) bonded, free A(offset+3) next to the B.

    The only candidate move is (a1, b2) -> (a3, b2), with
    delta E = FENE(new_bond) - FENE(old_bond).
    """
    x, y, z = origin
    a1, b2, a3 = offset + 1, offset + 2, offset + 3
    spec = {a1: (A, (x, y, z)),
            b2: (B, (x + old_bond, y, z)),
            a3: (A, (x + old_bond, y + new_bond, z))}
    return spec, [(a1, b2)], neighbours(a1, [b2, a3])


def assert_identities(stats):
    assert stats['n_sites'] == stats['n_proposed_shift'] + stats['n_proposed_swap']
    for kind, n_done in (('shift', 'n_shifts'), ('swap', 'n_swaps')):
        assert stats[f'n_proposed_{kind}'] == (stats[f'n_metropolis_rejected_{kind}']
                                               + stats[f'n_accepted_local_{kind}'])
        assert stats[f'n_accepted_local_{kind}'] == stats[f'n_conflict_rejected_{kind}'] + stats[n_done]
    assert stats['n_proposed_overstretched'] <= (stats['n_metropolis_rejected_shift']
                                                 + stats['n_metropolis_rejected_swap'])
    assert stats['n_options'] >= stats['n_sites']


def random_system(n_stickers=600, box=10.0, bonded_fraction=0.6, rc=1.12, seed=1):
    """Random A/B sticker melt with a brute-force neighbour list (identical on every rank),
    split into contiguous chunks across ranks like MC_exchange.neigh_list does."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, box, size=(n_stickers, 3))
    ids = np.arange(1, n_stickers + 1)
    types = np.where(ids % 2 == 1, A, B)
    pairs = []
    for i in range(0, int(bonded_fraction * n_stickers) - 1, 2):
        d = rng.normal(size=3)
        pos[i + 1] = (pos[i] + 0.97 * d / np.linalg.norm(d)) % box
        pairs.append((int(ids[i]), int(ids[i + 1])))
    atoms = make_atoms({int(ids[i]): (int(types[i]), tuple(pos[i])) for i in range(n_stickers)})

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
    return local, make_bonds(pairs), atoms, np.array([0, box, 0, box, 0, box])


# -------------------- Tests --------------------

def test_backward_compatible_return_shapes():
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0))
    neigh = neigh if COMM.Get_rank() == 0 else {}
    random.seed(0)
    out = complementary_bond_exchange(neigh, make_bonds(pairs), make_atoms(spec), BOX, T=1.0,
                                      sticker_types_A=A, sticker_types_B=B)
    assert len(out) == 2

    deleted, created, stats = run(neigh, make_bonds(pairs), make_atoms(spec))
    assert stats['total'] == stats['n_shifts'] + stats['n_swaps']
    assert stats['n_shifts'] + 2 * stats['n_swaps'] == len(deleted) == len(created)


def test_stat_types():
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0))
    neigh = neigh if COMM.Get_rank() == 0 else {}
    _, _, stats = run(neigh, make_bonds(pairs), make_atoms(spec))
    for key in COUNT_KEYS + ('n_shifts', 'n_swaps', 'total'):
        assert type(stats[key]) is int, key
    for key in SUM_KEYS:
        assert type(stats[key]) is float, key


def test_downhill_shift_is_accepted():
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0))
    neigh = neigh if COMM.Get_rank() == 0 else {}
    deleted, created, stats = run(neigh, make_bonds(pairs), make_atoms(spec))

    assert deleted == {1: 2} and created == {2: 3}
    assert stats['n_sites'] == stats['n_options'] == stats['n_proposed_shift'] == 1
    assert stats['n_accepted_local_shift'] == stats['n_shifts'] == 1
    assert stats['n_metropolis_rejected_shift'] == stats['n_proposed_overstretched'] == 0
    assert stats['sum_dE_proposed'] == pytest.approx(stats['sum_dE_accepted'])
    assert stats['sum_dE_accepted'] < 0
    assert_identities(stats)


def test_swap_counters():
    # A1-B2 and A3-B4 both stretched to 1.3; swapping partners gives two bonds of 0.9
    spec = {1: (A, (5.0, 5.0, 5.0)), 2: (B, (6.3, 5.0, 5.0)),
            3: (A, (6.3, 5.9, 5.0)), 4: (B, (5.0, 5.9, 5.0))}
    neigh = neighbours(1, [2, 3, 4]) if COMM.Get_rank() == 0 else {}
    deleted, created, stats = run(neigh, make_bonds([(1, 2), (3, 4)]), make_atoms(spec))

    assert stats['n_proposed_swap'] == stats['n_accepted_local_swap'] == stats['n_swaps'] == 1
    assert stats['n_proposed_shift'] == stats['n_shifts'] == 0
    assert set(deleted.items()) == {(1, 2), (3, 4)}
    assert set(created.items()) == {(1, 4), (2, 3)}
    assert_identities(stats)


def test_overstretched_proposal_is_counted_and_sum_stays_finite():
    # The new bond would be 2.0 > R0 = 1.5: raw FENE is inf, so the move is always rejected
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0), new_bond=2.0, old_bond=1.0)
    neigh = neigh if COMM.Get_rank() == 0 else {}
    deleted, _, stats = run(neigh, make_bonds(pairs), make_atoms(spec), T=1e6)

    assert deleted == {}
    assert stats['n_proposed_overstretched'] == 1
    assert stats['n_metropolis_rejected_shift'] == 1
    assert np.isfinite(stats['sum_dE_proposed']) and stats['sum_dE_proposed'] == 0.0
    assert stats['sum_dE_accepted'] == 0.0
    assert_identities(stats)


def test_existing_overstretched_bond_keeps_sums_finite():
    # The old bond is 1.6 > R0 (LAMMPS fene only warns below 2*R0): delta E = -inf, so the move
    # is accepted, but it is not "overstretched" (that means a new bond >= R0) and not summed
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0), new_bond=0.9, old_bond=1.6)
    neigh = neigh if COMM.Get_rank() == 0 else {}
    _, _, stats = run(neigh, make_bonds(pairs), make_atoms(spec))

    assert stats['n_shifts'] == 1
    assert stats['n_proposed_overstretched'] == 0
    assert stats['sum_dE_proposed'] == 0.0 and stats['sum_dE_accepted'] == 0.0
    assert_identities(stats)


def test_low_and_high_temperature_limits():
    # Uphill move: the bond would grow from 0.9 to 1.3, delta E = O(10)
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0), new_bond=1.3, old_bond=0.9)
    neigh = neigh if COMM.Get_rank() == 0 else {}
    bonds, atoms = make_bonds(pairs), make_atoms(spec)

    _, _, cold = run(neigh, bonds, atoms, T=1e-6)
    assert cold['n_metropolis_rejected_shift'] == 1 and cold['n_shifts'] == 0
    assert cold['sum_dE_proposed'] > 0

    _, _, hot = run(neigh, bonds, atoms, T=1e9)
    assert hot['n_metropolis_rejected_shift'] == 0 and hot['n_shifts'] == 1


def test_quarantined_site_is_counted():
    # Site 1 accepts (1,2)->(3,2); the site centred on atom 2 is then skipped
    spec, pairs, neigh = shift_site(0, (5.0, 5.0, 5.0))
    spec[4] = (A, (6.3, 4.1, 5.0))
    neigh = {**neigh, 2: {4: 1.0, 3: 1.0}} if COMM.Get_rank() == 0 else {}
    _, _, stats = run(neigh, make_bonds(pairs), make_atoms(spec))

    assert stats['n_skipped_quarantined'] == 1
    assert stats['n_shifts'] == 1
    assert_identities(stats)


@pytest.mark.parametrize('seed', range(5))
def test_identities_random_system(seed):
    neigh, bonds, atoms, box = random_system(seed=seed + 1)
    random.seed(seed * 1000 + COMM.Get_rank())
    deleted, _, stats = complementary_bond_exchange(neigh, bonds, atoms, box, T=1.0,
                                                    sticker_types_A=A, sticker_types_B=B,
                                                    return_stats=True)
    assert stats['n_sites'] > 0 and stats['n_proposed_shift'] > 0 and stats['n_proposed_swap'] > 0
    assert stats['n_proposed_overstretched'] > 0          # Rc = 1.12 allows new bonds > R0
    assert np.isfinite(stats['sum_dE_proposed'])
    assert stats['n_shifts'] + 2 * stats['n_swaps'] == len(deleted)
    assert_identities(stats)


@pytest.mark.parametrize('bond_shift, bond_swap', [(False, True), (True, False), (False, False)])
def test_mechanism_switches(bond_shift, bond_swap):
    neigh, bonds, atoms, box = random_system(seed=7)
    random.seed(COMM.Get_rank())
    _, _, stats = complementary_bond_exchange(neigh, bonds, atoms, box, T=1.0,
                                              sticker_types_A=A, sticker_types_B=B,
                                              bond_shift=bond_shift, bond_swap=bond_swap,
                                              return_stats=True)
    off = [k for k, on in (('shift', bond_shift), ('swap', bond_swap)) if not on]
    for kind in off:
        assert all(stats[k] == 0 for k in COUNT_KEYS if k.endswith(kind)), kind
    if not bond_shift and not bond_swap:
        assert all(stats[k] == 0 for k in COUNT_KEYS)
        assert all(stats[k] == 0.0 for k in SUM_KEYS)
    assert_identities(stats)


def test_counts_independent_of_rank_count_for_disjoint_sites():
    # Eight atom-disjoint downhill shift sites, one candidate each, dealt round-robin to ranks
    spec, pairs, neigh_all = {}, [], []
    for s in range(8):
        sp, pr, nb = shift_site(3 * s, (5.0 * (s + 1), 5.0, 5.0))
        spec.update(sp); pairs += pr; neigh_all.append(nb)
    rank, size = COMM.Get_rank(), COMM.Get_size()
    neigh = {k: v for i, nb in enumerate(neigh_all) if i % size == rank for k, v in nb.items()}
    _, _, stats = run(neigh, make_bonds(pairs), make_atoms(spec), seed=rank)

    assert stats['n_sites'] == stats['n_shifts'] == 8
    assert stats['n_conflict_rejected_shift'] == 0
    assert_identities(stats)


@pytest.mark.skipif(COMM.Get_size() < 2, reason='needs at least 2 MPI ranks')
def test_cross_rank_conflict_is_counted():
    # Ranks 0 and 1 each accept a shift that breaks the same bond A1-B2; root keeps rank 0's
    spec, pairs, _ = shift_site(0, (5.0, 5.0, 5.0))
    spec[4] = (A, (6.3, 4.2, 5.0))
    rank = COMM.Get_rank()
    neigh = {0: neighbours(1, [2, 3]), 1: neighbours(1, [2, 4])}.get(rank, {})
    deleted, created, stats = run(neigh, make_bonds(pairs), make_atoms(spec))

    assert stats['n_accepted_local_shift'] == 2
    assert stats['n_conflict_rejected_shift'] == 1
    assert stats['n_shifts'] == 1
    assert deleted == {1: 2} and created == {2: 3}
    assert_identities(stats)
