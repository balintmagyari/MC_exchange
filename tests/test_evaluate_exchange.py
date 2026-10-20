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
