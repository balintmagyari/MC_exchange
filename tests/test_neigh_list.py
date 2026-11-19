"""Tests for MC_exchange.neigh_list.

Run serially with `pytest tests/test_neigh_list.py` and in parallel with
`mpirun -np 4 python -m pytest tests/test_neigh_list.py`. Every test is valid on any
number of ranks: the scattered chunks are gathered and checked as one list.

Regression for BUG_neigh_list_periodic_faces.md: pairs straddling a periodic face
were never listed (v1.3.0), and boxes with lo != 0 lost most pairs.
"""
import numpy as np
import pytest
from mpi4py import MPI

from MC_exchange import neigh_list, calculate_distance_pbc

COMM = MPI.COMM_WORLD
L = 36.12
RC = 1.12
BOX = np.array([0.0, L, 0.0, L, 0.0, L])


# -------------------- Helpers --------------------

def make_atoms(pos: np.ndarray) -> np.ndarray:
    atoms = np.zeros(len(pos), dtype=[('id', 'i4'), ('type', 'i4'), ('mol', 'i4'),
                                      ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    atoms['id'] = np.arange(1, len(pos) + 1)
    atoms['type'] = 1
    atoms['x'], atoms['y'], atoms['z'] = np.asarray(pos, dtype=float).T
    return atoms


def gathered(atoms, box=BOX, rc=RC, **kwargs):
    """Run neigh_list, gather all ranks' chunks; returns (chunks, merged) identical on every rank."""
    local = neigh_list(atoms, box, Rc=rc, comm=COMM, **kwargs)
    chunks = COMM.allgather(local)
    merged = {}
    for chunk in chunks:
        merged.update(chunk)
    return chunks, merged


def pair_set(merged: dict) -> set:
    return {(min(a, b), max(a, b)) for a, nb in merged.items() for b in nb}


def brute_force_pairs(pos: np.ndarray, box: np.ndarray, rc: float) -> dict:
    """Independent O(N^2) reference: {(id_i, id_j): distance} with id_i < id_j."""
    lengths = box[1::2] - box[::2]
    ref = {}
    for i in range(len(pos)):
        d = pos[i + 1:] - pos[i]
        d -= lengths * np.round(d / lengths)
        r = np.sqrt((d ** 2).sum(axis=1))
        for k in np.where(r <= rc)[0]:
            ref[(i + 1, i + 2 + int(k))] = float(r[k])
    return ref


# -------------------- Tests --------------------

@pytest.mark.parametrize('offset', [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),          # faces
    (1, 1, 0), (0, 1, 1), (1, 0, 1),          # edges
    (1, 1, 1),                                # corner
])
def test_pair_across_periodic_boundary(offset):
    a = np.array([5.0, 5.0, 5.0])
    b = a.copy()
    for d in range(3):
        if offset[d]:
            a[d], b[d] = 0.1, L - 0.3
    _, merged = gathered(make_atoms([a, b]))
    assert pair_set(merged) == {(1, 2)}
    expected = calculate_distance_pbc(BOX, *a, *b)
    assert merged[1][2] == pytest.approx(expected, abs=1e-12)
    assert merged[2][1] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('box', [
    BOX,
    np.array([-L / 2, L / 2, -L / 2, L / 2, -L / 2, L / 2]),        # centred box (lo != 0)
    np.array([3.0, 3.0 + L, -7.0, -7.0 + 30.0, 1.5, 1.5 + 25.0]),   # anisotropic, shifted
])
def test_random_configuration_matches_brute_force(box):
    rng = np.random.default_rng(7)
    lo, hi = box[::2], box[1::2]
    pos = lo + rng.random((2000, 3)) * (hi - lo)
    chunks, merged = gathered(make_atoms(pos), box=box)

    ref = brute_force_pairs(pos, box, RC)
    assert pair_set(merged) == set(ref)
    for (i, j), r in ref.items():
        assert merged[i][j] == pytest.approx(r, abs=1e-12)
        assert merged[j][i] == pytest.approx(r, abs=1e-12)

    # Chunks are disjoint and cover every atom that has a neighbour
    keys = [k for chunk in chunks for k in chunk]
    assert len(keys) == len(set(keys))
    assert set(keys) == {k for p in ref for k in p}


def test_atoms_slightly_outside_box():
    # LAMMPS only remaps atoms into the box on reneighboring steps
    pos = [[-0.05, 5.0, 5.0], [L - 0.5, 5.0, 5.0],
           [L + 0.05, 20.0, 20.0], [0.4, 20.0, 20.0]]
    _, merged = gathered(make_atoms(pos))
    assert pair_set(merged) == {(1, 2), (3, 4)}
    assert merged[1][2] == pytest.approx(0.45, abs=1e-12)
    assert merged[3][4] == pytest.approx(0.35, abs=1e-12)


def test_cutoff_is_inclusive():
    pos = [[10.0, 10.0, 10.0], [10.0 + 1.0, 10.0, 10.0],     # pair at exactly rc = 1.0
           [20.0, 20.0, 20.0], [20.0 + 1.0 + 1e-9, 20.0, 20.0]]
    _, merged = gathered(make_atoms(pos), rc=1.0)
    assert pair_set(merged) == {(1, 2)}


def test_non_periodic_dimension_does_not_wrap():
    pos = [[5.0, 5.0, 0.1], [5.0, 5.0, L - 0.3]]
    _, merged = gathered(make_atoms(pos), pbc=[True, True, False])
    assert merged == {}
    _, merged = gathered(make_atoms(pos), pbc=[True, True, True])
    assert pair_set(merged) == {(1, 2)}


def test_python_types_and_sorted_order():
    rng = np.random.default_rng(3)
    _, merged = gathered(make_atoms(rng.random((500, 3)) * L))
    for a, nb in merged.items():
        assert type(a) is int
        assert list(nb) == sorted(nb)
        assert all(type(b) is int and type(r) is float for b, r in nb.items())


def test_rc_too_large_raises():
    with pytest.raises(ValueError):
        neigh_list(make_atoms([[1.0, 1.0, 1.0]]), np.array([0, 2.0, 0, 2.0, 0, 2.0]),
                   Rc=1.12, comm=COMM)


def test_atoms_are_owned_by_their_spatial_subdomain():
    # Spatially compact chunks keep cross-rank conflicts in the exchange step rare
    rng = np.random.default_rng(11)
    pos = rng.random((2000, 3)) * L
    chunks, _ = gathered(make_atoms(pos))
    grid = np.array(MPI.Compute_dims(COMM.Get_size(), [0, 0, 0]))
    for rank, chunk in enumerate(chunks):
        for atom_id in chunk:
            c = np.minimum((pos[atom_id - 1] / L * grid).astype(int), grid - 1)
            assert (c[0] * grid[1] + c[1]) * grid[2] + c[2] == rank


def test_failure_on_root_is_raised_on_every_rank():
    # Missing coordinate fields fail inside the pair search on rank 0 only;
    # every rank must raise instead of waiting forever in scatter
    bad = np.zeros(3, dtype=[('id', 'i4'), ('x', 'f8')])
    with pytest.raises(RuntimeError):
        neigh_list(bad, BOX, Rc=RC, comm=COMM)


def test_distances_bit_identical_to_calculate_distance_pbc():
    # The exchange step recomputes distances with calculate_distance_pbc on raw coordinates;
    # the list must agree exactly, also for shifted boxes and atoms outside the box
    box = np.array([-18.0123, 18.0123, -7.0, 29.12, 0.5, 36.62])
    rng = np.random.default_rng(5)
    lo, hi = box[::2], box[1::2]
    pos = lo + rng.random((3000, 3)) * (hi - lo)
    pos[::10] += (hi - lo) * rng.integers(-2, 3, size=(len(pos[::10]), 3))
    _, merged = gathered(make_atoms(pos), box=box)
    assert merged
    for a, nb in merged.items():
        for b, r in nb.items():
            assert r == calculate_distance_pbc(box, *pos[a - 1], *pos[b - 1])


def test_pair_at_exactly_rc_in_shifted_box():
    box = np.array([-18.0123, 18.0123, -18.0123, 18.0123, -18.0123, 18.0123])
    pos = np.array([[-0.0, 1.0, 1.0], [1.12, 1.0, 1.0]])
    r = calculate_distance_pbc(box, *pos[0], *pos[1])
    _, merged = gathered(make_atoms(pos), box=box, rc=r)
    assert pair_set(merged) == {(1, 2)}


def test_wrap_float_edge_case():
    # -1e-17 % L returns L in floating point; such atoms must still be found
    pos = [[-1e-17, 5.0, 5.0], [L - 0.5, 5.0, 5.0]]
    _, merged = gathered(make_atoms(pos))
    assert pair_set(merged) == {(1, 2)}


@pytest.mark.parametrize('n', [0, 1])
def test_empty_and_single_atom(n):
    chunks, merged = gathered(make_atoms(np.full((n, 3), 3.0)))
    assert merged == {} and all(c == {} for c in chunks)


def test_convert_np_false_keeps_numpy_types():
    rng = np.random.default_rng(9)
    _, merged_np = gathered(make_atoms(rng.random((500, 3)) * L), convert_np=False)
    _, merged_py = gathered(make_atoms(np.random.default_rng(9).random((500, 3)) * L))
    assert merged_np == merged_py
    a, nb = next(iter(merged_np.items()))
    assert isinstance(a, np.integer) and isinstance(next(iter(nb.values())), np.floating)


@pytest.mark.parametrize('box, rc, pbc', [
    (np.array([0, L, 0, L, 0, L, 0]), RC, [True] * 3),          # wrong size
    (np.array([0, L, 0, np.inf, 0, L]), RC, [True] * 3),        # non-finite
    (np.array([0, L, L, 0, 0, L]), RC, [True] * 3),             # hi < lo
    (BOX, RC, [True, True]),                                    # pbc length
    (BOX, 0.0, [True] * 3),
    (BOX, -1.0, [True] * 3),
    (BOX, float('nan'), [True] * 3),
    (BOX, L / 2, [True] * 3),                                   # exactly L/2
])
def test_invalid_input_raises_value_error(box, rc, pbc):
    with pytest.raises(ValueError):
        neigh_list(make_atoms([[1.0, 1.0, 1.0]]), box, Rc=rc, pbc=pbc, comm=COMM)


@pytest.mark.parametrize('bad', ['nan', 'duplicate'])
def test_bad_atoms_raise_on_every_rank(bad):
    atoms = make_atoms([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [2.0, 1.0, 1.0]])
    if bad == 'nan':
        atoms['x'][1] = np.nan
    else:
        atoms['id'][1] = atoms['id'][0]
    with pytest.raises(RuntimeError):
        neigh_list(atoms, BOX, Rc=RC, comm=COMM)


def test_unpicklable_exception_on_root_does_not_hang(monkeypatch):
    import sys
    nl_module = sys.modules['MC_exchange.neigh_list']     # The package re-exports the function name

    class Odd(Exception):
        def __init__(self, a, b):           # Cannot be rebuilt by unpickling
            super().__init__(a)
            self.f = lambda: b               # Cannot be pickled at all

    def broken(*args, **kwargs):
        raise Odd('boom', 2)

    monkeypatch.setattr(nl_module, '_find_pairs', broken)
    with pytest.raises(RuntimeError, match='boom'):
        neigh_list(make_atoms([[1.0, 1.0, 1.0]]), BOX, Rc=RC, comm=COMM)


def test_neigh_list_into_complementary_exchange():
    # End to end under MPI: real neigh_list split, exchange keeps the topology valid
    from MC_exchange import complementary_bond_exchange
    rng = np.random.default_rng(21)
    n = 1500
    pos = rng.random((n, 3)) * L / 3                     # dense enough for many sites
    box = np.array([0, L / 3] * 3)
    atoms = make_atoms(pos)
    atoms['type'] = np.where(atoms['id'] % 2 == 1, 2, 3)
    bonds, used = [], set()
    for i in range(0, n - 1, 2):                         # bond A-B neighbours where possible
        d = pos[i + 1] - pos[i]
        d -= (L / 3) * np.round(d / (L / 3))
        if np.linalg.norm(d) < 1.12:
            bonds.append((2, i + 1, i + 2)); used |= {i + 1, i + 2}
    bonds = np.array(bonds, dtype=[('type', 'i4'), ('atom 1', 'i4'), ('atom 2', 'i4')])
    atoms['type'][np.isin(atoms['id'], list(used)) & (atoms['type'] == 2)] = 4
    atoms['type'][np.isin(atoms['id'], list(used)) & (atoms['type'] == 3)] = 5

    import random
    random.seed(100 + COMM.Get_rank())
    local = neigh_list(atoms, box, Rc=1.12, comm=COMM)
    deleted, created, stats = complementary_bond_exchange(local, bonds, atoms, box, T=5.0,
                                                          sticker_types_A=[2, 4], sticker_types_B=[3, 5],
                                                          return_stats=True, comm=COMM)
    assert stats['n_sites'] > 0
    assert len(deleted) == len(created)
    new = {tuple(sorted(p)) for p in zip(bonds['atom 1'].tolist(), bonds['atom 2'].tolist())}
    new -= {tuple(sorted((int(a), int(b)))) for a, b in deleted.items()}
    new |= {tuple(sorted((int(a), int(b)))) for a, b in created.items()}
    ends = [a for p in new for a in p]
    assert len(new) == len(bonds) and len(ends) == len(set(ends))          # valency <= 1
    assert all((a % 2) != (b % 2) for a, b in new)                          # A-B only
