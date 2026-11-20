"""Tests for the LAMMPS data functions (gather_bonds / gather_atoms type filters and root
handling, get_box_dims, gathered coordinates) and for the LAMMPS -> gather_atoms -> neigh_list
-> exchange chain with stickers across the periodic faces of a shifted box.

Needs a working LAMMPS Python module; the tests are skipped if it cannot be loaded
(on macOS with Homebrew MPI, `DYLD_LIBRARY_PATH=/opt/homebrew/lib` may be needed).
Run serially with `pytest tests/test_data.py` and in parallel with
`mpirun -np 2 python -m pytest tests/test_data.py`.
"""
import numpy as np
import pytest
from mpi4py import MPI

import MC_exchange as mc

COMM = MPI.COMM_WORLD


def _make_lammps():
    try:
        from lammps import lammps
        return lammps(cmdargs=['-screen', 'none', '-log', 'none'], comm=COMM)
    except Exception as exc:        # ImportError, or OSError when liblammps cannot be loaded
        pytest.skip(f'LAMMPS not available: {exc}')


@pytest.fixture(scope='module')
def lmp():
    """Six atoms (types 1, 2 alternating) joined by three bonds of types 1, 2 and 3."""
    lmp = _make_lammps()
    lmp.commands_string("""
        units lj
        atom_style bond
        region box block 0 10 0 10 0 10
        create_box 2 box bond/types 3 extra/bond/per/atom 3
        mass * 1.0
        bond_style harmonic
        bond_coeff * 1.0 1.0
        create_atoms 1 single 1 1 1
        create_atoms 2 single 2 1 1
        create_atoms 1 single 1 3 1
        create_atoms 2 single 2 3 1
        create_atoms 1 single 1 5 1
        create_atoms 2 single 2 5 1
        create_bonds single/bond 1 1 2
        create_bonds single/bond 2 3 4
        create_bonds single/bond 3 5 6
    """)
    yield lmp
    lmp.close()


@pytest.mark.parametrize('bond_type, expected', [
    (None, [1, 2, 3]),
    (2, [2]),
    (np.int64(2), [2]),             # numpy integers used to fall through to "all bonds"
    ([1, 3], [1, 3]),               # list filter used to compare the column against itself
])
def test_gather_bonds_type_filter(lmp, bond_type, expected):
    bonds = mc.gather_bonds(lmp, bond_type)
    assert sorted(bonds['type'].tolist()) == expected


@pytest.mark.parametrize('atom_type, expected', [
    (1, [1, 3, 5]),
    (np.int64(1), [1, 3, 5]),
    ([1, 2], [1, 2, 3, 4, 5, 6]),
])
def test_gather_atoms_type_filter(lmp, atom_type, expected):
    atoms = mc.gather_atoms(lmp, atom_type)
    assert sorted(atoms['id'].tolist()) == expected


@pytest.mark.skipif(COMM.Get_size() < 2, reason='needs at least 2 MPI ranks')
def test_non_zero_root(lmp):
    # Every rank must receive the root's selection when root != 0
    bonds = mc.gather_bonds(lmp, [1, 3], root=1)
    atoms = mc.gather_atoms(lmp, 2, root=1)
    assert sorted(bonds['type'].tolist()) == [1, 3]
    assert sorted(atoms['id'].tolist()) == [2, 4, 6]


# -------------------- get_box_dims and gathered coordinates --------------------

def test_get_box_dims(lmp):
    box = mc.get_box_dims(lmp)
    assert isinstance(box, np.ndarray)
    assert box.tolist() == [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]


def test_gather_atoms_fields_and_coordinates(lmp):
    atoms = mc.gather_atoms(lmp)
    assert atoms.dtype.names == ('id', 'type', 'mol', 'x', 'y', 'z')
    assert atoms['id'].tolist() == [1, 2, 3, 4, 5, 6]                  # sorted by id
    assert atoms['type'].tolist() == [1, 2, 1, 2, 1, 2]
    expected = [(1, 1, 1), (2, 1, 1), (1, 3, 1), (2, 3, 1), (1, 5, 1), (2, 5, 1)]
    assert np.allclose(np.column_stack([atoms['x'], atoms['y'], atoms['z']]), expected)


def test_gather_bonds_pairs(lmp):
    bonds = mc.gather_bonds(lmp)
    pairs = {(int(t), *sorted((int(a), int(b)))) for t, a, b in
             zip(bonds['type'], bonds['atom 1'], bonds['atom 2'])}
    assert pairs == {(1, 1, 2), (2, 3, 4), (3, 5, 6)}


# -------------------- LAMMPS -> gather_atoms -> neigh_list -> exchange --------------------

@pytest.fixture(scope='module')
def lmp_faces():
    """A/B stickers (types 1/2, all bonds type 1) placed across the periodic faces of a shifted box,
    with the force field of Vitrimer_setup.lmp."""
    lmp = _make_lammps()
    lmp.commands_string("""
        units lj
        atom_style bond
        boundary p p p
        region box block -6 6 -6 6 -6 6
        create_box 2 box bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 4
        mass * 1.0
        pair_style lj/cut 1.122462048309373
        pair_modify shift yes
        pair_coeff * * 1.0 1.0
        bond_style fene
        bond_coeff 1 30.0 1.5 1.0 1.0
        special_bonds fene
        create_atoms 1 single  5.70  0.0  0.0
        create_atoms 2 single -5.75  0.0  0.0
        create_atoms 1 single  0.0  -5.80  2.0
        create_atoms 2 single  0.0   5.70  2.0
        create_atoms 2 single  0.5  -5.70  2.6
        create_atoms 1 single -5.6  -5.6  -5.6
        create_atoms 2 single  5.8   5.8   5.8
        create_atoms 1 single  0.0   0.0   0.0
        create_bonds single/bond 1 3 4
    """)
    yield lmp
    lmp.close()


def test_neigh_list_from_lammps_finds_pairs_across_faces(lmp_faces):
    atoms = mc.gather_atoms(lmp_faces, [1, 2])
    box = mc.get_box_dims(lmp_faces)
    local = mc.neigh_list(atoms, box, Rc=1.12)
    merged = {}
    for chunk in COMM.allgather(local):
        assert not set(chunk) & set(merged)                             # one owner per atom
        merged.update(chunk)

    xyz = {int(a['id']): (a['x'], a['y'], a['z']) for a in atoms}
    expected = {(i, j) for i in xyz for j in xyz
                if i < j and mc.calculate_distance_pbc(box, *xyz[i], *xyz[j]) <= 1.12}
    assert expected == {(1, 2), (3, 4), (3, 5), (4, 5), (6, 7)}         # x face, y face, corner
    assert {(min(a, b), max(a, b)) for a, nb in merged.items() for b in nb} == expected
    for a, nb in merged.items():
        for b, r in nb.items():
            assert r == mc.calculate_distance_pbc(box, *xyz[a], *xyz[b])


def test_exchange_across_face_is_applied_in_lammps(lmp_faces):
    # Bond 3-4 spans the y face; sticker 5 is free and closer to 3 than 4 is, so a shift is possible
    atoms = mc.gather_atoms(lmp_faces, [1, 2])
    bonds = mc.gather_bonds(lmp_faces, 1)
    box = mc.get_box_dims(lmp_faces)
    local = mc.neigh_list(atoms, box, Rc=1.12)
    deleted, created, stats = mc.complementary_bond_exchange(local, bonds, atoms, box, T=1e6,
                                                             sticker_types_A=1, sticker_types_B=2,
                                                             bond_swap=False, return_stats=True)
    assert stats['n_sites'] == 1                                        # the site spans the face
    assert len(deleted) == len(created)
    for a, b in created.items():
        assert {int(a), int(b)} in ({3, 4}, {3, 5})
