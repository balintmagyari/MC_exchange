"""Tests for the type filters and root handling of gather_bonds / gather_atoms.

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
