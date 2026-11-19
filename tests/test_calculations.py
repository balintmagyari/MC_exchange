"""Tests for MC_exchange.calculations: minimum-image distances and the bond / pair potentials.

The potentials are checked against closed-form values and against the energies LAMMPS itself
computes with the settings of Vitrimer_setup.lmp (bond_style fene, pair_style lj/cut 2^(1/6)
with pair_modify shift yes). The LAMMPS tests are skipped if LAMMPS cannot be loaded
(on macOS with Homebrew MPI, `DYLD_LIBRARY_PATH=/opt/homebrew/lib` may be needed).

Run serially with `pytest tests/test_calculations.py` and in parallel with
`mpirun -np 2 python -m pytest tests/test_calculations.py`.
"""
import math

import numpy as np
import pytest
from mpi4py import MPI

from MC_exchange.calculations import (calculate_distance_pbc, calculate_fene_potential,
                                      calculate_raw_fene_potential, calculate_lj_potential)

COMM = MPI.COMM_WORLD
BOX = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
K, R0, EPS, SIGMA = 30.0, 1.5, 1.0, 1.0
WCA = 2 ** (1 / 6)


def fene_ref(r):
    return -0.5 * K * R0 ** 2 * math.log(1 - (r / R0) ** 2)


def wca_ref(r):
    return 4 * EPS * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6) + EPS if r < WCA * SIGMA else 0.0


# -------------------- calculate_distance_pbc --------------------

@pytest.mark.parametrize('p1, p2, expected', [
    ((1, 1, 1), (2, 1, 1), 1.0),                   # plain distance
    ((1, 2, 3), (4, 6, 3), 5.0),                   # 3-4-5 triangle
    ((0.2, 5, 5), (9.9, 5, 5), 0.3),               # across the x face
    ((5, 0.1, 5), (5, 9.6, 5), 0.5),               # across the y face
    ((5, 5, 9.8), (5, 5, 0.4), 0.6),               # across the z face
    ((0.1, 0.1, 0.1), (9.9, 9.9, 9.9), math.sqrt(3 * 0.2 ** 2)),   # across a corner
    ((-0.3, 5, 5), (0.2, 5, 5), 0.5),              # coordinate below the box
    ((20.5, 5, 5), (0.2, 5, 5), 0.3),              # two box lengths away
    ((3, 3, 3), (3, 3, 3), 0.0),
])
def test_distance_pbc_values(p1, p2, expected):
    assert calculate_distance_pbc(BOX, *p1, *p2) == pytest.approx(expected, abs=1e-12)
    assert calculate_distance_pbc(BOX, *p2, *p1) == pytest.approx(expected, abs=1e-12)


def test_distance_pbc_half_box_is_largest_image_distance():
    assert calculate_distance_pbc(BOX, 0, 0, 0, 5, 0, 0) == pytest.approx(5.0)
    assert calculate_distance_pbc(BOX, 0, 0, 0, 5, 5, 5) == pytest.approx(math.sqrt(75))


def test_distance_pbc_shifted_and_anisotropic_box():
    box = np.array([-5.0, 5.0, 2.0, 22.0, -1.0, 2.0])     # lengths 10, 20, 3
    assert calculate_distance_pbc(box, -4.9, 3, 0, 4.9, 3, 0) == pytest.approx(0.2)
    assert calculate_distance_pbc(box, 0, 2.5, 0, 0, 21.5, 0) == pytest.approx(1.0)
    assert calculate_distance_pbc(box, 0, 5, -0.9, 0, 5, 1.9) == pytest.approx(0.2)
    assert calculate_distance_pbc(box, 0, 5, 0, 0, 15, 0) == pytest.approx(10.0)  # half of y, no wrap in x


def test_distance_pbc_accepts_one_element_arrays():
    # The exchange code passes atoms[...]['x'] slices of length 1
    a, b = np.array([0.2]), np.array([9.9])
    assert calculate_distance_pbc(BOX, a, np.array([5.0]), np.array([5.0]),
                                  b, np.array([5.0]), np.array([5.0])) == pytest.approx(0.3)
    assert isinstance(calculate_distance_pbc(BOX, a, 5.0, 5.0, b, 5.0, 5.0), float)


def test_distance_pbc_matches_brute_force_images():
    rng = np.random.default_rng(0)
    L = BOX[1::2] - BOX[::2]
    for _ in range(200):
        p1, p2 = rng.uniform(-10, 20, 3), rng.uniform(-10, 20, 3)
        images = [np.linalg.norm(p1 - p2 + L * np.array(s))
                  for s in np.ndindex(7, 7, 7) for s in [np.array(s) - 3]]
        assert calculate_distance_pbc(BOX, *p1, *p2) == pytest.approx(min(images), abs=1e-12)


# -------------------- calculate_raw_fene_potential --------------------

@pytest.mark.parametrize('r', [0.5, 0.8, 0.97, 1.0, 1.12, 1.3, 1.49])
def test_raw_fene_closed_form(r):
    assert calculate_raw_fene_potential(r) == pytest.approx(fene_ref(r), rel=1e-12)


@pytest.mark.parametrize('r', [1.5, 1.6, 10.0])
def test_raw_fene_is_infinite_at_and_beyond_r0(r):
    assert calculate_raw_fene_potential(r) == math.inf


def test_raw_fene_is_increasing_and_custom_parameters():
    rs = np.linspace(0.1, 1.49, 100)
    v = [calculate_raw_fene_potential(r) for r in rs]
    assert all(b > a for a, b in zip(v, v[1:]))
    assert calculate_raw_fene_potential(1.0, K=10.0, R0=2.0) == pytest.approx(-0.5 * 10 * 4 * math.log(0.75))
    assert calculate_raw_fene_potential(2.0, K=10.0, R0=2.0) == math.inf


# -------------------- calculate_lj_potential --------------------

@pytest.mark.parametrize('r', [0.8, 0.9, 1.0, 1.05, 1.12])
def test_lj_closed_form_inside_cutoff(r):
    assert calculate_lj_potential(r) == pytest.approx(wca_ref(r), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize('r', [WCA, 1.2, 2.5])
def test_lj_is_zero_at_and_beyond_cutoff(r):
    assert calculate_lj_potential(r) == 0


def test_lj_is_continuous_at_cutoff():
    assert calculate_lj_potential(WCA * (1 - 1e-9)) == pytest.approx(0.0, abs=1e-6)


def test_lj_custom_cutoff_is_shifted_to_zero():
    rc = 2.5
    shift = 4 * ((1 / rc) ** 12 - (1 / rc) ** 6)
    r = 1.5
    assert calculate_lj_potential(r, Rc=rc) == pytest.approx(4 * ((1 / r) ** 12 - (1 / r) ** 6) - shift)
    assert calculate_lj_potential(rc * (1 - 1e-9), Rc=rc) == pytest.approx(0.0, abs=1e-6)


# -------------------- calculate_fene_potential (FENE + WCA, used by perform_bond_swap) --------------------

@pytest.mark.parametrize('r', [0.8, 0.97, 1.05, 1.12])
def test_fene_with_wca_inside_wca_range(r):
    assert calculate_fene_potential(r) == pytest.approx(fene_ref(r) + wca_ref(r), rel=1e-12)


@pytest.mark.parametrize('r', [1.5, 2.0])
def test_fene_with_wca_is_huge_beyond_r0(r):
    assert calculate_fene_potential(r) == 10 ** 20


@pytest.mark.xfail(strict=True, reason='Known issue: calculate_fene_potential applies the WCA term '
                                       'beyond 2^(1/6) sigma, where LAMMPS bond_style fene does not. '
                                       'Only the deprecated perform_bond_swap uses it.')
@pytest.mark.parametrize('r', [1.2, 1.4])
def test_fene_with_wca_beyond_wca_range(r):
    assert calculate_fene_potential(r) == pytest.approx(fene_ref(r), rel=1e-12)


# -------------------- Cross-check against LAMMPS --------------------

def _make_lammps():
    try:
        from lammps import lammps
        return lammps(cmdargs=['-screen', 'none', '-log', 'none'], comm=COMM)
    except Exception as exc:        # ImportError, or OSError when liblammps cannot be loaded
        pytest.skip(f'LAMMPS not available: {exc}')


@pytest.fixture(scope='module')
def lmp():
    """Two bonded atoms with the force field of Vitrimer_setup.lmp, plus an unbonded pair far away."""
    lmp = _make_lammps()
    lmp.commands_string(f"""
        units lj
        atom_style bond
        boundary p p p
        region box block 0 20 0 20 0 20
        create_box 1 box bond/types 1 extra/bond/per/atom 1 extra/special/per/atom 2
        mass * 1.0
        pair_style lj/cut {WCA}
        pair_modify shift yes
        pair_coeff * * {EPS} {SIGMA}
        bond_style fene
        bond_coeff 1 {K} {R0} {EPS} {SIGMA}
        special_bonds fene
        create_atoms 1 single 5 5 5
        create_atoms 1 single 6 5 5
        create_atoms 1 single 15 15 15
        create_atoms 1 single 16 15 15
        create_bonds single/bond 1 1 2
        thermo_style custom step ebond evdwl
        thermo_modify norm no
    """)
    yield lmp
    lmp.close()


def _energies(lmp, r_bonded, r_pair):
    lmp.command(f'set atom 2 x {5 + r_bonded}')
    lmp.command(f'set atom 4 x {15 + r_pair}')
    lmp.command('run 0')
    return lmp.get_thermo('ebond'), lmp.get_thermo('evdwl')


@pytest.mark.parametrize('r', [0.8, 0.9, 0.97, 1.0, 1.05, 1.1, 1.12])
def test_potentials_match_lammps_inside_wca_range(lmp, r):
    ebond, evdwl = _energies(lmp, r, r)
    # bond_style fene includes the WCA term for bonded pairs; special_bonds fene removes the pair term
    assert ebond == pytest.approx(calculate_raw_fene_potential(r) + calculate_lj_potential(r), rel=1e-10)
    assert ebond == pytest.approx(calculate_fene_potential(r), rel=1e-10)
    assert evdwl == pytest.approx(calculate_lj_potential(r), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize('r', [1.15, 1.25, 1.4])
def test_potentials_match_lammps_beyond_wca_range(lmp, r):
    ebond, evdwl = _energies(lmp, r, r)
    assert ebond == pytest.approx(calculate_raw_fene_potential(r) + calculate_lj_potential(r), rel=1e-10)
    assert evdwl == pytest.approx(0.0, abs=1e-12)
    assert calculate_lj_potential(r) == 0
