import numpy as np

def calculate_distance_pbc(box_dims: np.ndarray,
                           atom_1_x: float | np.ndarray,
                           atom_1_y: float | np.ndarray,
                           atom_1_z: float | np.ndarray,
                           atom_2_x: float | np.ndarray,
                           atom_2_y: float | np.ndarray,
                           atom_2_z: float | np.ndarray) -> float:
    """
    Calculates distance between two atoms, while also adjusting for periodic boundary conditions.

    Parameters
    ----------
    box_dims : np.ndarray
        box dimensions read from LAMMPS data file.
    atom_1_x : float | np.ndarray
        x-coordinate of atom 1.
    atom_1_y : float | np.ndarray
        y-coordinate of atom 1.
    atom_1_z : float | np.ndarray
        z-coordinate of atom 1.
    atom_2_x : float | np.ndarray
        x-coordinate of atom 2.
    atom_2_y : float | np.ndarray
        y-coordinate of atom 2.
    atom_2_z : float | np.ndarray
        z-coordinate of atom 2.

    Returns
    ----------
    float : distance between the two atoms.
    """
    if isinstance(atom_1_x, np.ndarray):
        atom_1_x = atom_1_x.item()
    if isinstance(atom_1_y, np.ndarray):
        atom_1_y = atom_1_y.item()
    if isinstance(atom_1_z, np.ndarray):
        atom_1_z = atom_1_z.item()
    if isinstance(atom_2_x, np.ndarray):
        atom_2_x = atom_2_x.item()
    if isinstance(atom_2_y, np.ndarray):
        atom_2_y = atom_2_y.item()
    if isinstance(atom_2_z, np.ndarray):
        atom_2_z = atom_2_z.item()

    # Convert inputs to NumPy arrays
    atom_1 = np.array([atom_1_x, atom_1_y, atom_1_z])
    atom_2 = np.array([atom_2_x, atom_2_y, atom_2_z])
    box_lengths = np.array([
        box_dims[1] - box_dims[0],
        box_dims[3] - box_dims[2],
        box_dims[5] - box_dims[4]
    ])

    # Compute delta and apply periodic boundary conditions in one step
    delta = atom_1 - atom_2
    delta -= box_lengths * np.round(delta / box_lengths)

    # Compute distance
    distance = np.linalg.norm(delta)
    return float(distance)

def calculate_fene_potential(distance: float,
                             K: float = 30.0, R0: float = 1.5, eps: float = 1.0, sigma: float = 1.0) -> float:
    """
    Calculates the FENE potential of two bonding atoms in the system based on the distance between them.

    Parameters
    -----

    distance : float
            Distance between the two atoms. Make sure to adjust for PBC.
    K : float
            Bond coefficient or spring constant. Defaults to 30.0
    R0 : float
            Cut-off distance of the potential. Defaults to 1.5
    eps : float
            Lennard-Jones energy term. MUST match that of the non-bonding LJ term. Defaults to 1.0
    sigma : float
            Zero-crossing distance of the Lennard-Jones potential. Defaults to 1.0

    Returns
    --------

    V : float
            Calculated FENE potential.
    """
    if distance >= R0:
        V = 10**20      # 10^20 stands for infinite FENE potential here.
    else:
        V = -0.5 * K * R0**2 * np.log(1 - (distance/R0)**2) + 4*eps * ((sigma/distance)**12 - (sigma/distance)**6) + eps
    return V

def calculate_raw_fene_potential(distance: float,
                             K: float = 30.0, R0: float = 1.5, eps: float = 1.0, sigma: float = 1.0):
    """
    Calculates the raw FENE potential (no LJ included) of two bonding atoms in the system based on the distance between them.

    Parameters
    -----

    distance : float
            Distance between the two atoms. Make sure to adjust for PBC.
    K : float
            Bond coefficient or spring constant. Defaults to 30.0
    R0 : float
            Cut-off distance of the potential. Defaults to 1.5
    eps : float
            Lennard-Jones energy term. MUST match that of the non-bonding LJ term. Defaults to 1.0
    sigma : float
            Zero-crossing distance of the Lennard-Jones potential. Defaults to 1.0

    Returns
    --------

    V : float
            Calculated FENE potential.
    """
    if distance >= R0:
        V = np.nan      # 10^20 stands for infinite FENE potential here.
    else:
        V = -0.5 * K * R0**2 * np.log(1 - (distance/R0)**2)
    return V

def calculate_lj_potential(distance: float,
                           Rc: float = 2**(1/6), eps: float = 1.0, sigma: float = 1.0):
    if distance >= Rc:
        V = 0
    else:
        V = 4 * eps * ((sigma/distance)**12 - (sigma/distance)**6 - (sigma/Rc)**12 + (sigma/Rc)**6)
    return V