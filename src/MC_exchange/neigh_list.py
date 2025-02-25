import math
import numpy as np

from mpi4py import MPI

from calculations import calculate_distance_pbc


def _calculate_subdomain_boundaries(box_dims: np.ndarray,
                                   pbc: list[bool] = [True, True, True],
                                   comm: MPI.Intracomm = MPI.COMM_WORLD 
                                   ) -> np.ndarray:
    """
    Based on the simulation box dimensions, calculate the boundaries 
    of the subdomain on the current process.

    Parameters
    ----------
    box_dims : np.ndarray
        Simulation box dimensions obtained from LAMMPS. 1D array of size 6,
        in the following order: [xlo, xhi, ylo, yhi, zlo, zhi].
    pbc : list[bool]
        Periodic boundary conditions in the x, y and z directions.
        Default is [True, True, True].
    comm : MPI.Intracomm 
        MPI communicator (default is MPI.COMM_WORLD).

    Returns
    -------
    np.ndarray (3 x 2)
        Boundaries of the subdomain expressed in the same units as the simulation box size.
    """
    box_dims = box_dims.flatten().reshape(3, 2)
    subdomain_bounds = np.empty((3, 2), dtype=np.float64)

    mpi_grid = MPI.Compute_dims(comm.size, [0,0,0])             # 3D grid construction
    cart_comm = comm.Create_cart(mpi_grid, periods=pbc)         # New cartesian communicator
    mpi_coord = cart_comm.Get_coords(comm.rank)                # Translates rank to logical coordinates

    for dim in range(3):
        # Calculate bounds for this process's subdomain
        chunk_size = box_dims[dim,1] / mpi_grid[dim]
        subdomain_bounds[dim,0] = mpi_coord[dim] * chunk_size
        subdomain_bounds[dim,1] = (mpi_coord[dim] + 1) * chunk_size

    return subdomain_bounds

def _local_atoms_and_subdomain(
        atom_data: np.ndarray,
        box_dims: np.ndarray,
        Rc: float = 1.5,
        pbc: list[bool] = [True, True, True],
        comm: MPI.Intracomm = MPI.COMM_WORLD
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function filters the complete list of atoms (that should have been broadcasted to all processes),
    such that the return array will contain atoms which are within the subdomain belonging to the process, but also ghost atoms.

    The ghost atom communication is such that ghost atoms are communicated in the positive coordinate dimensions. 
    This is done in order to preserve computing power later on when the neighbor lists are constructed.

    The returned array is of the same size as the input atom_data array, and also retrains its structured array features.

    Parameters
    ----------
    atom_data : np.ndarray
        Structured array of all atoms' data that have been broadcasted to all processes.
    
    subdomain_bounds : np.ndarray
        Boundaries of the subdomain of the current process.

    box_dims : np.ndarray
        Simulation box dimensions.

    Rc : float
        Neighbor list Rc.

    Returns
    -------
    tuple(np.ndarray)
        Two arrays. The first array is a structured array of atoms' data 
        that belong to the process plus also its ghost atoms. The second array is the subdomain's
        boundaries, in a (3,2) array shape.
    """
    box_dims = box_dims.flatten().reshape(3, 2)

    box_xlo = box_dims[0,0]
    box_xhi = box_dims[0,1]
    box_ylo = box_dims[1,0]
    box_yhi = box_dims[1,1]
    box_zlo = box_dims[2,0]
    box_zhi = box_dims[2,1]

    subdomain_bounds = _calculate_subdomain_boundaries(box_dims=box_dims, pbc=pbc, comm=comm)
    
    domain_xlo = subdomain_bounds[0,0]
    domain_xhi = subdomain_bounds[0,1]
    domain_ylo = subdomain_bounds[1,0]
    domain_yhi = subdomain_bounds[1,1]
    domain_zlo = subdomain_bounds[2,0]
    domain_zhi = subdomain_bounds[2,1]

    x_mask = np.ones(len(atom_data['x']), dtype=bool)
    y_mask = np.ones(len(atom_data['y']), dtype=bool)
    z_mask = np.ones(len(atom_data['z']), dtype=bool)

    if domain_xlo == box_xlo and domain_xhi != box_xhi:
        x_mask = (atom_data['x'] < domain_xhi + Rc) | \
                 (atom_data['x'] >= box_xhi - Rc)
    elif domain_xlo != box_xlo and domain_xhi == box_xhi:
        x_mask = (atom_data['x'] >= domain_xlo - Rc) | \
                 (atom_data['x'] < box_xlo + Rc)
    elif domain_xlo != box_xlo and domain_xhi != box_xhi:
        x_mask = (atom_data['x'] < domain_xhi + Rc) & \
                 (atom_data['x'] >= domain_xlo - Rc)

    if domain_ylo == box_ylo and domain_yhi != box_yhi:
        y_mask = (atom_data['y'] < domain_yhi + Rc) | \
                 (atom_data['y'] >= box_yhi - Rc)
    elif domain_ylo != box_ylo and domain_yhi == box_yhi:
        y_mask = (atom_data['y'] >= domain_ylo - Rc) | \
                 (atom_data['y'] < box_ylo + Rc)
    elif domain_ylo != box_ylo and domain_yhi != box_yhi:
        y_mask = (atom_data['y'] < domain_yhi + Rc) & \
                 (atom_data['y'] >= domain_ylo - Rc)

    if domain_zlo == box_zlo and domain_zhi != box_zhi:
        z_mask = (atom_data['z'] < domain_zhi + Rc) | \
                 (atom_data['z'] >= box_zhi - Rc)
    elif domain_zlo != box_zlo and domain_zhi == box_zhi:
        z_mask = (atom_data['z'] >= domain_zlo - Rc) | \
                 (atom_data['z'] < box_zlo + Rc)
    elif domain_zlo != box_zlo and domain_zhi != box_zhi:
        z_mask = (atom_data['z'] < domain_zhi + Rc) & \
                 (atom_data['z'] >= domain_zlo - Rc)

    main_filtered_data = atom_data[x_mask & y_mask & z_mask]

    return main_filtered_data, subdomain_bounds

def neigh_list(
        global_atoms: np.ndarray,
        box_dims: np.ndarray,
        Rc: float = 1.5,
        pbc: list[bool] = [True, True, True],
        comm: MPI.Intracomm = MPI.COMM_WORLD,
        convert_np: bool = True
        ) -> dict:
    """
    
    Parameters
    ----------
    atoms : np.ndarray
        Atoms data to use to construct neighbor list.
    box_dims : np.ndarray
        Dimensions of the entire simulation box.
    subdomain_dims : np.ndarray (3x2)
        Dimensions of the subdomain on the current process.
    Rc : float
        Cut-off radius for neighbor list construction. Default is 1.5
    convert_np : bool
        Choice to convert np.int and np.float data types to regular Python int and floats.
        Default is True.

    Returns
    -------
    dict
        
    """
    # Receive local atoms and subdomain bounds from supporting function
    atoms, subdomain_dims = _local_atoms_and_subdomain(global_atoms, box_dims, Rc, pbc, comm)

    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = subdomain_dims.flatten()
    subdomain_lengths = np.array([x_hi - x_lo, y_hi - y_lo, z_hi - z_lo])

    cell_size = Rc
    num_cells = np.ceil(subdomain_lengths / cell_size).astype(int)

    cell_dict = {}
    for data in atoms:
        x_idx = int((data['x'] - x_lo) / cell_size) % num_cells[0]
        y_idx = int((data['y'] - y_lo) / cell_size) % num_cells[1]
        z_idx = int((data['z'] - z_lo) / cell_size) % num_cells[2]
        cell_idx = (x_idx, y_idx, z_idx)
        cell_dict.setdefault(cell_idx, []).append(data['id'])

    # Prepare neighbor offsets (including the cell itself)
    neighbor_offsets = [-1, 0, 1]
    neighbor_cells = [
        (dx, dy, dz)
        for dx in neighbor_offsets
        for dy in neighbor_offsets
        for dz in neighbor_offsets
    ]

    neighbor_list = {}

    for cell_idx, atoms_ids in cell_dict.items():
        # Get neighboring cells
        neighbors = []
        for offset in neighbor_cells:
            neighbor_idx = (
                (cell_idx[0] + offset[0]) % num_cells[0],
                (cell_idx[1] + offset[1]) % num_cells[1],
                (cell_idx[2] + offset[2]) % num_cells[2],
            )
            neighbors.extend(cell_dict.get(neighbor_idx, []))

        for atom_id1 in atoms_ids:
            neighbor_info = {}

            atom_1_x = atoms[atoms['id'] == atom_id1]['x'].item()
            atom_1_y = atoms[atoms['id'] == atom_id1]['y'].item()
            atom_1_z = atoms[atoms['id'] == atom_id1]['z'].item()

            for atom_id2 in neighbors:
                if atom_id1 == atom_id2:
                    continue

                atom_2_x = atoms[atoms['id'] == atom_id2]['x'].item()
                atom_2_y = atoms[atoms['id'] == atom_id2]['y'].item()
                atom_2_z = atoms[atoms['id'] == atom_id2]['z'].item()

                dist = calculate_distance_pbc(box_dims,
                                              atom_1_x, atom_1_y, atom_1_z,
                                              atom_2_x, atom_2_y, atom_2_z)
                
                if convert_np:
                    atom_id1 = int(atom_id1)
                    atom_id2 = int(atom_id2)
                    dist = float(dist)

                if dist <= Rc:
                    neighbor_info[atom_id2] = dist

            if neighbor_info:
                neighbor_list[atom_id1] = neighbor_info

    gathered_neigh_lists: list[dict] = comm.gather(neighbor_list, root=0)

    mpi_rank = comm.Get_rank()
    if mpi_rank == 0:
        complete_neigh_list = {}
        for neigh_dict in gathered_neigh_lists:
            for key, inner_dict in neigh_dict.items():
                # Merge nested dictionaries if key exists
                if key in complete_neigh_list:
                    complete_neigh_list[key].update(inner_dict)
                else:
                    complete_neigh_list[key] = inner_dict  # Create new entry

        # Scatter Complete Neighbor list across all processes
        items = list(complete_neigh_list.items())
        num_chunks = comm.Get_size()
        chunk_size = math.ceil(len(items) / num_chunks)

        # Create chunks as sub-dictionaries
        scatter_list: list[dict] = [
            dict(items[i*chunk_size : (i+1)*chunk_size])
            for i in range(num_chunks)
        ]
    else:
        scatter_list = None

    local_neigh_list: dict = comm.scatter(scatter_list, root=0)     # Scattered local neighbor list

    return local_neigh_list