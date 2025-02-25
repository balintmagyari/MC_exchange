import numpy as np

from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype

from lammps import lammps
from lammps.numpy_wrapper import numpy_wrapper

# -------------------- Functions --------------------

def get_box_dims(lmp) -> np.ndarray:
    """
    Extracts the dimensions of the simulation box.

    Paramaters
    ----------
    lmp: Any
        Instance of the LAMMPS Python class.

    Returns
    -------
    np.array
        Box dimensions in the form: [xlo, xhi, ylo, yhi, zlo, zhi]
    """
    box_dims = lmp.extract_box()

    boxlo: list = box_dims[0]         # Lower bounds of the simulation box (x, y and z directions)
    boxhi: list = box_dims[1]         # Upper bounds of the simulation box (x, y and z directions)

    # xy_tilt: int = box_dims[2]       # The xy tilt factor for triclinic boxes. For orthogonal boxes this is 0.0
    # yz_tilt: int = box_dims[3]       # The yz tilt factor for triclinic boxes. For orthogonal boxes this is 0.0
    # xz_tilt: int = box_dims[4]       # The xz tilt factor for triclinic boxes. For orthogonal boxes this is 0.0

    # periodicity: list = box_dims[5]   # Indicates periodicity in each dimension (x, y and z)
    # box_change: int = box_dims[6]     # Flag indicating whether the box has changed during simulation (0: unchanged, 1: changed)

    box_dims = np.array([boxlo[0], boxhi[0], boxlo[1], boxhi[1], boxlo[2], boxhi[2]])

    return box_dims

def gather_bonds(lmp,
                 bond_type: int | list[int] | None = None,
                 comm = MPI.COMM_WORLD,
                 root: int = 0) -> np.ndarray:
    """
    Gathers bond data from the LAMMPS simulation.
     
    Parameters
    ----------
    lmp : Any
        Instance of the LAMMPS Python class.
    bond_type : int | list[int] | None
        Bond type(s) to filter for. If None, no filter is applied (default is None).
    comm : Any 
        MPI communicator (default is MPI.COMM_WORLD).
    root : int
        Processor number of the root process (default is 0).

    Returns
    -------
    np.ndarray
        Bond data.
    """
    rank = comm.Get_rank()

    bonds = numpy_wrapper(lmp).gather_bonds()
    types = bonds[:,0]
    atom1s = bonds[:,1]
    atom2s = bonds[:,2]

    bond_datatype = np.dtype([
        ('type', 'int32'),
        ('atom 1', 'int32'),
        ('atom 2', 'int32')
    ])

    bond_data = np.zeros(len(bonds), dtype=bond_datatype)

    bond_data['type'] = types
    bond_data['atom 1'] = atom1s
    bond_data['atom 2'] = atom2s

    if rank == root:
        if isinstance(bond_type, int):
            selected_bond_data: np.ndarray = bond_data[bond_data['type'] == bond_type]
        elif isinstance(bond_type, list):
            selected_bond_data: np.ndarray = bond_data[np.isin(bond_data['type'], bond_data)]
        else:
            selected_bond_data = bond_data.copy()

        selected_desc = selected_bond_data.dtype.descr
        selected_datashape = selected_bond_data.shape
    
    else:
        selected_desc = None
        selected_datashape = None

    # -------------------- Broadcasting of selected bond data --------------------

    # 2. Broadcast Dtype Description
    selected_desc = comm.bcast(selected_desc, root=root)
    reconstructed_dtype = np.dtype(selected_desc)                 # All ranks rebuild dtype

    mpi_dtype = from_numpy_dtype(reconstructed_dtype)
    mpi_dtype.Commit()  # Required for communication

    # 3. Broadcast Array Shape
    selected_datashape = comm.bcast(selected_datashape, root=0)

    # 4. Prepare Buffers for Bcast
    if rank == root:
        # Force contiguous memory layout
        selected_bond_data = np.ascontiguousarray(selected_bond_data)
    else:
        # Allocate empty array with reconstructed dtype/shape
        selected_bond_data = np.empty(selected_datashape, dtype=reconstructed_dtype)

    # 5. Broadcast the Data
    comm.Bcast([selected_bond_data, mpi_dtype], root=0)
    mpi_dtype.Free()  # Critical to prevent leaks

    return selected_bond_data

def gather_atoms(lmp, 
                 atom_type: int | list[int] | None = None,
                 comm = MPI.COMM_WORLD,
                 root: int = 0) -> np.ndarray | None:
    """
    Extracts atom information from each processor, combines the data into one structured numpy array and 
    broadcasts this array to all processes.

    If specified, using the atom_type parameter, the data can be filtered to only include specified atom 
    types. The data filtration takes place solely on the root process, before the filtered data is 
    broadcasted to all other processes. This is done in order to avoid computational error on separate 
    processes.
    
    The following data is collected for each atom, organized in a Numpy structured array:
        - atom id
        - atom type
        - molecule id
        - x-, y-, and z-coordinate inside the simulation box

    Parameters
    ----------
    lmp: Any
        Instance of the LAMMPS Python class.
    atom_type : int | list[int] | None
        Atom type(s) to filter for. If None, no filter is applied (default is None).
    comm : Any 
        MPI communicator (default is MPI.COMM_WORLD).
    root : int
        Processor number of the root process (default is 0).

    Returns
    -------
    np.array
        Returns the requested atom data of **all** atoms in the system, on all processes.
    """
    rank = comm.Get_rank()
    
    # Get local atom count (excluding ghosts)
    nlocal = lmp.extract_global("nlocal")

    # Extract local data
    local_ids = lmp.numpy.extract_atom("id")[:nlocal].astype(np.int64)
    local_types = lmp.numpy.extract_atom("type")[:nlocal].astype(np.int32)
    local_mols = lmp.numpy.extract_atom("molecule")[:nlocal].astype(np.int32)
    local_x = lmp.numpy.extract_atom("x")[:nlocal].copy()

    # Gather data sizes
    send_counts = np.array(comm.gather(nlocal, root=root))

    if rank == root:
        displacements = np.insert(np.cumsum(send_counts), 0, 0)[:-1]
        total_atoms = np.sum(send_counts)
        
        # Create receive buffers
        global_ids = np.empty(total_atoms, dtype=np.int64)
        global_types = np.empty(total_atoms, dtype=np.int32)
        global_mols = np.empty(total_atoms, dtype=np.int32)
        global_x = np.empty(total_atoms*3, dtype=np.float64)
    else:
        global_ids = global_types = global_mols = global_x = None
        displacements = None

    # Gather atom IDs
    comm.Gatherv(
        sendbuf=local_ids,
        recvbuf=(global_ids, send_counts, displacements, MPI.INT64_T),
        root=root
    )

    # Gather atom types
    comm.Gatherv(
        sendbuf=local_types,
        recvbuf=(global_types, send_counts, displacements, MPI.INT),
        root=root
    )

    # Gather molecules
    comm.Gatherv(
        sendbuf=local_mols,
        recvbuf=(global_mols, send_counts, displacements, MPI.INT),
        root=root
    )

    # Gather coordinates (flattened array)
    comm.Gatherv(
        sendbuf=local_x.ravel(),
        recvbuf=(global_x, 3*send_counts if rank == 0 else None, 3*displacements if rank == 0 else None, MPI.DOUBLE),
        root=root
    )

    if rank == root:
        # Sort all data by atom ID
        sort_idx = np.argsort(global_ids)
        
        # Reshape coordinates
        sorted_x = global_x.reshape(-1,3)[sort_idx]
        
        # Create structured array
        atom_datatypes = [('id', 'i4'), ('type', 'i4'), ('mol', 'i4'),
                          ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        
        atom_data = np.zeros(len(global_ids), dtype=atom_datatypes)
        
        atom_data['id'] = global_ids[sort_idx]
        atom_data['type'] = global_types[sort_idx]
        atom_data['mol'] = global_mols[sort_idx]
        atom_data['x'] = sorted_x[:,0]
        atom_data['y'] = sorted_x[:,1]
        atom_data['z'] = sorted_x[:,2]

        if isinstance(atom_type, int):
            selected_atom_data: np.ndarray = atom_data[atom_data['type'] == atom_type]
        elif isinstance(atom_type, list):
            selected_atom_data: np.ndarray = atom_data[np.isin(atom_data['type'], atom_type)]
        else:
            selected_atom_data = atom_data

        selected_desc = selected_atom_data.dtype.descr                         # Serialize the dtype for broadcasting
        selected_datashape = selected_atom_data.shape                           # Shape of the data

    else:
        selected_desc = None
        selected_datashape = None 

    # -------------------- Broadcasting of selected atom data --------------------

    # 2. Broadcast Dtype Description
    selected_desc = comm.bcast(selected_desc, root=root)
    reconstructed_dtype = np.dtype(selected_desc)                 # All ranks rebuild dtype

    mpi_dtype = from_numpy_dtype(reconstructed_dtype)
    mpi_dtype.Commit()  # Required for communication

    # 3. Broadcast Array Shape
    selected_datashape = comm.bcast(selected_datashape, root=root)

    # 4. Prepare Buffers for Bcast
    if rank == root:
        # Force contiguous memory layout
        selected_atom_data = np.ascontiguousarray(selected_atom_data)
    else:
        # Allocate empty array with reconstructed dtype/shape
        selected_atom_data = np.empty(selected_datashape, dtype=reconstructed_dtype)

    # 5. Broadcast the Data
    comm.Bcast([selected_atom_data, mpi_dtype], root=0)
    mpi_dtype.Free()  # Critical to prevent leaks

    return selected_atom_data