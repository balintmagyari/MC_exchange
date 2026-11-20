import numpy as np

from mpi4py import MPI
from scipy.spatial import cKDTree


def _check_box(box_dims: np.ndarray, Rc: float, pbc: list[bool]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the box and cut-off; return (box lower bounds, box lengths, periodic mask)."""
    box_dims = np.asarray(box_dims, dtype=np.float64).flatten()
    if box_dims.size != 6 or not np.all(np.isfinite(box_dims)):
        raise ValueError(f'Invalid box dimensions (need 6 finite values xlo, xhi, ylo, yhi, zlo, zhi): {box_dims}')
    lo = box_dims[::2]
    L = box_dims[1::2] - lo
    if np.any(L <= 0):
        raise ValueError(f'Invalid box dimensions (hi must be larger than lo): {box_dims}')

    periodic = np.asarray(pbc, dtype=bool)
    if periodic.shape != (3,):
        raise ValueError(f'pbc must have one entry per dimension (3), got {pbc!r}')

    if not (np.isfinite(Rc) and Rc > 0):
        raise ValueError(f'Rc must be a positive finite number, got {Rc!r}')
    if np.any(periodic & (Rc >= L / 2)):
        raise ValueError(f'Rc = {Rc} must be smaller than half the periodic box length (L = {L}).')
    return lo, L, periodic


def _find_pairs(atoms: np.ndarray,
                box_dims: np.ndarray,
                Rc: float,
                pbc: list[bool]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find all atom pairs within Rc (inclusive), using the minimum image convention
    in periodic dimensions.

    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        (pairs, distances, pos, span): pairs is an (M, 2) array of row indices into atoms (i < j),
        distances is an (M,) array of the corresponding minimum-image distances, computed from
        the raw coordinates exactly like calculations.calculate_distance_pbc, pos is the (N, 3)
        array of positions shifted into [0, L) in periodic dimensions and into [0, extent] in
        non-periodic ones, and span is the length of each dimension used for the process grid
        (box length if periodic, max(extent, Rc) otherwise).
    """
    lo, L, periodic = _check_box(box_dims, Rc, pbc)

    raw = np.column_stack([atoms['x'], atoms['y'], atoms['z']]).astype(np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError('Atom coordinates contain NaN or inf.')
    ids = np.asarray(atoms['id'])
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Atom ids are not unique.')

    pos = raw - lo
    boxsize = np.empty(3, dtype=np.float64)
    span = L.copy()

    for d in range(3):
        if periodic[d]:
            # Wrap into [0, L): LAMMPS coordinates can lie slightly outside the box
            pos[:, d] %= L[d]
            pos[pos[:, d] >= L[d], d] = 0.0            # float edge case: -1e-17 % L == L
            boxsize[d] = L[d]
        else:
            # Non-periodic: shift to >= 0 and use a box so large that no pair wraps around
            if len(pos):
                pos[:, d] -= pos[:, d].min()
                extent = pos[:, d].max()
            else:
                extent = 0.0
            boxsize[d] = 2.0 * (extent + Rc) + 1.0
            span[d] = max(extent, Rc)

    if len(pos) < 2:
        return np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=np.float64), pos, span

    # Search slightly beyond Rc so that round-off inside the tree cannot drop a pair at Rc;
    # the exact decision is made below with the same formula the exchange code uses.
    tree = cKDTree(pos, boxsize=boxsize)
    pairs = tree.query_pairs(Rc * (1.0 + 1e-9), output_type='ndarray')

    # Minimum-image distances from the raw coordinates, same formula as
    # calculations.calculate_distance_pbc, so both give bit-identical values
    delta = raw[pairs[:, 0]] - raw[pairs[:, 1]]
    delta[:, periodic] -= L[periodic] * np.round(delta[:, periodic] / L[periodic])
    distances = np.linalg.norm(delta, axis=1)

    keep = distances <= Rc
    return pairs[keep], distances[keep], pos, span


def _owner_ranks(pos: np.ndarray, span: np.ndarray, size: int) -> np.ndarray:
    """
    Rank that owns each position, from a regular 3D process grid (MPI.Compute_dims) over
    the box, numbered row-major like MPI.Intracomm.Create_cart. Keeping each rank's atoms
    spatially compact keeps cross-rank conflicts in the exchange step rare.
    """
    grid = np.array(MPI.Compute_dims(size, [0, 0, 0]), dtype=np.int64)
    coords = np.floor(pos / span * grid).astype(np.int64)
    coords = np.clip(coords, 0, grid - 1)
    return (coords[:, 0] * grid[1] + coords[:, 1]) * grid[2] + coords[:, 2]


def neigh_list(
        global_atoms: np.ndarray,
        box_dims: np.ndarray,
        Rc: float = 1.5,
        pbc: list[bool] = [True, True, True],
        comm: MPI.Intracomm = MPI.COMM_WORLD,
        convert_np: bool = True
        ) -> dict:
    """
    Construct the neighbor list of all atoms and distribute it across processes.

    The pair search is done on the root process with a k-d tree (scipy.spatial.cKDTree),
    using the minimum image convention in periodic dimensions. Coordinates are wrapped
    into the box first, so atoms slightly outside the box are handled correctly. The
    complete neighbor list is then split spatially: each atom goes to the process whose
    subdomain of a regular process grid (MPI.Compute_dims) contains it, so every atom
    appears on exactly one process and each process holds a compact region of the box.
    Within a process, atoms and their neighbors are sorted by id.

    This is a collective call: every process of comm must call it with the same box_dims,
    Rc and pbc. Only the root's global_atoms is used. If the root fails, every process
    raises a RuntimeError.

    Parameters
    ----------
    global_atoms : np.ndarray
        Structured array of all atoms (fields 'id', 'x', 'y', 'z'), identical on all processes
        (as returned by gather_atoms). Atom ids must be unique.
    box_dims : np.ndarray
        Dimensions of the entire simulation box: [xlo, xhi, ylo, yhi, zlo, zhi].
    Rc : float
        Cut-off radius for neighbor list construction (inclusive). Must be smaller than half
        the box length in periodic dimensions. Default is 1.5
    pbc : list[bool]
        Periodic boundary conditions in the x, y and z directions. Default is [True, True, True].
        Note that the exchange functions always use fully periodic distances.
    comm : MPI.Intracomm
        MPI communicator (default is MPI.COMM_WORLD).
    convert_np : bool
        Choice to convert np.int and np.float data types to regular Python int and floats.
        Default is True.

    Returns
    -------
    dict
        Local neighbor list {atom_id: {neighbor_id: distance}} of this process. Only atoms
        with at least one neighbor are included; the relation is symmetric over all processes.
        Distances are identical to calculations.calculate_distance_pbc.
    """
    _check_box(box_dims, Rc, pbc)          # On every rank, so a bad input cannot hang the scatter

    size = comm.Get_size()
    scatter_list: list | None = None
    error: str | None = None

    if comm.Get_rank() == 0:
        root_exc: Exception | None = None
        try:
            pairs, distances, pos, span = _find_pairs(global_atoms, box_dims, Rc, pbc)
            ids = np.asarray(global_atoms['id'])
            owner = _owner_ranks(pos, span, size)

            # Symmetric (both directions), sorted by owner rank, atom id and neighbor id
            row_a = np.concatenate([pairs[:, 0], pairs[:, 1]])
            row_b = np.concatenate([pairs[:, 1], pairs[:, 0]])
            dist = np.concatenate([distances, distances])
            id_a, id_b, rank_a = ids[row_a], ids[row_b], owner[row_a]
            order = np.lexsort((id_b, id_a, rank_a))
            id_a, id_b, rank_a, dist = id_a[order], id_b[order], rank_a[order], dist[order]

            if convert_np:
                id_a, id_b, dist = id_a.tolist(), id_b.tolist(), dist.tolist()
            rank_a = rank_a.tolist()

            scatter_list = [{} for _ in range(size)]
            for a, b, r, d in zip(id_a, id_b, rank_a, dist):
                scatter_list[r].setdefault(a, {})[b] = d
        except Exception as exc:          # Hand the error to every rank instead of hanging in scatter
            root_exc = exc
            error = f'{type(exc).__name__}: {exc}'     # A plain string always pickles
            scatter_list = [None] * size

    error = comm.bcast(error, root=0)
    local_neigh_list = comm.scatter(scatter_list, root=0)     # Scattered local neighbor list
    if error is not None:
        message = f'neigh_list failed on rank 0: {error}'
        if comm.Get_rank() == 0:
            raise RuntimeError(message) from root_exc
        raise RuntimeError(message)

    return local_neigh_list
