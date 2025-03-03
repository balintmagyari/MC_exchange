import lammps
from mpi4py import MPI
import MC_exchange as mc
import warnings

"""
This is an example file containing a function defining a loop that incorporates the Python package named MC_exchange.
"""

def loop(lmp: lammps, 
         n: int, 
         ts: float, 
         Nts: int, 
         P_coeff: float,
         comm = MPI.COMM_WORLD, 
         sticker_atom_type: int = 3,
         cross_link_bond_type: int = 2,
         print_progress: bool = False
         ) -> None:
    """
    Performs loop with bond exchange reactions.

    Parameters
    ----------
    lmp: lammps
        Instance of the LAMMPS Python class.
    n: int
        number of loops to run.
    ts: float
        LAMMPS simulation timesteps
    Nts: int
        Number of steps to run in each loop (i.e. # timesteps between bond exchange reactions). This variable is also known as tau_c.
    comm : Any 
        MPI communicator (default is MPI.COMM_WORLD).
    sticker_atom_type : int
        Numerical atom type of the stickers (vitrimeric beads) (default is 3).
    cross_link_bond_type : int
        Numerical bond type of the reversible cross-link bonds (default is 2).

    Returns
    -------
    None
    """
    mpi_rank = comm.Get_rank()

    for i in range(n):
        comm.Barrier()
        exchange_start = MPI.Wtime()
        if mpi_rank == 0 and i%1 == 0:
            print(f'----------------------------------\nRunning loop {i+1} of {n}\n', flush=True)

        # -------------------- Data extraction from LAMMPS --------------------
        stickers = mc.gather_atoms(lmp, sticker_atom_type, comm)
        cross_links = mc.gather_bonds(lmp, cross_link_bond_type)
        box_dims = mc.get_box_dims(lmp)

        # -------------------- Neighbor list construction --------------------
        nl_start_time = MPI.Wtime()
        local_neigh_list = mc.neigh_list(stickers, box_dims, Rc=1.12, convert_np=True)
        nl_end_time = MPI.Wtime()

        # if mpi_rank == 0 and print_progress and i%1 == 0:
        #     print(f'\nParallel Neighbor list construction time on {comm.Get_size()} procs: {round(nl_end_time-nl_start_time, 3)} sec\n', flush=True)

        # -------------------- Bond exchange --------------------
        
        complete_bonds_to_delete, complete_bonds_to_create = mc.perform_bond_exchange(local_neigh_list, cross_links, stickers, box_dims, P_coeff=P_coeff)

        # if mpi_rank == 0:
        #     with open(f'datas/b2_delete_{i}.json', 'w') as file:
        #         json.dump(complete_bonds_to_delete, file, indent=4, sort_keys=True)
        #     with open(f'datas/b2_create_{i}.json', 'w') as file:
        #         json.dump(complete_bonds_to_create, file, indent=4, sort_keys=True)

            # print(f'Bonds to delete: {len(complete_bonds_to_delete)}', flush=True)
            # print(f'Bonds to create: {len(complete_bonds_to_create)}\n', flush=True)

        # -------------------- Bond deletion and creation --------------------
        di = 0
        for atom1, atom2 in complete_bonds_to_delete.items():
            atom1 = int(atom1)
            atom2 = int(atom2)
            lmp.command(f'group bondpair id {atom1} {atom2}')
            lmp.command(f'delete_bonds bondpair bond {cross_link_bond_type} remove special')
            lmp.command('group bondpair delete')
            di += 1

        n_bonds_to_create = len(complete_bonds_to_create)
        ci = 0
        for atom1, atom2 in complete_bonds_to_create.items():
            atom1 = int(atom1)
            atom2 = int(atom2)
            ci += 1

            if ci < n_bonds_to_create:
                lmp.command(f'create_bonds single/bond {cross_link_bond_type} {atom1} {atom2} special no')
            elif ci == n_bonds_to_create:
                lmp.command(f'create_bonds single/bond {cross_link_bond_type} {atom1} {atom2} special yes')
                if mpi_rank == 0 and print_progress and ci == di:
                    print(f'Number of exchanged bonds: {ci}\n', flush=True)
                elif mpi_rank == 0 and print_progress and ci != di:
                    warnings.warn('Number of bonds deleted is not equal to number of bonds created!')

        comm.Barrier()
        exchange_end = MPI.Wtime()
        # -------------------- LAMMPS MD Run --------------------
        lmp.command(f'timestep {ts}')
        lmp.command(f'run {Nts}')

        comm.Barrier()
        MD_end = MPI.Wtime()

        if mpi_rank == 0:
            print(f'Bond exchange time: {round(exchange_end-exchange_start, 3)} sec', flush=True)
            print(f'Molecular dynamics time: {round(MD_end-exchange_end, 3)} sec\n', flush=True)