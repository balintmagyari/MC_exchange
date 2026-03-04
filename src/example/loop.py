import MC_exchange as mc
from mpi4py import MPI
from lammps import lammps

def loop(lmp: lammps, 
         n: int, 
         ts: float, 
         Nts: int, 
         P_coeff: float,
         alpha: float = 1.0,
         comm = MPI.COMM_WORLD, 
         sticker_atom_type: int | list[int] = 3,
         cross_link_bond_type: int = 2,
         print_progress: bool = False
         ) -> None:
    """
    Performs loop with bond exchange reactions.

    Parameters
    ----------
    lmp : lammps
        Instance of the LAMMPS Python class.
    n : int
        number of loops to run.
    ts : float
        LAMMPS simulation timesteps
    Nts : int
        Number of steps to run in each loop (i.e. # timesteps between bond exchange reactions). This variable is also known as tau_c.
    P_coeff : float
        Upper limit of the random number drawn between (0, P_coeff) during MC exchange. Max value of P_coeff should be 1.0.
    alpha : float
        Arbitrary parameter used to control the energy barrier, similar to the role of a catalyst during a chemical reaction.
    comm : Any 
        MPI communicator (default is MPI.COMM_WORLD).
    sticker_atom_type : int | list[int]
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
        # if mpi_rank == 0 and i%1 == 0:
        #     print(f'-------- Running loop {i+1} of {n} --------\n', flush=True)

        # -------------------- Data extraction from LAMMPS --------------------
        stickers = mc.gather_atoms(lmp, sticker_atom_type, comm)
        cross_links = mc.gather_bonds(lmp, cross_link_bond_type)
        box_dims = mc.get_box_dims(lmp)

        # if mpi_rank == 0:
        #     print(f'Number of cross-link bonds: {len(cross_links)}', flush=True)

        # -------------------- Neighbor list construction --------------------
        local_neigh_list = mc.neigh_list(stickers, box_dims, Rc=1.12, convert_np=True)

        # -------------------- Bond exchange --------------------
        temp = lmp.extract_compute('thermo_temp', 0, 0)
        assert isinstance(temp, float)
        complete_bonds_to_delete, complete_bonds_to_create = mc.evaluate_bond_exchange(local_neigh_list, cross_links, stickers, box_dims, 
                                                                                      alpha=alpha, P_coeff=P_coeff, T=temp, bond_shift=True, bond_swap=True)
        n_bonds_to_delete = len(complete_bonds_to_delete)
        n_bonds_to_create = len(complete_bonds_to_create)
        
        # if mpi_rank == 0 and i == 1:
        #     with open('bond_to_delete.json', 'w') as f:
        #         json.dump(complete_bonds_to_delete, f)
        #     with open('bond_to_create.json', 'w') as f:
        #         json.dump(complete_bonds_to_create, f)
        #     print(f'Number of bonds to delete: {n_bonds_to_delete}', flush=True)
        #     print(f'Number of bonds to create: {n_bonds_to_create}\n', flush=True)

        # -------------------- Bond deletion and creation --------------------
        ci = 0
        if n_bonds_to_delete == n_bonds_to_create and n_bonds_to_delete > 0:
            di = 0
            for atom1, atom2 in complete_bonds_to_delete.items():
                atom1 = int(atom1)
                atom2 = int(atom2)
                lmp.command(f'group bondpair id {atom1} {atom2}')
                lmp.command(f'delete_bonds bondpair bond {cross_link_bond_type} remove special')
                lmp.command('group bondpair delete')
                di += 1

            for atom1, atom2 in complete_bonds_to_create.items():
                atom1 = int(atom1)
                atom2 = int(atom2)
                ci += 1

                if ci < n_bonds_to_create:
                    lmp.command(f'create_bonds single/bond {cross_link_bond_type} {atom1} {atom2} special no')
                elif ci == n_bonds_to_create:
                    lmp.command(f'create_bonds single/bond {cross_link_bond_type} {atom1} {atom2} special yes')
                    # if mpi_rank == 0 and print_progress and ci == di:
                    #     print(f'Number of exchanged bonds: {ci}\n', flush=True)

        elif n_bonds_to_delete == n_bonds_to_create and n_bonds_to_delete == 0 and mpi_rank == 0 and print_progress:
            print('No exchange performed! Number of exchanged bonds: 0\n', flush=True)

            # print(f'Bonds to delete: \t {complete_bonds_to_delete}', flush=True)
            # print(f'Bonds to create: \t {complete_bonds_to_create}\n', flush=True)

        elif mpi_rank == 0 and print_progress:
            print(f'Number of bonds deleted {n_bonds_to_delete} is not equal to number of bonds created {n_bonds_to_create}! Therefore, no exchange performed at all!', flush=True)

            # if temporary_counter == 0:
            #     # Suppose your dictionary is called my_dict and its keys are strings representing integers
            #     complete_bonds_to_delete = {int(k): v for k, v in sorted(complete_bonds_to_delete.items(), key=lambda item: int(item[0]))}
            #     complete_bonds_to_create = {int(k): v for k, v in sorted(complete_bonds_to_create.items(), key=lambda item: int(item[0]))}
                
            #     with open("delete_bonds.json", 'w') as f:
            #         json.dump(complete_bonds_to_delete, f)
            #     with open("create_bonds.json", 'w') as f:
            #         json.dump(complete_bonds_to_create, f)

            #     temporary_counter += 1

        comm.Barrier()
        exchange_end = MPI.Wtime()
        # -------------------- LAMMPS MD Run --------------------
        lmp.command(f'timestep {ts}')
        lmp.command(f'run {Nts}')

        comm.Barrier()
        MD_end = MPI.Wtime()

        if mpi_rank == 0 and print_progress and i == 0:
            # print('loop,timestep,time,exchanged_bonds,exchange_time,md_time', flush=True)
            print('----- Statistics being printed -----', flush=True)

        if mpi_rank == 0 and print_progress:
            # print(f'{i+1},{(i+1)*Nts},{(i+1)*Nts*ts},{ci},{round(exchange_end-exchange_start, 4)},{round(MD_end-exchange_end, 4)}', flush=True)

            print(f'Loop: {i+1} ---> timestep={(i+1)*Nts} ---> time={(i+1)*Nts*ts}\n\
                Number of exchanged bonds: {ci}\n\
                Time taken to perform exchange: {round(exchange_end-exchange_start, 4)} \t Time for MD: {round(MD_end-exchange_end, 4)}\n', flush=True)


            # # print(f'Temperature: {round(temp, 3)}', flush=True)
            # print(f'Bond exchange time: {round(exchange_end-exchange_start, 3)} sec', flush=True)
            # print(f'Molecular dynamics time: {round(MD_end-exchange_end, 3)} sec\n', flush=True)

def main(temp: float,
         tauc: int,
         N_loops: int,
         N_atom_coords: int = 200,
         N_frames: int = 50,
         alpha: float = 1.0,
         sticker_atom_type: int | list[int] = 3,
         seed: int = 111
         ) -> None:
    """
    Main script to perform bond exchange dynamics of vitrimers.

    Parameters
    ----------
    temp : float
        Desired temperature to be set for MD simulations.
    tauc : int
        Number of MD timesteps between subsequent bond exchanges.
    N_loops : int
        Number of loops (bond exchange + MD steps) to perform.

        Total number of MD steps = tauc * N_loops

    N_atom_coords : int
        Number of unwrapped atom coordinates snapshots to take during the entire simulation. Default is 200.
    N_frames : int
        Number of snapshots to take during the entire simulation for Ovito visualization. Default is 50.
    seed : int
        Random seed to use with Langevin thermostat. Default is 111111.
    
    Returns
    -------
    None
    """
    comm = MPI.COMM_WORLD       # Specify MPI Communicator

    # ----- Initial setup containing LAMMPS code that is universal across different runs -----
    lmp = lammps(cmdargs=['-screen', 'none',
                          '-log', 'none',
                          '-var', 'TEMP', str(temp),
                          '-var', 'SEED', str(seed)],
                          comm=comm)                # Instantiate LAMMPS class
    
    # lmp.command(f'log logs/Teq_T={temp}_tc={tauc}_a={alpha}_s={seed}.lammps')
    lmp.file("Vitrimer_setup.lmp")                                # Read and run basic LAMMPS file with initial setup

    # ----- Temperature equilibration -----
    comm.Barrier()

    lmp.command('timestep 0.005')
    lmp.command('run 1000')
    lmp.command('reset_timestep 0')

    loop(lmp, 
         n=2, 
         ts=0.005, 
         Nts=tauc, 
         P_coeff=1, 
         alpha=alpha,
         comm=comm, 
         sticker_atom_type=sticker_atom_type,
         print_progress=False)               # Temp equilib. loop
    
    lmp.command(f'reset_timestep 0')
    comm.Barrier()
 
    # ----- Measurement run -----
    # lmp.command(f'log logs/Vitrimer_T={temp}_tc={tauc}_a={alpha}_s={seed}.lammps')
    lmp.command(f'thermo {tauc}')

    lmp.command(f'dump d_atomcoords all custom {int((N_loops*tauc) / N_atom_coords)} Atom_coords_T={temp}_tc={tauc}_a={alpha}_s={seed}.dump id mol xu yu zu')
    lmp.command(f'dump d_frames all atom {int((N_loops*tauc) / N_frames)} Frames_T={temp}_tc={tauc}_a={alpha}_s={seed}.dump')
    lmp.command('dump_modify d_atomcoords sort id')
    lmp.command('dump_modify d_frames sort id')

    loop(lmp,
         n=N_loops,
         ts=0.01,
         Nts=tauc,
         P_coeff=1,
         alpha=alpha,
         comm=comm,
         sticker_atom_type=sticker_atom_type,
         print_progress=True)               # Measurement loop
    comm.Barrier()

    lmp.command(f'write_data Vitrimer_T={temp}_tc={tauc}_a={alpha}_s={seed}.data')

    lmp.command(f'undump d_atomcoords')
    lmp.command(f'undump d_frames')
    lmp.command(f'reset_timestep 0')
    lmp.close()
    comm.Barrier()

    MPI.Finalize()

if __name__=="__main__":
    # parser = argparse.ArgumentParser(prog="Vitrimer Bond Exchange",
    #                                  description='Perform molecular dynamics simulations of vitrimers')
    
    # parser.add_argument('--temperature', type=float, required=True, help='Temperature value')
    # parser.add_argument('--tauc', type=int, required=True, help='Number of MD steps between bond exchanges')
    # parser.add_argument('--loops', type=int, required=True, help='Number of loops to perform')

    # parser.add_argument('--atomcoords', type=int, default=200, help='Number of atom coordinate dump files to save')
    # parser.add_argument('--frames', type=int, default=50, help='Number of frames to save for Ovito illustration')
    # parser.add_argument('--stress', type=int, default=10, help='Frequency of stress measurements expressed as number of timesteps')
    # parser.add_argument('--seed', type=int, default=111, help='Random seed')
    # parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter of MC exchange')

    # args = parser.parse_args()

    # temp = args.temperature
    # tauc = args.tauc
    # loops = args.loops
    # atomcoords = args.atomcoords
    # frames = args.frames
    # stress_freq = args.stress
    # alpha = args.alpha
    # seed = args.seed

    main(temp=1.0, 
         tauc=1000, 
         N_loops=2000, 
         N_atom_coords=10, 
         N_frames=10, 
         alpha=1,
         sticker_atom_type=[2,3,4],
         seed=111)
    
    # main(temp=temp, 
    #      tauc=tauc, 
    #      N_loops=loops, 
    #      N_atom_coords=atomcoords, 
    #      N_frames=frames, 
    #      stress_freq=stress_freq, 
    #      alpha=alpha,
    #      sticker_atom_type=[2,3,4],
    #      seed=seed)

