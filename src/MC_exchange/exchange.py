import numpy as np
import warnings
import random

from typing import Union

from mpi4py import MPI
from .calculations import calculate_distance_pbc, calculate_fene_potential, calculate_lj_potential


def perform_bond_exchange(sticker_neighbor_list: dict, 
                          bonds: np.ndarray,
                          atoms: np.ndarray,
                          box_dims: np.ndarray,
                          T: float,
                          cut_off: float | None = None,
                          P_coeff: float = 1.0,
                          kB: float = 1.0,
                          comm: MPI.Intracomm = MPI.COMM_WORLD,
                          return_stats: bool = False
                          ) -> Union[tuple[dict, dict], tuple[dict, dict, int, int, int, int]]:
    """
    Evaluate bond exchange dynamics on the local process, gather combined data on which
    bonds to delete and create, and broadcast that combined data to all processes.

    If return_stats = True, statistics about the number of exchanges is returned.

    Parameters
    ---------
    sticker_neighbor_list : dict
        Dictionary containing the local neighbor list on the current process.
    bonds : np.ndarray
        Bond data organized as a structured numpy array with columns: ['type', 'atom 1', 'atom 2'].
    atoms : np.ndarray
        Atom data organized as a structured numpy array with columns: ['id', 'type', 'mol', 'x', 'y', 'z'].
    box_dims : np.ndarray
        Box dimensions in the form: [xlo, xhi, ylo, yhi, zlo, zhi].
    T : float
        Current temperature of the simulation.
    cut_off : float | None
        Cut-off distance for bond exchange dynamics to be considered. If None, the cut-off distance from the
        neighbor list is used. Default is None.
    P_coeff : float
        Defines the maximum of the range between 0 and P_coeff, from which a random number is drawn during the
        Monte Carlo exchange. Default is 1.0.
    kB : float
        Boltzmann's constant. Default is 1.0.
    comm : MPI.Intracomm 
        MPI communicator. Default is MPI.COMM_WORLD.
    return_stats : bool
        Whether to return bond exchange statistics (number of exchanges, etc.). Default is False.

    Returns
    -------
    tuple[dict, dict] | tuple[dict, dict, int, int, int, int]

    Depending on the value of return_stats. If return_stats = False, bonds_to_delete and bonds_to_create dictionaries are returned. 
    If return_stats = True, bonds_to_delete, bonds_to_create, total_N_possible, total_N_exchanges, total_N_deltaU_exchanges, total_N_MC_exchanges are returned.

    total_N_possible : Total number of possible bond exchanges based on distance criterion.
    total_N_exchanges : Total number of actual bond exchanges to be perfomed. total_N_exchanges = total_N_deltaU_exchanges + total_N_MC_exchanges.
    total_N_deltaU_exchanges : Total number of bond exchanges performed due to a reduces potential.
    total_N_MC_exchanges : Total number of bond exchanges performed through Monte Carlo evaluation.
    """
    
    # Counters for statistics
    N_possible = 0              # Possible total number of exchanges given solely the distance criterion.
    N_exchanges = 0             # Number of actualy exchanges happening. N_exchanges = N_deltaU_exchanges + N_MC_exchanges
    N_deltaU_exchanges = 0      # Number of exchanges due to reduction in interaction potential.
    N_MC_exchanges = 0          # Number of exchanges due to random probability acceptance.

    already_exchanged_atoms = []    # List used to prevent the same pair of atoms taking part in bond exchange more than once
    bonds_to_delete = {}            # Final dictionary containing atom1 and atom2 as key: value pairs between which bonds should be created after bond exchange
    bonds_to_create = {}            # Final dictionary containing atom1 and atom2 as key: value pairs between which bonds should be created after bond exchange

    # Iterate over the neighbor list
    for atom_main, neighbors_data in sticker_neighbor_list.items():
        # atom_main -> atom_id, neighbors_data -> dictionary of atom_id: distance (from atom_main)

        # Skip to next iteration if there aren't enough stickers in the vicinity.
        # It is assumed that for a successful bond exchange, two pairs of bonds (i.e. 4 stickers) need to be within cut-off distance from each other.
        if len(neighbors_data) < 3:
            continue

        if atom_main in already_exchanged_atoms:
            continue
        
        for neighbor_id, distance in neighbors_data.items():
            if atom_main >= neighbor_id:
                continue    # Avoid double counting

            if cut_off is not None and distance > cut_off:
                continue    # Continue if atom is outside cut-off distance from atom_main

            if neighbor_id in already_exchanged_atoms:
                continue

            # Find cross-link bond of atom_main and the sticker to which it is cross-linked to.
            bond_1 = None
            atom_c = None

            for bond in bonds:
                # bond_type = bond['type']
                bond_atom_1 = bond['atom 1']
                bond_atom_2 = bond['atom 2']

                if bond_atom_1 == atom_main and bond_atom_2 != neighbor_id:
                    atom_c = bond_atom_2
                    break
                elif bond_atom_2 == atom_main and bond_atom_1 != neighbor_id:
                    atom_c = bond_atom_1
                    break

            # Skip if no suitable bond is found for atom_main
            if atom_c is None:
                continue

            if atom_c in already_exchanged_atoms:
                continue

            # Check if atom_c is in the neighbor list of atom_main and if it is within cut-off distance.
            # Should almost always be part of the neighbor list since atom_main is bonded to atom_c.
            if atom_c not in neighbors_data or (cut_off is not None and neighbors_data[atom_c] > cut_off):
                continue
            
            # Find cross-link bond of neighbor_id and the sticker to which it is cross-linked to.
            atom_d = None

            for bond in bonds:
                # bond_type = bond['type']
                bond_atom_1 = bond['atom 1']
                bond_atom_2 = bond['atom 2']

                if bond_atom_1 == neighbor_id and bond_atom_2 != atom_main:
                    atom_d = bond_atom_2
                    break
                elif bond_atom_2 == neighbor_id and bond_atom_1 != atom_main:
                    atom_d = bond_atom_1
                    break

            # Skip if no suitable bond is found for neighbor_id
            if atom_d is None:
                continue

            if atom_d in already_exchanged_atoms:
                continue

            # Check if atom_d is in the neighbor list of atom_main and if it is within cut-off distance.
            # If atom_d is not in the neighbor list of the main atom cross-linking is not possible.
            if atom_d not in neighbors_data or (cut_off is not None and neighbors_data[atom_d] > cut_off):
                continue

            if atom_c == atom_d:
                continue
                # warnings.warn('Atom C and Atom D are identical!!!')

            # -------------------- Exchange is considered from this point onwards --------------------

            neighbor_id = int(neighbor_id)
            atom_c = int(atom_c)
            atom_d = int(atom_d)

            N_possible += 1     # If all above if statements are false, based on the distance criterion, exchange can occur.

            # Gather single atom data and collect coordinates of neighbor atom, atom c, and atom d.
            atom_neighbor_data = atoms[atoms['id'] == neighbor_id]
            atom_neighbor_x = atom_neighbor_data['x']; atom_neighbor_y = atom_neighbor_data['y']; atom_neighbor_z = atom_neighbor_data['z']

            atom_c_data = atoms[atoms['id'] == atom_c]
            atom_c_x = atom_c_data['x']; atom_c_y = atom_c_data['y']; atom_c_z = atom_c_data['z']

            atom_d_data = atoms[atoms['id'] == atom_d]
            atom_d_x = atom_d_data['x']; atom_d_y = atom_d_data['y']; atom_d_z = atom_d_data['z']


            # Construct dictionary of distances between each pair of atoms of the 4 stickers
            distances = {
                f'{atom_main}-{neighbor_id}': neighbors_data[neighbor_id], 
                f'{atom_main}-{atom_c}': neighbors_data[atom_c], 
                f'{atom_main}-{atom_d}': neighbors_data[atom_d], 
                f'{neighbor_id}-{atom_c}': calculate_distance_pbc(box_dims, atom_neighbor_x, atom_neighbor_y, atom_neighbor_z,
                                                                  atom_c_x, atom_c_y, atom_c_z),
                f'{neighbor_id}-{atom_d}': calculate_distance_pbc(box_dims, atom_neighbor_x, atom_neighbor_y, atom_neighbor_z,
                                                                  atom_d_x, atom_d_y, atom_d_z),
                f'{atom_c}-{atom_d}': calculate_distance_pbc(box_dims, atom_c_x, atom_c_y, atom_c_z,
                                                                  atom_d_x, atom_d_y, atom_d_z)
            }

            # with open("datas/distances.json", 'w') as file:
            #     json.dump(distances, file)

            # Calculate the FENE potential between each pair of atoms based on the distance between those atoms.
            fene_potentials = {
                f'{atom_main}-{neighbor_id}': calculate_fene_potential(distances[f'{atom_main}-{neighbor_id}']), 
                f'{atom_main}-{atom_c}': calculate_fene_potential(distances[f'{atom_main}-{atom_c}']), 
                f'{atom_main}-{atom_d}': calculate_fene_potential(distances[f'{atom_main}-{atom_d}']), 
                f'{neighbor_id}-{atom_c}': calculate_fene_potential(distances[f'{neighbor_id}-{atom_c}']),
                f'{neighbor_id}-{atom_d}': calculate_fene_potential(distances[f'{neighbor_id}-{atom_d}']),
                f'{atom_c}-{atom_d}': calculate_fene_potential(distances[f'{atom_c}-{atom_d}'])
            }

            # Calculate the LJ non-bonding potential between each pair of atoms based on the distance between those atoms.
            lj_potentials = {
                f'{atom_main}-{neighbor_id}': calculate_lj_potential(distances[f'{atom_main}-{neighbor_id}']), 
                f'{atom_main}-{atom_c}': calculate_lj_potential(distances[f'{atom_main}-{atom_c}']), 
                f'{atom_main}-{atom_d}': calculate_lj_potential(distances[f'{atom_main}-{atom_d}']), 
                f'{neighbor_id}-{atom_c}': calculate_lj_potential(distances[f'{neighbor_id}-{atom_c}']),
                f'{neighbor_id}-{atom_d}': calculate_lj_potential(distances[f'{neighbor_id}-{atom_d}']),
                f'{atom_c}-{atom_d}': calculate_lj_potential(distances[f'{atom_c}-{atom_d}'])
            }

            # with open("datas/FENE_potentials.json", 'w') as file:
            #     json.dump(fene_potentials, file)

            # with open("datas/LJ_potentials.json", 'w') as file:
            #     json.dump(lj_potentials, file)

            # Potential energy of old configuration
            U_old = fene_potentials[f'{atom_main}-{atom_c}'] + fene_potentials[f'{neighbor_id}-{atom_d}'] + \
                    lj_potentials[f'{atom_main}-{neighbor_id}'] + lj_potentials[f'{atom_main}-{atom_d}'] + \
                    lj_potentials[f'{neighbor_id}-{atom_c}'] + lj_potentials[f'{atom_c}-{atom_d}']

            # Potential energy of the first new configuration
            U_new_1 = fene_potentials[f'{atom_main}-{neighbor_id}'] + fene_potentials[f'{atom_c}-{atom_d}'] + \
                      lj_potentials[f'{atom_main}-{atom_c}'] + lj_potentials[f'{atom_main}-{atom_d}'] + \
                      lj_potentials[f'{neighbor_id}-{atom_c}'] + lj_potentials[f'{neighbor_id}-{atom_d}']

            # Potential energy of the second new configuration
            U_new_2 = fene_potentials[f'{atom_main}-{atom_d}'] + fene_potentials[f'{neighbor_id}-{atom_c}'] + \
                      lj_potentials[f'{atom_main}-{neighbor_id}'] + lj_potentials[f'{atom_main}-{atom_c}'] + \
                      lj_potentials[f'{neighbor_id}-{atom_d}'] + lj_potentials[f'{atom_c}-{atom_d}']
            
            # Defining the atoms of the possible new bonds
            if U_new_1 <= U_new_2:
                new_bond1_atom_1 = atom_main
                new_bond1_atom_2 = neighbor_id
                new_bond2_atom_1 = atom_c
                new_bond2_atom_2 = atom_d
            else:
                new_bond1_atom_1 = atom_main
                new_bond1_atom_2 = atom_d
                new_bond2_atom_1 = neighbor_id
                new_bond2_atom_2 = atom_c

            U_new = min(U_new_1, U_new_2)

            delta_U = U_new - U_old

            # Acceptance probability. If change in potential is negative, the acceptance probability automatically becomes 1.0.
            if T != 0:
                P_accept = np.exp(-delta_U/(kB * T)) if delta_U > 0 else 1.0
            else:
                P_accept = 0

            bond_exchange = False
            if P_accept == 1:
                N_deltaU_exchanges += 1
                bond_exchange = True
                # print('Bond exchange happens naturally due to a negative delta U.')

            else:
                ran = random.uniform(0, P_coeff)
                if P_accept >= ran:
                    N_MC_exchanges += 1
                    bond_exchange = True
                    # print('Bond exchange happens due to Metropolis acceptance criterion.')

            # Swapping bonds if bond exchange was deemed plausible
            if bond_exchange:
                bonds_to_delete[min(atom_main, atom_c)] = max(atom_main, atom_c)
                bonds_to_delete[min(neighbor_id, atom_d)] = max(neighbor_id, atom_d)

                bonds_to_create[min(new_bond1_atom_1, new_bond1_atom_2)] = max(new_bond1_atom_1, new_bond1_atom_2)
                bonds_to_create[min(new_bond2_atom_1, new_bond2_atom_2)] = max(new_bond2_atom_1, new_bond2_atom_2)

                already_exchanged_atoms.append(atom_main)
                already_exchanged_atoms.append(neighbor_id)
                already_exchanged_atoms.append(atom_c)
                already_exchanged_atoms.append(atom_d)

                N_exchanges += 1        

    # print(f"\nTotal number of POSSIBLE bond exchanges given distance criterion: {N_possible}", flush=True)
    # print(f"Total number of ACTUAL bond exchanges performed: {N_exchanges}", flush=True)
    # print(f"Number of bond exchanges due to reduced potential: {N_deltaU_exchanges}", flush=True)
    # print(f"Number of bond exchanges due to random acceptance: {N_MC_exchanges}\n", flush=True)

    # -------------------- Gathering data from each process, combining them into one complete set of data and broadcasting it back to all processes--------------------

    gathered_bonds_to_delete = comm.gather(bonds_to_delete, root=0)
    gathered_bonds_to_create = comm.gather(bonds_to_create, root=0)

    # Summing bond exchange statistics on root process
    if return_stats:
        total_N_possible = comm.reduce(N_possible, op=MPI.SUM, root=0)
        total_N_exchanges = comm.reduce(N_exchanges, op=MPI.SUM, root=0)
        total_N_deltaU_exchanges = comm.reduce(N_deltaU_exchanges, op=MPI.SUM, root=0)
        total_N_MC_exchanges = comm.reduce(N_MC_exchanges, op=MPI.SUM, root=0)
    
    mpi_rank = comm.Get_rank()
    if mpi_rank == 0:
        complete_bonds_to_delete = {}
        complete_bonds_to_create = {}

        for d in gathered_bonds_to_delete:
            complete_bonds_to_delete.update(d)

        for d in gathered_bonds_to_create:
            complete_bonds_to_create.update(d)
    else:
        complete_bonds_to_delete = None
        complete_bonds_to_create = None

        if return_stats:
            total_N_possible = None
            total_N_exchanges = None
            total_N_deltaU_exchanges = None
            total_N_MC_exchanges = None

    # Broadcasting bonds to delete/create dictionaries
    complete_bonds_to_delete = comm.bcast(complete_bonds_to_delete, root=0)
    complete_bonds_to_create = comm.bcast(complete_bonds_to_create, root=0)

    # Broadcasting bond exchange statistics
    if return_stats:
        total_N_possible = comm.bcast(total_N_possible, root=0)
        total_N_exchanges = comm.bcast(total_N_exchanges, root=0)
        total_N_deltaU_exchanges = comm.bcast(total_N_deltaU_exchanges, root=0)
        total_N_MC_exchanges = comm.bcast(total_N_MC_exchanges, root=0)

    if return_stats:
        return complete_bonds_to_delete, complete_bonds_to_create, total_N_possible, total_N_exchanges, total_N_deltaU_exchanges, total_N_MC_exchanges
    else:
        return complete_bonds_to_delete, complete_bonds_to_create