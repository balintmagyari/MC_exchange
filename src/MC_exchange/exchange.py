import numpy as np
import warnings
import random

from itertools import combinations

from typing import Union

from mpi4py import MPI
from .calculations import calculate_distance_pbc, calculate_fene_potential, calculate_lj_potential, calculate_raw_fene_potential


def perform_bond_exchange(sticker_neighbor_list: dict, 
                          bonds: np.ndarray,
                          atoms: np.ndarray,
                          box_dims: np.ndarray,
                          T: float,
                          cut_off: float | None = None,
                          alpha: float = 1.0,
                          P_coeff: float = 1.0,
                          kB: float = 1.0,
                          comm: MPI.Intracomm = MPI.COMM_WORLD,
                          return_stats: bool = False
                          ) -> tuple:
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
    alpha : float
        Arbitrary parameter used to control the energy barrier, similar to the role of a catalyst during a chemical reaction.
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

            delta_U = alpha * (U_new - U_old)               # Change in potential, adjusted by alpha

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
        assert gathered_bonds_to_delete is not None # For type checker
        assert gathered_bonds_to_create is not None # For type checker
        complete_bonds_to_delete = {}
        complete_bonds_to_create = {}

        for d in gathered_bonds_to_delete:
            complete_bonds_to_delete.update(d)

        for d in gathered_bonds_to_create:
            complete_bonds_to_create.update(d)
    else:
        complete_bonds_to_delete = {}
        complete_bonds_to_create = {}

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
    
def three_four_atom_bond_exchange(sticker_neighbor_list: dict, 
                          bonds: np.ndarray,
                          atoms: np.ndarray,
                          box_dims: np.ndarray,
                          T: float,
                          alpha: float = 1.0,
                          P_coeff: float = 1.0,
                          kB: float = 1.0,
                          stopper_exchange: bool = True,
                          double_bonded_exchange: bool = True,
                          comm: MPI.Intracomm = MPI.COMM_WORLD
                          ) -> tuple:
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
    alpha : float
        Arbitrary parameter used to control the energy barrier, similar to the role of a catalyst during a chemical reaction.
    P_coeff : float
        Defines the maximum of the range between 0 and P_coeff, from which a random number is drawn during the
        Monte Carlo exchange. Default is 1.0.
    kB : float
        Boltzmann's constant. Default is 1.0.
    comm : MPI.Intracomm 
        MPI communicator. Default is MPI.COMM_WORLD.

    Returns
    -------
    tuple[dict, dict] | tuple[dict, dict, int, int, int, int]

    bonds_to_delete and bonds_to_create dictionaries are returned. 
    """

    already_exchanged_atoms = set()    # Set used to prevent the same pair of atoms taking part in bond exchange more than once
    bonds_to_delete = {}               # Final dictionary containing atom1 and atom2 as key: value pairs between which bonds should be created after bond exchange
    bonds_to_create = {}               # Final dictionary containing atom1 and atom2 as key: value pairs between which bonds should be created after bond exchange

    # Loop through entries in neighbor list
    for atom_main, neighbors_data in sticker_neighbor_list.items():
        sticker_ids = []                # Combined list of sticker IDs, will include atom_main
        stopper_bond_exchange = False   # Boolean whether to perform BER including 1 pair of bonded stickers and a nearby 'free' sticker
        paired_bond_exchange = False    # Boolean whether to perform BER between two pairs of bonded sticker

        if atom_main in already_exchanged_atoms:    # Skip to next iteration of atom_main has already been exchanged
            continue

        sticker_ids.append(atom_main)
        for neighbor_id in neighbors_data.keys():
            if neighbor_id in already_exchanged_atoms or neighbor_id < atom_main:      # Skip to next iteration of neighbor_id has already been exchanged or is lower in value than atom_main to prevent double counting
                continue
            # if neighbor_id < atom_main:      # Avoid double counting
            #     continue
            sticker_ids.append(neighbor_id)

        n_stickers = len(sticker_ids)   # Number of stickers that can potentially be exchanged

        # Continue with next iteration if the total number of stickers (including atom_main) is not enough for BER.
        if n_stickers < 3:
            continue

        linked_pairs = []   # List holding data on which atoms are bonded from the sticker_ids list
        for id1, id2 in combinations(sticker_ids, 2):
            # TODO: check this statement. Something unexpected happens where 
            if np.any(
                ((bonds['atom 1'] == id1) & (bonds['atom 2'] == id2)) |
                ((bonds['atom 1'] == id2) & (bonds['atom 2'] == id1))
            ):
                linked_pairs.append((id1, id2))
                # print(f'Linked pair appended: ({id1, id2})', flush=True)

        n_pairs = len(linked_pairs)

        if n_pairs == 0:      # Continue with next iteration of main loop if no pair is found
            continue

        if n_pairs == 1:      # Evaluate 3 sticker bond exchange if only 1 pair of sticker is linked
            stopper_bond_exchange = True

        if n_pairs == 2:       # Evaluate 4 sticker BER if two pairs of sticker is linked
            id1, id2 = linked_pairs[0]
            id3, id4 = linked_pairs[1]

            # Test to see if the four atom IDs are unique
            if len({id1, id2, id3, id4}) == 4:
                paired_bond_exchange = True
            else:
                pass
                # print(f'\nWrongly made linked_pairs: {linked_pairs}', flush=True)
                # print(f'Sticker IDs: {sticker_ids}\n', flush=True)

        if stopper_bond_exchange and stopper_exchange:
            id1, id2 = linked_pairs[0]

            # Saving coordinates of id1 and id2 atoms for later use
            id1_data = atoms[atoms['id'] == id1]
            id1_x = id1_data['x']; id1_y = id1_data['y']; id1_z = id1_data['z']
            id2_data = atoms[atoms['id'] == id2]
            id2_x = id2_data['x']; id2_y = id2_data['y']; id2_z = id2_data['z']

            # Remove bonded sticker ids from sticker_ids list
            sticker_ids.remove(id1); sticker_ids.remove(id2)

            distances = {}
            distances[f'{id1}-{id2}'] = calculate_distance_pbc(box_dims, id1_x, id1_y, id1_z, id2_x, id2_y, id2_z)
            for free_sticker in sticker_ids:

                # TODO: write an if statement here that checks whether the free_sticker is bonded to another sticker that may be outside the considered neighbor list (i.e. outside the sphere with radius Rc)
                if np.any((bonds['atom 1'] == free_sticker) | (bonds['atom 2'] == free_sticker)):
                    sticker_ids.remove(free_sticker)
                    # print(f'Removed sticker \t {free_sticker}', flush=True)
                    continue

                # if np.any((bonds['atom 1'] == free_sticker) | (bonds['atom 2'] == free_sticker)):
                #     warnings.warn('Supposed free sticker is also bonded to another sticker!!!')

                free_sticker_data = atoms[atoms['id'] == free_sticker]
                free_sticker_x = free_sticker_data['x']; free_sticker_y = free_sticker_data['y']; free_sticker_z = free_sticker_data['z']

                distances[f'{id1}-{free_sticker}'] = calculate_distance_pbc(box_dims, id1_x, id1_y, id1_z, free_sticker_x, free_sticker_y, free_sticker_z)
                distances[f'{id2}-{free_sticker}'] = calculate_distance_pbc(box_dims, id2_x, id2_y, id2_z, free_sticker_x, free_sticker_y, free_sticker_z)
            
            # print(f'Distances dictionary: \t {distances}', flush=True)

            if len(distances) < 2:
                continue

            # print(f'Exchange evaluated using: \t {distances}\n', flush=True)

            fene_old = calculate_raw_fene_potential(distances[f'{id1}-{id2}'])
            new_fene_potentials = []        # FENE potentials of all possible NEW configurations
            for free_sticker in sticker_ids:
                potential1 = calculate_raw_fene_potential(distances[f'{id1}-{free_sticker}'])
                potential2 = calculate_raw_fene_potential(distances[f'{id2}-{free_sticker}'])

                new_fene_potentials.append([id1, free_sticker, potential1])
                new_fene_potentials.append([id2, free_sticker, potential2])
            
            new_fene_potentials = np.array(new_fene_potentials)

            min_row_idx = np.argmin(new_fene_potentials[:, 2])
            min_row = new_fene_potentials[min_row_idx]
            new_sticker1 = min_row[0]; new_sticker2 = min_row[1]; fene_new = min_row[2]

            delta_U = alpha * (fene_new - fene_old)

            # Acceptance probability. If change in potential is negative, the acceptance probability automatically becomes 1.0.
            if T != 0:
                P_accept = np.exp(-delta_U/(kB * T)) if delta_U > 0 else 1.0
            else:
                P_accept = 0

            bond_exchange = False
            if P_accept == 1:
                bond_exchange = True
                # print('Bond exchange happens naturally due to a negative delta U.')

            else:
                ran = random.uniform(0, P_coeff)
                if P_accept >= ran:
                    bond_exchange = True
                    # print('Bond exchange happens due to Metropolis acceptance criterion.')

            if bond_exchange:
                # print(f'Exchange granted!\nExchange between {id1}-{id2} original to {new_sticker1}-{new_sticker2}', flush=True)
                id1 = int(id1)
                id2 = int(id2)
                new_sticker1 = int(new_sticker1)
                new_sticker2 = int(new_sticker2)

                bonds_to_delete[min(id1, id2)] = max(id1, id2)
                bonds_to_create[min(new_sticker1, new_sticker2)] = max(new_sticker1, new_sticker2)

                # print(f'Bonds to delete on proc {comm.Get_rank()}: {bonds_to_delete}', flush=True)
                # print(f'Bonds to create on proc {comm.Get_rank()}: {bonds_to_create}', flush=True)

                # Add originally bonded sticker that is now free to the already_exchanged_atoms list so that it is not considered for another exchange
                if new_sticker1 == id1:
                    already_exchanged_atoms.add(id2)
                elif new_sticker1 == id2:
                    already_exchanged_atoms.add(id1)
                else:
                    warnings.warn('Neither atoms of the new bond has been bonded before!')

                # Add newly bonded atoms to already_exchanged_atoms
                already_exchanged_atoms.add(new_sticker1)
                already_exchanged_atoms.add(new_sticker2)

        if paired_bond_exchange and double_bonded_exchange:
            id1, id2 = linked_pairs[0]
            id3, id4 = linked_pairs[1]

            # print('Traditional bond exchange considered!', flush=True)

            # Saving coordinates of id1 and id2 atoms for later use
            id1_data = atoms[atoms['id'] == id1]
            id1_x = id1_data['x']; id1_y = id1_data['y']; id1_z = id1_data['z']
            id2_data = atoms[atoms['id'] == id2]
            id2_x = id2_data['x']; id2_y = id2_data['y']; id2_z = id2_data['z']
            id3_data = atoms[atoms['id'] == id3]
            id3_x = id3_data['x']; id3_y = id3_data['y']; id3_z = id3_data['z']
            id4_data = atoms[atoms['id'] == id4]
            id4_x = id4_data['x']; id4_y = id4_data['y']; id4_z = id4_data['z']

            # Remove bonded sticker ids from sticker_ids list
            sticker_ids.remove(id1); sticker_ids.remove(id2)

            distances = {}
            distances[f'{id1}-{id2}'] = calculate_distance_pbc(box_dims, id1_x, id1_y, id1_z, id2_x, id2_y, id2_z)
            distances[f'{id1}-{id3}'] = calculate_distance_pbc(box_dims, id1_x, id1_y, id1_z, id3_x, id3_y, id3_z)
            distances[f'{id1}-{id4}'] = calculate_distance_pbc(box_dims, id1_x, id1_y, id1_z, id4_x, id4_y, id4_z)
            distances[f'{id2}-{id3}'] = calculate_distance_pbc(box_dims, id2_x, id2_y, id2_z, id3_x, id3_y, id3_z)
            distances[f'{id2}-{id4}'] = calculate_distance_pbc(box_dims, id2_x, id2_y, id2_z, id4_x, id4_y, id4_z)
            distances[f'{id3}-{id4}'] = calculate_distance_pbc(box_dims, id3_x, id3_y, id3_z, id4_x, id4_y, id4_z)
            
            fene_1to2 = calculate_raw_fene_potential(distances[f'{id1}-{id2}'])
            fene_1to3 = calculate_raw_fene_potential(distances[f'{id1}-{id3}'])
            fene_1to4 = calculate_raw_fene_potential(distances[f'{id1}-{id4}'])
            fene_2to3 = calculate_raw_fene_potential(distances[f'{id2}-{id3}'])
            fene_2to4 = calculate_raw_fene_potential(distances[f'{id2}-{id4}'])
            fene_3to4 = calculate_raw_fene_potential(distances[f'{id3}-{id4}'])

            U_old = fene_1to2 + fene_3to4; U_new1 = fene_1to3 + fene_2to4; U_new2 = fene_1to4 + fene_2to3

            if U_new1 <= U_new2:
                U_new = U_new1
                new_id1 = int(id1); new_id2 = int(id3)
                new_id3 = int(id2); new_id4 = int(id4)
            else:
                U_new = U_new2
                new_id1 = int(id1); new_id2 = int(id4) 
                new_id3 = int(id2); new_id4 = int(id3)

            # U_new = min(U_new1, U_new2)

            delta_U = alpha * (U_new - U_old)

            # Acceptance probability. If change in potential is negative, the acceptance probability automatically becomes 1.0.
            if T != 0:
                P_accept = np.exp(-delta_U/(kB * T)) if delta_U > 0 else 1.0
            else:
                P_accept = 0

            bond_exchange = False
            if P_accept == 1:
                bond_exchange = True
                # print('Bond exchange happens naturally due to a negative delta U.')

            else:
                ran = random.uniform(0, P_coeff)
                if P_accept >= ran:
                    bond_exchange = True
                    # print('Bond exchange happens due to Metropolis acceptance criterion.')

            if bond_exchange:
                id1 = int(id1); id2 = int(id2); id3 = int(id3); id4 = int(id4)

                # Add original bonds to delete to dicitonary, organize such that key is smaller than value in the dictionary
                bonds_to_delete[min(id1, id2)] = max(id1, id2)
                bonds_to_delete[min(id3, id4)] = max(id3, id4)

                # Add original bonds to create to dicitonary, organize such that key is smaller than value in the dictionary
                bonds_to_create[min(new_id1, new_id2)] = max(new_id1, new_id2)
                bonds_to_create[min(new_id3, new_id4)] = max(new_id3, new_id4)

                # Add newly bonded atoms to already_exchanged_atoms
                already_exchanged_atoms.add(new_id1)
                already_exchanged_atoms.add(new_id2)
                already_exchanged_atoms.add(new_id3)
                already_exchanged_atoms.add(new_id4)

# -------------------- Gathering data from each process, combining them into one complete set of data and broadcasting it back to all processes--------------------

    bonds_to_delete_list = list(bonds_to_delete.items())        # Create list to prevent silent overwriting when combining dictionaries
    bonds_to_create_list = list(bonds_to_create.items())        # Create list to prevent silent overwriting when combining dictionaries

    gathered_bonds_to_delete = comm.gather(bonds_to_delete_list, root=0)
    gathered_bonds_to_create = comm.gather(bonds_to_create_list, root=0)

    mpi_rank = comm.Get_rank()
    if mpi_rank == 0:
        # Flatten lists
        assert gathered_bonds_to_delete is not None # For type checker
        assert gathered_bonds_to_create is not None # For type checker
        all_bonds_to_delete = [pair for sublist in gathered_bonds_to_delete for pair in sublist]
        all_bonds_to_create = [pair for sublist in gathered_bonds_to_create for pair in sublist]
        
        # Conflict resolution: ensure each atom appears only once
        used_atoms = set()
        filtered_bonds_to_delete = []
        filtered_bonds_to_create = []

        for (a1, a2), (b1, b2) in zip(all_bonds_to_delete, all_bonds_to_create):
            if a1 in used_atoms or a2 in used_atoms or b1 in used_atoms or b2 in used_atoms:
                continue
            filtered_bonds_to_delete.append((a1, a2))
            filtered_bonds_to_create.append((b1, b2))
            used_atoms.update([a1, a2, b1, b2])

        # Convert back to dicts if needed
        complete_bonds_to_delete = dict(filtered_bonds_to_delete)
        complete_bonds_to_create = dict(filtered_bonds_to_create)
    else:
        complete_bonds_to_delete = {}
        complete_bonds_to_create = {}

    # Broadcasting bonds to delete/create dictionaries
    complete_bonds_to_delete = comm.bcast(complete_bonds_to_delete, root=0)
    complete_bonds_to_create = comm.bcast(complete_bonds_to_create, root=0)

    return complete_bonds_to_delete, complete_bonds_to_create