import numpy as np
import warnings
import random

from itertools import combinations

import warnings

from mpi4py import MPI
import sys
from .calculations import calculate_distance_pbc, calculate_raw_fene_potential, calculate_lj_potential

from .error_handling import mpi_abort_on_exception

# Assign the hook at the very beginning of your script
sys.excepthook = mpi_abort_on_exception

def complementary_bond_exchange(sticker_neighbor_list: dict, 
                                bonds: np.ndarray,
                                atoms: np.ndarray,
                                box_dims: np.ndarray,
                                T: float,
                                sticker_types_A: list[int],
                                sticker_types_B: list[int],
                                alpha: float = 1.0,
                                P_coeff: float = 1.0,
                                kB: float = 1.0,
                                bond_shift: bool = True,
                                bond_swap: bool = True,
                                comm: MPI.Intracomm = MPI.COMM_WORLD,
                                ) -> tuple:
    """
    Evaluate bond exchange dynamics on the local process, gather combined data on which
    bonds to delete and create, and broadcast that combined data to all processes.

    This function performs bond exchange between stickers of complementary types.
    It is used to replicate real-life sticker-sticker chemical reactions, where 
    the chemistry of the two sticker types compliments each other to form the 
    reversible chemical bond.

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
    bond_shift : bool
        Whether to allow bond shift bond exchange reactions (BERs) to occur. These BERs occur between a bonded pair of
        stickers and a free sticker.
    bond_swap : bool
        Whether to allow bond swap bond exchange reactions (BERs) to occur. These BERs occur between two pairs of 
        bonded stickers where bonding partners are swapped.
    comm : MPI.Intracomm 
        MPI communicator. Default is MPI.COMM_WORLD.

    Returns
    -------
    tuple[dict, dict]

    bonds_to_delete and bonds_to_create dictionaries are returned. 
    """

    already_exchanged_atoms = set()    # Set used to prevent the same pair of atoms taking part in bond exchange more than once
    bonds_to_delete = {}               # Final dictionary containing atom1 and atom2 as key: value pairs between which bonds should be created after bond exchange
    bonds_to_create = {}               # Final dictionary containing atom1 and atom2 as key: value pairs between which bonds should be created after bond exchange

    # Create a fast lookup set for existing bonds (O(1) lookup time)
    fast_bond_set = set(zip(bonds['atom 1'], bonds['atom 2']))
    fast_bond_set.update(zip(bonds['atom 2'], bonds['atom 1'])) # Add reverse pairs

    id_to_idx = {atom_id: idx for idx, atom_id in enumerate(atoms['id'])}

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
            sticker_ids.append(neighbor_id)

        n_stickers = len(sticker_ids)   # Number of stickers that can potentially be exchanged

        # Continue with next iteration if the total number of stickers (including atom_main) is not enough for BER.
        if n_stickers < 3:
            continue

        linked_pairs = []   
        for id1, id2 in combinations(sticker_ids, 2):
            if (id1, id2) in fast_bond_set:
                linked_pairs.append((id1, id2))

        n_pairs = len(linked_pairs)

        if n_pairs == 0:      # Continue with next iteration of main loop if no pair is found
            continue

        elif n_pairs == 1:      # Evaluate 3 sticker bond exchange if only 1 pair of sticker is linked
            stopper_bond_exchange = True

        elif n_pairs == 2:       # Evaluate 4 sticker BER if two pairs of sticker is linked
            id1, id2 = linked_pairs[0]
            id3, id4 = linked_pairs[1]

            # Test to see if the four atom IDs are unique
            if len({id1, id2, id3, id4}) == 4:
                paired_bond_exchange = True
            else:
                raise RuntimeError(f"Topology violation occured before this loop. One of the atoms in {linked_pairs} is bonded to two other atoms. This is not allowed, as stickers are only meant to bond to a maximum of one other sticker at a time.")

        if stopper_bond_exchange and bond_shift:
            id1, id2 = linked_pairs[0]

            # Remove bonded sticker ids from sticker_ids list
            sticker_ids.remove(id1); sticker_ids.remove(id2)        # We know that these two are bonded
            sticker_ids = [s for s in sticker_ids if not np.any((bonds['atom 1'] == s) | (bonds['atom 2'] == s))]       # Remove any other stickers that are bonded but their bonding partner is not in the neighbor list

            if len(sticker_ids) == 0: # Continue to next iteration of main loop if no free stickers are left to consider for BER
                continue

            # Instantly grab the index and use it to grab the atom type
            idx1 = id_to_idx[id1]; idx2 = id_to_idx[id2]
            id1_type = atoms['type'][idx1]; id2_type = atoms['type'][idx2]

            # Raise error if id1_type == id2_type, which should not be allowed in complementary bond exchanges
            if id1_type == id2_type:
                raise ValueError(f'Bonded stickers {id1} and {id2} are of the same atom type! This should not be allowed')
            
            A_group = []; B_group = []

            if id1_type in sticker_types_A:
                A_group.append(id1)
            elif id1_type in sticker_types_B:
                B_group.append(id1)
            else:
                raise ValueError(f'Atom {id1} type does not match any of the specified sticker types.')

            if id2_type in sticker_types_A:
                A_group.append(id2)
            elif id2_type in sticker_types_B:
                B_group.append(id2)
            else:
                raise ValueError(f'Atom {id2} type does not match any of the specified sticker types.')
            
            for s in sticker_ids:
                # Instant index lookup
                idx_s = id_to_idx[s]
                s_type = atoms['type'][idx_s]
                
                if s_type in sticker_types_A:
                    A_group.append(s)
                elif s_type in sticker_types_B:
                    B_group.append(s)
                else:
                    raise ValueError(f'Type of atom {s} is not in specified sticker types.')
                
            # 1. Identify the roles of the bonded pair
            bound_A = id1 if id1 in A_group else id2
            bound_B = id2 if id1 in A_group else id1

            # 2. Isolate the "free" attacking stickers
            free_A_stickers = [s for s in A_group if s != bound_A]
            free_B_stickers = [s for s in B_group if s != bound_B]

            potential_exchanges = []

            # 3. Evaluate attacks by free Group A stickers
            for free_A in free_A_stickers:
                # free_A wants to bond with bound_B. bound_A becomes free.
                exchange_data = {
                    'old_bond': (bound_A, bound_B),
                    'new_bond': (free_A, bound_B),
                    'leaving_atom': bound_A,
                    'attacking_atom': free_A
                }
                potential_exchanges.append(exchange_data)

            # 4. Evaluate attacks by free Group B stickers
            for free_B in free_B_stickers:
                # free_B wants to bond with bound_A. bound_B becomes free.
                exchange_data = {
                    'old_bond': (bound_A, bound_B),
                    'new_bond': (bound_A, free_B),
                    'leaving_atom': bound_B,
                    'attacking_atom': free_B
                }
                potential_exchanges.append(exchange_data)

            if len(potential_exchanges) == 0:
                continue

            # Select ONE trial move at random to maintain detailed balance
            trial_move = random.choice(potential_exchanges)

            old_bond = trial_move['old_bond']
            new_bond = trial_move['new_bond']
            leaving_atom = trial_move['leaving_atom']
            attacking_atom = trial_move['attacking_atom']

            # Extract coordinates instantly using the index mapping
            idx_old_1 = id_to_idx[old_bond[0]]
            old_1_x, old_1_y, old_1_z = atoms['x'][idx_old_1], atoms['y'][idx_old_1], atoms['z'][idx_old_1]
            idx_old_2 = id_to_idx[old_bond[1]]
            old_2_x, old_2_y, old_2_z = atoms['x'][idx_old_2], atoms['y'][idx_old_2], atoms['z'][idx_old_2]

            dist_old = calculate_distance_pbc(box_dims,
                                              old_1_x, old_1_y, old_1_z,
                                              old_2_x, old_2_y, old_2_z,
                                              )
            
            # Extract coordinates instantly using the index mapping
            idx_new_1 = id_to_idx[new_bond[0]]
            new_1_x, new_1_y, new_1_z = atoms['x'][idx_new_1], atoms['y'][idx_new_1], atoms['z'][idx_new_1]
            idx_new_2 = id_to_idx[new_bond[1]]
            new_2_x, new_2_y, new_2_z = atoms['x'][idx_new_2], atoms['y'][idx_new_2], atoms['z'][idx_new_2]

            dist_new = calculate_distance_pbc(box_dims,
                                              new_1_x, new_1_y, new_1_z,
                                              new_2_x, new_2_y, new_2_z,
                                              )
            
            fene_old = calculate_raw_fene_potential(dist_old)
            fene_new = calculate_raw_fene_potential(dist_new)
            lj_old = calculate_lj_potential(dist_old)
            lj_new = calculate_lj_potential(dist_new)

            # Calculate energy deltas
            # If the new distance exceeds the maximum FENE extension R_0, 
            # calculate_raw_fene_potential should return np.inf, automatically rejecting the move.
            delta_E_fene = fene_new - fene_old
            
            # Subtracted because forming a bond REMOVES LJ, and breaking a bond ADDS LJ.
            delta_E_lj = lj_new - lj_old
            
            delta_E = delta_E_fene - delta_E_lj

            # Apply the activation energy modifier (alpha)
            # In some vitrimer models, this mimics the barrier the catalyst helps overcome
            modified_delta_E = delta_E * alpha
            
            # Metropolis-Hastings Acceptance Criterion
            accept = False
            
            if modified_delta_E < 0:
                accept = True
            else:
                # Calculate Boltzmann probability
                probability = np.exp(-modified_delta_E / (kB * T))
                
                # Draw a random number between 0 and P_coeff
                # If P_coeff is 1.0, this is a standard MC acceptance check
                if random.uniform(0, P_coeff) < probability:
                    accept = True

            if accept:
                # Add to the action dictionaries
                bonds_to_delete[old_bond[0]] = old_bond[1]
                bonds_to_create[new_bond[0]] = new_bond[1]
                
                # Mark all three atoms involved in this 3-body shift as exchanged
                already_exchanged_atoms.update([old_bond[0], old_bond[1], trial_move['attacking_atom']])
                
                # print(f"Accepted shift: Broken {old_bond}, Formed {new_bond} | dE: {delta_E:.3f} on process {comm.Get_rank()}", flush=True)

        if paired_bond_exchange and bond_swap:
            id1, id2 = linked_pairs[0]
            id3, id4 = linked_pairs[1]

            # 1. Instant index lookups for all four atoms
            idx1 = id_to_idx[id1]; idx2 = id_to_idx[id2]
            idx3 = id_to_idx[id3]; idx4 = id_to_idx[id4]

            type1 = atoms['type'][idx1]; type2 = atoms['type'][idx2]
            type3 = atoms['type'][idx3]; type4 = atoms['type'][idx4]

            # 2. Map atoms to A and B groups for Bond 1
            if type1 in sticker_types_A and type2 in sticker_types_B:
                bond1_A, bond1_B = id1, id2
            elif type2 in sticker_types_A and type1 in sticker_types_B:
                bond1_A, bond1_B = id2, id1
            else:
                raise ValueError(f"Bond 1 ({id1}, {id2}) violates complementary A-B pairing.")

            # 3. Map atoms to A and B groups for Bond 2
            if type3 in sticker_types_A and type4 in sticker_types_B:
                bond2_A, bond2_B = id3, id4
            elif type4 in sticker_types_A and type3 in sticker_types_B:
                bond2_A, bond2_B = id4, id3
            else:
                raise ValueError(f"Bond 2 ({id3}, {id4}) violates complementary A-B pairing.")

            # 4. Define the ONLY valid proposed topology
            # A from Bond 1 pairs with B from Bond 2
            # A from Bond 2 pairs with B from Bond 1
            potential_exchanges = [{
                'old_bonds': ((bond1_A, bond1_B), (bond2_A, bond2_B)),
                'new_bonds': ((bond1_A, bond2_B), (bond2_A, bond1_B))
            }]

            # 5. Extract coordinates, calculate energies, and apply Metropolis...

# -------------------- Gathering data from each process, combining them into one complete set of data and broadcasting it back to all processes--------------------

    bonds_to_delete_list = list(bonds_to_delete.items())        # Create list to prevent silent overwriting when combining dictionaries
    bonds_to_create_list = list(bonds_to_create.items())        # Create list to prevent silent overwriting when combining dictionaries

    gathered_bonds_to_delete = comm.gather(bonds_to_delete_list, root=0)
    gathered_bonds_to_create = comm.gather(bonds_to_create_list, root=0)

    mpi_rank = comm.Get_rank()
    if mpi_rank == 0:
        assert gathered_bonds_to_delete is not None # For type checker
        assert gathered_bonds_to_create is not None # For type checker
        # Flatten lists
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