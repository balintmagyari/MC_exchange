import numpy as np
import warnings
import random
import math
from itertools import combinations

# System related imports
from mpi4py import MPI
import sys

# Imports from other modules
from .calculations import calculate_distance_pbc, calculate_fene_potential, calculate_lj_potential, calculate_raw_fene_potential
from .error_handling import mpi_abort_on_exception

# Assign the hook at the very beginning of your script
sys.excepthook = mpi_abort_on_exception

def perform_bond_swap(sticker_neighbor_list: dict, 
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
    Perform 'bond swap' style bond exchange in vitrimers. This is the bond exchange reaction where 
    two pairs of bonded stickers come into close contact and exchange partners.

    This function evaluates bond exchange dynamics on the local process, gather combined data on which
    bonds to delete and create, and broadcast that combined data to all processes.

    If return_stats = True, statistics about the number of exchanges is returned, and thus the length
    of the returned tuple is changed accordingly.

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
    warnings.warn(
        "perform_bond_swap() is deprecated and will be removed in version 2.0.0."
        "Use evaluate_bond_exchange() instead.",
        category=DeprecationWarning,
        stacklevel=2
    )
    
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
    
def evaluate_bond_exchange(sticker_neighbor_list: dict, 
                          bonds: np.ndarray,
                          atoms: np.ndarray,
                          box_dims: np.ndarray,
                          T: float,
                          alpha: float = 1.0,
                          P_coeff: float = 1.0,
                          kB: float = 1.0,
                          bond_shift: bool = True,
                          bond_swap: bool = True,
                          comm: MPI.Intracomm = MPI.COMM_WORLD,
                          return_stats: bool = False
                          ) -> tuple:
    """
    Evaluate bond exchange dynamics on the local process, gather combined data on which
    bonds to delete and create, and broadcast that combined data to all processes.

    Accepted moves are gathered whole on the root process, where a move sharing an atom with
    a move from another rank is rejected as a unit, so a swap is never applied halfway.

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
    return_stats : bool
        If True, returns a dictionary of exchange statistics with the same keys as
        complementary_bond_exchange(). Default is False.

    Returns
    -------
    tuple
        If return_stats is False:
            - bonds_to_delete (dict): Keys and values represent the lower and higher IDs of bonds to break.
            - bonds_to_create (dict): Keys and values represent the lower and higher IDs of new bonds to form.

        If return_stats is True:
            - bonds_to_delete (dict): Dictionary of bonds targeted for deletion.
            - bonds_to_create (dict): Dictionary of bonds targeted for creation.
            - exchange_stats (dict): Global values, summed over all ranks and identical on every rank.
              Counts are int, sums are float. X is 'shift' or 'swap'.
                'n_shifts', 'n_swaps', 'total': successful reactions after conflict resolution.
                'n_sites': neighbourhoods that proposed a trial move (one proposal each).
                'n_options': candidate moves evaluated across all sites. A shift site has two per
                    free sticker (it may bond to either sticker of the pair), a swap site has two
                    (the two re-pairings). The lowest-energy candidate is the one proposed.
                'n_proposed_X': proposals per mechanism.
                'n_proposed_overstretched': proposals with delta U = +inf (a new bond >= R0);
                    always rejected and also included in 'n_metropolis_rejected_X'.
                'n_metropolis_rejected_X': proposals rejected by the Metropolis test, including
                    those with delta U = nan (old and new configuration both contain a bond >= R0).
                'n_accepted_local_X': proposals accepted on their rank, before conflict resolution.
                'n_conflict_rejected_X': locally accepted moves dropped for sharing an atom with a
                    move from another rank.
                'n_skipped_quarantined': sites skipped because their central sticker had already
                    reacted in this call.
                'sum_dE_proposed': sum of finite delta U (scaled by alpha) over all proposals.
                'sum_dE_accepted': sum of finite delta U (scaled by alpha) over locally accepted moves.
              These identities hold:
                n_sites = n_proposed_shift + n_proposed_swap
                n_proposed_X = n_metropolis_rejected_X + n_accepted_local_X
                n_accepted_local_X = n_conflict_rejected_X + n_shifts (or n_swaps)
    """

    already_exchanged_atoms = set()    # Set used to prevent the same pair of atoms taking part in bond exchange more than once
    local_accepted_moves = []          # Accepted moves, each kept whole so that a swap is never split during conflict resolution

    # Local acceptance counters, as in complementary_bond_exchange(). Shift/swap pairs are indexed by kind: 0 = shift, 1 = swap
    n_sites = 0                         # Neighbourhoods that proposed a trial move
    n_options = 0                       # Candidate moves evaluated across all sites
    n_proposed = [0, 0]
    n_overstretched = 0                 # Proposals with an infinite delta U (a new bond >= R0)
    n_metropolis_rejected = [0, 0]
    n_accepted_local = [0, 0]
    n_skipped_quarantined = 0
    sum_dE_proposed = 0.0               # Finite delta U only
    sum_dE_accepted = 0.0

    # Loop through entries in neighbor list
    for atom_main, neighbors_data in sticker_neighbor_list.items():
        sticker_ids = []                # Combined list of sticker IDs, will include atom_main
        stopper_bond_exchange = False   # Boolean whether to perform BER including 1 pair of bonded stickers and a nearby 'free' sticker
        paired_bond_exchange = False    # Boolean whether to perform BER between two pairs of bonded sticker

        if atom_main in already_exchanged_atoms:    # Skip to next iteration of atom_main has already been exchanged
            n_skipped_quarantined += 1
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

        elif n_pairs == 1:      # Evaluate 3 sticker bond exchange if only 1 pair of sticker is linked
            stopper_bond_exchange = True

        elif n_pairs == 2:       # Evaluate 4 sticker BER if two pairs of sticker is linked
            id1, id2 = linked_pairs[0]
            id3, id4 = linked_pairs[1]

            # Test to see if the four atom IDs are unique
            if len({id1, id2, id3, id4}) == 4:
                paired_bond_exchange = True
            else:
                pass
                # print(f'\nWrongly made linked_pairs: {linked_pairs}', flush=True)
                # print(f'Sticker IDs: {sticker_ids}\n', flush=True)

        if stopper_bond_exchange and bond_shift:
            id1, id2 = linked_pairs[0]

            # Remove bonded sticker ids from sticker_ids list
            sticker_ids.remove(id1); sticker_ids.remove(id2)

            sticker_ids = [s for s in sticker_ids if not np.any((bonds['atom 1'] == s) | (bonds['atom 2'] == s))]

            if len(sticker_ids) == 0: # Continue to next iteration of main loop if no free stickers are left to consider for BER
                continue

            # Saving coordinates of id1 and id2 atoms for later use
            id1_data = atoms[atoms['id'] == id1]
            id1_x = id1_data['x']; id1_y = id1_data['y']; id1_z = id1_data['z']
            id2_data = atoms[atoms['id'] == id2]
            id2_x = id2_data['x']; id2_y = id2_data['y']; id2_z = id2_data['z']

            distances = {}
            distances[f'{id1}-{id2}'] = calculate_distance_pbc(box_dims, id1_x, id1_y, id1_z, id2_x, id2_y, id2_z)

            fene_old = calculate_raw_fene_potential(distances[f'{id1}-{id2}'])
            new_fene_potentials = []        # FENE potentials of all possible NEW configurations
            for free_sticker in sticker_ids:
                free_sticker_data = atoms[atoms['id'] == free_sticker]
                free_sticker_x = free_sticker_data['x']; free_sticker_y = free_sticker_data['y']; free_sticker_z = free_sticker_data['z']

                distances[f'{id1}-{free_sticker}'] = calculate_distance_pbc(box_dims, id1_x, id1_y, id1_z, free_sticker_x, free_sticker_y, free_sticker_z)
                distances[f'{id2}-{free_sticker}'] = calculate_distance_pbc(box_dims, id2_x, id2_y, id2_z, free_sticker_x, free_sticker_y, free_sticker_z)

                potential1 = calculate_raw_fene_potential(distances[f'{id1}-{free_sticker}'])
                potential2 = calculate_raw_fene_potential(distances[f'{id2}-{free_sticker}'])

                new_fene_potentials.append([id1, free_sticker, potential1])
                new_fene_potentials.append([id2, free_sticker, potential2])
            
            new_fene_potentials = np.array(new_fene_potentials)

            min_row_idx = np.argmin(new_fene_potentials[:, 2])
            min_row = new_fene_potentials[min_row_idx]
            new_sticker1 = min_row[0]; new_sticker2 = min_row[1]; fene_new = min_row[2]

            delta_U = alpha * (fene_new - fene_old)

            n_sites += 1
            n_options += len(new_fene_potentials)
            n_proposed[0] += 1

            # Only finite delta U is summed. +inf (a new bond >= R0) is always rejected and counted as overstretched.
            dU_finite = math.isfinite(delta_U)
            if dU_finite:
                sum_dE_proposed += delta_U
            elif delta_U > 0:
                n_overstretched += 1

            # Acceptance probability. If change in potential is negative, the acceptance probability automatically becomes 1.0, also at T = 0.
            if delta_U <= 0:
                P_accept = 1.0
            elif T != 0:
                P_accept = np.exp(-delta_U/(kB * T))
            else:
                P_accept = 0

            bond_exchange = False
            if math.isnan(delta_U):
                # inf - inf: the old and the new bond are both >= R0, so the move is rejected
                bond_exchange = False

            elif P_accept == 1:
                bond_exchange = True
                # print('Bond exchange happens naturally due to a negative delta U.')

            elif P_accept > 0:
                ran = random.uniform(0, P_coeff)
                if P_accept >= ran:
                    bond_exchange = True
                    # print('Bond exchange happens due to Metropolis acceptance criterion.')

            if not bond_exchange:
                n_metropolis_rejected[0] += 1

            if bond_exchange:
                n_accepted_local[0] += 1
                if dU_finite:
                    sum_dE_accepted += delta_U

                # print(f'Exchange granted!\nExchange between {id1}-{id2} original to {new_sticker1}-{new_sticker2}', flush=True)
                id1 = int(id1)
                id2 = int(id2)
                new_sticker1 = int(new_sticker1)    # Always id1 or id2
                new_sticker2 = int(new_sticker2)    # The free sticker

                local_accepted_moves.append({
                    'old_bonds': [(id1, id2)],
                    'new_bonds': [(new_sticker1, new_sticker2)],
                    'involved_atoms': [id1, id2, new_sticker2]
                })

                # Quarantine the originally bonded pair and the free sticker so that none of them is considered for another exchange
                already_exchanged_atoms.update([id1, id2, new_sticker2])

        if paired_bond_exchange and bond_swap:
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

            n_sites += 1
            n_options += 2              # The two possible re-pairings
            n_proposed[1] += 1

            # Only finite delta U is summed. +inf (a new bond >= R0) is always rejected and counted as overstretched.
            dU_finite = math.isfinite(delta_U)
            if dU_finite:
                sum_dE_proposed += delta_U
            elif delta_U > 0:
                n_overstretched += 1

            # Acceptance probability. If change in potential is negative, the acceptance probability automatically becomes 1.0, also at T = 0.
            if delta_U <= 0:
                P_accept = 1.0
            elif T != 0:
                P_accept = np.exp(-delta_U/(kB * T))
            else:
                P_accept = 0

            bond_exchange = False
            if math.isnan(delta_U):
                # inf - inf: the old and the new configuration both contain a bond >= R0, so the move is rejected
                bond_exchange = False

            elif P_accept == 1:
                bond_exchange = True
                # print('Bond exchange happens naturally due to a negative delta U.')

            elif P_accept > 0:
                ran = random.uniform(0, P_coeff)
                if P_accept >= ran:
                    bond_exchange = True
                    # print('Bond exchange happens due to Metropolis acceptance criterion.')

            if not bond_exchange:
                n_metropolis_rejected[1] += 1

            if bond_exchange:
                n_accepted_local[1] += 1
                if dU_finite:
                    sum_dE_accepted += delta_U

                id1 = int(id1); id2 = int(id2); id3 = int(id3); id4 = int(id4)

                # Both deletions and both creations are kept in one move, so that the swap is applied or rejected as a whole
                local_accepted_moves.append({
                    'old_bonds': [(id1, id2), (id3, id4)],
                    'new_bonds': [(new_id1, new_id2), (new_id3, new_id4)],
                    'involved_atoms': [id1, id2, id3, id4]
                })

                # Add newly bonded atoms to already_exchanged_atoms
                already_exchanged_atoms.update([id1, id2, id3, id4])

# -------------------- Gathering data from each process, combining them into one complete set of data and broadcasting it back to all processes--------------------

    # Gather the lists of entire moves from all MPI ranks
    gathered_moves = comm.gather(local_accepted_moves, root=0)

    mpi_rank = comm.Get_rank()
    if mpi_rank == 0:
        assert gathered_moves is not None # For type checker

        # Flatten the list of moves
        all_moves = [move for sublist in gathered_moves for move in sublist]

        used_atoms = set()
        filtered_bonds_to_delete = []
        filtered_bonds_to_create = []

        global_shifts = 0
        global_swaps = 0
        n_conflict_rejected = [0, 0]    # Counted independently, so n_accepted_local = n_conflict_rejected + n_X is a real check

        # Conflict resolution at the MOVE level: a move sharing any atom with an already approved move
        # (from another rank) is rejected as a whole, so a swap is never applied halfway
        for move in all_moves:
            if any(atom in used_atoms for atom in move['involved_atoms']):
                n_conflict_rejected[0 if len(move['old_bonds']) == 1 else 1] += 1
                continue

            # Organize such that key is smaller than value in the final dictionaries
            for ob in move['old_bonds']:
                filtered_bonds_to_delete.append((min(ob), max(ob)))

            for nb in move['new_bonds']:
                filtered_bonds_to_create.append((min(nb), max(nb)))

            used_atoms.update(move['involved_atoms'])

            # Count the officially approved move
            if len(move['old_bonds']) == 1:
                global_shifts += 1
            elif len(move['old_bonds']) == 2:
                global_swaps += 1

        # Convert back to dicts
        complete_bonds_to_delete = dict(filtered_bonds_to_delete)
        complete_bonds_to_create = dict(filtered_bonds_to_create)
    else:
        complete_bonds_to_delete = {}
        complete_bonds_to_create = {}

        global_shifts = 0
        global_swaps = 0
        n_conflict_rejected = [0, 0]

    # Broadcasting bonds to delete/create dictionaries
    complete_bonds_to_delete = comm.bcast(complete_bonds_to_delete, root=0)
    complete_bonds_to_create = comm.bcast(complete_bonds_to_create, root=0)

    # Broadcast the final statistics, including the conflict counters only known on root
    global_shifts, global_swaps, n_conflict_rejected = comm.bcast(
        (global_shifts, global_swaps, n_conflict_rejected), root=0)

    # Sum the local acceptance counters over all ranks in one collective. Every rank must reach it,
    # so it is unconditional. float64 is exact for integer counts up to 2**53.
    local_counters = np.array([n_sites, n_options,
                               n_proposed[0], n_proposed[1],
                               n_overstretched,
                               n_metropolis_rejected[0], n_metropolis_rejected[1],
                               n_accepted_local[0], n_accepted_local[1],
                               n_skipped_quarantined,
                               sum_dE_proposed, sum_dE_accepted], dtype=np.float64)
    global_counters = np.empty_like(local_counters)
    comm.Allreduce(local_counters, global_counters, op=MPI.SUM)

    if return_stats:
        exchange_stats = {
            'n_shifts': global_shifts,
            'n_swaps': global_swaps,
            'total': global_shifts + global_swaps,
            'n_sites': int(global_counters[0]),
            'n_options': int(global_counters[1]),
            'n_proposed_shift': int(global_counters[2]),
            'n_proposed_swap': int(global_counters[3]),
            'n_proposed_overstretched': int(global_counters[4]),
            'n_metropolis_rejected_shift': int(global_counters[5]),
            'n_metropolis_rejected_swap': int(global_counters[6]),
            'n_accepted_local_shift': int(global_counters[7]),
            'n_accepted_local_swap': int(global_counters[8]),
            'n_conflict_rejected_shift': n_conflict_rejected[0],
            'n_conflict_rejected_swap': n_conflict_rejected[1],
            'n_skipped_quarantined': int(global_counters[9]),
            'sum_dE_proposed': float(global_counters[10]),
            'sum_dE_accepted': float(global_counters[11]),
        }

        assert global_shifts + global_swaps*2 == len(complete_bonds_to_delete)

        return complete_bonds_to_delete, complete_bonds_to_create, exchange_stats
    else:
        return complete_bonds_to_delete, complete_bonds_to_create

def complementary_bond_exchange(neighbor_list: dict, 
                                bonds: np.ndarray,
                                atoms: np.ndarray,
                                box_dims: np.ndarray,
                                T: float,
                                sticker_types_A: list[int] | int,
                                sticker_types_B: list[int] | int,
                                kB: float = 1.0,
                                K: float = 30.0,
                                R0: float = 1.5,
                                eps: float = 1,
                                sigma: float = 1,
                                Rc: float = 2 ** (1 / 6),
                                bond_shift: bool = True,
                                bond_swap: bool = True,
                                return_stats: bool = False,
                                comm: MPI.Intracomm = MPI.COMM_WORLD,
                                ) -> tuple:
    """
    Evaluates Monte Carlo bond exchange dynamics for a vitrimer network across MPI processes.

    This function performs reversible bond exchanges between active stickers of complementary 
    types (Group A and Group B). It evaluates local neighborhoods to propose physically valid 
    3-body associative bond shifts and/or 4-body bond swaps. Proposed reactions are accepted 
    or rejected based on the Metropolis-Hastings criterion using the change in the raw FENE
    potential. This assumes LAMMPS 'special_bonds fene' (lj 0 1 1) with 'bond_style fene',
    under which the WCA/LJ contribution of a pair is the same whether it is bonded or not
    and therefore cancels from the energy change.
    
    To maintain topological integrity in a parallelized environment, accepted moves are 
    gathered on the root process, evaluated for cross-processor spatial conflicts, and 
    resolved before broadcasting the final, unified topology updates back to all ranks.

    Parameters
    ----------
    neighbor_list : dict
        Local neighbor list on the current process, structured as {atom_main: {neighbor_id: distance_data}}.
    bonds : np.ndarray
        Existing topological bond data organized as a structured numpy array with columns: 
        ['type', 'atom 1', 'atom 2'].
    atoms : np.ndarray
        Atom coordinate and type data organized as a structured numpy array with columns: 
        ['id', 'type', 'mol', 'x', 'y', 'z'].
    box_dims : np.ndarray
        Simulation box dimensions in the form: [xlo, xhi, ylo, yhi, zlo, zhi] for periodic 
        boundary condition calculations.
    T : float
        Current temperature of the simulation for Boltzmann probability weighting.
    sticker_types_A : list[int] | int
        The LAMMPS atom type(s) designated as Group A complementary stickers.
    sticker_types_B : list[int] | int
        The LAMMPS atom type(s) designated as Group B complementary stickers.
    kB : float, optional
        Boltzmann constant. Default is 1.0.
    K : float, optional
        Spring constant parameter for the FENE potential calculation. Default is 30.0.
    R0 : float, optional
        Maximum allowed bond extension parameter for the FENE potential. Default is 1.5.
    eps : float, optional
        Unused; kept for backwards compatibility. The LJ term cancels under 'special_bonds fene'.
    sigma : float, optional
        Unused; kept for backwards compatibility. The LJ term cancels under 'special_bonds fene'.
    Rc : float, optional
        Unused; kept for backwards compatibility. The LJ term cancels under 'special_bonds fene'.
    bond_shift : bool, optional
        If True, evaluates 3-body associative bond exchange reactions between a bonded pair 
        and a free sticker. Default is True.
    bond_swap : bool, optional
        If True, evaluates 4-body exchange reactions between two existing bonded pairs. 
        Default is True.
    return_stats : bool, optional
        If True, returns a dictionary of exchange statistics: the number of successful shifts and
        swaps after MPI conflict resolution, plus attempt/rejection counters for acceptance rates.
        Default is False.
    comm : MPI.Intracomm, optional
        The MPI intracommunicator instance for parallel data gathering and broadcasting. 
        Default is MPI.COMM_WORLD.

    Returns
    -------
    tuple
        If return_stats is False:
            - bonds_to_delete (dict): Keys and values represent the lower and higher IDs of bonds to break.
            - bonds_to_create (dict): Keys and values represent the lower and higher IDs of new bonds to form.
            
        If return_stats is True:
            - bonds_to_delete (dict): Dictionary of bonds targeted for deletion.
            - bonds_to_create (dict): Dictionary of bonds targeted for creation.
            - exchange_stats (dict): Global values, summed over all ranks and identical on every rank.
              Counts are int, sums are float. X is 'shift' or 'swap'.
                'n_shifts', 'n_swaps', 'total': successful reactions after conflict resolution.
                'n_sites': neighbourhoods that proposed a trial move (one proposal each).
                'n_options': candidate moves enumerated across all sites.
                'n_proposed_X': proposals per mechanism.
                'n_proposed_overstretched': proposals with delta E = +inf (a new bond >= R0);
                    always rejected and also included in 'n_metropolis_rejected_X'.
                'n_metropolis_rejected_X': proposals rejected by the Metropolis test.
                'n_accepted_local_X': proposals accepted on their rank, before conflict resolution.
                'n_conflict_rejected_X': locally accepted moves dropped for sharing an atom with a
                    move from another rank.
                'n_skipped_quarantined': sites skipped because their central sticker had already
                    reacted in this call.
                'sum_dE_proposed': sum of finite delta E over all proposals.
                'sum_dE_accepted': sum of finite delta E over locally accepted moves.
              These identities hold:
                n_sites = n_proposed_shift + n_proposed_swap
                n_proposed_X = n_metropolis_rejected_X + n_accepted_local_X
                n_accepted_local_X = n_conflict_rejected_X + n_shifts (or n_swaps)

    Raises
    ------
    ValueError
        If an atom identified in a linked pair violates the complementary A-B bonding rules.
    RuntimeError
        If an atom is detected to possess a valency greater than 1, indicating a corrupted 
        topology passed from the main simulation array.
    """
    # ---------------------------------------------------------
    # Input Normalization & Optimization
    # ---------------------------------------------------------
    # Convert sticker types to sets for instantaneous O(1) lookups.
    # If an int is passed, wrap it in a set.
    if isinstance(sticker_types_A, int):
        A_set = {sticker_types_A}
    else:
        A_set = set(sticker_types_A)

    if isinstance(sticker_types_B, int):
        B_set = {sticker_types_B}
    else:
        B_set = set(sticker_types_B)

    already_exchanged_atoms = set()    # Set used to prevent the same pair of atoms taking part in bond exchange more than once
    local_accepted_moves = []

    # Local acceptance counters (plain Python numbers, cheaper than numpy element updates in the loop).
    # Shift/swap pairs are indexed by kind: 0 = shift, 1 = swap
    n_sites = 0                         # Neighbourhoods that proposed a trial move
    n_options = 0                       # Candidate moves enumerated across all sites
    n_proposed = [0, 0]
    n_overstretched = 0                 # Proposals with an infinite delta E (a new bond >= R0)
    n_metropolis_rejected = [0, 0]
    n_accepted_local = [0, 0]
    n_skipped_quarantined = 0
    sum_dE_proposed = 0.0               # Finite delta E only
    sum_dE_accepted = 0.0

    # Create a fast lookup set for existing bonds (O(1) lookup time)
    fast_bond_set = set(zip(bonds['atom 1'], bonds['atom 2']))
    fast_bond_set.update(zip(bonds['atom 2'], bonds['atom 1'])) # Add reverse pairs
    id_to_idx = {atom_id: idx for idx, atom_id in enumerate(atoms['id'])}

    # Create an O(1) lookup set of ALL atoms currently bonded in the simulation
    all_bonded_atoms = set(bonds['atom 1']).union(set(bonds['atom 2']))

    def get_group(atom_id):
        idx = id_to_idx[atom_id]
        atype = atoms['type'][idx]
        if atype in A_set: 
            return 'A'
        elif atype in B_set: 
            return 'B'
        else: 
            raise ValueError(f"Atom {atom_id} type {atype} is not an A or B sticker.")

    # Loop through entries in neighbor list
    for atom_main, neighbors_data in neighbor_list.items():
        sticker_ids = []                # Combined list of sticker IDs, will include atom_main

        if atom_main in already_exchanged_atoms:    # Skip to next iteration of atom_main has already been exchanged
            n_skipped_quarantined += 1
            continue

        sticker_ids.append(atom_main)
        for neighbor_id in neighbors_data.keys():
            if neighbor_id in already_exchanged_atoms or neighbor_id < atom_main:      # Skip to next iteration of neighbor_id has already been exchanged or is lower in value than atom_main to prevent double counting
                continue
            sticker_ids.append(neighbor_id)

        if len(sticker_ids) < 3:
            continue

        linked_pairs = []   
        for id1, id2 in combinations(sticker_ids, 2):
            if (id1, id2) in fast_bond_set:
                linked_pairs.append((id1, id2))

        if len(linked_pairs) == 0:
            continue

        # ---------------------------------------------------------
        # 1. Isolate Roles and Setup Pooling
        # ---------------------------------------------------------
        free_stickers = [s for s in sticker_ids if s not in all_bonded_atoms]
        
        potential_exchanges = []

        # ---------------------------------------------------------
        # 2. Evaluate all possible 3-Body Shifts
        # ---------------------------------------------------------
        if bond_shift and len(free_stickers) > 0:
            for bond in linked_pairs:
                g1, g2 = get_group(bond[0]), get_group(bond[1])
                
                if g1 == 'A' and g2 == 'B':
                    bound_A, bound_B = bond[0], bond[1]
                elif g1 == 'B' and g2 == 'A':
                    bound_A, bound_B = bond[1], bond[0]
                else:
                    raise ValueError(f"Bond {bond} violates complementary pairing.")

                for free_s in free_stickers:
                    free_group = get_group(free_s)
                    if free_group == 'A':
                        potential_exchanges.append({
                            'old_bonds': [(bound_A, bound_B)],
                            'new_bonds': [(free_s, bound_B)],
                            'involved_atoms': [bound_A, bound_B, free_s]
                        })
                    elif free_group == 'B':
                        potential_exchanges.append({
                            'old_bonds': [(bound_A, bound_B)],
                            'new_bonds': [(bound_A, free_s)],
                            'involved_atoms': [bound_A, bound_B, free_s]
                        })

        # ---------------------------------------------------------
        # 3. Evaluate all possible 4-Body Swaps
        # ---------------------------------------------------------
        if bond_swap and len(linked_pairs) >= 2:
            for bond1, bond2 in combinations(linked_pairs, 2):
                if len({bond1[0], bond1[1], bond2[0], bond2[1]}) != 4:
                    # Find the exact atom ID that is double-bonded
                    shared_atom = set(bond1).intersection(set(bond2)).pop()
                    raise RuntimeError(
                        f"Topology Valency Violation: Atom {shared_atom} is illegally bonded to multiple partners. "
                        f"Evaluated pairs: {bond1} and {bond2}. Stickers are restricted to a maximum of one bond."
                    )
                
                g1_0, g1_1 = get_group(bond1[0]), get_group(bond1[1])
                g2_0, g2_1 = get_group(bond2[0]), get_group(bond2[1])

                if g1_0 == g1_1:
                    raise ValueError(f'Bonded stickers {g1_0} and {g1_1} are of the same atom type! This should not be allowed')
                if g2_0 == g2_1:
                    raise ValueError(f'Bonded stickers {g2_0} and {g2_1} are of the same atom type! This should not be allowed')

                b1_A, b1_B = (bond1[0], bond1[1]) if g1_0 == 'A' else (bond1[1], bond1[0])
                b2_A, b2_B = (bond2[0], bond2[1]) if g2_0 == 'A' else (bond2[1], bond2[0])

                potential_exchanges.append({
                    'old_bonds': [(b1_A, b1_B), (b2_A, b2_B)],
                    'new_bonds': [(b1_A, b2_B), (b2_A, b1_B)],
                    'involved_atoms': [b1_A, b1_B, b2_A, b2_B]
                })

        # ---------------------------------------------------------
        # 4. Detailed Balance & Metropolis Evaluation
        # ---------------------------------------------------------
        if len(potential_exchanges) == 0:
            continue

        n_sites += 1
        n_options += len(potential_exchanges)

        trial_move = random.choice(potential_exchanges)
        kind = 0 if len(trial_move['old_bonds']) == 1 else 1
        n_proposed[kind] += 1

        fene_old, fene_new = 0.0, 0.0

        # With special_bonds fene (lj 0 1 1), a bonded pair feels FENE + WCA (WCA from bond_style fene)
        # and a non-bonded pair feels WCA (from pair_style lj/cut). The WCA term is therefore identical
        # before and after the move and cancels, leaving only the raw FENE contribution in delta E.

        # Sum potentials for ALL bonds being broken
        for o_bond in trial_move['old_bonds']:
            idx_1, idx_2 = id_to_idx[o_bond[0]], id_to_idx[o_bond[1]]
            x1, y1, z1 = atoms['x'][idx_1], atoms['y'][idx_1], atoms['z'][idx_1]
            x2, y2, z2 = atoms['x'][idx_2], atoms['y'][idx_2], atoms['z'][idx_2]

            dist = calculate_distance_pbc(box_dims, x1, y1, z1, x2, y2, z2)
            fene_old += calculate_raw_fene_potential(distance = dist,
                                                     K = K,
                                                     R0 = R0)

        # Sum potentials for ALL bonds being formed
        for n_bond in trial_move['new_bonds']:
            idx_1, idx_2 = id_to_idx[n_bond[0]], id_to_idx[n_bond[1]]
            x1, y1, z1 = atoms['x'][idx_1], atoms['y'][idx_1], atoms['z'][idx_1]
            x2, y2, z2 = atoms['x'][idx_2], atoms['y'][idx_2], atoms['z'][idx_2]

            dist = calculate_distance_pbc(box_dims, x1, y1, z1, x2, y2, z2)
            fene_new += calculate_raw_fene_potential(distance = dist,
                                                     K = K,
                                                     R0 = R0)

        # Delta E Calculation
        delta_E = fene_new - fene_old

        # The raw FENE potential is infinite for a bond >= R0, so a single such proposal would turn
        # the sums into +/-inf. Only finite delta E is summed. delta E = +inf (a new bond >= R0) is
        # always rejected and counted as overstretched. -inf/nan need an existing bond >= R0, which
        # LAMMPS fene allows with a "FENE bond too long" warning; those are left out of the sums only.
        dE_finite = math.isfinite(delta_E)      # math.isfinite: ~15x cheaper than np.isfinite on a scalar
        if dE_finite:
            sum_dE_proposed += delta_E
        elif delta_E > 0:
            n_overstretched += 1

        # Metropolis Acceptance Criterion
        accept = False
        if delta_E < 0:
            accept = True
        else:
            probability = np.exp(-delta_E / (kB * T))
            if random.uniform(0, 1) < probability:
                accept = True

        if not accept:
            n_metropolis_rejected[kind] += 1

        # ---------------------------------------------------------
        # 5. Topology Updates
        # ---------------------------------------------------------
        # Instead of breaking the move apart, save the entire move as a unit
        if accept:
            n_accepted_local[kind] += 1
            if dE_finite:
                sum_dE_accepted += delta_E

            # Note: You need to define local_accepted_moves = [] at the very top of your function
            # right beneath where you defined already_exchanged_atoms = set()
            local_accepted_moves.append(trial_move)
            
            # Mark all atoms involved in this reaction as exchanged to quarantine them locally
            already_exchanged_atoms.update(trial_move['involved_atoms'])

# -------------------- Gathering data from each process, combining them into one complete set of data and broadcasting it back to all processes--------------------

    # Gather the lists of entire moves from all MPI ranks
    gathered_moves = comm.gather(local_accepted_moves, root=0)

    mpi_rank = comm.Get_rank()
    if mpi_rank == 0:
        assert gathered_moves is not None # For type checker
        
        # Flatten the list of moves
        all_moves = [move for sublist in gathered_moves for move in sublist]

        used_atoms = set()
        filtered_bonds_to_delete = []
        filtered_bonds_to_create = []

        # Initialize global counters on the root process
        global_shifts = 0
        global_swaps = 0
        n_conflict_rejected = [0, 0]    # Counted independently, so n_accepted_local = n_conflict_rejected + n_X is a real check

        # Conflict resolution: evaluate at the MOVE level, not the bond level
        for move in all_moves:
            # Check if ANY atom in this move conflicts with a move accepted by another rank
            conflict = False
            for atom in move['involved_atoms']:
                if atom in used_atoms:
                    conflict = True
                    break
            
            # If cross-rank conflict exists, reject the entire move
            if conflict:
                # print('Cross-processor conflict was found! Move rejected', flush=True)
                n_conflict_rejected[0 if len(move['old_bonds']) == 1 else 1] += 1
                continue
                
            # If no conflict, apply all deletions and creations using the undirected min/max standard
            for ob in move['old_bonds']:
                lower_id = min(ob[0], ob[1])
                higher_id = max(ob[0], ob[1])
                filtered_bonds_to_delete.append((lower_id, higher_id))
                
            for nb in move['new_bonds']:
                lower_id = min(nb[0], nb[1])
                higher_id = max(nb[0], nb[1])
                filtered_bonds_to_create.append((lower_id, higher_id))
                
            # Quarantine the atoms globally
            used_atoms.update(move['involved_atoms'])

            # Count the officially approved move
            if len(move['old_bonds']) == 1:
                global_shifts += 1
            elif len(move['old_bonds']) == 2:
                global_swaps += 1

        complete_bonds_to_delete = dict(filtered_bonds_to_delete)
        complete_bonds_to_create = dict(filtered_bonds_to_create)
    else:
        complete_bonds_to_delete = {}
        complete_bonds_to_create = {}

        global_shifts = 0
        global_swaps = 0
        n_conflict_rejected = [0, 0]

    # Broadcasting bonds to delete/create dictionaries
    complete_bonds_to_delete = comm.bcast(complete_bonds_to_delete, root=0)
    complete_bonds_to_create = comm.bcast(complete_bonds_to_create, root=0)

    # Broadcast the final statistics, including the conflict counters only known on root
    global_shifts, global_swaps, n_conflict_rejected = comm.bcast(
        (global_shifts, global_swaps, n_conflict_rejected), root=0)

    # Sum the local acceptance counters over all ranks in one collective. Every rank must reach it,
    # so it is unconditional. float64 is exact for integer counts up to 2**53.
    local_counters = np.array([n_sites, n_options,
                               n_proposed[0], n_proposed[1],
                               n_overstretched,
                               n_metropolis_rejected[0], n_metropolis_rejected[1],
                               n_accepted_local[0], n_accepted_local[1],
                               n_skipped_quarantined,
                               sum_dE_proposed, sum_dE_accepted], dtype=np.float64)
    global_counters = np.empty_like(local_counters)
    comm.Allreduce(local_counters, global_counters, op=MPI.SUM)

    if return_stats:
        exchange_stats = {
            'n_shifts': global_shifts,
            'n_swaps': global_swaps,
            'total': global_shifts + global_swaps,
            'n_sites': int(global_counters[0]),
            'n_options': int(global_counters[1]),
            'n_proposed_shift': int(global_counters[2]),
            'n_proposed_swap': int(global_counters[3]),
            'n_proposed_overstretched': int(global_counters[4]),
            'n_metropolis_rejected_shift': int(global_counters[5]),
            'n_metropolis_rejected_swap': int(global_counters[6]),
            'n_accepted_local_shift': int(global_counters[7]),
            'n_accepted_local_swap': int(global_counters[8]),
            'n_conflict_rejected_shift': n_conflict_rejected[0],
            'n_conflict_rejected_swap': n_conflict_rejected[1],
            'n_skipped_quarantined': int(global_counters[9]),
            'sum_dE_proposed': float(global_counters[10]),
            'sum_dE_accepted': float(global_counters[11]),
        }

        assert global_shifts + global_swaps*2 == len(complete_bonds_to_delete)

        return complete_bonds_to_delete, complete_bonds_to_create, exchange_stats
    else:
        return complete_bonds_to_delete, complete_bonds_to_create