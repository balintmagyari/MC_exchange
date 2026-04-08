# MC-exchange Python Package

> [!NOTE]
> Documentation under development. Once Balint Magyari has more time he will continue writing a detailed documentation for the package.

The MC-exchange Python package has been developed to support the dynamic bond exchange reactions (BERs) during a molecular dynamics (MD) simulation run using LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) software. Using the [LAMMPS Python package](https://docs.lammps.org/Python_head.html), a MD simulation can be run directly from a Python script, allowing an interface between Python and LAMMPS. The MC-exchange Python package uses this to gather the necessary data from the LAMMPS MD simulation, analyze the data, determine which bonds to exchanged, and finally communicate the altered bond data back to LAMMPS, where the actual bond exchange can take place.

These simulations are useful for studying the effects of dynamic bonds of associative covalently adaptive polymer networks.

The package is written with the ease of use in mind. However, users must be warned that a thorough understanding of the LAMMPS Python package is necessary to be able to gather the necessary bits of data prior to performing bond exchange reactions using the MC-exchange Python library.

## General Use

The package can be installed from PyPI distribution archive using:

```zsh
pip install MC-exchange
```

After successful installation of the MC-exchange Python package, it can be imported into a script as follows:

```python
import MC_exchange as mc
```

The above statement imports all user-necessary functionalities of the package. 

## Package organization

The MC-exchange Python package is organized into modules. Currently, there are three modules dedicated to the base functionalities, and one module where these functionalities are combined to perform bond exchange reactions. The three base modules are `calculations.py`, `data.py` and `neigh_list.py`. All functions necessary to perform dynamic bond exchange reactions are imported direcly, thus the user does not need to search these modules for the required functionalities. Nevertheless, we provide a brief overview of the different modules, so that the user can better understand the structure of the library. 

### Base modules

Currently, three base modules are responsible for the primary functionalities of the package. *data.py* is the module responsible for gathering data from the LAMMPS simulations. *calculations.py* contains calculations that use the data obtained by the functions of *data.py* to compute properties that can be necessary during bond exchange reactions. Lastly, *neigh_list.py* contain the functionalities required to construct spacial neighbor lists using atom data.

## Functionalities

Currently, in version 1.1.0, there are two main functions used to perform associative bond exchange reactions. Firstly,
the `evaluate_bond_exchange()` function is used to perform two types of bond exchange reactions between *stickers* of
any kind. The two types of bond exchange reactions allowed are *bond shifts* and *bond swaps*. The first is an 3-body
exchange reaction between a bonded sticker pair and a 'free' sticker where a single bond migrates, while the latter is a
4-body exchange between two pairs of bonded stickers requiring the rearrangement of two bonds. In either case, the
number of total bonds in the system does not change.

Secondly, the `complementary_bond_exchange` function, as the name suggests, allows reactions to occur between stickers
that are of different types (i.e., A-B crosslinks). In the function, these types can be defined according to LAMMPS'
numerical type convention under arguments `sticker_types_A` and `sticker_types_B`.

Both of the above mentioned bond exchange functions allow *bond shifts* and *bond swaps* to be turned on and off
according to the user's requirements.

The software is designed such that it works when LAMMPS simulations are run parallel (i.e., on multiple processors).
Communication between processes is handled by the `mpi4py` Python package. 

## Example script

In order to run the following script...

Users are encouraged to clone the [repository](https://github.com/balintmagyari/MC_exchange) of this project to access all source files and example code.

To run the example script, make sure that the Python environment into which MC-exchange was installed is activated and you are located in the 
`src/example/` directory. To run the example script simply run the following command in the terminal:

```zsh
mpiexec -np {n_procs} python3 loop.py
```

where `{n_procs}` is the number of parallel processes the user wishes to launch.

**Note**, this requires that the user has installed **OpenMPI** or **MPICH** to the local machine.

## Developer

**Balint Magyari**: PhD Student at University of Naples Federico II 

For any questions please contact <balint.magyari@unina.it>.
