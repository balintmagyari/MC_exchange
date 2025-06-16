# MC-Exchange Python Package

The MC-Exchange Python package has been developed to support the dynamic bond exchange reactions (BERs) during a molecular dynamics (MD) simulation run using LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) software. Using the [LAMMPS Python package](https://docs.lammps.org/Python_head.html), a MD simulation can be run directly from a Python script, allowing an interface between Python and LAMMPS. The MC-Exchange Python package uses this to gather the necessary data from the LAMMPS MD simulation, analyze the data, determine which bonds to exchanged, and finally communicate the altered bond data back to LAMMPS, where the actual bond exchange can take place.

## General Use

After successful installation of the MC-Exchange Python package, it can be imported into a script as follows:

```python
import MC_exchange as mc
```

The above statement imports all functionalities of the package. 

## Package organization

The MC-Exchange Python package is organized into modules. Currently, there are three modules dedicated to the base functionalities, and one module where these functionalities are combined to perform bond exchange reactions. The three base modules are *calculations.py*, *data.py* and *neigh_list.py*.

### Base modules

Currently, three base modules are responsible for the primary functionalities of the package. *data.py* is the module responsible for gathering data from the LAMMPS simulations. *calculations.py* contains calculations that use the data obtained by the functions of *data.py* to compute properties that can be necessary during bond exchange reactions. Lastly, *neigh_list.py* contain the functionalities required to construct spacial neighbor lists using atom data.

## Functionalities




