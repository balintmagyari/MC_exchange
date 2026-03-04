# MC-Exchange Python Package

The MC-Exchange Python package has been developed to support the dynamic bond exchange reactions (BERs) during a molecular dynamics (MD) simulation run using LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) software. Using the [LAMMPS Python package](https://docs.lammps.org/Python_head.html), a MD simulation can be run directly from a Python script, allowing an interface between Python and LAMMPS. The MC-Exchange Python package uses this to gather the necessary data from the LAMMPS MD simulation, analyze the data, determine which bonds to exchanged, and finally communicate the altered bond data back to LAMMPS, where the actual bond exchange can take place.

The package is written with the ease of use in mind. However, users must be warned that a thorough understanding of the LAMMPS Python package is necessary to be able to gather the necessary bits of data prior to performing bond exchange reactions using the MC-Exchange Python library.

## General Use

To install the package, users are asked to clone the repository. Once at the root level of the repository, the distribution files can be generated, and subsequently the package can be installed as follows:

<details>
<summary><b>macOS / Linux</b></summary>

```bash
python3 -m pip install --upgrade build
python3 -m build
pip install dist/mc_exchange-1.0.0-py3-none-any.whl
```
</details>

<details>
<summary><b>Windows</b></summary>

```cmd
py -m pip install --upgrade build
py -m build
pip install dist/mc_exchange-1.0.0-py3-none-any.whl
```
</details>

After successful installation of the MC-Exchange Python package, it can be imported into a script as follows:

```python
import MC_exchange as mc
```

The above statement imports all functionalities of the package. 

## Package organization

The MC-Exchange Python package is organized into modules. Currently, there are three modules dedicated to the base functionalities, and one module where these functionalities are combined to perform bond exchange reactions. The three base modules are `calculations.py`, `data.py` and `neigh_list.py`. 

### Base modules

Currently, three base modules are responsible for the primary functionalities of the package. *data.py* is the module responsible for gathering data from the LAMMPS simulations. *calculations.py* contains calculations that use the data obtained by the functions of *data.py* to compute properties that can be necessary during bond exchange reactions. Lastly, *neigh_list.py* contain the functionalities required to construct spacial neighbor lists using atom data.

## Functionalities

Currently, in version 1.0.0, there is a single function that the user should learn how to use called `evaluate_bond_exchange()`. This single function combines the
functionalities of the base modules into one easy to use function for evaluating associative bond exchange reactions. Users are encouraged the take a closer
look at the scripts under the `src/example/` to learn how the integration between LAMMPS and Python works.

The software is designed such that it works when LAMMPS simulations are run parallel (i.e., on multiple processors). Communication between processes is handled
by the `mpi4py` Python package. 

## Example script

To run the example script, make sure that the Python environment into which MC-Exchange was installed is activated and you are located in the 
`src/example/` directory. To run the example script simply run the following command in the terminal:

```zsh
mpiexec -np {n_procs} python3 loop.py
```

where `{n_procs}` is the number of parallel processes the user wishes to launch.

**Note**, this requires that the user has installed **OpenMPI** or **MPICH** to the local machine.

## Developer

**Balint Magyari**: PhD Student at University of Naples Federico II 

For any questions please contact <balint.magyari@unina.it>.
