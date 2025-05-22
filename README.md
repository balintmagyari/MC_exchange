# MC-Exchange Python Package

The MC-Exchange Python package has been developed to support the dynamic bond exchange reactions (BERs) during a molecular dynamics (MD) simulation run using LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator). Using the [LAMMPS Python package](https://docs.lammps.org/Python_head.html), a MD simulation can be run directly from a Python script, allowing an interface between Python and LAMMPS. The MC-Exchange Python package uses this to gather necessary data from the LAMMPS MD simulation, analyze the data, determine which bonds are exchanged, and finally communicate the altered bond data back to LAMMPS.

## General Use

After successful installation of the MC-Exchange Python package, it can be imported into a script as follows:

```python
import MC_exchange as mc
```




