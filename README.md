# Non-abelian-topology
A ground state simulation for the non-abelian D4 topological order Hamiltonian: Transform-map neural network (TM-NN)
<img width="613" height="494" alt="image" src="https://github.com/user-attachments/assets/3e2609eb-b072-4a92-ac38-6e0f54f1feb8" />



Key challenge in simulation: the ground state wavefunction has non-trivial sign structure, i.e. alternating plus and minus sign depending on the configuration snapshot. Solution: Beside the spin configuration, a path information - the sequence of hexagonal flips to land on the current configuration starting from a reference. To run the code:
 
 `python GS.py --L l`
 
with l (divisible by three) is the linear size of the lattice. The non-abelian effect is observed through braiding operations, simulated by

`python braiding.py --L l`

The code GS.py has to run first the generate the ground state wavefunction. After a full braid, two plaquette terms flip sign, as opposed to an abelian order where a full braid never changes sign.
