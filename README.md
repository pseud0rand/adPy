# adPy
*A Bare-bones Implementation of Automatic Differentiation in Python*

The Python scripts are simple implementations of some of the concepts discussed in "Introduction to Automatic Differentiation and MATLAB Object-Oriented Programming" by Richard D. Neidinger (https://doi.org/10.1137/080743627)

The scripts were written in the process of learning about automatic differentiation. They are based on operator overloading, and only require NumPy. 

The adpy_reverse_mode.py is the reverse form, containing the back-propagation step in which local derivatives are collected. This is done using dicionaries, and dynamic creation of key labels that correspond to symbolic representations of local derivatives.

Only a few special functions are implemented. More can be easily added at the end of each script, following the form of others.
