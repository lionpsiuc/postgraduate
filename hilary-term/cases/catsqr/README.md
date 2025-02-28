# Case Studies in High-Performance Computing

## Assignment 1 - Communication-Avoiding Methods

### Mathematical Background

We explain the idea surrounding a communication-avoiding tall-skinny QR (TSQR) factorisation. Here are the steps taken:

1. **Matrix Partitioning**: We are given a tall, narrow matrix $W\in\mathbb{R}^{m\times n}$ with $m\gg n$. This matrix is divided into four blocks of rows, which would be distributed across four processors, with $W$ now being $[W_0\ W_1\ W_2\ W_3]^\intercal$, where each block $W_i$ has $\frac{m}{4}$ rows.
2. **Local QR Factorisation**: Each processor performs a local QR decomposition on its block, $W_i=Q_iR_i$, where $Q_i\in\mathbb{R}^{\frac{m}{4}\times n}$ is orthogonal and $R_i\in\mathbb{R}^{n\times n}$ is upper triangular. Moreover, this step avoids inter-processor communication, keeping computation local.
3. **Reduction Step**: The upper triangular factors $R_i$ from each local QR decomposition are collected and combined into a new small matrix given by $R=[R_0\ R_1\ R_2\ R_3]^\intercal\in\mathbb{R}^{4n\times n}$. Another QR factorisation is performed on this reduced matrix, giving us $R=Q\prime R_{\text{final}}$. Here, $Q\prime\in\mathbb{R}^{4n\times n}$ is another orthogonal matrix, and $R_{\text{final}}\in\mathbb{R}^{n\times n}$ is the final upper triangular factor.
4. **Constructing the Final $Q$**: Since $W=QR$, the final orthogonal matrix is $Q=[Q_0\ Q_1\ Q_2\ Q_3]^\intercal\cdot Q\prime\in\mathbb{R}^{m\times n}$. The local orthogonal matrices $Q_i$ are updated by multiplying with $Q\prime$. The final decomposition is thus $W=QR_{\text{final}}$.

### Folder Structure and Usage Details

The repository is organised in such a way as to address the requirements of the assignment (i.e., a separate directory for the Python implementation and for the C implementation).

#### C Implementation

Located in `c`, where this folder contains the C code required for the assignment, addressing the second and third question. It is organised as follows:

- **`tsqr.c`**: Implements the communication-avoiding TSQR factorisation using LAPACK routines along with MPI for parallel processing. This file defines a method which performs the local QR factorisations and the subsequent reduction step as described in the lectures.
- **`timing.c`**: Serves as a test driver that runs the communication-avoiding TSQR implementation for various matrix dimensions. It times the execution of the factorisation for different values of $m$ (number of rows) and $n$ (number of columns), and outputs the results to a file named `timing.txt`.
- **`plot.py`**: A Python script that reads the timing data from `timing.txt` and generates a plot. The plot visualises the scaling behaviour of the TSQR algorithm.
- **`Makefile`**: Automates the compilation process for the C files. See [here](#how-to-run) for details on how to compile and run.

#### Python Implementation

Located in `python`. Its structure is as follows:

- **`tsqr.ipynb`**: A Jupyter Notebook that implements the communication-avoiding TSQR factorisation. In this version, the input matrix is divided into four blocks (using Python's slicing capabilities) and the QR decompositions are computed for each block without any parallel programming.

##### Dependencies

The required Python packages are listed in `requirements.txt`. Make sure these are installed (e.g., via `pip install -r requirements.txt`) before running any Python scripts. These are also essential for running `plot.py`.

### How to Run

#### C Implementation

1. **Prior to a Full Compilation and Execution**: Navigate to the `c` folder and run:

```bash
make clean
```

This is to ensure `timing.png` is deleted, allowing a fresh run to commence.

2. **Compilation**: Once more, ensure you are in the `c` folder and run:

```bash
make
```

This will generate `timing` and `tsqr`, two executables for their respective source files.

3. **Execution and Timing**: We first focus on `tsqr`. In order to run it, use the following command:

```bash
mpirun -np 4 ./tsqr
```

This will print out the input, $Q$, $R$, and reconstructed input matrix. Moreover, we print the error of the reconstructed matrix as a way to verify that the algorithm generated a suitable $Q$ and $R$ matrix. To change the dimensions of the matrix, simply change the inputs of the `tsqr` method within `main`. Moving on to `timing`, run using the following:

```bash
mpirun -np 4 ./timing
```

Note that we do not recommend changing the values of `m` and `n` here, as they were chosen to allow for a clear plot to be generated, in order to show the scaling properties of the algorithm. This will generate a file called `timing.txt` containing the execution times for different matrix sizes.

4. **Plot Generation**: With the timing data available, run the following command:

```bash
python3 plot.py
```

This will produce a plot called `timing.png` showing how the performance varies with different values of $m$ and $n$.

5. **Cleaning**: Once finished, simply run the following:

```bash
make clean
```

This will remove all executables along with any generated residual files (e.g., `timing.png` and `timing.txt`).

#### Python Implementation

Ensure you are in `python` and open the Jupyter Notebook `tsqr.ipynb` within your own environment (e.g., Jupyter Notebook, Jupyter Lab, etc.).
