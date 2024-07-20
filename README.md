### MBL-Classical-Shadow

## Contents

This repository contains code for the paper **[Efficient Classical Shadow Tomography through Many-body Localization Dynamics, arXiv:2309.01258](https://arxiv.org/abs/2309.01258)**. This package consists of several codes for MBL-based classical shadow tomography.

- Packages

  - `itensor_setup.jl`: Constructs the double operator space for the shadow norm calculation. Additionally, it provides the 24 single qubit Clifford gates and wraps them to the MPO.
  
  - `sample_b_shadow_MBL.jl`: Samples the unknown state (GHZ or ZXZ) using the MBL-based classical shadow protocol. Obtains a series of measurement results $|b\rangle$.
  
  - `OA_fixed_double_op_state_MPS_forsample.jl`: Obtains the shadow norm $||O_A||_\text{sh}^2$ for the operator $O_A$ with fixed positions on the lattice.
  
  - `reconstruct_from_sample_b.jl`: Using the previously calculated shadow norm, this program reconstructs the expectation value and variance of $O_A$ using the MBL-based classical shadow protocol.

  - `OA_aver_double_op_state_MPS.jl`: Obtains the average shadow norm $||O_A||_\text{sh}^2$ for the operator $O_A$, with fixed length $k$ but different positions on the lattice.

  - `OA_sh_MBL_ED.jl`: Calculates the shadow norm using exact diagonalization, providing a concrete benchmark for the MPS calculation.

  - `transfer_data_mma.jl`: Transfers the JLD2 format data to the CSV format.

## Dependence

- Julia
  - Julia 1.10.4 (or above)
  - Packages: `LinearAlgebra`, `JLD2`, `DelimitedFiles`, `Plots`, `Distributions`, `Random`, `Statistics`, `ITensors`, `LaTeXStrings`, `SparseArrays`

## Running the program

- **Full MBL-based classical shadow**
    - Run the following commands sequentially in the Julia REPL:
        ```julia
        include("sample_b_shadow_MBL.jl");
        include("OA_fixed_double_op_state_MPS_forsample.jl");
        include("reconstruct_from_sample_b.jl");
        ```  
    - Run `transfer_data_mma.jl` to transfer the `.jld2` data file to a `.csv` data file. Use the command in the Julia REPL: 
        ```julia
        include("transfer_data_mma.jl");
        ```
      The result can reproduce Fig. 4 and Fig. 5 in the paper.

- **Check the shadow norm complexity scaling**
    - Run `OA_aver_double_op_state_MPS.jl` to obtain the scaling for the shadow norm at different time scales, which can reproduce Fig. 3 in the paper.

- **Shadow norm benchmark in exact diagonalization**
    - Run `OA_sh_MBL_ED.jl` to calculate the shadow norm using exact diagonalization. Change the parameters in the file if you want to investigate other cases. The result can act as a benchmark for the MPS calculation.

## Contact

Email: tgzhou.physics@gmail.com