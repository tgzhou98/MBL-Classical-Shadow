using Plots
using LaTeXStrings
using Distributions
using JLD2
using Dates
using Test

include("itensor_setup.jl")

function GHZ_psi_gen(s, nsite)
    psi = randomMPS(s,2)
    for ind in 1:1
        M = zeros(ComplexF64, 2, 2)
        M[1,:] = [1.0, 0.0]
        M[2,:] = [0.0, 1.0]

        for i in 1:2
            for j in 1:2
                (psi[ind].tensor)[i,j] = M[i,j]
            end
        end
    end
    for ind in 2:nsite - 1
        M = zeros(ComplexF64, 2, 2, 2)
        M[:,1,:] = [1.0 0; 0 0]
        M[:,2,:] = [0 0; 0 1.0]
        for i in 1:2
            for j in 1:2
                for k in 1:2
                    (psi[ind].tensor)[i,j,k] = M[i,j,k]
                end
            end
        end
    end
    for ind in nsite:nsite
        M = zeros(ComplexF64, 2, 2)
        M[:,1] = [1.0, 0.0]
        M[:,2] = [0.0, 1.0]
        for i in 1:2
            for j in 1:2
                (psi[ind].tensor)[i,j] = M[i,j]
            end
        end
    end
    normalize!(psi)
    
    #####################################################
    # Test module
    # Check the expectation value ⟨∏ S_z^⊗k⟩
    psi_GHZ_test = deepcopy(psi)

    # Test Start
    k_len = 8
    op_k_pos_arr = ones(Int64, k_len)
    for i in 1:k_len÷2
        op_k_pos_arr[2*(i-1)+1] = nsite ÷ 2 + 1 - i
        op_k_pos_arr[2*(i-1)+2] = nsite ÷ 2 + i
    end
    OA_expect_k_test = zeros(ComplexF64, k_len)
    for (k_ind, k_pos) in enumerate(op_k_pos_arr)

        psi_GHZ_test = apply(op("Sz", s[k_pos]),psi_GHZ_test ; cutoff)
        OA_expect_k_test[k_ind] = inner(psi, psi_GHZ_test) * 2^(k_ind)
    end
    println("GHZ test $(OA_expect_k_test)")
    #####################################################

    return psi
end

function ZXZ_psi_gen(s, nsite, cutoff)
    state = ["Up" for n=1:nsite]
    psi0 = MPS(s,state)

    gates = ITensor[]
    for j in 1:(nsite)
        if j < nsite - 1
            s1 = s[j]
            s2 = s[j+1]
            s3 = s[j+2]
        elseif j == nsite - 1
            s1 = s[j]
            s2 = s[j+1]
            s3 = s[1]
        elseif j == nsite
            s1 = s[j]
            s2 = s[1]
            s3 = s[2]
        end
        hj = 1/2*(op("Sz", s1) * op("Sx", s2) * op("Sz", s3) * 2^3 + op("Id", s1) * op("Id", s2) * op("Id", s3))
        push!(gates, hj)
    end
    psi_ZXZ = apply(gates, psi0; cutoff)
    normalize!(psi_ZXZ)

    # Test module
    #########################################################
    # Test 1
    # Check the expectation value ⟨∏ S_z^⊗k⟩
    psi_ZXZ_test = deepcopy(psi_ZXZ)

    # Test Start
    k_len = 8
    op_k_pos_arr = ones(Int64, k_len)
    for i in 1:k_len÷2
        op_k_pos_arr[2*(i-1)+1] = nsite ÷ 2 + 1 - i
        op_k_pos_arr[2*(i-1)+2] = nsite ÷ 2 + i
    end
    OA_expect_k_test = zeros(ComplexF64, k_len)
    for (k_ind, k_pos) in enumerate(op_k_pos_arr)

        psi_ZXZ_test = apply(op("Sz", s[k_pos]),psi_ZXZ_test ; cutoff)
        OA_expect_k_test[k_ind] = inner(psi_ZXZ, psi_ZXZ_test) * 2^(k_ind)
    end
    println("ZXZ test $(OA_expect_k_test)")
    #########################################################

    #########################################################
    # Test 2
    # Check the expectation value ⟨Z_i X_{i+1} Z_{i+2}⟩
    psi_ZXZ_test = deepcopy(psi_ZXZ)

    gates_stab = ITensor[]
    for j in 1:(nsite - 2)
        s1 = s[j]
        s2 = s[j + 1]
        s3 = s[j + 2]

        hj = op("Sz", s1) * op("Sx", s2) * op("Sz", s3)
        push!(gates_stab, hj)
    end
    # psi_ZXZ = apply(gates, psi0; cutoff)

    println("Normalizartion $(inner(psi_ZXZ, psi_ZXZ) )")

    stab_expect_k_test = zeros(ComplexF64, length(gates_stab))
    for i in 1:length(gates_stab)
        psi_ZXZ_test_new = apply(gates_stab[i], psi_ZXZ_test ; cutoff)
        stab_expect_k_test[i] = inner(psi_ZXZ, psi_ZXZ_test_new) * 2^(3)
    end
    println("ZXZ Stablizer test $(stab_expect_k_test)")

    #########################################################

    return psi_ZXZ
end

function sample_ensemble(nsite::Int64, cutoff::Float64, tau::Float64, ttotal::Float64,
    Nrealization::Int64, Delta::Float64, W_disorder::Float64, h_arr::Array{Float64, 1}, total_dirname::String, state_name::String)
    

    clif_num = 24

    file_extension = "N$(nsite)_W$(W_disorder)_Nrand$(Nrealization)_tau$(tau)"


    # load_flag = true
    load_flag = false
    if load_flag
        # Load previous data
        println("Load clif_ind_layer_rand")
        clif_ind_layer_rand = JLD2.load("$(total_dirname)/sample_result_$(file_extension)_config.jld2", "clif_ind_layer_rand")
    else
        # New single qubit clifford layer
        clif_ind_layer_rand = zeros(Int64, Nrealization, nsite, 2)
        for nreal in 1:Nrealization
            clif_ind_layer_rand[nreal,:,1] = rand(1:clif_num, nsite)
            clif_ind_layer_rand[nreal,:,2] = rand(1:clif_num, nsite)
        end
    end

    # Site basis
    s = siteinds("S=1/2", nsite)

    JLD2.save("$(total_dirname)/sample_result_$(file_extension)_config.jld2", "clif_ind_layer_rand",clif_ind_layer_rand, "h_arr", h_arr)

    # Construction of Analog MBL Hamiltonian using Trotter decomposition
    # Make gates (1,2),(2,3),(3,4),...
    gates = ITensor[]
    for j in 1:(nsite - 1)
        s1 = s[j]
        s2 = s[j + 1]
        hj = (Delta * op("Sz", s1) * op("Sz", s2) +
            1 / 2 * op("S+", s1) * op("S-", s2) +
            1 / 2 * op("S-", s1) * op("S+", s2)) +
            h_arr[j] / 2 * op("Sz", s1) * op("Id", s2) + h_arr[j + 1] / 2 * op("Id", s1) * op("Sz", s2)
        if j == 1
            hj += h_arr[j] / 2 * (op("Sz", s1) * op("Id", s2))
        elseif j == nsite - 1
            hj += h_arr[j + 1] / 2 * (op("Sz", s2) * op("Id", s1))
        end
        Gj = exp(-im * tau / 2 * hj)
        push!(gates, Gj)
    end
    # Include gates in reverse order too, for a second order Trotter decomposition
    # (N,N-1),(N-1,N-2),...
    append!(gates, reverse(gates))

    # Generate Initial State for the classical shadow
    if state_name == "GHZ"
        psi = GHZ_psi_gen(s, nsite)
    elseif state_name == "ZXZ"
        psi = ZXZ_psi_gen(s, nsite, cutoff)
    end

    psi_init = deepcopy(psi)

    if load_flag
        # Load previous data
        println("Load shadow_sample_arr")
        shadow_sample_arr = JLD2.load("$(total_dirname)/sample_result_$(file_extension).jld2", "shadow_sample_arr")
    else
        # Generate new data
        shadow_sample_arr = zeros(Int64, Nrealization, nsite)
    end

    nreal_start = 1
    for nreal in nreal_start:Nrealization
        
        # Single sites random clifford gates in realization `nreal`. It includes both initial layer and ending layer
        clif_ind_layer_init = clif_ind_layer_rand[nreal,:,1]
        clif_ind_layer_end  = clif_ind_layer_rand[nreal,:,2]

        # initial clifford layer
        gates_clif_init = ITensor[]
        for j in 1:nsite
            clif_ind = clif_ind_layer_init[j]
            push!(gates_clif_init, op("cli$(clif_ind)", s[j]))
        end

        # ending clifford layer
        gates_clif_end = ITensor[]
        for j in 1:nsite
            clif_ind = clif_ind_layer_end[j]
            push!(gates_clif_end, op("cli$(clif_ind)", s[j]))
        end


        # ###################################
        # # Start evolution
        psi = deepcopy(psi_init)

        # Initial single site twirling
        psi = apply(gates_clif_init, psi; cutoff)
        # MBL circuits
        for (i, t) in enumerate(0.0:tau:ttotal)
            t≈ttotal && break

            psi = apply(gates, psi; cutoff)
            normalize!(psi)
        end
        # for i in 1:nsite
        #     psi = apply(gates_clif_end[i], psi; cutoff)
        # end
        psi = apply(gates_clif_end, psi; cutoff)
        normalize!(psi)

        # Sample from MPS and use left canaonical 
        # https://itensor.discourse.group/t/the-use-of-orthogonalize/56
        # https://itensor.github.io/ITensors.jl/stable/examples/MPSandMPO.html

        # After orthogonalized at site 1
        orthogonalize!(psi,1)
        # Sample the MPS state with probability
        v1 = ITensors.sample(psi)
        if nreal % 500 == 1
            println("nreal=$(nreal)")
            println("Time = $(now())")
            println(v1)
            println("$(file_extension)")

            JLD2.save("$(total_dirname)/sample_result_$(file_extension).jld2", "shadow_sample_arr",shadow_sample_arr)
        end
        if nreal == 1
            println("nreal 1, Time = $(now())")
        end
        if nreal == 2
            println("nreal 2, Time = $(now())")
        end

        shadow_sample_arr[nreal,:] = v1
    end

    println("$(file_extension)")
    JLD2.save("$(total_dirname)/sample_result_$(file_extension).jld2", "shadow_sample_arr",shadow_sample_arr)


    return shadow_sample_arr
end

function main_sample()
    nsite = 30                        # Site Number
    cutoff = 1E-14                    # cutoff δ_χ for MPS
    tau = 0.1                         # Trotter decomposition time step
    ttotal = 2.0                      # Total evolution time of MBL Hamiltonian
    Delta = 1.0                       # XXZ parameter, XX+YY+Delta*ZZ
    Nrealization = 50000              # Total random realization number            
    W_disorder_arr = [5.0, 10.0]      # The disorder potential W*Z for the XXZ model                    
    state_name_arr = ["GHZ", "ZXZ"]   # The unknown initial state for classical shadow protocal                    

    for state_name in state_name_arr
        for W_disorder in W_disorder_arr
            total_dirname = "tJ$(ttotal)_sample_$(state_name)"
            if !isdir(total_dirname)
                mkdir(total_dirname)
            end
            # h_arr = JLD2.load("tJ$(ttotal)_sample_$(state_name)/sample_result_N$(nsite)_W$(W_disorder)_Nrand$(50000)_tau$(tau)_config.jld2", "h_arr")
            # Generate random Z field for the XXZ model, with disorder strength W
            h_arr = rand(Uniform(-W_disorder, W_disorder), nsite)
            display(h_arr)
            file_extension = "N$(nsite)_W$(W_disorder)_Nrand$(Nrealization)_tau$(tau)"
            JLD2.save("$(total_dirname)/sample_result_$(file_extension)_config.jld2", "h_arr", h_arr)

            shadow_sample_arr = sample_ensemble(nsite, cutoff, tau, ttotal, Nrealization, Delta, W_disorder, h_arr, total_dirname, state_name)
        end
    end
end

main_sample()