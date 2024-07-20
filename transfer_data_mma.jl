using JLD2
using DelimitedFiles

# Convert all JLD2 data to the csv file

let 
    nsite = 30                       # Site Number
    cutoff = 1E-14                   # cutoff δ_χ for MPS
    tau = 0.1                        # Trotter decomposition time step
    ttotal = 2.0                     # Total evolution time of MBL Hamiltonian
    Delta = 1.0                      # XXZ parameter, XX+YY+Delta*ZZ
    Nrealization = 50000             # Total random realization number            
    W_disorder_arr = [5.0, 10.0]     # The disorder potential W*Z for the XXZ model                    
    state_name_arr = ["GHZ", "ZXZ"]  # The unknown initial state for classical shadow protocal                    
    k_len = 8                        # The maximal length of operator O_A
    maxdim_val = 200                 # The maximal bond dimension for the shadow norm calculation

    for W_disorder in W_disorder_arr
        
        for state_name in state_name_arr
            
            file_extension = "N$(nsite)_W$(W_disorder)_Nrand$(Nrealization)_tau$(tau)"
            total_dirname = "tJ$(ttotal)_sample_$(state_name)"
            
            OA_expect = JLD2.load("$(total_dirname)/$(file_extension)_reconstruction_plotdata.jld2", "OA_expect")
            std2OA = JLD2.load("$(total_dirname)/$(file_extension)_reconstruction_plotdata.jld2", "std2OA")

            writedlm("$(total_dirname)/plotdata_$(file_extension)_$(state_name)_OA_expect.csv", OA_expect, ',')
            writedlm("$(total_dirname)/plotdata_$(file_extension)_$(state_name)_std2OA.csv", std2OA, ',')
            
            file_extension_OA_sh = "N$(nsite)_W$(W_disorder)_tau$(tau)"
            OA_sh_t_all = JLD2.load("$(total_dirname)/OA_sh_fixOpdata_allk_type_$(file_extension_OA_sh)_chi$(maxdim_val).jld2", "OA_sh_t_all")

            OA_expect_k_realizationall = JLD2.load("$(total_dirname)/$(file_extension)_reconstruction_result.jld2", "OA_expect_k_realizationall")
            OA_expect_all_anystate = OA_expect_k_realizationall
            for i in 1:k_len
                OA_expect_all_anystate[:,i] = OA_expect_all_anystate[:,i] / OA_sh_t_all[21,i]
            end

            OA_expect_anystate_k_result_mean = real.(mean(OA_expect_all_anystate, dims=1)[1,:])
            OA_expect_anystate_k_result_var =  real.(var(OA_expect_all_anystate, dims=1)[1,:])

            M = 50000
            std2OA = 2*sqrt.(OA_expect_anystate_k_result_var/M)

            OA_expect = ((OA_expect_anystate_k_result_mean))
            # OA_var_div_Osh = ((OA_expect_anystate_k_result_var) ./ ( (1.0 / (M)) ./OA_sh_t_all[21,:] ))
            println("<O_A> = $(OA_expect_anystate_k_result_mean)")
            println("var O_A = $(OA_expect_anystate_k_result_var)")
            println("std O_A = $(2*sqrt.(OA_expect_anystate_k_result_var/M))")
            println("(|OA|_sh^2/M)  =$(  (1.0 / (M)) ./OA_sh_t_all[21,:] )")
            println("(|OA|_sh^2) = $(1.0 ./OA_sh_t_all[21,:] )")
            println("(log(|OA|_sh^2)) = $( -log.(3, OA_sh_t_all[21,:]) )")


            writedlm("$(total_dirname)/plotdata_$(file_extension)_$(state_name)_var.csv", hcat(log.(3, OA_expect_anystate_k_result_var), -log.(3, OA_sh_t_all[21,:])), ',')
        end
    end
end