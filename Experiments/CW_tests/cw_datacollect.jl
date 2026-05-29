using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using IterTools
using LaTeXStrings
using GraphicalModelLearning
using ProgressMeter
using JLD2
using DataFrames
using Random,Graphs
include("../../samplers.jl")
include("../../utils.jl")
include("cw_utils.jl")


filename="data_cw.jld2"
repeats = 5

if isfile(filename)
    display("Existing file found")
    jldopen(filename) do file
        global df = file["df"]
    end
else
    df =  DataFrame(n_spins = Int64[] ,J = Float64[], h = Float64[], sampler=Symbol[] ,n_samples = [], J_err = [], h_err = [],sample_mag = [] )
end
nspins_list = [5000]
sampler_list = [:Glauber]
Jns_list = [1.2]
h_list = [0.04]
M_list = [2^x for x in 22:2:32]

arguments = product(nspins_list, Jns_list, h_list, sampler_list, M_list) |> collect

Optimizer = NLP(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>1))

for arg in arguments, _ in 1:repeats
    global (ns,Jns,h, sampler_id,M) = arg
    J = Jns/ns
    @info (ns,J,h, sampler_id, M)
    E_sm(m) =   -0.5*J*( m^2 - ns) + h*m
    Energy_exact(σ) = E_sm(sum(σ))

    if sampler_id == :Exact
        samples_exact = CW_ExactSampler(E_sm,M, ns)
        J_est, h_est = CW_learn_conditional_RPLE(samples_exact)
        sample_mag = sum(k[2]*(v/M) for (k,v) in samples_exact)
    end
    n_restarts = max(5, div(6000*M,2^30))
   # n_restarts = min(n_restarts,1500)

    if  sampler_id == :Glauber
        samples_MCMC = CW_Glauber_sampler_quadratic(J,h, M, ns, init=:up, burnin=10^5, restarts = n_restarts, keep_every =4)
        J_est, h_est = CW_learn_conditional_RPLE(samples_MCMC)

        sample_mag = sum(k[2]*(v/M) for (k,v) in samples_MCMC)
        @show sample_mag
        @show  J_est*ns, h_est
        @assert sample_mag > 1000
    end
    J_err = abs(J-J_est)
    h_err = abs(h-h_est)
    push!(df, [ns, J,h, sampler_id, M, J_err, h_err, sample_mag])

    jldopen(filename,"w") do file
        file["df"] = df
    end
end



