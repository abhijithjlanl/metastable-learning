using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, LaTeXStrings, IterTools
using DataFrames
using Random
using GraphicalModelLearning
include("../samplers.jl")
include("../utils.jl")
filename = "data_energies_high.jld2"

include("potts_tensors.jl")
include("potts_samplers.jl")

repeats = 3

if isfile(filename)
    display("Existing file found")
    jldopen(filename) do file
        global df = file["df"]
    end
else
    df =  DataFrame(n_spins = Int64[] ,β = Float64[], q = Int64[], active_edges=[],sampler=Symbol[], model_type = [] ,n_samples = [],sample_energy = [])
end

nspins_list = 60
β_list = range(0.3,1.5,11)

sampler_list = [:GlauberMin, :GlauberRand]
M_list = [10^6]
model_list = [:Ferro1, :Ferro2]
q_list = [2,3,4]


arguments = product(nspins_list, β_list, q_list, sampler_list, model_list, M_list) |> collect


for arg in arguments, _ in 1:repeats
    global (ns, β,q, sampler_id, model_type, n_samples) = arg
    @show arg
    @assert iseven(ns)
    n_edges = div(3 * ns, 2)
    global Edges = random_p_hyergraph(ns, n_edges, p=3)
    while length(Edges) != n_edges
        Edges = []
        global Edges = random_p_hyergraph(ns, n_edges, p=3)
    end
    

    if model_type == :Ferro1
          global Φ = threealphabet_ferro1(q)
        elseif model_type == :Ferro2
          global Φ = threealphabet_ferro2(q)
    end



    Energy(σ) = HypergraphPottsEnergy(σ, Edges, Φ, -1)

    if sampler_id == :GlauberMin
        global samples, energy_estim =  Glauber_sampler(Energy, n_samples, ns, q; order_parameters = [Energy], init = () -> Int.(ones(ns)), β=β, restarts=100)
    elseif  sampler_id == :GlauberRand
        global samples, energy_estim =  Glauber_sampler(Energy, n_samples, ns, q; order_parameters = [Energy], β=β, restarts=100)
    end

    push!(df, [ns, β,q, Edges[:], sampler_id, model_type, n_samples, energy_estim[1] ])
    @show energy_estim[1]

    jldopen(filename,"w") do file
        file["df"] = df
    end
    end


