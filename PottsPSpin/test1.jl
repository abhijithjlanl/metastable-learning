using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, LaTeXStrings
using DataFrames
using Random
using GraphicalModelLearning, BenchmarkTools
include("../samplers.jl")
include("../utils.jl")
#filename = "Ferro_init_minima.jld2"

include("potts_tensors.jl")
include("potts_samplers.jl")


ns = 50
n_edges = 3*ns//2
N = 2^20
q =2
J = -1

Edges = random_p_hyergraph(ns, n_edges,p=3)

#=
Energy(σ) = FerroPottsEnergy(σ,-1,Edges)
samples = exact_sampler(Energy, N, ns, q) |> countmap
samples_glb = Glauber_sampler(Energy, N, ns, q) |> countmap
@info l1_norm_error(samples, samples_glb)/N
=#

ϕ  =  threealphabet_ferro1(q)

Energy(σ) = HypergraphPottsEnergyOptimized(σ, Edges,ϕ ,J)

β_list = range(0.5,1.2,8)
e_list1 = []
e_list2 = []

P = plot(show = true)

for β in β_list
    samples_1, energy1 =  Glauber_sampler(Energy, N, ns, q; order_parameters = [(x) -> Energy(x)/N], init = () -> Int.(ones(ns)), β=β)
    samples_2,energy2 =  Glauber_sampler(Energy, N, ns, q; order_parameters = [(x) -> Energy(x)/N], β=β)
    @info β, energy1, energy2
    push!(e_list1, energy1[1]/length(Edges))
    push!(e_list2, energy2[1]/length(Edges))
end
plot!(β_list, e_list1, show=true, label="Glauber, min start", seriestype=:scatter)
plot!(β_list, e_list2, show=true, label="Glauber, random start", seriestype=:scatter)

####Sanity check with old code

Energy(σ) = FerroEnergy(σ, Edges)
bin_energy_list1 = []
bin_energy_list2 = []

for β in β_list
    E_exact =  avg_energy(Energy, ns, β)
    samples_1, energy3 =  Glauber_sampler(Energy, N, ns; order_parameters = [(x) -> Energy(x)/N], init = () -> Int.(ones(ns)), β=β)
    samples_2,energy4 =  Glauber_sampler(Energy, N, ns; order_parameters = [(x) -> Energy(x)/N], β=β)
    @info β, energy3, energy4
    push!(bin_energy_list1, energy3[1])
    push!(bin_energy_list2, energy4[1])

end
plot!(β_list, bin_energy_list1 ./length(Edges), show=true, label="BinGlauber, min start", seriestype=:scatter)
plot!(β_list, bin_energy_list2 ./length(Edges), show=true, label="BinGlauber, random start", seriestype=:scatter)


