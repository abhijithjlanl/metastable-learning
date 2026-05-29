using Combinatorics
include("samplers.jl")

#    for e in Edges
#        E += J*reduce(*, σ[e])
#    end

n_spins = 20
n_edges = n_spins//2
Edges = []
while length(Edges) != n_edges
    global Edges = random_p_hyergraph(n_spins, n_edges,p=3)
end

for i in 1:n_spins
    n_terms =0
    for e in Edges
        if i in e
            n_terms += 1
        end
    end
    @show i, n_terms
end


Energy(σ) = FerroEnergy(σ, Edges)
#Energy(σ) = SpinGlassEnergy(σ, Edges, 2*rand(length(Edges)) .- 1)
β = 4
E_avg =  avg_energy(Energy, n_spins, β)
@show E_avg
for n_samples in 4 .^ Array(1:8)
    @show n_samples
   # local S1 = exact_sampler(Energy, n_samples, n_spins, β = β)
    local S2 = Gibbs_sampler(Energy, n_samples, n_spins, β = β,
                             order_parameters= [x->(Energy(x)/n_samples)],return_samples = false )
    #@show sample_TV(S1, S2)
    @show S2, E_avg
end


