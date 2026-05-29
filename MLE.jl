
using StatsBase
using LinearAlgebra
include("samplers.jl")


function GradE(σ,Edges)
    return map(x -> reduce(*, σ[x]), Edges )
end

function exactgrads(Edges, Energy, n_spins; β = 1)
    f(x) = GradE(x, Edges)
    return  expectation_val(f, Energy, n_spins; β = β)
end




function MLE(Data, GradEfn::Function, n_steps, schedule::Function, Sampler, J_init;metrics = [], γ = 0.9)
    J = J_init
    ∇E_data = GradEfn.(Data) |> mean
    v = zeros(length(J))
    for t in 1:n_steps
        ##Using NAG Arxiv:1609.04747
        J_temp = J + schedule(t)*v
        NewSamp = Sampler(J_temp, Data)
        ∇E_model = GradEfn.(NewSamp) |> mean
        v = γ*v + schedule(t)* (∇E_model - ∇E_data)
        J = J + v
        @show t, norm(∇E_data - ∇E_model), [f(J) for f in metrics]
    end
    return J
end


function error_in_couplings(Edges1,J1, Edges2, J2)
    D1 = Dict(Edges1 .=> J1)
    D2 = Dict(Edges2 .=> J2)
    err =0.0
    for k in Edges1
        if k in Edges2
            err = max(abs(D1[k] - D2[k]), err)
        else
            err = max(abs(D1[k]),err)
        end
    end
    for k in Edges2
        if !(k in Edges1)
            err = max(abs(D2[k]),err)
        end
    end
    return  err
end

function MLE_test()
    ns = 6
    n_edges = 9
    β = 1
    order = 2
    Edge = random_p_hyergraph(ns, n_edges,p=order)
    while length(Edges) != n_edges
        Edges = []
        Edges = random_p_hyergraph(ns, n_edges,p=order)
    end
  #  @show Edges

    Energy(σ) = FerroEnergy(σ, Edges)
    Data = exact_sampler(Energy, 2^12, ns,β = β)
    global All_edges = combinations(1:ns,order)

    J_init = rand(length(All_edges))

    GradEFn(x) = GradE(x, All_edges) ##Looking at all possible 3 body terms
    function Temp_sampler(J, Data; n_samples = 2^8)
        Efn(x) = SpinGlassEnergy(x,All_edges, J)
        return Gibbs_sampler(Efn, n_samples, ns, restarts = n_samples, burnin = 1, init = ()->rand(Data))
    end

    function Exact_test_sampler(J, Data; n_samples = 2^10)
        Efn(x) = SpinGlassEnergy(x,All_edges, J)
        return exact_sampler(Efn, n_samples, ns,β = β)
    end



    schedule(t) = 0.001/(1+ (t/50))
    err(x) = error_in_couplings(Edges, -β*ones(length(Edges)), All_edges,x)

    J = MLE(Data, GradEFn, 50000, schedule ,Exact_test_sampler, J_init, metrics = [err] )

    return Edges, J
end

Edges, J = MLE_test()
D1 = Dict(All_edges .=> J)
D2 = Dict(Edges .=> -1*ones(length(Edges)))

Energy_exact(x) = FerroEnergy(x,Edges)
Energy_learned(x) = SpinGlassEnergy(x, All_edges, J)



