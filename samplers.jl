using StatsBase
using Combinatorics
using ProgressMeter
using GraphicalModelLearning
##Gibbs distributions are exp(-βE)


function Exact_partition_fn(Energy, n_spins; β = 1)
    Z = 0.0
    for k in 0:(2^n_spins) - 1
        σ = digits(k, base=2, pad = n_spins)
        σ = 2*σ .- 1
        Z += exp(-β*Energy(σ))
    end
    return Z
end


function expectation_val(f, Energy, n_spins; β = 1)
    F = 0.0
    Z = 0.0
    for k in 0:(2^n_spins) - 1
        σ = digits(k, base=2, pad = n_spins)
        σ = 2*σ .- 1
        wt = exp(-β*Energy(σ))
        F = F .+ f(σ)*wt
        Z += wt
    end
    return  F/Z
end

avg_energy(Energy, n_spins, β) = expectation_val(Energy, Energy, n_spins, β = β)

function exact_sampler(E::Function, N::Int, p::Int; β = 1)
    states = Array(0:(2^p)-1)
    states = map( x -> digits(x, base=2, pad=p) |> reverse, states)
    states = map(x -> 2*x .- 1, states)
    wts =  exp.(-β*E.(states))
    return  wsample(states, wts, N)
end

function Glauber_sampler(E::Function,delE::Function, N::Int,p::Int;β = 1, restarts = 16, burnin = 10^4, order_parameters = [], return_samples = true, init = false, throw_away = 10)
    samples =  Dict()
    negβ = -1*β
    if init == false
        init = () -> rand([1,-1],p)
    end
    ns_list = repeat([div(N,restarts)], restarts)
    ord_p = zeros(length(order_parameters))
    if rem(N,restarts) > 0
        push!(ns_list,rem(N,restarts))
    end
   for k in 1:restarts
        state = init()[:]
        E_state = E(state)
        #push!(samples, deepcopy(state))
        for t = 1:((throw_away*ns_list[k])+burnin)
            i = rand(1:p)
            s_i = state[i]
            dE = delE(i, state)
            #=
              f_state = copy(state)
              f_state[i] = -1*f_state[i]
              @assert (E(state) - E(f_state)) == dE
            =#
            prob2 = exp(negβ*dE*s_i)
            fs = wsample([1,-1],[prob2,1])
            ##There is something non-trival above
            ##Actually the weights must be exp(E/2T) and exp(-E/2T)
            ##But this is same as by making it exp(E/T) and 1
            if fs != s_i
                E_state = E_state - dE
                state[i] = fs
            end
            if (t >  burnin) && ( (t - burnin)%throw_away==0)
                if return_samples
                    if haskey(samples, state)
                        samples[copy(state)] += 1
                    else
                        samples[copy(state)] = 1
                    end
                end
                for (i,x) in enumerate(order_parameters)
                    ord_p[i] += x(state)
                end
            end
        end
    end
    if length(order_parameters) == 0
      return samples
    elseif return_samples == false
        return ord_p
    end
   # @assert sum(values(samples)) == N
    return samples, ord_p
end



function Glauber_sampler(E::Function,N::Int,p::Int;β = 1, restarts = 16, burnin = 10^4, order_parameters = [], return_samples = true, init = false, throw_away = 10)
    samples =  []
#    β_by_2 = β/2
    if init == false
        init = () -> rand([1,-1],p)
    end
    ns_list = repeat([div(N,restarts)], restarts)
    ord_p = zeros(length(order_parameters))
    if rem(N,restarts) > 0
        push!(ns_list,rem(N,restarts))
    end
  @showprogress  for k in 1:restarts
#   for k in 1:restarts
        state = init()
        E_state = E(state)
        #push!(samples, deepcopy(state))
        for t = 1:((throw_away*ns_list[k])+burnin)-1
              i = rand(1:p)
             # flipped_state = deepcopy(state)
              state[i] = -1*state[i]
              E_flip = E(state)
              dE = E_state - E_flip
              prob2 = exp(β*dE*state[i])
              fs = wsample([1,-1],[prob2,1])
            ##There is something non-trival above
            ##Actually the weights must be exp(E/2T) and exp(-E/2T)
            ##But this is same as by making it exp(E/T) and 1, and saves extra compute
            if fs == state[i]
                E_state = E_flip
            else
                state[i] = fs
            end


            if (t >  burnin-1)&& (t%throw_away===0)
                if return_samples
                    push!(samples, deepcopy(state))
                end
                for (i,x) in enumerate(order_parameters)
                    ord_p[i] += x(state)
                end
            end
        end
    end
    if length(order_parameters) == 0
      return samples
    elseif return_samples == false
        return ord_p
    end
    return samples[1:n_samples], ord_p
end



function Gibbs_sampler(E,N,p;β = 1, restarts = 16, burnin = 10^4, order_parameters = [], return_samples = true, init = false, throw_away = 10)
    samples =  []
    if init == false
        init = () -> rand([1,-1],p)
    end
    ns_list = repeat([div(N,restarts)], restarts)
    ord_p = zeros(length(order_parameters))
    if rem(N,restarts) > 0
        push!(ns_list,rem(N,restarts))
    end
  @showprogress  for k in 1:restarts
#   for k in 1:restarts
        state = init()
        #push!(samples, deepcopy(state))
        for t = 1:((throw_away*ns_list[k])+burnin)-1
            #state = samples[end]
            for i in 1:p
                flipped_state = deepcopy(state)
                flipped_state[i] = -1*state[i]
                dE = E(state) - E(flipped_state)
                prob = 1/(1 + exp(β*dE*state[i]))
                state[i] = wsample([1,-1],[prob,1-prob])
            end

            if (t >  burnin-1)&& (t%throw_away===0)
                if return_samples
                    push!(samples, deepcopy(state))
                end
                for (i,x) in enumerate(order_parameters)
                    ord_p[i] += x(state)
                end
            end
        end
    end
    if length(order_parameters) == 0
      return samples
    elseif return_samples == false
        return ord_p
    end
    return samples[1:n_samples], ord_p
end

function MetropolisHastings_sampler(E,N,p;β = 1, restarts = 64, burnin = 1000, order_parameters = [], return_samples = true, init = false)
    samples =  []
    if init == false
        init = () -> rand([1,-1],p)
    end
    ns_list = repeat([div(N,restarts)], restarts)
    ord_p = zeros(length(order_parameters))
    if rem(N,restarts) > 0
        push!(ns_list,rem(N,restarts))
    end
    for k in 1:restarts
        state = init()
        #push!(samples, deepcopy(state))
        for t = 1:(ns_list[k]+burnin)-1
            #state = samples[end]
            α = rand(1:p)
            flipped_state = deepcopy(state)
            flipped_state[α] = -1*state[α]
            dE = E(flipped_state) - E(state)
            if dE <= 0
                state = flipped_state
            else
                prob = exp(-β*dE)
                if rand() < prob
                    state = flipped_state
                end
            end

            if t >  burnin-1
                if return_samples
                    push!(samples, deepcopy(state))
                end
                for (i,x) in enumerate(order_parameters)
                    ord_p[i] += x(state)
                end
            end
        end
    end
    if length(order_parameters) == 0
      return samples
    elseif return_samples == false
        return ord_p
    end
    return samples, ord_p
end




#=
function Gibbs_sampler(E,N,p;β = 1)
    samples =  []
    state = rand([1,-1],p)
    push!(samples, deepcopy(state))
    for t = 1:N-1
        #state = samples[end]
        for i in 1:p
            flipped_state = deepcopy(state)
            flipped_state[i] = -1*state[i]
            dE = E(state) - E(flipped_state)
            prob = 1/(1 + exp(β*dE*state[i]))
            state[i] = wsample([1,-1],[prob,1-prob])
        end
        push!(samples, deepcopy(state))
    end
    return samples
end
=#

function normalize(S::Dict)
    S = convert(Dict{Vector{Int64}, Float64}, S)
    L = sum(values(S))
    map!(x ->x/L, values(S))
    return  S
end


function sample_TV(samples1, samples2)
    S1 =  countmap(samples1) |> normalize
    S2 = countmap(samples2) |> normalize
    TV = 0.0
    for s  in keys(S1)
        if s in keys(S2)
            TV += abs(S1[s] - S2[s])
        else
             TV += S1[s]
        end
    end

    for s in keys(S2)
        if !(s in keys(S1))
            TV += S2[s]
        end
    end
    return  TV
end
#=

for n_samples in 4 .^ Array(1:9)
    @show n_samples
    local S1 = exact_sampler(Energy, n_samples, n_spins)
    local S2 = Gibbs_sampler(Energy, n_samples, n_spins)
    @show sample_TV(S1, S2)
end
=#
function random_p_hyergraph( nnodes, avg_edges; p = 3)
    node_list = Array(1:nnodes)
    possible_edges =  combinations(1:nnodes, p)
    prob = avg_edges/length(possible_edges)
    Edges = []
    for e in possible_edges
        if rand() < prob
            push!(Edges, e)
        end
    end
    return Edges
end

function SpinGlassEnergy(σ, Edges, J::Vector)
    T = map(x -> reduce(*, σ[x]), Edges )
    return  J'*T
end
function Energy(σ, F)
    E = 0
    for edge in keys(F)
        E += F[edge]*reduce(*,[σ[u] for u in edge ])
    end
    return E
end


Energy(σ, F::FactorGraph) = Energy(σ, F.terms)


FerroEnergy(σ, Edges; J= 1) =  SpinGlassEnergy(σ, Edges, -J*ones(length(Edges)))
AntiFerroEnergy(σ, Edges; J = 1) =  SpinGlassEnergy(σ, Edges, J*ones(length(Edges)))

function energy_hist(n_spins, Efn)
    E_list = []
    for k in 0:(2^n_spins) - 1
        σ = digits(k, base=2, pad = n_spins)
        σ = 2*σ .- 1
        push!(E_list, Efn(σ))
    end
    return E_list
 
end










