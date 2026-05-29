using GraphicalModelLearning, LinearAlgebra, DataFrames
using JuMP,Ipopt
using Plots
import Base.*

function data_info(samples::Array{T,2}) where T <: Real
    (num_conf, num_row) = size(samples)
    num_spins = num_row - 1
    num_samples = sum(samples[1:num_conf,1])
    return num_conf, num_spins, num_samples
end




function  learn_logsqdiv(edge_list, data_dist; method= Ipopt.Optimizer(), Linf_bound=10, L1_local_bound=Inf) 
    states = collect(keys(data_dist))
    num_spins = length(states[1])
    num_states = length(states)

    nodal_stat = Dict{Tuple,Any}()
    for edge in edge_list
        nodal_stat[edge] = [prod(states[k][i] for i in edge) for k in 1:num_states]
    end

    model = Model(method.solver, add_bridges=false)
    JuMP.@variable(model,-Linf_bound<= x[edge_list]<=Linf_bound)
    JuMP.@variable(model, F)
    @objective(model, Min,
                 sum( data_dist[states[ind]]*( sum(x[edge]*nodal_stat[edge][ind] for edge in edge_list)- F
                       - log(data_dist[states[ind]]) )^2
                                       for ind in 1:num_states))
    if L1_local_bound != Inf
        JuMP.@variable(model, z[edge_list])
        for inter in edge_list
            @constraint(model, z[inter] >=  x[inter]) #z_plus
            @constraint(model, z[inter] >= -x[inter]) #z_minus
        end
        for current_spin in 1:num_spins
            involved_edges = [edge  for edge  in edge_list if current_spin in  edge  ]
            @constraint(model, sum(z[inter] for inter in involved_edges) <= L1_local_bound )
        end

    end

    JuMP.optimize!(model)
    reconstruction = Dict{Tuple, Float64}(Tuple.(edge_list) .=> JuMP.value.(x)[edge_list])
    inter_order = max(length(edge_list)...)
    Graph_learned =FactorGraph(inter_order, num_spins, :spin, reconstruction)
    return Graph_learned
    
end



function  learn_logsqdiv_explicit(edge_list, data_dist; method= Ipopt.Optimizer(), Linf_bound=10, L1_local_bound=Inf) 
    ##Compute the Coefficient matrix explicitly before passing to jump
    num_spins = length(collect(keys(data_dist))[1])
    data_dist = dict_2_array(data_dist, num_spins)
    num_states = size(data_dist)[1]
    states = [data_dist[i,2:end] for i in 1:num_states]
    probs = data_dist[:,1]
    logprobs = log.(probs)
    plogp = probs .* logprobs
    num_states = length(states)
    push!(edge_list,())
    num_vars = length(edge_list)

    A_mat = Matrix{Float64}(I,length(edge_list), length(edge_list))
    for (I,edge1) in enumerate(edge_list), (J,edge2) in enumerate(edge_list)
        if I < J
            edge = [i for i in edge1]
            [push!(edge,i) for i in edge2]
            edge_stats =   [prod(states[k][i] for i in edge) for k in 1:num_states]
            A_mat[I,J] = probs'*edge_stats 
            A_mat[J,I] = A_mat[I,J]
        end
    end

    b_vec = zeros(length(edge_list))
    for (I,edge) in enumerate(edge_list)
        if edge == ()
            b_vec[I] = -2*sum(plogp)
        else
            edge_stats =   [prod(states[k][i] for i in edge) for k in 1:num_states]
            b_vec[I] =  -2*plogp'*edge_stats
        end
    end
                  


    model = Model(method.solver, add_bridges=false)
    JuMP.@variable(model,x[1:num_vars])
    for ind in 1:(num_vars-1)
      @constraint(model, x[ind] <= Linf_bound)
      @constraint(model, x[ind] >= -1*Linf_bound)
    end

    @objective(model, Min, x'*A_mat*x + b_vec'*x)
              
    if L1_local_bound != Inf
        JuMP.@variable(model, z[1:num_vars])
        for inter in 1:num_vars
            @constraint(model, z[inter] >=  x[inter]) #z_plus
            @constraint(model, z[inter] >= -x[inter]) #z_minus
        end
        for current_spin in 1:num_spins
            involved_edges = [edge  for edge  in edge_list if current_spin in  edge  ]
            positions = [ind for ind in 1:num_vars if edge_list[ind] in involved_edges]
            @constraint(model, sum(z[inter] for inter in positions) <= L1_local_bound )
        end

    end

    JuMP.optimize!(model)
    reconstruction = Dict{Tuple, Float64}()
    for inter in 1:(num_vars-1)
        @assert edge_list[inter] != ()
        reconstruction[edge_list[inter]]= JuMP.value(x[inter])
    end
    
    inter_order = max(length(edge_list)...)
    println("Learned Free Eneregy $(-1*JuMP.value(x[num_vars]))")
    Graph_learned =FactorGraph(inter_order, num_spins, :spin, reconstruction)
    return Graph_learned
    
end
function  learn_globalrise(edge_list, data_dist; method= Ipopt.Optimizer(), Linf_bound=10, L1_local_bound=Inf) 
    states = collect(keys(data_dist))
    num_spins = length(states[1])
    num_states = length(states)

    nodal_stat = Dict{Tuple,Any}()
    for edge in edge_list
        nodal_stat[edge] = [prod(states[k][i] for i in edge) for k in 1:num_states]
    end

    model = Model(method.solver)
    JuMP.@variable(model,-Linf_bound<= x[edge_list]<=Linf_bound)
    JuMP.@variable(model, z[edge_list])
    JuMP.@variable(model, F)
    @NLobjective(model, Min,
                 sum( data_dist[states[ind]]*( exp(sum(-1*x[edge]*nodal_stat[edge][ind] for edge in edge_list) ))
                                       for ind in 1:num_states))
    if L1_local_bound != Inf
        for inter in edge_list
            @constraint(model, z[inter] >=  x[inter]) #z_plus
            @constraint(model, z[inter] >= -x[inter]) #z_minus
        end
        for current_spin in 1:num_spins
            involved_edges = [edge  for edge  in edge_list if current_spin in  edge  ]
            @constraint(model, sum(z[inter] for inter in involved_edges) <= L1_local_bound )
        end

    end

    JuMP.optimize!(model)
    reconstruction = Dict{Tuple, Float64}(Tuple.(edge_list) .=> JuMP.value.(x)[edge_list])
    inter_order = max(length(edge_list)...)
    Graph_learned =FactorGraph(inter_order, num_spins, :spin, reconstruction)
    return Graph_learned
    
end





function learn_structured(edge_list, samples::Matrix{T}, formulation::multiRISE, method::NLP;
        Linf_bound = 10, L1_local_bound=Inf) where T<:Real

    num_conf, num_spins, num_samples = data_info(samples)
    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)
    inter_order = formulation.interaction_order
    @assert max(length.(edge_list)...)  == inter_order
    reconstruction = Dict{Tuple,Real}()

    for current_spin = 1:num_spins
        nodal_stat = Dict{Tuple,Array{Real,1}}()
        involved_edges = [edge  for edge  in edge_list if current_spin in  edge  ]
        for edge in involved_edges
            nodal_stat[edge] = [prod(samples[k,i+1] for i in edge) for k in 1:num_conf]
        end
        ################################

        model = Model(method.solver)
        JuMP.@variable(model,-Linf_bound<= x[keys(nodal_stat)]<=Linf_bound)
        JuMP.@variable(model, z[keys(nodal_stat)])

        @NLobjective(model, Min,
            sum((samples[k,1]/num_samples)*exp(-sum(x[inter]*stat[k] for (inter,stat) = nodal_stat)) for k=1:num_conf) +
            lambda*sum(z[inter] for inter = keys(nodal_stat) if length(inter)>1)
        )

        for inter in keys(nodal_stat)
            @constraint(model, z[inter] >=  x[inter]) #z_plus
            @constraint(model, z[inter] >= -x[inter]) #z_minus
        end
        if L1_local_bound != Inf
            involved_edges = [edge  for edge  in edge_list if current_spin in  edge  ]
            @constraint(model, sum(z[inter] for inter in involved_edges) <= L1_local_bound )
        end



        JuMP.optimize!(model)
        if JuMP.termination_status(model) != JuMP.MOI.LOCALLY_SOLVED
            return nothing
        end

        nodal_reconstruction = JuMP.value.(x)
        for inter = keys(nodal_stat)
            reconstruction[inter] = deepcopy(nodal_reconstruction[inter])
        end
    end

    if formulation.symmetrization
        reconstruction_list = Dict{Tuple,Vector{Real}}()
        for (k,v) in reconstruction
            key = tuple(sort([i for i in k])...)
            if !haskey(reconstruction_list, key)
                reconstruction_list[key] = Vector{Real}()
            end
            push!(reconstruction_list[key], v)
        end

        reconstruction = Dict{Tuple,Real}()
        for (k,v) in reconstruction_list
            reconstruction[k] = mean(v)
        end
    end
    return FactorGraph(inter_order, num_spins, :spin, reconstruction)
end

function  learn_structured_sumoflocal(edge_list, samples::Matrix{T}, formulation::multiRISE, method::NLP; Linf_bound=10, L1_local_bound = Inf) where T<:Real
    num_conf, num_spins, num_samples = data_info(samples)
    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)
    inter_order = formulation.interaction_order
    @assert max(length.(edge_list)...)  == inter_order
    reconstruction = Dict{Tuple,Real}()
    involved_edges = Dict{Int,Any}()
    for current_spin = 1:num_spins
        involved_edges[current_spin] = [edge  for edge  in edge_list if current_spin in  edge  ]
    end
    nodal_stat = Dict{Tuple, Array{Real,1}}()
    for edge in edge_list
            nodal_stat[edge] = [prod(samples[k,i+1] for i in edge) for k in 1:num_conf]
    end

       ################################

        model = Model(method.solver)
        JuMP.@variable(model, -Linf_bound<=x[keys(nodal_stat)]<=Linf_bound)
        JuMP.@variable(model, z[keys(nodal_stat)])

        @NLobjective(model, Min,
                     sum(
                     sum((samples[k,1]/num_samples)*exp(-sum(x[inter]*nodal_stat[inter][k] for inter = involved_edges[u])) for k=1:num_conf)
                     for u in 1:num_spins)+
            lambda*sum(z[inter] for inter = keys(nodal_stat) if length(inter)>1)
        )

        for inter in keys(nodal_stat)
            @constraint(model, z[inter] >=  x[inter]) #z_plus
            @constraint(model, z[inter] >= -x[inter]) #z_minus
        end
        if L1_local_bound != Inf
            for current_spin in 1:num_spins
                involved_edges = [edge  for edge  in edge_list if current_spin in  edge  ]
                @constraint(model, sum(z[inter] for inter in involved_edges) <= L1_local_bound )
            end
        end


        JuMP.optimize!(model)
        if JuMP.termination_status(model) != JuMP.MOI.LOCALLY_SOLVED
            return nothing
        end

        nodal_reconstruction = JuMP.value.(x)
        for inter = keys(nodal_stat)
            reconstruction[inter] = deepcopy(nodal_reconstruction[inter])
        end

        return FactorGraph(inter_order, num_spins, :spin, reconstruction)
end



function  learn_sumoflocal_PLE(edge_list, samples::Matrix{T}, lambda::Real, method::NLP; Linf_bound=10, L1_local_bound = Inf) where T<:Real
    num_conf, num_spins, num_samples = data_info(samples)
 #   inter_order = formulation.interaction_order
    inter_order = max(length.(edge_list)...)
    reconstruction = Dict{Tuple,Real}()
    involved_edges = Dict{Int,Any}()
    for current_spin = 1:num_spins
        involved_edges[current_spin] = [edge  for edge  in edge_list if current_spin in  edge  ]
    end
    nodal_stat = Dict{Tuple, Array{Real,1}}()
    for edge in edge_list
            nodal_stat[edge] = [-2*prod(samples[k,i+1] for i in edge) for k in 1:num_conf]
    end

       ################################

        model = Model(method.solver)
        JuMP.@variable(model, -Linf_bound<=x[keys(nodal_stat)]<=Linf_bound)
        JuMP.@variable(model, z[keys(nodal_stat)]) ##For L1 regularization

        for inter in keys(nodal_stat)
          @constraint(model, z[inter] >=  x[inter]) #z_plus
          @constraint(model, z[inter] >= -x[inter]) #z_minus
        end

       normalization = 1.0/(num_samples*num_spins)

        @NLobjective(model, Min,
                     normalization*sum(
                     sum((samples[k,1])*
                         log( 1+
                            exp(sum(x[inter]*nodal_stat[inter][k] for inter = involved_edges[u]))) for k=1:num_conf)
                     for u in 1:num_spins)
#           + lambda*sum(z[inter] for inter = keys(nodal_stat))
        )
        if L1_local_bound != Inf
            for current_spin in 1:num_spins
              #  involved_edges = [edge  for edge  in edge_list if current_spin in  edge  ]
                @constraint(model, sum(z[inter] for inter in involved_edges[current_spin]) <= L1_local_bound )
            end
        end


        JuMP.optimize!(model)
        if JuMP.termination_status(model) != JuMP.MOI.LOCALLY_SOLVED
            println("Optimization failed!")
            return nothing
        end

        nodal_reconstruction = JuMP.value.(x)
        for inter = keys(nodal_stat)
            reconstruction[inter] = deepcopy(nodal_reconstruction[inter])
        end

        return FactorGraph(inter_order, num_spins, :spin, reconstruction)
end






function  learn_structured_constrained(edge_list, ground_states, samples::Matrix{T}, formulation::multiRISE, method::NLP; Linf_bound=10) where T<:Real
    num_conf, num_spins, num_samples = data_info(samples)
    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)
    inter_order = formulation.interaction_order
    @assert max(length.(edge_list)...)  == inter_order
    reconstruction = Dict{Tuple,Real}()
    involved_edges = Dict{Int,Any}()
    for current_spin = 1:num_spins
        involved_edges[current_spin] = [edge  for edge  in edge_list if current_spin in  edge  ]
    end
    nodal_stat = Dict{Tuple, Array{Real,1}}()
    for edge in edge_list
            nodal_stat[edge] = [prod(samples[k,i+1] for i in edge) for k in 1:num_conf]
    end

       ################################

        model = Model(method.solver)
        JuMP.@variable(model, -Linf_bound<=x[keys(nodal_stat)]<=Linf_bound)
        JuMP.@variable(model, z[keys(nodal_stat)])

        @NLobjective(model, Min,
                     sum(
                     sum((samples[k,1]/num_samples)*exp(-sum(x[inter]*nodal_stat[inter][k] for inter = involved_edges[u])) for k=1:num_conf)
                     for u in 1:num_spins)+
            lambda*sum(z[inter] for inter = keys(nodal_stat) if length(inter)>1)
        )
        ngs = length(ground_states)
        for ig  in 1:ngs-1
            gs = ground_states[ig]
            gs_next = ground_states[ig+1]
            stats_gs = [ prod(gs[i] for i in edge) for edge in edge_list]
            stats_gs_next = [ prod(gs_next[i] for i in edge) for edge in edge_list]

            stats_gs = Dict(edge_list .=> stats_gs)
            stats_gs_next = Dict(edge_list .=> stats_gs_next)

            @constraint(model, sum((stats_gs[edge] - stats_gs_next[edge] )*x[edge] for edge in edge_list) <= 0.001)
            @constraint(model, sum((stats_gs[edge] - stats_gs_next[edge] )*x[edge] for edge in edge_list) >= -0.001)
           end
            

        for inter in keys(nodal_stat)
            @constraint(model, z[inter] >=  x[inter]) #z_plus
            @constraint(model, z[inter] >= -x[inter]) #z_minus
        end

        JuMP.optimize!(model)
        if JuMP.termination_status(model) != JuMP.MOI.LOCALLY_SOLVED
            return nothing
        end

        nodal_reconstruction = JuMP.value.(x)
        for inter = keys(nodal_stat)
            reconstruction[inter] = deepcopy(nodal_reconstruction[inter])
        end

        return FactorGraph(inter_order, num_spins, :spin, reconstruction)
end






function learn_two_step(samples::Matrix{T}, formulation::multiRISE, method::NLP, threshold::Real; plot_histrogram = true) where T<:Real
    F1 = learn(samples, formulation, method)
    Edge_list = []
    for k in keys(F1.terms)
        if abs(F1.terms[k]) > threshold
            push!(Edge_list, k)
        end
    end
    if plot_histrogram
        histogram([abs.(values(Graph_learned.terms))...], yscale=:log10)
        vline!([threshold], color=:red, label="threshold")
    end
    p = max(length.(Edge_list))
    M = multiRISE(1e-4,true,p) ##Reg is zero as we already have the structure here

    F2 = learn_structured(Edge_list, samples, formulation, method)
    return F2
end

function learn_two_step(Edge_list, samples::Matrix{T}, formulation::multiRISE, method::NLP, threshold::Real; plot_histrogram = true) where T<:Real
    F1 = learn_structured(Edge_list, samples, formulation, method)
    Edge_list_new = []
    for k in keys(F1.terms)
        if abs(F1.terms[k]) > threshold
            push!(Edge_list_new, k)
        end
    end
    if plot_histrogram
        histogram([abs.(values(F1.terms))...], yscale=:log10)
        vline!([threshold], color=:red, label="threshold")
    end
    p = max(length.(Edge_list))
    M = multiRISE(1e-4,true,p) ##Reg is zero as we already have the structure here
    F2 = learn_structured(Edge_list_new, samples, formulation, method)
    return F2
end




function dict_2_array(A,n)
    S = zeros(length(A),n+1)
    f = 1
    for (k,v) in A
        S[f,2:n+1] .= k
        S[f,1] = v
        f += 1
    end

    return S

end

function l1_norm(dict::Dict)
    norm = 0.0

    for value in values(dict)
        norm += abs(value)
    end

    return norm
end

function max_norm(dict::Dict)
    norm = 0.0

    for value in values(dict)
        norm = max(norm,abs(value))
    end

    return norm
end




function  *(a::Real, F::FactorGraph)
    for k in keys(F.terms)
        F.terms[k] = a*F.terms[k]
    end
  end




function l1_norm_error(dict1::Dict, dict2::Dict)
    error = 0.0
    all_keys = union(keys(dict1), keys(dict2))
    for key in all_keys
        value1 = get(dict1, key, 0)
        value2 = get(dict2, key, 0)
        error += abs(value1 - value2)
    end
    return error
end
function max_norm_error(dict1::Dict, dict2::Dict; return_key = false)
    error = 0.0
    k = ()
    all_keys = union(keys(dict1), keys(dict2))
    for key in all_keys
        value1 = get(dict1, key, 0)
        value2 = get(dict2, key, 0)
        ev = abs(value1 - value2)
        if  ev > error
            k = key
            error =  ev
        end
    end
    if return_key
        return k,error
    end
    return error
end

function all_spin_states(ns)
    states = Array(0:(2^ns)-1)
    states = map( x -> digits(x, base=2, pad=ns) |> reverse, states)
    states = map(x -> 2*x .- 1, states)
    return  states
end

function  full_distribution(Energy ,ns,β)
    states = all_spin_states(ns)
    wts =  exp.(-1*β*Energy.(states))
    wts = wts/max(wts...)   ##Normalizing to avoid overflow
    prob = wts/sum(wts)
    return Dict(states .=> prob)
end


function TV_exact(Energy1, Energy2, ns)
    states = Array(0:(2^ns)-1)
    states = map( x -> digits(x, base=2, pad=ns) |> reverse, states)
   states = map(x -> 2*x .- 1, states)
    wts1 =  exp.(-1*Energy1.(states))
    wts1 = wts1/max(wts1...)   ##Normalizing to avoid overflow
    prob1 = wts1/sum(wts1)
    wts2 =  exp.(-1*Energy2.(states))
    wts2 =  wts2/max(wts2...)
    prob2 = wts2/sum(wts2)
    return 0.5*sum(abs.(prob1 - prob2))
end

function Error_p_body(p,Energy1, Energy2,ns; ord = Inf)
    all_p_body = []
    for k in 1:p
        push!(all_p_body,Tuple.(collect(combinations(1:ns,k)))...)
    end
    states = Array(0:(2^ns)-1)
    states = map( x -> digits(x, base=2, pad=ns) |> reverse, states)
    states = map(x -> 2*x .- 1, states)
    wts1 =  exp.(-1*Energy1.(states))
    wts1 = wts1/max(wts1...) #Normalizing to avoid overflow
    prob1 = wts1/sum(wts1)
    wts2 =  exp.(-1*Energy2.(states))
    wts2 = wts2/max(wts2...)
    prob2 = wts2/sum(wts2)
    AA = zeros(2^ns, length(all_p_body))
    for (i,y)  in enumerate(all_p_body), (j,x) in  enumerate(states)
        AA[j,i] = reduce(*,[x[k] for k in y])
    end
    E1 = prob1'*AA
    E2 = prob2'*AA
    return  norm(E1-E2, ord)

end


function find_n_star(df, n_spins,n_edges, β_list, err::Symbol, thresh;ord = 2, outlier_detection = nothing)

    plots_list = []
    n_star= []
    for β in β_list
        @show β
        df_plot = df[(df.β .== β).& (df.n_spins .== n_spins), :]
        p =plot(df_plot.n_samples, df_plot[!,err], seriestype=:scatter, xscale=:log2, mswidth = 0.1, markersize = 1, label="data")

        gdf = DataFrames.groupby(df_plot, [:n_spins, :n_edges, :β, :n_samples])
        df_avg = combine(gdf, [err] .=> mean
                         , renamecols = false)

        plot!(p,df_avg.n_samples, df_avg[!,err], seriestype=:scatter, xscale=:log2, color=:red, mswidth= 0.1, markersize=2, label="Mean")
        hline!(p,[thresh], color=:red,label=nothing)
        xlabel!(p,L"n")
        ylabel!(p,L"%$err")
        title!(p, "β = $β", titlefontsize =8)

        X = df_avg.n_samples .|> log
        v = sortperm(X)
        X = X[v]
        Y = df_avg[!,err]
        Y = Y[v]
        if outlier_detection != nothing
            inds = [ i for i in 1:length(Y) if outlier_detection(Y[i])]
            X = X[inds]
            Y = Y[inds]
        end
        f = Polynomials.fit(X,Y,ord)
        plot!(p,exp.(X), f.(X), xscale=:log2, color=:blue, label="Fit")
        nn = roots(f - thresh)
        nn = [real(x) for x in nn if isreal(x)]
        nn = [x for x in  nn if 0 < x < max(X...)] ##To avoid spurious roots
        @show β, nn
        push!(n_star, exp(min(nn...)) )
        push!(plots_list, deepcopy(p))
    end
    P1 =plot(plots_list..., size=(800,600))
#    savefig(P1,"test.pdf")
#    display(P1)
    return P1, n_star


end
function Free_energy(Energy_fn,ns;β =1)
    states = all_spin_states(ns)
    Z = sum( exp.(-β*Energy_fn.(states)))
    return log(Z)
end


function symmetric_difference_size(list1, list2)::Int
    set1 = Set(list1)
    set2 = Set(list2)
    diff1 = setdiff(set1, set2)
    diff2 = setdiff(set2, set1)
    return length(union(diff1, diff2))
end



function plot_model(F::FactorGraph)
    
end

"""
  log_histogram(samples, f, domain)

    This applies the function f (order parmeter) to the samples and gives the log frequency. Returns a dataframe. `domain` is an array that has the total domain of the function

"""
function log_histogram(samples::Dict, f::Function, domain; bins=nothing)
    Fs = Dict()
    for (k, v) in samples
        fk = f(k)
        if !(fk in keys(Fs))
            Fs[fk] = v
        else
            Fs[fk] += v
        end
    end

    for (k, v) in Fs
        Fs[k] = log(v)
    end

    for x in domain
        if !(x in keys(Fs))
            Fs[x] = 0.0
        end
    end
    if bins != nothing
        Fs = merge_bins(Fs, bins)
    end
    df = DataFrame(x=collect(keys(Fs)), y=collect(values(Fs)))
    a = sortperm(df, :x)
    return df[a, :]

end


function coarse_grain_histogram(values::Vector, probabilities::Vector, num_bins::Int)
    bin_edges = range(minimum(values), maximum(values), length=num_bins+1)
    binned_values = [mean(values[inbin]) for inbin in [(bin_edges[i] .<= values .< bin_edges[i+1]) for i in 1:num_bins]]
    binned_probabilities = [sum(probabilities[inbin]) for inbin in [(bin_edges[i] .<= values .< bin_edges[i+1]) for i in 1:num_bins]]
    return binned_values, binned_probabilities
end



function merge_bins(data::Dict{Any,Any}, num_bins::Int)
    keys_sorted = sort(collect(keys(data)))  # Sort the keys
    n = length(keys_sorted)

    # Calculate approximately how many keys per bin
    bin_size = ceil(Int, n / num_bins)

    # Resulting dictionary of bins
    binned_data = Dict{Any,Any}()

    for i in 1:num_bins
        # Determine the range of indices for this bin
        start_idx = (i - 1) * bin_size + 1
        end_idx = min(i * bin_size, n)

        # Merge keys in the current bin
        bin_key = mean(keys_sorted[start_idx:end_idx])
        bin_value = sum(data[k] for k in keys_sorted[start_idx:end_idx])

        binned_data[bin_key] = bin_value
    end

    return binned_data
end


function bisection(f, a, b; tol=1e-10, max_iter=1000)
    if f(a) * f(b) >= 0
        error("Function does not change sign over the interval. Bisection method cannot be applied.")
    end

    for _ in 1:max_iter
        mid = (a + b) / 2  # Calculate the midpoint of the interval
        if abs(f(mid)) < tol  # Check if the midpoint is close enough to be considered a root
            return mid
        elseif f(a) * f(mid) < 0  # Determine the subinterval containing the root
            b = mid
        else
            a = mid
        end
    end

    error("Maximum iterations reached without convergence.")
end






