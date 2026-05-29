using GraphicalModelLearning

# Potts analogue of `learn_sumoflocal_PLE`:
# one scalar coupling per candidate hyperedge, with a fixed local tensor `tensor`.
# The learned coefficients are "effective couplings" (they absorb the inverse temperature β).
function learn_sumoflocal_PLE(
    edge_list,
    samples::Matrix{T},
    q::Int,
    tensor,
    lambda::Real,
    method::NLP;
    Linf_bound = 10,
    L1_local_bound = Inf,
) where {T<:Real}
    num_conf = size(samples, 1)
    num_spins = size(samples, 2) - 1
    num_samples = sum(samples[:, 1])
    inter_order = maximum(length.(edge_list))

    @assert ndims(tensor) == inter_order
    @assert all(size(tensor, d) == q for d in 1:inter_order)

    # `dict_2_array` stores counts in column 1 and spin states in columns 2:end.
    counts = Float64.(samples[:, 1])
    states = round.(Int, samples[:, 2:end])

    involved_edges = Dict{Int, Any}()
    for u in 1:num_spins
        involved_edges[u] = [edge for edge in edge_list if u in edge]
    end

    # For each (vertex, edge), precompute tensor values for all candidate states of that vertex.
    # This keeps the nonlinear objective readable and avoids repeated tensor indexing logic.
    local_stat = Dict{Tuple{Int, Tuple}, Matrix{Float64}}()
    local_obs_stat = Dict{Tuple{Int, Tuple}, Vector{Float64}}()
    for u in 1:num_spins
        for edge in involved_edges[u]
            edge_t = Tuple(edge)
            pos_u = findfirst(==(u), edge_t)
            vals = zeros(num_conf, q)
            vals_obs = zeros(num_conf)
            for k in 1:num_conf
                edge_states = [states[k, i] for i in edge_t]
                for a in 1:q
                    edge_states[pos_u] = a
                    vals[k, a] = tensor[Tuple(edge_states)...]
                end
                vals_obs[k] = vals[k, states[k, u]]
            end
            local_stat[(u, edge_t)] = vals
            local_obs_stat[(u, edge_t)] = vals_obs
        end
    end

    model = Model(method.solver)
    JuMP.@variable(model, -Linf_bound <= x[edge_list] <= Linf_bound)
    JuMP.@variable(model, z[edge_list] >= 0)

    for edge in edge_list
        JuMP.@constraint(model, z[edge] >= x[edge])
        JuMP.@constraint(model, z[edge] >= -x[edge])
    end

    normalization = 1.0 / (num_samples * num_spins)

    # Negative pseudo-log-likelihood for q-state conditionals:
    # log(sum_a exp(-E_u(a | rest))) + E_u(observed | rest)
    JuMP.@NLobjective(
        model,
        Min,
        normalization * sum(
            sum(
                counts[k] * (
                    log(
                        sum(
                            exp(
                                -sum(
                                    x[inter] * local_stat[(u, inter)][k, a] for inter in involved_edges[u]
                                )
                            ) for a in 1:q
                        )
                    ) +
                    sum(x[inter] * local_obs_stat[(u, inter)][k] for inter in involved_edges[u])
                ) for k in 1:num_conf
            ) for u in 1:num_spins
        ) + lambda * sum(z[edge] for edge in edge_list)
    )

    if L1_local_bound != Inf
        for u in 1:num_spins
            JuMP.@constraint(model, sum(z[edge] for edge in involved_edges[u]) <= L1_local_bound)
        end
    end

    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    if status != JuMP.MOI.LOCALLY_SOLVED && status != JuMP.MOI.OPTIMAL
        println("Optimization failed! status = ", status)
        return nothing
    end

    reconstruction = Dict{Tuple, Real}()
    nodal_reconstruction = JuMP.value.(x)
    for edge in edge_list
        reconstruction[Tuple(edge)] = deepcopy(nodal_reconstruction[edge])
    end

    return FactorGraph(inter_order, num_spins, :spin, reconstruction)
end
