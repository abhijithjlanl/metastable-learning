using Pkg
const SCRIPT_DIR = @__DIR__
Pkg.activate(joinpath(SCRIPT_DIR, "..", "..", ".."))

using GraphicalModelLearning
using LaTeXStrings
using Random
using StatsBase
using Combinatorics
using JLD2

include(joinpath(SCRIPT_DIR, "..", "..", "..", "samplers.jl"))
include(joinpath(SCRIPT_DIR, "..", "..", "..", "utils.jl"))
include(joinpath(SCRIPT_DIR, "..", "..", "..", "PottsPSpin", "potts_tensors.jl"))
include(joinpath(SCRIPT_DIR, "..", "..", "..", "PottsPSpin", "potts_samplers.jl"))
include(joinpath(SCRIPT_DIR, "..", "potts_sumoflocal_PLE.jl"))

# ---------------------------------------------------------------
# Fixed model parameters
# ---------------------------------------------------------------
const ns            = 24
const q             = 3
const β             = 1.2
const J_edge        = -1.0
const true_eff_coup = β * J_edge       # -1.2
const n_edges       = div(3 * ns, 2)   # 36
const prior_mult    = 2                # |prior| = 72
const support_thr   = 0.1 * abs(true_eff_coup)   # 0.12
const model_type    = :Ferro1

# Sweep parameters
const M_values = [1_000, 3_000, 10_000, 30_000, 100_000]
const n_trials = 5
const M_big    = 100_000

Optimizer = NLP(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
function build_prior(Edges_t, ns, n_edges, prior_mult)
    all_p_body   = Tuple.(collect(combinations(1:ns, 3)))
    edges_prior  = collect(Edges_t)
    target       = prior_mult * n_edges
    while length(edges_prior) < target
        e = rand(all_p_body)
        e ∉ edges_prior && push!(edges_prior, e)
    end
    return edges_prior
end

function support_stats(true_edges, learned_terms; atol = 1e-6)
    true_set    = Set(Tuple.(true_edges))
    learned_act = Set(Tuple(k) for (k, v) in learned_terms if abs(v) > atol)
    tp  = length(intersect(true_set, learned_act))
    fp  = length(setdiff(learned_act, true_set))
    fn  = length(setdiff(true_set, learned_act))
    prec = (tp + fp) == 0 ? 0.0 : tp / (tp + fp)
    rec  = (tp + fn) == 0 ? 0.0 : tp / (tp + fn)
    return prec, rec
end

function run_learning(edge_list, samples_cm, ns, q, Φ, Optimizer, max_deg_prior)
    X = dict_2_array(samples_cm, ns)
    F = learn_sumoflocal_PLE(
        edge_list, X, q, Φ, 0.0, Optimizer;
        L1_local_bound = abs(true_eff_coup) * max_deg_prior,
    )
    return F
end

# ---------------------------------------------------------------
# Result arrays  shape: (n_trials, length(M_values))
# ---------------------------------------------------------------
errs_min  = fill(NaN, n_trials, length(M_values))
errs_rand = fill(NaN, n_trials, length(M_values))
prec_min  = fill(NaN, n_trials, length(M_values))
prec_rand = fill(NaN, n_trials, length(M_values))
rec_min   = fill(NaN, n_trials, length(M_values))
rec_rand  = fill(NaN, n_trials, length(M_values))

# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
Φ = model_type == :Ferro1 ? threealphabet_ferro1(q) : threealphabet_ferro2(q)

for trial in 1:n_trials
    @show trial

    # Draw a fresh random hypergraph
    local Edges = random_p_hyergraph(ns, n_edges, p = 3)
    while length(Edges) != n_edges
        Edges = random_p_hyergraph(ns, n_edges, p = 3)
    end
    Edges_t = Tuple.(Edges)

    Energy(σ) = HypergraphPottsEnergy(σ, Edges, Φ, J_edge)
    TrueF     = Dict(e => true_eff_coup for e in Edges_t)

    # Build candidate prior (same for all M in this trial)
    edge_list     = build_prior(Edges_t, ns, n_edges, prior_mult)
    max_deg_prior = maximum(sum(i ∈ e for e in edge_list) for i in 1:ns)

    # ---- collect samples at the largest M first, then subsample ----
    # GlauberMin: 4 restarts from all-ones
    samples_min_big = Glauber_sampler(
        Energy, M_big, ns, q;
        β = β, restarts = 4, init = () -> ones(Int, ns),
    )
    # GlauberRand: 16 restarts from random states
    n_restarts_rand  = max(8, div(16 * M_big, 10^5))
    samples_rand_big = Glauber_sampler(
        Energy, M_big, ns, q;
        β = β, restarts = n_restarts_rand,
    )

    # Sweep M values (subsample the big arrays)
    for (mi, M) in enumerate(M_values)
        @show trial, M

        sub_min  = countmap(samples_min_big[1:M])
        sub_rand = countmap(samples_rand_big[1:M])

        # Learning for GlauberMin
        F_min = run_learning(edge_list, sub_min, ns, q, Φ, Optimizer, max_deg_prior)
        if F_min !== nothing
            errs_min[trial, mi] = max_norm_error(TrueF, F_min.terms)
            p, r = support_stats(Edges_t, F_min.terms; atol = support_thr)
            prec_min[trial, mi] = p
            rec_min[trial, mi]  = r
        end

        # Learning for GlauberRand
        F_rand = run_learning(edge_list, sub_rand, ns, q, Φ, Optimizer, max_deg_prior)
        if F_rand !== nothing
            errs_rand[trial, mi] = max_norm_error(TrueF, F_rand.terms)
            p, r = support_stats(Edges_t, F_rand.terms; atol = support_thr)
            prec_rand[trial, mi] = p
            rec_rand[trial, mi]  = r
        end
    end

    # Save incrementally after each trial
    outfile = joinpath(SCRIPT_DIR, "results_panelb.jld2")
    jldopen(outfile, "w") do f
        f["M_values"]      = M_values
        f["β"]             = β
        f["errs_min"]      = errs_min
        f["errs_rand"]     = errs_rand
        f["prec_min"]      = prec_min
        f["prec_rand"]     = prec_rand
        f["rec_min"]       = rec_min
        f["rec_rand"]      = rec_rand
        f["n_trials_done"] = trial
    end
    @show "Trial $trial saved."
end

println("Done. Results saved to results_panelb.jld2")
println("Shapes — errs_min: ", size(errs_min), "  errs_rand: ", size(errs_rand))
