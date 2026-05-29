using Pkg
const SCRIPT_DIR = @__DIR__
Pkg.activate(joinpath(SCRIPT_DIR, "..", "..", ".."))

using Random
using StatsBase
using JLD2

include(joinpath(SCRIPT_DIR, "..", "..", "..", "samplers.jl"))
include(joinpath(SCRIPT_DIR, "..", "..", "..", "PottsPSpin", "potts_tensors.jl"))
include(joinpath(SCRIPT_DIR, "..", "..", "..", "PottsPSpin", "potts_samplers.jl"))

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
const n        = 12
const q        = 3
const J_edge   = -1.0
const n_edges  = div(3n, 2)   # 18
const n_trials = 5
const M        = 100_000

const β_values = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

Φ = threealphabet_ferro1(q)

# Result arrays: (n_trials, length(β_values))
energy_exact = fill(NaN, n_trials, length(β_values))
energy_rand  = fill(NaN, n_trials, length(β_values))
energy_min   = fill(NaN, n_trials, length(β_values))

# ---------------------------------------------------------------
# Precompute all 3^n states once (shared across β and trials)
# ---------------------------------------------------------------
all_states = [reverse(digits(x, base = q, pad = n)) .+ 1 for x in 0:q^n-1]
println("Total states: ", length(all_states))

# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
for trial in 1:n_trials
    @show trial

    # Fresh random hypergraph
    local Edges = random_p_hyergraph(n, n_edges, p = 3)
    while length(Edges) != n_edges
        Edges = random_p_hyergraph(n, n_edges, p = 3)
    end

    Energy(σ) = HypergraphPottsEnergy(σ, Edges, Φ, J_edge)

    # Precompute all energies for this graph (used by exact at every β)
    all_energies = [Energy(s) for s in all_states]

    for (bi, β) in enumerate(β_values)
        @show trial, β

        # --- Exact thermal average ---
        wts = exp.(-β .* all_energies)
        Z   = sum(wts)
        energy_exact[trial, bi] = sum(all_energies .* wts) / (Z * n)

        # --- GlauberRand: 16 restarts from random states ---
        samples_rand = Glauber_sampler(Energy, M, n, q; β = β, restarts = 16)
        cm_rand = countmap(samples_rand)
        energy_rand[trial, bi] = sum(Energy(k) * v for (k, v) in cm_rand) / (M * n)

        # --- GlauberMin: 4 restarts from all-ones (ferromagnetic ground state) ---
        samples_min = Glauber_sampler(
            Energy, M, n, q;
            β = β, restarts = 4, init = () -> ones(Int, n),
        )
        cm_min = countmap(samples_min)
        energy_min[trial, bi] = sum(Energy(k) * v for (k, v) in cm_min) / (M * n)
    end

    # Incremental save after each trial
    outfile = joinpath(SCRIPT_DIR, "results.jld2")
    jldopen(outfile, "w") do f
        f["β_values"]     = β_values
        f["energy_exact"] = energy_exact
        f["energy_rand"]  = energy_rand
        f["energy_min"]   = energy_min
        f["n_trials_done"] = trial
    end
    @show "Trial $trial saved."
end

println("Done. Results saved to results.jld2")
