using Pkg
const SCRIPT_DIR = @__DIR__
Pkg.activate(joinpath(SCRIPT_DIR, "..", "..", ".."))

using JLD2
using Plots
using LaTeXStrings
using Statistics

plot_font = "Computer Modern"
default(fontfamily = plot_font, linewidth = 2, framestyle = :box, grid = true)

# ---------------------------------------------------------------
# Load results
# ---------------------------------------------------------------
infile = joinpath(SCRIPT_DIR, "results.jld2")
β_values, energy_exact, energy_rand, energy_min, n_trials_done =
    jldopen(infile) do f
        f["β_values"],
        f["energy_exact"],
        f["energy_rand"],
        f["energy_min"],
        f["n_trials_done"]
    end

n_done = n_trials_done
energy_exact = energy_exact[1:n_done, :] .* 12
energy_rand  = energy_rand[1:n_done, :]  .* 12
energy_min   = energy_min[1:n_done, :]   .* 12
println("Using $n_done completed trials.")

# ---------------------------------------------------------------
# Compute mean ± SE over trials
# ---------------------------------------------------------------
se(v) = vec(std(v, dims = 1)) ./ sqrt(n_done)

mean_exact = vec(mean(energy_exact, dims = 1))
mean_rand  = vec(mean(energy_rand,  dims = 1))
mean_min   = vec(mean(energy_min,   dims = 1))

se_exact = se(energy_exact)
se_rand  = se(energy_rand)
se_min   = se(energy_min)

# ---------------------------------------------------------------
# Colors
# ---------------------------------------------------------------
col_exact = colorant"seagreen"
col_rand  = colorant"darkorange"
col_min   = colorant"steelblue"

fsize = 12

# ---------------------------------------------------------------
# Plot E vs β
# ---------------------------------------------------------------
β_ticks = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

fig = plot(;
    xlabel         = L"\beta",
    ylabel         = L"E",
    guidefontsize  = fsize,
    tickfontsize   = fsize - 2,
    legend         = :bottomleft,
    legendfontsize = fsize - 2,
    ylims          = (-50, 2),
    xticks         = β_ticks,
    size           = (580, 420),
    margin         = 7Plots.mm,
    bottom_margin  = 9Plots.mm,
)

plot!(fig, β_values, mean_exact;
    ribbon      = se_exact,
    fillalpha   = 0.35,
    color       = col_exact,
    linewidth   = 1.5,
    markershape = :diamond,
    markersize  = 5,
    marker      = stroke(1, col_exact),
    markercolor = col_exact,
    label       = "n=12, Exact",
)

plot!(fig, β_values, mean_rand;
    ribbon      = se_rand,
    fillalpha   = 0.35,
    color       = col_rand,
    linewidth   = 1.5,
    markershape = :circle,
    markersize  = 4,
    marker      = stroke(1, col_rand),
    markercolor = col_rand,
    linestyle   = :dash,
    label       = "n=12, Glauber (rand init)",
)

plot!(fig, β_values, mean_min;
    ribbon      = se_min,
    fillalpha   = 0.35,
    color       = col_min,
    linewidth   = 1.5,
    markershape = :circle,
    markersize  = 4,
    marker      = stroke(1, col_min),
    markercolor = col_min,
    linestyle   = :dot,
    label       = "n=12, Glauber (min init)",
)


infile = joinpath(SCRIPT_DIR, "results_n24.jld2")
β_values, energy_rand, energy_min, n_trials_done =
    jldopen(infile) do f
        f["β_values"],
        f["energy_rand"],
        f["energy_min"],
        f["n_trials_done"]
    end

n_done = n_trials_done
energy_rand  = energy_rand[1:n_done, :] .* 24
energy_min   = energy_min[1:n_done, :]  .* 24
println("Using $n_done completed trials.")

# ---------------------------------------------------------------
# Compute mean ± SE over trials
# ---------------------------------------------------------------
se(v) = vec(std(v, dims = 1)) ./ sqrt(n_done)

mean_rand  = vec(mean(energy_rand,  dims = 1))
mean_min   = vec(mean(energy_min,   dims = 1))

se_rand  = se(energy_rand)
se_min   = se(energy_min)

plot!(fig, β_values, mean_rand;
    ribbon      = se_rand,
    fillalpha   = 0.35,
    color       = col_rand,
    linewidth   = 1.5,
    markershape = :utriangle,
    markersize  = 4,
    marker      = stroke(1, col_rand),
    markercolor = col_rand,
    linestyle   = :dash,
    label       = "n=24, Glauber (rand init)",
)

plot!(fig, β_values, mean_min;
    ribbon      = se_min,
    fillalpha   = 0.35,
    color       = col_min,
    linewidth   = 1.5,
    markershape = :utriangle,
    markersize  = 4,
    marker      = stroke(1, col_min),
    markercolor = col_min,
    linestyle   = :dot,
    label       = "n=24, Glauber (min init)",
)




# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
outpdf = joinpath(SCRIPT_DIR, "n12_24_energy_sweep.pdf")
savefig(fig, outpdf)
println("Figure saved to $outpdf")
display(fig)
