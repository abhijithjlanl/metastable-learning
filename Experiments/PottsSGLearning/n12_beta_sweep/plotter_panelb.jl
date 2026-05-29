using Pkg
const SCRIPT_DIR = @__DIR__
Pkg.activate(joinpath(SCRIPT_DIR, "..", "..", ".."))

using JLD2
using Plots
using LaTeXStrings
using Statistics

# ---------------------------------------------------------------
# Global style — match error_cw.pdf
# ---------------------------------------------------------------
plot_font = "Computer Modern"
default(fontfamily = plot_font, linewidth = 2, framestyle = :box, grid = true)

fsize = 12

# ---------------------------------------------------------------
# Load results
# ---------------------------------------------------------------
infile = joinpath(SCRIPT_DIR, "results_panelb.jld2")
M_values, β, errs_min, errs_rand, prec_min, prec_rand, rec_min, rec_rand, n_trials_done =
    jldopen(infile) do f
        f["M_values"], f["β"],
        f["errs_min"],  f["errs_rand"],
        f["prec_min"],  f["prec_rand"],
        f["rec_min"],   f["rec_rand"],
        f["n_trials_done"]
    end

n_done    = n_trials_done
errs_rand = errs_rand[1:n_done, :]
prec_rand = prec_rand[1:n_done, :]
rec_rand  = rec_rand[1:n_done, :]
println("Using $n_done completed trials (β = $β).")

# ---------------------------------------------------------------
# Colors
# ---------------------------------------------------------------
col_rand = colorant"steelblue"

# ---------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------
M_float    = Float64.(M_values)
geomean(v) = exp.(vec(mean(log.(max.(v, 1e-12)), dims = 1)))
se(v)      = vec(std(v, dims = 1)) ./ sqrt(n_done)

gmean_err_rand = geomean(errs_rand)

mean_prec_rand = vec(mean(prec_rand, dims = 1))
mean_rec_rand  = vec(mean(rec_rand,  dims = 1))
se_prec_rand   = se(prec_rand)
se_rec_rand    = se(rec_rand)

# M^{-1/2} reference anchored at first GlauberRand geometric-mean point
ref_slope = gmean_err_rand[end] .* sqrt.(M_float[end] ./ M_float)

# ---------------------------------------------------------------
# Axis ticks
# ---------------------------------------------------------------
xtick = ([1e3, 1e4, 1e5], [L"10^3", L"10^4", L"10^5"])
ybtick = ([0.4, 0.6, 0.8, 1.0], [L"0.4", L"0.6", L"0.8", L"1.0"])

# ---------------------------------------------------------------
# Top panel — max-norm error (log-log)
# ---------------------------------------------------------------
pa = plot(;
          xscale         = :log10,
          yscale         = :log10,
          ylabel         = L"\max_{e \in \texttt{E}'} |\hat{\theta}_e - \theta^*_e|",
          xlabel         = "",
          xformatter     = _ -> "",
          title    = "Parameter learning, " * L"n = 24, \beta = 1.2 ",

          guidefontsize  = fsize-2,
          tickfontsize   = fsize - 2,
          legend         = :topright,
          legendfontsize = fsize - 4,
          xticks         = xtick,
          titlefontsize = fsize-2
)

plot!(pa, M_float, gmean_err_rand;
      ribbon      = se(errs_rand),
      fillalpha   = 0.35,
      color       = col_rand,
      linewidth   = 2,
      markershape = :circle,
      markersize  = 6,
      marker      = stroke(1, col_rand),
      markercolor = col_rand,
      label       = nothing,
      )

plot!(pa, M_float, ref_slope;
      color     = :black,
      linestyle = :dash,
      linewidth = 1.5,
      label     = L"\sqrt{M}",
      )

# ---------------------------------------------------------------
# Bottom panel — Precision and Recall (linear scale, 0–1.15)
# ---------------------------------------------------------------
pb = plot(;
          xscale         = :log10,
          ylabel         = "Precision / Recall",
          xlabel         = "Number of samples " * L"(M)",
          ylims          = (0.4, 1.10),
          guidefontsize  = fsize-2,
          tickfontsize   = fsize - 2,
          legend         = :bottomright,
          legendfontsize = fsize - 4,
          xticks         = xtick,
          yticks         = ybtick,
          title =    "Structure learning, " * L"n = 24, \beta = 1.2",
  titlefontsize =  fsize-2

)

plot!(pb, M_float, mean_prec_rand;
    ribbon      = se_prec_rand,
    fillalpha   = 0.25,
    color       = col_rand,
    linewidth   = 2,
    linestyle   = :solid,
    markershape = :utriangle,
    markersize  = 6,
    marker      = stroke(1, col_rand),
    markercolor = col_rand,
    label       = "Precision",
)

plot!(pb, M_float, mean_rec_rand;
    ribbon      = se_rec_rand,
    fillalpha   = 0.25,
    color       = col_rand,
    linewidth   = 2,
    linestyle   = :dash,
    markershape = :utriangle,
    markersize  = 6,
    marker      = stroke(1, col_rand),
    markercolor = col_rand,
    label       = "Recall",
)

hline!(pb, [1.0]; linestyle = :dot, color = :black, linewidth = 1.2, label = false)

# ---------------------------------------------------------------
# Combine and save
# ---------------------------------------------------------------
fig = plot(pa, pb;
    layout        = (2, 1),
    size          = (560, 420),
    margin        = 2Plots.mm,
    bottom_margin = 2Plots.mm,
)

outpdf = joinpath(SCRIPT_DIR, "n12_24_panelb.pdf")
savefig(fig, outpdf)
println("Figure saved to $outpdf")
display(fig)
