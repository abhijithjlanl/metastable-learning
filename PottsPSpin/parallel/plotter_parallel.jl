using Pkg
const SCRIPT_DIR = @__DIR__
Pkg.activate(joinpath(SCRIPT_DIR, "..", ".."))

using JLD2
using DataFrames
using Measures
using Plots
using Statistics

const INPUT_FILE = joinpath(SCRIPT_DIR, "data_energies_high_parallel.jld2")
const OUTPUT_FILE = joinpath(SCRIPT_DIR, "potts_metastable_parallel.pdf")

if !isfile(INPUT_FILE)
    error("Missing input file: $INPUT_FILE")
end

df = jldopen(INPUT_FILE, "r") do file
    file["df"]
end

transform!(
    df,
    [:sample_energy, :active_edges] => ByRow((energy, edges) -> energy / length(edges)) => :normalized_energy,
)

stderror(x) = std(x) / sqrt(length(x))

plot_font = "Computer Modern"
default(fontfamily = plot_font, linewidth = 2, framestyle = :box, label = nothing, grid = true)

q_list = sort(unique(df.q))
ns = maximum(df.n_spins)
sampler_list = [:GlauberMin, :GlauberRand]
model_list = [:Ferro1, :Ferro2]
#n_samples_target = maximum(df.n_samples)

df = df[(df.n_spins .== ns) .& (df.n_samples .== 10000), :]

plots_dict = Dict()

for q in q_list
    for model in model_list
        p = plot(title = "$model, $q")
        for sampler in sampler_list
            df_plot = df[
                (df.model_type .== model) .&
                (df.sampler .== sampler) .&
                (df.q .== q),
                :,
            ]

            if nrow(df_plot) == 0
                continue
            end

            gdf = groupby(df_plot, :β)
            df_summary = combine(
                gdf,
                :normalized_energy => mean => :normalized_energy_mean,
                :normalized_energy => stderror => :normalized_energy_stderror,
                nrow => :count,
            )
            sort!(df_summary, :β)

            plot!(
                p,
                df_summary.β,
                df_summary.normalized_energy_mean,
                ribbon = df_summary.normalized_energy_stderror,
                marker = stroke(),
                label = String(sampler),
                xlabel = "β",
                ylabel = "<Energy>/|Edges|",
            )
        end
        plots_dict[(model, q)] = plot(p, margin = 5mm)
    end
end

rows = model_list
cols = q_list
ps = [plots_dict[(row, col)] for row in rows, col in cols]

plot(ps..., layout = size(ps), size = (1600, 800), show = false)
savefig(OUTPUT_FILE)

println("Saved plot to $OUTPUT_FILE")
