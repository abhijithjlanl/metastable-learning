using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, LaTeXStrings, IterTools
using DataFrames
using Random
using GraphicalModelLearning
using Measures
include("../samplers.jl")
include("../utils.jl")
filename = "data_energies.jld2"

include("potts_tensors.jl")
include("potts_samplers.jl")

if isfile(filename)
    display("Existing file found")
    jldopen(filename) do file
        global df = file["df"]
    end
end

transform!(df, [:sample_energy, :active_edges] => 
    ByRow((E, edges) -> E ./ length(edges)) => :normalized_energy
           )


stderror(x) = std(x)/sqrt(length(x))
plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=true)
#scalefontsizes(1.5)

q_list = [2,3,4,5]
ns = 40

sampler_list = [:GlauberMin, :GlauberRand]


df = df[(df.n_spins .== ns) .& (df.β .> 0.5 ),:]
plots_dict = Dict()


for q in q_list
    for model in [:Ferro1, :Ferro2]
        plots_dict[(model,q)] = plot(title="$model, $q")
        for sampler  in sampler_list
            df_plot = df[(df.model_type .== model) .&  (df.n_samples .== 10^5).& (df.sampler .== sampler) .& (df.q .== q), : ]
            gdf = DataFrames.groupby(df_plot, :β)
            df_plot = combine(gdf, [:normalized_energy] .=> [mean stderror length])

            plot!(plots_dict[(model,q)], df_plot.β, df_plot.normalized_energy_mean, ribbon=df_plot.normalized_energy_stderror, marker=stroke(), label=String(sampler), xlabel="β", ylabel="<Energy>/|Edges|")

        end
    end
end

    # 1. Extract and sort unique symbols for axes
    rows = sort(unique(first.(keys(plots_dict))))
    cols = sort(unique(last.(keys(plots_dict))))

    # 2. Build matrix of plots (with titles for labels)
    #    Matches rows to vertical axis, cols to horizontal axis
    ps = [plot(plots_dict[(r,c)], margin=5mm) for r in rows, c in cols]

    # 3. Plot the grid
    plot(ps..., layout=size(ps), size=(1600, 800), show=true)

    savefig("potts_metastable.pdf")
   



