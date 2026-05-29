using Pkg
using LaTeXStrings
Pkg.activate("../../LS_env")
include("../../utils.jl")
include("cw_utils.jl")
using Polynomials
using Plots
using Plots.Measures


using JLD2
using DataFrames
using StatsBase

stderror(x) = std(x)/sqrt(length(x))

filename = "data_cw.jld2"
if isfile(filename)
    display("Existing file found")
    jldopen(filename) do file
        global df = file["df"]
    end
end

plot_font = "Computer Modern"
default(fontfamily=plot_font,
    linewidth=2, framestyle=:box, label=nothing, grid=true)
#scalefontsizes(1.5)


ns = 5000
J =  1.2/ns
h = 0.04
algo_list = [:Glauber]
P = plot()
df = df[(df.n_spins .== ns) .& (df.J .== J) .& (df.h .== h),:]
xtick = ([ 10^7,10^8, 10^9 ],[ L"10^7", L"10^8", L"10^9"])
ytick = ([0.5,  0.1,0.02], [L"0.5", L"10^{-1}",L"2~\times 10^{-2}"])

for S in algo_list
    df_plot = df[df.sampler.==S, :]
    df_plot = sort(df, :n_samples)
    gdf = DataFrames.groupby(df_plot, :n_samples)
    df_plot = combine(gdf, [:J_err,:h_err] .=> [mean stderror length])
    @show df_plot


    plot!(P, df_plot.n_samples, ns*df_plot.J_err_mean, ribbon=ns*df_plot.J_err_stderror, show=true, label=L"|J - \hat{J}~| ", xscale=:log10, yscale=:log10, marker=stroke(), xticks=xtick, yticks = ytick)
    plot!(P, df_plot.n_samples, df_plot.h_err_mean, ribbon=df_plot.h_err_stderror, show=true, label=L"|h - \hat{h}~|", xscale=:log10, yscale=:log10, marker=stroke(), xticks=xtick, yticks = ytick, left_margin = 2Plots.mm)
    global C =  geomean(df_plot.h_err_mean)/(geomean(df_plot.n_samples))^(-1/2)
end
#plot!(P, 5*10^6:10^8:10^9, 0.25*C*( 5*10^6:10^8:10^9).^(-1/2), linestyle=:dash, xscale=:log10, yscale=:log10, color=:grey)
xlabel!("Number of samples (M)")
ylabel!("Error")
savefig("error_cw.pdf")



gdf = DataFrames.groupby(df, [:n_samples,:sampler])
df_mag = combine(gdf, :sample_mag .=> [mean stderror])
@show df_mag


##Making the Histograom
E_sm(m) =   -0.5*J*( m^2 - ns) + h*m
function bin_entropy(p)
    if p ≈ 0.0 ||  p ≈ 1.0
        return 0.0
    else
        return   -p*log(p) - (1-p)*log(1-p)
    end
end
m_list = -ns:2:ns
log_p(m) = -1*E_sm(m) + ns*bin_entropy(0.5*(ns + m)/ns)
#log_p(m) = -1*E_sm(m) + log(binomial(ns, div(ns+m,2)))

P2 = plot()
largest_states =  log_p.(m_list)
x_max = max(largest_states...)
x_diff = [y - x_max for y in largest_states]
logZ_approx = x_max + log(sum(exp.(x_diff)))
Mp_list = log_p.(m_list) .- logZ_approx#.+ log(M)

xtick = ([10^7,10^8, 10^9 ],[ L"10^7", L"10^8", L"10^9"])

le = log10(exp(1))

plot!(P2, m_list, Mp_list, show = true, ylabel = L"\log~(~\mu(\sum_i \sigma_i))", xlabel=L"\sum_i \sigma_i", left_margin = 2Plots.mm) 

savefig("prob_cw.pdf")

P3 = plot()
plot!(P3,df_mag.n_samples, df_mag.sample_mag_mean, ribbon = df_mag.sample_mag_stderror, show=true, xticks=xtick, xscale=:log10, marker=stroke(), left_margin = 2Plots.mm)

xlabel!("Number of samples (M)")
ylabel!(L"\langle \sum_i \sigma_i \rangle_M")
savefig("mag_cw.pdf")

P4 = plot()











