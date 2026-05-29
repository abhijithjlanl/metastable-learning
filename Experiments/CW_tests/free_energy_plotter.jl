using Pkg
using LaTeXStrings
Pkg.activate("../../LS_env")
include("../../utils.jl")
include("cw_utils.jl")
using Polynomials
using Plots
using Plots.Measures
using Colors


using JLD2
using DataFrames
using StatsBase

ColorBlue = HSV(210,36,0.5) |> RGB
ColorGreen = HSV(91,9,0.5) |> RGB


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

plot!(P2, m_list, -1*Mp_list, show = true, ylabel = L"-\log~(~\mu(\sum_i \sigma_i))", xlabel=L"\sum_i \sigma_i", left_margin = 2Plots.mm, color=:black, label="Free energy") 

filename =  "Glauber_J=12E-1_h=4E-2_large.jld2"

M = 4*10^9

if isfile(filename)
        display("Existing CW samples file found")
        samples_cw = JLD2.load(filename)["samples_cw"]
    else
        global samples_cw = CW_Glauber_sampler_quadratic(J,h,  M, ns; init=:up, burnin =10^6, restarts= 4,keep_every = 5)

        @save  filename samples_cw
    end
m_vals = [x[2] for x in keys(samples_cw)] |> unique
p_vals = [ (get(samples_cw, [1,m],0) + get(samples_cw, [-1,m],0))/(4*10^9) for m in m_vals]
nbins = ((max(m_vals...) - min(m_vals...))/50.0) |> round |> Int
m_vals, p_vals =  coarse_grain_histogram(m_vals, p_vals, nbins)

Ptwin = twinx(P2)
bar!(Ptwin,m_vals, p_vals, show = true, fillalpha=0.3,lw = 0.0001, color =ColorGreen, linecolor=ColorGreen, label="Glauber dynamics")

p = Binomial(ns)
m_vals = collect(-ns:2:ns)
p_dict = Dict(m_vals .=> pdf(p) )
m_wts = [exp(log_p(m)-logZ_approx) for m in m_vals]
m_wts = round.(m_wts*M)
m_wts = m_wts/sum(m_wts)

nbins = ((max(m_vals...) - min(m_vals...))/50.0) |> round |> Int
m_vals, p_vals =  coarse_grain_histogram(m_vals, m_wts, nbins)
bar!(Ptwin,m_vals, p_vals, show = true, fillalpha = 0.3,lw = 0.00001, linecolor=ColorBlue, color =ColorBlue, label="Exact sampler", ylims = (0.0,0.4), legend_position = :topright, ylabel="Empirical probabilities")

savefig("FE_histogram.pdf")

#Exact bars



#=

filename =  "Exact_J=12E-1_h=4E-2.jld2"



if isfile(filename)
    display("Existing Exact samples file found")
    jldopen(filename) do file
        global samples_exact = file["df"]
    else
        global  samples_exact= CW_ExactSampler(E_sm, 10^9, ns; logZ = logZ_approx)

        @save  filename samples_exact
    end
end
=#




