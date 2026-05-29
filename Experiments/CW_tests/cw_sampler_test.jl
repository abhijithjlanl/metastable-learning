
using Pkg
Pkg.activate("../../LS_env")
using LaTeXStrings
using StatsBase
using Random
using ForwardDiff
include("../../utils.jl")
include("cw_utils.jl")
gr()


##Increasing number of spins to O(100) makes learning very hard
##Not sure if meatstability criterion is true in case h is very large. We start from all down and an high value of h promotes spin flips.
#Starting from approximate detailed balance. Can we find the criterion on h and J that will give metastability


ns = 5000
J = 1.2/ns
h = 0.04

E_sm(m) =   -0.5*J*( m^2 - ns) + h*m
Energy_exact(σ) = E_sm(sum(σ))

#logZ_gaussian = 0.5*log(π/(0.5*J )) + 0.5*h*h/J - 0.5*J*ns

#M_list = [10^5,4*10^6,10^7, 4*10^7, 16*10^7]
M_list = [10^9]
TV_list = []

for M in M_list
    @show M
    #   global samples_exact = CW_ExactSampler(E_sm,M, ns; logZ = abs(E_sm(-ns)))
    n_restarts = max(20, div(6000 * M, 2^30))
   @time global samples_MCMC = CW_Glauber_sampler_quadratic(J,h, M, ns, init=:up, burnin=10^5, restarts = n_restarts, keep_every =4)
#  global df_exact = log_histogram(samples_exact, x->x[2], -ns:2:ns)
   global df_MCMC = log_histogram(samples_MCMC, x->x[2], -ns:2:ns, bins=nothing)
#    push!(TV_list, 0.5*l1_norm_error(samples_exact, samples_MCMC)/M)
end
#=
P2 = plot(M_list, TV_list, xscale = :log10, yscale = :log10, show=true,label= "Error")
C = TV_list .* sqrt.(M_list) |> geomean
plot!(P2, M_list, C*sqrt.(1 ./M_list), linestyle=:dot, label="1/sqrt(M)")
title!("ns = $ns")
xlabel!("M")
ylabel!("TV")
=##

#P1 = plot(df_exact.:x/ns, df_exact.:y, seriestype=:bar, show=false, label="Exact samples", bar_width = 1/ns, linewidth = 0.0)
#mvals = -ns:2:ns |> collect
#E_vals = 
P1 = plot()
plot!(P1, df_MCMC.:x, df_MCMC.:y, seriestype=:bar, label="MCMC samples")
title!("ns = $ns, J = $J , h = $h")
display(P1)

#P3 = plot(P1,P2, show = true)
#display(P3)
#@show J_est, h_est = CW_learn_conditional_RPLE(samples_MCMC, mag_sign = false, local_l1 = abs(J)*(ns-1) + abs(h))
@show J_est, h_est = CW_learn_conditional_RPLE(samples_MCMC)
#@show J_est, h_est = CW_learn_conditional_RPLE(samples_exact)


###Plot the other solution 

E_sm(m) = -0.5 * J * (m^2 - ns) + h * m
grad_E_sm(m) = -1 * J * m + h
function bin_entropy(p)
    if p ≈ 0.0 || p ≈ 1.0
        return 0.0
    else
        return -p * log(p) - (1 - p) * log(1 - p)
    end
end

grad_bin_entropy(p) = log((1-p)/p)
m_list = -ns:2:ns
log_p(m) = -1 * E_sm(m) + ns * bin_entropy(0.5 * (ns + m) / ns)
f(x) = log_p(x*ns)
grad_f(x) = ForwardDiff.derivative(f,x)
hess_f(x) = ForwardDiff.derivative(grad_f, x)
#grad_f(m) = -1 * grad_E_sm(m) + (grad_bin_entropy(0.5 * (ns + m) / ns) * 0.5)
#hess_f(m) = ForwardDiff.derivative(grad_f,m)

m_0 = bisection(grad_f, 0.3,0.7) 
a =  hess_f(m_0)
#gauss_meta(m) = inv(2*a)*(m-m_0)^2
#plot!(P1,gauss_meta, -ns, ns)






