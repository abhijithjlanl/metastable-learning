using Pkg
Pkg.activate("../../LS_env")
include("CW_1D_utils.jl")
using Plots
using ArnoldiMethod
using SparseArrays
using LaTeXStrings
using StatsBase
using Random
using ForwardDiff

include("../../utils.jl")
samples = CW_1D_dynamics(500,1.02,0.000,0,10^5)
histogram(samples)

J = 1.2
h = 0.01
#=
gap_list = BigFloat[]
ns_list = 100:1000

ns_list =  ns_list[round.(Int, range(1, stop=length(ns_list), length=10))]

for ns = ns_list
    P = CW_transition_matrix(ns, J, h)
    gap = partialschur(I - P, nev=2, maxdim=20, which=SR(), restarts=500, tol=1e-50)[1].eigenvalues[2]
    push!(gap_list, gap)
end
plot(ns_list, gap_list, yscale=:log10,show=true)
slope = (log10(gap_list[end]) - log10(gap_list[end-3]))/(ns_list[end] - ns_list[end-3])
C = gap_list[end]/10^(slope*ns_list[end])
plot!(x->C*10^(slope*x), ns_list[1],ns_list[end])
=#
#######Computing the minima
E_sm(m) = 0.5 * J * (m^2) - h * m
grad_E_sm(m) = J * m - h
function bin_entropy(p)
    if p ≈ 0.0 || p ≈ 1.0
        return 0.0
    else
        return -p * log(p) - (1 - p) * log(1 - p)
    end
end

ns =10000
P = CW_transition_matrix(ns, J, h)
P_s = partialschur(I - P', nev=2, maxdim=20, which=SR(), restarts=500, tol=1e-50)
gap = P_s[1].eigenvalues[2]
@show gap


grad_bin_entropy(p) = log((1 - p) / p)
#m_list = -ns:2:ns
log_p(m,ns) = ns*E_sm(m) + ns * bin_entropy(0.5 * (1 + m))
f(x) = log_p(x,ns)/ns
grad_f(x) = ForwardDiff.derivative(f, x)
hess_f(x) = ForwardDiff.derivative(grad_f, x)
third_d_f(x) = ForwardDiff.derivative(hess_f, x)
fourth_d_f(x) = ForwardDiff.derivative(third_d_f, x)
m_0 = bisection(grad_f, 0.3,0.9) 
a =  -1*hess_f(m_0)
@assert a > 0
ν = zeros(BigFloat,ns+1)
δ = 2/ns
S(m) = bin_entropy(0.5*(1+m))
Z = sqrt((2*π)/(ns*a))*inv(δ)
#=
for i in 1:ns+1
    m = -1 + (i - 1) * δ
    ν[i] = exp(ns * (Ψ(m) - S(m))) / Z
end

P = CW_transition_matrix_sparse(ns, J, h)
D = ν.*P
D = abs.(D .- D')


##Metastability 1
global η1 = 0.0
for i in 1:ns+1
    m = -1 + (i - 1) * δ
    if -1 < m < 1
        C = (0.5 * ns * (1 + m) * D[i - 1, i]) + 0.5 * (ns * (1 - m) * D[i + 1, i])
      elseif  m == 1
        C = ns*D[i-1,1]
      elseif m == -1
        C = ns*D[i+1,1]
  end
    global η1 += exp(ns*S(m))*C
end


##Metastability 2
global  η2 = 0.0
p_plus(m) = 0.5 * (1 - m) * prob(J, h, m, ns, -1)
p_minus(m) = 0.5 * (1 + m) * prob(J, h, m, ns, +1)
ratio_minus(m) = (0.5*ns*(1+m) +1)/(0.5*ns*(1-m))
ratio_plus(m) = (0.5*ns*(1-m) +1)/(0.5*ns*(1+m))

for i in 1:ns+1
    m = -1 + (i - 1) * δ
    if -1 < m < 1
        C = 0.5 * ns * (1 - m) * abs(p_minus(m + δ) * exp(ns * Ψ(m + δ)) * ratio_minus(m) - p_plus(m) * exp(ns * Ψ(m))) + 0.5 * ns * (1 + m) * abs(p_plus(m - δ) * exp(ns * Ψ(m - δ)) * ratio_plus(m) - p_minus(m) * exp(ns * Ψ(m)))
    elseif m == 1
        C =  0.5 * ns * (1 + m) * abs(p_plus(m - δ) * exp(ns * Ψ(m - δ)) * ratio_plus(m) - p_minus(m) * exp(ns * Ψ(m)))
    elseif m == -1
        C = 0.5 * ns * (1 - m) * abs(p_minus(m + δ) * exp(ns * Ψ(m + δ)) * ratio_minus(m) - p_plus(m) * exp(ns * Ψ(m))) 
    end
    global η2 +=  C
end

@show η2/Z

##Metastability 3
=#


scale = δ
#H(m) = ns*E_sm(m)
#var_a = 1/(sqrt(a)*(ns)^(1/4))
#Ψ(m) = f(m)*(1*(abs(m-m_0) < var_a ) -100*(abs(m-m_0) >= var_a))
b = third_d_f(m_0) 
c = fourth_d_f(m_0)

Ψ(m) = -0.5*a*(m-m_0)^2 + (1/6)*b*(m - m_0)^3 + (1/24)*c*(m-m_0)^4
#Ψ(m)  = f(m)
H(m) = ns * (Ψ(m) - S(m))
Z = sum(exp(ns*Ψ(m)) for m in -1:δ:1)

global η3 = 0.0
for i in 1:ns+1
    m = -1 + (i - 1) * δ
    C = 0.0
    C_p(x) = 0.5 * (1 + x) * abs(tanh((J * x - h) - (J / ns)) - tanh(0.5 * (H(x) - H(x - δ))))
    C_m(x) = 0.5 * (1 - x) * abs(tanh(-1 * (J * x - h) - (J / ns)) - tanh(0.5 * (H(x) - H(x + δ))))
    if -1 < m < 1
        C += (exp(ns * Ψ(m)) / Z) * (C_p(m) + C_m(m))
    elseif m == 1
        C += (exp(ns * Ψ(m)) / Z) * (C_p(1))
    elseif m == -1
        C += (exp(ns * Ψ(m)) / Z) * (C_m(-1))
    end
    global η3 += 0.25*C

end


@show η3


##Some plots




