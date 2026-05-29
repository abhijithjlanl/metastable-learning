using Pkg
Pkg.activate("../../LS_env")
include("CW_1D_utils.jl")
using Plots
using ArnoldiMethod
using SparseArrays
using LaTeXStrings
using StatsBase
using Random
using Measures
using ForwardDiff
include("../../utils.jl")
function bin_entropy(p)
    @assert p >= 0
    if p ≈ 0.0 || p ≈ 1.0
        return 0.0
    else
        return -p * log(p) - (1 - p) * log(1 - p)
    end
end




Jh_list = [(1.1, 0.0), (1.2, 0.01), (1.4, 0.0), (1.4, 0.05)]
ns_list = 100:1600
η_dict = Dict()
ns_list =  ns_list[round.(Int, range(1, stop=length(ns_list), length=10))]


for (J,h) in Jh_list
    η_list = BigFloat[]
    for ns  in ns_list
      E_sm(m) = 0.5 * J * (m^2) - h * m
      grad_E_sm(m) = J * m - h

      grad_bin_entropy(p) = log((1 - p) / p)
      #m_list = -ns:2:ns
      log_p(m,ns) = ns*E_sm(m) + ns * bin_entropy(0.5 * (1 + m))
      f(x) = log_p(x,ns)/ns
      grad_f(x) = ForwardDiff.derivative(f, x)
      hess_f(x) = ForwardDiff.derivative(grad_f, x)
      third_d_f(x) = ForwardDiff.derivative(hess_f, x)
      fourth_d_f(x) = ForwardDiff.derivative(third_d_f, x)
      global m_0 = bisection(grad_f, 0.3,0.9) 
      a =  hess_f(m_0)
      b = third_d_f(m_0) 
      c = fourth_d_f(m_0)
#      @assert a < 0

      δ = 2/ns
     # width = 1/(-a*J)^(0.5)
        width = 0.25*m_0/sqrt(-1*a)
        @assert width < 1.0  print("$width")
        @show  (width, m_0, J)

        #Ψ(m) = 0.5*a*(m-m_0)^2 + (1/6)*b*(m - m_0)^3 + (1/24)*c*(m-m_0)^4
        global Ψ(m) = f(m) * (abs(m - m_0) < width) - 1e15 * (abs(m - m_0) >= width)
        H(m) = ns * (Ψ(m) - S(m))
        S(m) = bin_entropy(0.5 * (1 + m))
        Z = sum(exp(ns * BigFloat(Ψ(m))) for m in -1:δ:1)

        global η3 = BigFloat(0.0)
        for i in 1:ns+1
          m = -1 + (i - 1) * δ
          C = 0.0
          C_p(x) = 0.5 * (1 + x) * abs(tanh((J * x - h) - (J / ns)) - tanh(0.5 * (H(x) - H(x - δ))))
          C_m(x) = 0.5 * (1 - x) * abs(tanh(-1 * (J * x - h) - (J / ns)) - tanh(0.5 * (H(x) - H(x + δ))))
          if 2 < i < ns
              C += (exp(ns * Ψ(m) - log(Z))) * (C_p(m) + C_m(m))
          elseif i == ns+1
              C += (exp(ns * Ψ(m) - log(Z))) * (C_p(1))
          elseif i == 1
              C += (exp(ns * Ψ(m)-log(Z))) * (C_m(-1))
          end
          global η3 += 0.25*C
      end
      push!(η_list, η3)
    end
    η_dict[(J,h)] = copy(η_list)
end
xtick = ([100, 200, 400, 800, 1600], ["100", "200", "400", "800", "1600"])
Pl = plot(show=true, yscale=:log10, xlabel=L"Number of spins $(n)$", ylabel=L"Strong metastability $(\eta)$", legend=:bottomleft, xticks=xtick)

shapes = [:circle, :star, :xcross, :star4]

for (i, k) in enumerate(Jh_list)
    plot!(Pl, ns_list, η_dict[k], label=L"J,h= %$k", shape=shapes[i])
end

plot!(Pl, x -> 0.01 * exp(-0.02 * x), ns_list[1], ns_list[end], linestyle=:dash, label=L"\exp(-0.02n)",
    title="Truncated free-energy", legend_position=:bottomleft)



Pl2 = plot(x->-1*Ψ(x), -1, 1,
    frame=:box,
    xlabel=L"m",
    ylabel=L"\Phi(m)",
    label=L"(J,h) = (1.4, 0.05)"
          # yscale=:log10
)
vline!(Pl2, [m_0], style=:dash, label=false,
    xticks=([-1.0, -0.5, 0.0, 0.5, m_0, 1.0], ["-1.0", "-0.5", "0.0", "0.5", L"m_0", "1.0"]),
    yticks=([0, 1e14, 1e15], [L"0", L"10^{14}", L"10^{15}"]))

#   inset=bbox(0.15,0.47,0.3,0.3),
plot(Pl, Pl2, layout=(2, 1), size=(600, 800), margin=3mm)
savefig("truncated_metastable_cw.pdf")




