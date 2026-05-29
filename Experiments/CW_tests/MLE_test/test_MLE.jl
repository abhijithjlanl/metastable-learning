
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
using LaTeXStrings
using StatsBase
using Random
using ForwardDiff
include("../../../utils.jl")
include("../cw_utils.jl")
gr()

ns = 5000
J = 1.2/ns
h = 0.04


plot_font = "Computer Modern"
default(fontfamily=plot_font,
    linewidth=2, framestyle=:box, label=nothing, grid=true)
#scalefontsizes(1.5)




##Making the Histograom
E_sm(m,J,h,ns) = -0.5 * J * (m^2 - ns) + h * m
function bin_entropy(p)
    if p ≈ 0.0 || p ≈ 1.0
        return 0.0
    else
        return -p * log(p) - (1 - p) * log(1 - p)
    end
end
log_p(m,J,h,ns) = -1 * E_sm(m,J,h,ns) + ns * bin_entropy(0.5 * (ns + m) / ns)

function approx_logZ(J,h,ns)
    m_list = -ns:2:ns
    largest_states = log_p.(m_list,J,h,ns)
    x_max = max(largest_states...)
    x_diff = [y - x_max for y in largest_states]
    return  x_max + log(sum(exp.(x_diff)))
end



function neg_loglikelihood(J, h, samples, ns)
    num_samples = sum(values(samples))
    mean_energy = sum([v * E_sm(k[2], J, h, ns)/num_samples for (k, v) in samples])
    return (mean_energy + approx_logZ(J, h, ns))/ns
end


function neg_pseudologlikelihood(J, h, samples, ns)
    num_samples = sum(values(samples))
    return sum(v * log(1 + exp(-2 * J * k[1] * (k[2] - k[1]) + 2 * k[1] * h)) for
               (k, v) in samples) / num_samples
end



M = 2^32

n_restarts = max(5, div(6000 * M, 2^30))
#n_restarts = 1

#samples_MCMC = CW_Glauber_sampler_quadratic(J, h, M, ns, init=:up, burnin=10^5, restarts=n_restarts, keep_every=4)

#neg_loglikelihood(0, 0, samples_MCMC, ns)

cm = cgrad(:turbo, rev=false)
labelstr = "True model"

function cmap_scaling(x)
    if x < 0.6
        return  100#*log(x + 0.01)
    elseif return 1
        end
end


#scale = 1.5

x = range(0.8*J,1.2*J, length=200)
y = range(-0.3*h, 2 * h, length=200)
z = neg_pseudologlikelihood.(x', y, (samples_MCMC,), ns)
p1 = contourf(x * ns, y, z, levels=30, color=cm)
#p1 = surface(x * ns, y, z, levels=20, color=cm, title="PLE loss", show=true)

wsJ = 0.005
wsh =  0.2
hlineseg(x,yi,yf) = [x*ones(100), range(yi,yf, length=100)]
vlineseg(y, xi, xf) = [range(xi, xf, length=100), y*ones(100)]
clrln = :grey
lsln = :dash
#=
plot!(p1, hlineseg((1 - wsJ)J * ns, (1 - wsh) * h, (1 + wsh) * h)..., color=clrln, ls=lsln, label=false)
plot!(p1, hlineseg((1 + wsJ)*J * ns, (1 - wsh) * h, (1 + wsh) * h)..., color=clrln, ls = lsln,label=false)
plot!(p1, vlineseg((1 + wsh) * h, (1 - wsJ) * ns * J, (1 + wsJ) * ns * J)..., color=clrln, ls=lsln, label=false)
plot!(p1, vlineseg((1 - wsh) * h, (1 - wsJ) * ns * J, (1 + wsJ) * ns * J)..., color=clrln, ls=lsln, label=false)
=#
scatter!(p1, [J * ns], [h], markershape=:star, label=labelstr, xlabel="J", ylabel="h", legend_postion=:topleft, mc = :white, ms = 6)
plot(p1, size=(600, 400))
val,inds = findmin(z)
scatter!(p1,[x[inds[2]]*ns], [y[inds[1]]], markershape=:circle, mc=:white, ms=6, label = "Minimum point")
savefig("PLE_loss.pdf")

z = neg_loglikelihood.(x', y, (samples_MCMC,), ns)
p2 = contourf(x * ns, y, z, levels=25, color=cm)
scatter!(p2, [J * ns], [h], markershape=:star, label=labelstr, xlabel="J", ylabel="h", legend_postion=:topleft, mc = :white, ms = 6)

val, inds = findmin(z)

scatter!(p2, [x[inds[2]]] * ns, [y[inds[1]]], markershape=:circle, mc=:white, ms=6, label="Minimum point")

savefig("MLE_loss.pdf")



x = range((1 - wsJ) * J, (1 + wsJ) * J, length=50)
y = range((1 - wsh) * h, (1 + wsh) * h, length=100)
z = neg_pseudologlikelihood.(x', y, (samples_MCMC,), ns)
p5 = contourf(x * ns, y, z, color=cm, levels=25)

scatter!(p5, [J * ns], [h], markershape=:star,mc = :white, label=labelstr, xlabel="J", ylabel="h", legend_postion = :topleft, ms = 6)

val, inds = findmin(z)

scatter!(p5, [x[inds[2]] * ns], [y[inds[1]]], markershape=:circle, mc=:white, ms=6, label="Minimum point")


savefig("PLE_losszoom.pdf")


#=
p3 = plot(y, neg_pseudologlikelihood.(J, y, (samples_MCMC,), ns))
p4 = plot(y, neg_loglikelihood.(J, y, (samples_MCMC,), ns))


z = neg_loglikelihood.(x', y, (samples_MCMC,), ns)
p6 = contourf(x, y, z, levels=25, title="MLE",color=cm)
scatter!(p6, [J], [h], markershape=:star, label = labelstr)


plot(p1, p2, p3, p4, p5,p6, show=true, size=(1800, 800))
=#
