import Distributions.Binomial
import Distributions.pdf
using StatsBase, LinearAlgebra
using SparseArrays


prob(J, h, m, n, s) = 0.5 * (1 - tanh(s * (J * m - h) - (J / n)))

function CW_1D_dynamics(ns, J, h, m_start, n_samples; burnin=100000)
    ##M lies between -1 and 1
    ##Sampling from exp(n(0.5*J*m^2 - h*m))
    m_t = m_start
    samples = []
    for _ in 1:(burnin+n_samples)
        p_plus = 0.5 * (1 - m_t) * prob(J, h, m_t, ns, -1)
        p_minus = 0.5 * (1 + m_t) * prob(J, h, m_t, ns, +1)
        delta = wsample([1, 0, -1], [p_plus, 1 - (p_plus + p_minus), p_minus])
        m_t = m_t + (2 * delta / ns)
        push!(samples, m_t)
    end
    return samples[burnin+1:end]
end

function CW_transition_matrix(ns, J, h)
    n_mag = ns + 1
    A = zeros(n_mag, n_mag)
    delta = 2 / ns
    for i in 2:(n_mag-1)
        m = -1 + (i - 1) * delta
        #@show m
        p_plus = 0.5 * (1 - m) * prob(J, h, m, ns, -1)
        p_minus = 0.5 * (1 + m) * prob(J, h, m, ns, +1)
        A[i,i+1] = p_plus
        A[i,i-1] = p_minus
        A[i,i] = 1 - (p_plus + p_minus)
    end
    A[1, 2] = 0.5 * 2 * prob(J, h, -1, ns, -1)
    A[1, 1] =  1 - A[1,2]
    A[n_mag, n_mag-1] = 0.5*2*prob(J,h,1,ns,+1)
    A[n_mag,n_mag] = 1 - A[n_mag,n_mag-1]
    return A
end

function CW_transition_matrix_sparse(ns, J, h)
    n_mag = ns + 1
    delta = 2 / ns
    rows = []
    cols = []
    vals = BigFloat[]
    function push_sparse(i,j,k)
        push!(rows,i)
        push!(cols, j)
        push!(vals,k)
    end
    for i in 2:(n_mag-1)
        m = -1 + (i - 1) * delta
        #@show m
        p_plus = 0.5 * (1 - m) * prob(J, h, m, ns, -1)
        p_minus = 0.5 * (1 + m) * prob(J, h, m, ns, +1)
        push_sparse(i,i+1,p_plus)
        push_sparse(i, i - 1, p_minus)
        push_sparse(i, i , 1.0 - (p_minus + p_plus))
    end
    push_sparse(1,2, prob(J, h, -1, ns, -1))
    push_sparse(1, 1, 1 -  prob(J, h, -1, ns, -1))
    push_sparse(n_mag, n_mag - 1,  prob(J, h, 1, ns, +1))
    push_sparse(n_mag, n_mag, 1 -  prob(J, h, 1, ns, +1))
   # @show rows
   # @show cols
    return sparse(rows, cols, vals, n_mag, n_mag)
end



function compute_η_CW(ns, J, h, m_0, a)
    prob(J, h, m, n, s) = 0.5 * (1 + tanh(s * (J * m - h) - (J / n)))
    p_plus(m) = 0.5 * (1 - m) * prob(J, h, m, ns, -1)
    p_minus(m) = 0.5 * (1 + m) * prob(J, h, m, ns, +1)
    end

