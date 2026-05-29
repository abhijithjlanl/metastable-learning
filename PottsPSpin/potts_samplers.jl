

function exact_sampler(E::Function, N::Int, p::Int, q::Int; β = 1)
    states = Array(0:(q^p)-1)
    states = map( x -> digits(x, base=q, pad=p) |> reverse, states)
    states = map(x -> x.+1, states)
    wts =  exp.(-β*E.(states))
    return  wsample(states, wts, N)
end



function Glauber_sampler(E::Function,N::Int,p::Int, q::Int;β = 1, restarts = 16, burnin = 10^4, order_parameters = [], return_samples = true, init = false, throw_away = 10)
    samples =  []
#    β_by_2 = β/2
    if init == false
        init = () -> rand(1:q,p)
    end
    ns_list = repeat([div(N,restarts)], restarts)
    ord_p = zeros(length(order_parameters))
    if rem(N,restarts) > 0
        push!(ns_list,rem(N,restarts))
    end

    @showprogress  for k in 1:length(ns_list)
#   for k in 1:restarts
        state = init()
        E_state = E(state)
        #push!(samples, deepcopy(state))
        for t = 1:((throw_away*ns_list[k])+burnin)-1
              i = rand(1:p)
              wts = zeros(q)
              for j in 1:q
                  state[i] = j
                  wts[j] = exp(-β*E(state))
              end
              state[i] = wsample(1:q, wts)



            if (t >  burnin-1)&& (t%throw_away===0)
                if return_samples
                    push!(samples, deepcopy(state))
                end
                for (i,x) in enumerate(order_parameters)
                    ord_p[i] += x(state)
                end
            end
        end
    end
    if length(order_parameters) == 0
      return samples[1:N]
    elseif return_samples == false
        return ord_p
    end
    return samples[1:N], ord_p
end

#=

function Gibbs_sampler(E::Function,N::Int,p::Int, q::Int;β = 1, restarts = 16, burnin = 10^4, order_parameters = [], return_samples = true, init = false, throw_away = 10)
    samples =  []
#    β_by_2 = β/2
    if init == false
        init = () -> rand(1:q,p)
    end
    ns_list = repeat([div(N,restarts)], restarts)
    ord_p = zeros(length(order_parameters))
    if rem(N,restarts) > 0
        push!(ns_list,rem(N,restarts))
    end

    @showprogress  for k in 1:length(ns_list)
#   for k in 1:restarts
        state = init()
        E_state = E(state)
        #push!(samples, deepcopy(state))
        for t = 1:((throw_away*ns_list[k])+burnin)-1
              i = rand(1:p)
              wts = zeros(q)
              for j in 1:q
                  state[i] = j
                  wts[j] = exp(-β*E(state))
              end
              state[i] = wsample(1:q, wts)



            if (t >  burnin-1)&& (t%throw_away===0)
                if return_samples
                    push!(samples, deepcopy(state))
                end
                for (i,x) in enumerate(order_parameters)
                    ord_p[i] += x(state)
                end
            end
        end
    end
    if length(order_parameters) == 0
      return samples[1:N]
    elseif return_samples == false
        return ord_p
    end
    return samples[1:N], ord_p
end
=#




