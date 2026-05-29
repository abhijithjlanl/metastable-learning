
import Distributions.Binomial
import Distributions.pdf


function CW_Glauber_sampler(E, n_samples, n_spins; burnin = 100000, init=false, β = 1 )
    samples =  []
    if init == false
        init = () -> rand([1,-1],n_spins)
        elseif init == :down
            init = () -> Int.(-1*ones(n_spins))
        elseif  init == :up
            init = () -> Int.(ones(n_spins))
    end
    state = init()
    for t = 1:(n_samples + burnin)
        i = rand(1:n_spins)
        flipped_state = deepcopy(state)
        flipped_state[i] = -1*state[i]
        dE = E(state) - E(flipped_state)
        prob = 1/(1 + exp(β*dE*state[i]))
        state[i] = wsample([1,-1],[prob,1-prob])
        if t >  burnin
            push!(samples, [state[1], sum(state)] )
        end
    end
    return samples
end

function CW_Glauber_sampler_fast(E_sm, n_samples, n_spins; burnin = 100000, init=false, β = 1 )
    ##Glauber dynamics with reused energy evaluation
    samples =  []
    if init == false
        init = () -> rand([1,-1],n_spins)
    elseif init == :down
        init = () -> Int.(-1*ones(n_spins))
    elseif  init == :up
        init = () -> Int.(ones(n_spins))
    end
    state = init()
    m_state = sum(state)
    Estate = E_sm(m_state)
    for t = 1:(n_samples + burnin)
        i = rand(1:n_spins)

        m_state_flipped = m_state  - 2*state[i]
        state[i] = -1*state[i]
        Estate_flipped = E_sm(m_state_flipped)
        dE = Estate - Estate_flipped

        prob = exp(0.5*β*dE*state[i]) ###Remember that spin i got flipped
        fs =  wsample([1,-1],[prob,1/prob])
        if fs == state[i]
            Estate = Estate_flipped
            m_state = m_state_flipped
        else
            state[i] = fs
        end

        if t >  burnin
            push!(samples, [state[1], sum(state)] )
        end
    end
    return samples
end


function CW_Glauber_sampler_quadratic(J,h, n_samples, n_spins; burnin = 10^5, init=false, β = 1, output = :histogram, keep_every = 10, restarts = 1)
    ##Glauber dynamics with reused energy evaluation
    samples =  []
    if output == :histogram
        samples = Dict()
    end
    nsamp_list = repeat([div(n_samples, restarts)], restarts)
    if n_samples % restarts != 0
        push!(nsamp_list, rem(n_samples, restarts))
    end
        @assert sum(nsamp_list) == n_samples

    if init == false
        init = () -> rand([1,-1],n_spins)
    elseif init == :down
        init = () -> Int.(-1*ones(n_spins))
    elseif  init == :up
        init = () -> Int.(ones(n_spins))
    end
    E_sm(m) =   -0.5*J*( m^2 - ns) + h*m
    for k in 1:restarts
      state = init()
      m_state = sum(state)
      Estate = E_sm(m_state)
      for t = 1:(keep_every*(nsamp_list[k]) + burnin)
          i = rand(1:n_spins)
          s_i = state[i]

          dEby2 = -1*s_i*(J*(m_state - s_i) - h)
          prob = exp(-β*dEby2*s_i)
          fs =  wsample([1,-1],[prob,1/prob])
          if fs == -s_i
              Estate = Estate - 2*dEby2
              m_state = m_state  - 2*s_i
              state[i] = fs
          end

          if (t >  burnin) & ((t - burnin)%keep_every == 0)
              samp = [state[1], m_state]
              if output == :histogram
                  if haskey(samples, samp)
                      samples[copy(samp)] +=1
                  else
                      samples[copy(samp)] = 1
                  end
              else
                push!(samples, samp)
              end
          end
      end
    end
    return samples
end





function CW_ExactSampler(E_sm, n_samples, n_spins; init=false, β = 1, logZ = 0.0 )
    ##logZ  here is a shift to prevent overflow
    p = Binomial(n_spins)
    m_vals = -n_spins:2:n_spins
    p_dict = Dict(m_vals .=> pdf(p) )
    m_wts = [exp(-β * E_sm(x) - logZ) * p_dict[x] for x in m_vals]
m_wts = [-β * E_sm(x) - logZ  for x in m_vals]
    s1 = wsample(m_vals, m_wts, n_samples)
    s2 = [wsample([1,-1],[n_spins+x, n_spins-x]) for x in s1]
    return map( x->[x...], zip(s2,s1)) |> countmap
end








function CW_learn_conditional(samples)
    S = countmap(samples)
    num_configs = length(collect(keys(S)))
    num_spins = collect(keys(S))[1] |> length
    num_samples = sum(values(S))
    ################################
    Optimizer = NLP(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0, "tol"=>1e-10))
    model = Model(Optimizer.solver)
    JuMP.@variable(model,J )
    JuMP.@variable(model,h )
    @NLobjective(model,  Min, sum( v*exp(-J*k[1]*(k[2] - k[1]) + k[1]*h ) for
                                      (k,v) in S)/num_samples )

    JuMP.optimize!(model)
    return JuMP.value(J), JuMP.value(h)
end
    CW_learn_conditional_RPLE(S::Array) =  CW_learn_conditional_RPLE(countmap(S))

function CW_learn_conditional_RPLE(S::Dict; local_l1 = Inf, mag_sign = false, lambda = 0.0)
    num_configs = length(collect(keys(S)))
    num_spins = collect(keys(S))[1] |> length
    num_samples = sum(values(S))
    ################################
    Optimizer = NLP(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0, "tol"=>1e-10))
    model = Model(Optimizer.solver)
    JuMP.@variable(model,J )
    JuMP.@variable(model,h )
    if mag_sign == :positive
        @constraint(model, h >= 0 )
    end
    if mag_sign == :negative
        @constraint(model, h <= 0 )
    end

    JuMP.@variable(model,zJ  )
    JuMP.@variable(model,zh )
    @constraint(model, zJ >= J)
    @constraint(model, zJ >= -J)
    @constraint(model, zh >= h)
    @constraint(model, zh >= -h)
    if local_l1 != Inf
        @constraint(model, zh + zJ*(num_spins-1) <= local_l1)
    end
    @NLobjective(model,  Min, sum( v*log(1 + exp(-2*J*k[1]*(k[2] - k[1]) + 2*k[1]*h)) for
                                      (k,v) in S)/num_samples  + lambda*(zh + zJ*(num_spins-1)) )


    JuMP.optimize!(model)
    return JuMP.value(J), JuMP.value(h)
end


