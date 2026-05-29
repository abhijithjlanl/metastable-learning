

function FerroPottsEnergy(σ,J,edges)
    E = 0.0
    for e in edges
        if σ[e[1]] == σ[e[2]]
            E += J
        end
    end
    return E
end


function HypergraphPottsEnergy(σ,edges, tensor, J)
    return sum(J*tensor[σ[e]...] for e in edges)
end


function HypergraphPottsEnergyOptimized(σ, edges, tensor, J)
    # 1. Initialize an accumulator with the correct type (likely Float64 or Complex)
    total_energy = zero(eltype(tensor))
    
    # 2. Iterate through edges
    for e in edges
        # 3. Access indices directly to avoid allocating a new vector σ[e]
        #    Assuming the tensor is qxqxq, we know the hyperedge size is 3.
        @inbounds i, j, k = e[1], e[2], e[3]
        
        # 4. Index the tensor directly (no splatting)
        @inbounds total_energy += tensor[σ[i], σ[j], σ[k]]
    end
    
    # 5. Multiply by J once at the end, rather than inside the loop
    return J * total_energy
end




function  threealphabet_ferro1(q)
    ##1's are the ground state. Otherwise partiy of violations have to be even
    A = fill(-1, q, q, q)
    targets = [[1,i,i] for i in 1:q]
    for i in 1:q, j in 1:q, k in 1:q
        sort([i,j,k]) ∈ targets && (A[i,j,k] = 1)
    end
    return A
end



function threealphabet_ferro2(q)
    ##1's are the ground state. Otherwise partiy of violations have to be even
    A = fill(0, q, q, q)
    f(i) = (i == 1) ? 1 : -1
    for i in 1:q, j in 1:q, k in 1:q
        A[i,j,k] =  f(i)*f(j)*f(k)
    end
    return A
    end





