function _iscompatible(p1::Vector{Int}, p2::Vector{Int})::Bool
    for (i, j) in zip(p1, p2)
        if i == 0 || j == 0
            continue
        end
        @assert i > 0 && j > 0
        if i != j
            return false
        end
    end
    return true
end

function contract(A::MPOTree, B::MPOTree, leftprefix::Vector{Int}, rightprefix::Vector{Int}; maxdim=100)::MPO
    leftmpo = MPO[]
    leftinner_prefix = Vector{Int}[]
    for n in eachindex(A.mpos)
        if _iscompatible(leftprefix, A.leftprefix[n])
            push!(leftmpo, A.mpos[n])
            push!(leftinner_prefix, A.rightmpo_prefix[n])
        end
    end

    rightmpo = MPO[]
    rightinner_prefix = Vector{Int}[]
    for n in eachindex(B.mpos)
        if _iscompatible(rightprefix, B.rightprefix[n])
            push!(rightmpo, B.mpos[n])
            push!(rightinner_prefix, B.leftmpo_prefix[n])
        end
    end

    contraction_pairs = Tuple{MPO,MPO}[]
    prefix_contracted = Tuple{Int,Int}[]
    for n in eachindex(leftmpo), m in eachindex(rightmpo)
        if _iscompatible(leftinner_prefix[n], rightinner_prefix[m])
            push!(contraction_pairs, (leftmpo[n], rightmpo[m]))
            push!(prefix_contracted, collect(zip(leftinner_prefix[n], rightinner_prefix[m])))
        end
    end

    contracted_mpo = MPO[]
    for n in eachindex(contraction_pairs)
        res = contract(contraction_pairs[n][1], contraction_pairs[n][2]; maxdim=maxdim)
        push!(contracted_mpo, res)
    end

    res = contracted_mpo[1]
    for n in 2:length(contracted_mpo)
        res = +(res, contracted_mpo[n]; maxdim=maxdim)
    end

    return res
end


mutable struct MPOPatchCreator{T} <: AbstractPatchCreator{T,MPS}
    A::MPOTree
    B::MPOTree
    rtol::Float64
    maxbonddim::Int
    verbosity::Int
    maxval::Float64
    atol::Float64
end

