function contract(
    A::MPOTree,
    B::MPOTree,
    leftprefix::Vector{Int},
    rightprefix::Vector{Int};
    maxdim = 100,
)
    leftmpo = TTO[]
    leftinner_prefix = Vector{Int}[]
    for n in eachindex(A.mpos)
        if _iscompatible(leftprefix, A.leftprefix[n])
            push!(leftmpo, A.mpos[n])
            push!(leftinner_prefix, A.rightmpo_prefix[n])
        end
    end

    rightmpo = TTO[]
    rightinner_prefix = Vector{Int}[]
    for n in eachindex(B.mpos)
        if _iscompatible(rightprefix, B.rightprefix[n])
            push!(rightmpo, B.mpos[n])
            push!(rightinner_prefix, B.leftmpo_prefix[n])
        end
    end

    contraction_pairs = Tuple{TTO,TTO}[]
    prefix_contracted = Tuple{Int,Int}[]
    for n in eachindex(leftmpo), m in eachindex(rightmpo)
        if _iscompatible(leftinner_prefix[n], rightinner_prefix[m])
            push!(contraction_pairs, (leftmpo[n], rightmpo[m]))
            push!(
                prefix_contracted,
                collect(zip(leftinner_prefix[n], rightinner_prefix[m])),
            )
        end
    end

    contracted_mpo = TTO[]
    for n in eachindex(contraction_pairs)
        res = contract(contraction_pairs[n][1], contraction_pairs[n][2]; maxdim = maxdim)
        push!(contracted_mpo, res)
    end

    res = contracted_mpo[1]
    for n = 2:length(contracted_mpo)
        res = +(res, contracted_mpo[n]; maxdim = maxdim)
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