function contract_mpo_mpo(M1::MPO, M2::MPO; alg::String="densitymatrix", kwargs...)::MPO
    if alg == "densitymatrix"
        return contract_densitymatrix(M1, M2; kwargs...)
    elseif alg == "fit"
        return contract_fit(M1, M2; kwargs...)
    elseif alg == "tci2"
        return contract_tci2(M1, M2; kwargs...)
    else
        error("Unknown algorithm: $alg")
    end

end

function contract_tci2(
    M1::MPO, M2::MPO;
    cutoff=0.0, tolerance=0.0,
    maxdim=typemax(Int), maxbonddim=typemax(Int)
)::MPO
    contractinds =
        setdiff(commoninds.(M1, M2), Ref(linkinds(M1)), Ref(linkinds(M2))) |> flatten
    sites1 = setdiff.(siteinds(M1), Ref(contractinds))
    sites2 = setdiff.(siteinds(M2), Ref(contractinds))

    tt1 = TensorTrain{Float64,4}(
        M1; sites=[[s..., c] for (s, c) in zip(sites1, contractinds)])
    tt2 = TensorTrain{Float64,4}(
        M2; sites=[[c, s...] for (s, c) in zip(sites2, contractinds)])

    res = TCIA.contract_TCI(
        tt1, tt2;
        tolerance=min(cutoff, tolerance),
        maxbonddim=min(maxdim, maxbonddim)
    )
    return MPO(res; sites=collect(zip(sites1, sites2)))
end


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
