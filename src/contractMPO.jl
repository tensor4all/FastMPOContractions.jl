function contract_mpo_mpo(M1::MPO, M2::MPO; alg::String = "densitymatrix", kwargs...)::MPO
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

function _mpo_to_tt(
    ::Type{V},
    mpo::ITensors.MPO,
    sites1,
    sites2,
)::TCI.TensorTrain{V,4} where {V}
    links = linkinds(mpo)

    sitedims = [(prod(dim.(sites1[i])), prod(dim.(sites2[i]))) for i = 1:length(mpo)]

    Tfirst = reshape(
        Array(mpo[1], sites1[1]..., sites2[1]..., links[1]),
        1,
        sitedims[1]...,
        dim(links[1]),
    )

    Tlast = reshape(
        Array(mpo[end], links[end], sites1[end]..., sites2[end]...),
        dim(links[end]),
        sitedims[end]...,
        1,
    )

    tensors = [Tfirst]
    for i = 2:length(mpo)-1
        push!(
            tensors,
            reshape(
                Array(mpo[i], links[i-1], sites1[i]..., sites2[i]..., links[i]),
                dim(links[i-1]),
                sitedims[i]...,
                dim(links[i]),
            ),
        )
    end
    push!(tensors, Tlast)

    return TCI.TensorTrain{V,4}(Array{V,4}.(copy.(tensors)))
end


function contract_tci2(
    M1::MPO,
    M2::MPO;
    cutoff = 0.0,
    tolerance = 0.0,
    maxdim = typemax(Int),
    maxbonddim = typemax(Int),
)::MPO
    contractinds = setdiff.(commoninds.(M1, M2), Ref(linkinds(M1)), Ref(linkinds(M2)))
    sites1 = setdiff.(siteinds(M1), Ref(contractinds))
    sites2 = setdiff.(siteinds(M2), Ref(contractinds))

    sitesleft = setdiff.(sites1, contractinds)
    sitesright = setdiff.(sites2, contractinds)

    T = promote_type(eltype.(M1)..., eltype.(M2)...)

    tt1 = _mpo_to_tt(T, M1, sitesleft, contractinds)
    tt2 = _mpo_to_tt(T, M2, contractinds, sitesright)

    res = TCIA.contract_TCI(
        tt1,
        tt2;
        tolerance = max(cutoff, tolerance),
        maxbonddim = min(maxdim, maxbonddim),
    )
    return _tt_to_mpo(res, sitesleft, sitesright)
end


function _tt_to_mpo(tt::TCI.TensorTrain{V,4}, sites1, sites2)::MPO where {V}
    linkdims = [size(tt[n], 4) for n = 1:length(tt)-1]

    links = [Index(linkdims[l], "Link,l=$l") for l = 1:length(linkdims)]

    Tfirst = ITensor(dropdims(tt[1]; dims = 1), sites1[1]..., sites2[1]..., links[1])
    Tlast =
        ITensor(dropdims(tt[end]; dims = 4), links[end], sites1[end]..., sites2[end]...)

    tensors = ITensor[Tfirst]
    for n = 2:length(tt)-1
        push!(tensors, ITensor(tt[n], links[n-1], sites1[n]..., sites2[n]..., links[n]))
    end
    push!(tensors, Tlast)

    return MPO(tensors)
end

function _iscompatible(p1::Vector{Int}, p2::Vector{Int})::Bool
    all(p1 .>= 0) || error("p1 must be non-negative")
    all(p2 .>= 0) || error("p2 must be non-negative")

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
