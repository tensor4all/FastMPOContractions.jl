#===
Taken from ITensors.jl, which is licensed under the Apache license, with some modifications to support MPO-MPS contraction.
===#
function contract_densitymatrix(A::AbstractMPS, ψ::AbstractMPS; kwargs...)
    n = length(A)
    n != length(ψ) &&
        throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(ψ))) do not match"))
    if n == 1
        return MPS([A[1] * ψ[1]])
    end

    ψ_out = similar(ψ)
    cutoff::Float64 = get(kwargs, :cutoff, 1e-13)
    requested_maxdim::Int = get(kwargs, :maxdim, maxlinkdim(A) * maxlinkdim(ψ))
    mindim::Int = max(get(kwargs, :mindim, 1), 1)
    normalize::Bool = get(kwargs, :normalize, false)

    any(i -> isempty(i), siteinds(commoninds, A, ψ)) &&
        error("In `contract(A::MPO, x::MPS)`, `A` and `x` must share a set of site indices")

    # In case A and ψ have the same link indices
    A = sim(linkinds, A)

    ψ_c = dag(ψ)
    A_c = dag(A)

    # To not clash with the link indices of A and ψ
    sim!(linkinds, A_c)
    sim!(linkinds, ψ_c)
    sim!(siteinds, commoninds, A_c, ψ_c)

    # In case ψ has a dangling bond on each site
    all_linkinds = vcat(linkinds(A), linkinds(A_c), linkinds(ψ), linkinds(ψ_c))
    dangling_inds_ψ = [uniqueinds(ψ[i], A[i], all_linkinds) for i = 1:n]
    sim_dangling_inds_ψ = [sim.(dangling_inds_ψ[i]) for i = 1:n]

    # A version helpful for making the density matrix
    simA_c = sim(siteinds, uniqueinds, A_c, ψ_c)

    # Store the left environment tensors
    E = Vector{ITensor}(undef, n - 1)

    E[1] = ψ[1] * A[1] * A_c[1] * ψ_c[1]
    for j = 2:(n-1)
        E[j] = E[j-1] * ψ[j] * A[j] * A_c[j] * ψ_c[j]
    end
    R = ψ[n] * A[n]
    simR_c = replaceinds(ψ_c[n], dangling_inds_ψ[n], sim_dangling_inds_ψ[n]) * simA_c[n]
    ρ = E[n-1] * R * simR_c
    l = linkind(ψ, n - 1)
    ts = isnothing(l) ? "" : tags(l)
    function _siteinds(Ai_, ψi_)
        return vcat(uniqueinds(Ai_, ψi_, all_linkinds), uniqueinds(ψi_, Ai_, all_linkinds))
    end
    Lis = _siteinds(A[n], ψ[n])
    Ris = _siteinds(
        simA_c[n],
        replaceinds(ψ_c[n], dangling_inds_ψ[n], sim_dangling_inds_ψ[n]),
    )
    F = eigen(ρ, Lis, Ris; ishermitian = true, tags = ts, kwargs...)
    D, U, Ut = F.D, F.V, F.Vt
    l_renorm, r_renorm = F.l, F.r
    ψ_out[n] = Ut
    R = R * dag(Ut) * ψ[n-1] * A[n-1]
    simR_c =
        simR_c *
        U *
        replaceinds(ψ_c[n-1], dangling_inds_ψ[n-1], sim_dangling_inds_ψ[n-1]) *
        simA_c[n-1]
    for j in reverse(2:(n-1))
        # Determine smallest maxdim to use
        cip = commoninds(ψ[j], E[j-1])
        ciA = commoninds(A[j], E[j-1])
        prod_dims = dim(cip) * dim(ciA)
        maxdim = min(prod_dims, requested_maxdim)

        s = _siteinds(A[j], ψ[j])
        s̃ = _siteinds(
            simA_c[j],
            replaceinds(ψ_c[j], dangling_inds_ψ[j], sim_dangling_inds_ψ[j]),
        )
        ρ = E[j-1] * R * simR_c
        l = linkind(ψ, j - 1)
        ts = isnothing(l) ? "" : tags(l)
        Lis = IndexSet(s..., l_renorm)
        Ris = IndexSet(s̃..., r_renorm)
        F = eigen(ρ, Lis, Ris; ishermitian = true, maxdim = maxdim, tags = ts, kwargs...)
        D, U, Ut = F.D, F.V, F.Vt
        l_renorm, r_renorm = F.l, F.r
        ψ_out[j] = Ut
        R = R * dag(Ut) * ψ[j-1] * A[j-1]
        simR_c =
            simR_c *
            U *
            replaceinds(ψ_c[j-1], dangling_inds_ψ[j-1], sim_dangling_inds_ψ[j-1]) *
            simA_c[j-1]
    end
    if normalize
        R ./= norm(R)
    end
    ψ_out[1] = R
    setleftlim!(ψ_out, 0)
    setrightlim!(ψ_out, 2)
    return ψ_out
end
