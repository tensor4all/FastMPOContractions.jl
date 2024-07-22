# Only temporary, until the performance issune in ITensorNetworks.jl is fixed.
import ITensors.ITensorMPS: AbstractProjMPO, makeL!, makeR!, set_nsite!, OneITensor
import Base: copy

"""
Contract M1 and M2, and return the result as an MPO.
"""
function contract_fit(M1::MPO, M2::MPO; init = nothing, kwargs...)::MPO
    M2_ = MPS([M2[v] for v in eachindex(M2)])
    if init === nothing
        init_MPO::MPO = ITensors.contract(M1, M2; alg = "zipup", kwargs...)
        init = MPS([init_MPO[v] for v in eachindex(init_MPO)])
    else
        init = MPS([init[v] for v in eachindex(M2)])
    end
    M12_ = contract_fit(M1, M2_; init_mps = init, kwargs...)
    M12 = MPO([M12_[v] for v in eachindex(M1)])

    return M12
end


# To support MPO-MPO contraction
# Taken from https://github.com/shinaoka/ITensorTDVP.jl/commit/23e09395cce66215b256aeeaa993fe2c64a0f1c8
function contract_fit(A::MPO, psi0::MPS; init_mps = psi0, nsweeps = 1, kwargs...)::MPS
    n = length(A)
    n != length(psi0) && throw(
        DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi0))) do not match"),
    )
    if n == 1
        return MPS([A[1] * psi0[1]])
    end

    any(i -> isempty(i), siteinds(commoninds, A, psi0)) &&
        error("In `contract(A::MPO, x::MPS)`, `A` and `x` must share a set of site indices")

    # In case A and psi0 have the same link indices
    A = sim(linkinds, A)

    # Fix site and link inds of init_mps
    init_mps = deepcopy(init_mps)
    init_mps = sim(linkinds, init_mps)
    Ai = siteinds(A)
    init_mpsi = siteinds(init_mps)
    for j = 1:n
        ti = nothing
        for i in Ai[j]
            if !hasind(psi0[j], i)
                ti = i
                break
            end
        end
        if ti !== nothing
            ci = commoninds(init_mpsi[j], A[j])[1]
            replaceind!(init_mps[j], ci => ti)
        end
    end

    reduced_operator = ITensorTDVP.ReducedContractProblem(psi0, A)
    return ITensorTDVP.alternating_update(
        reduced_operator,
        init_mps;
        updater = ITensorTDVP.contract_operator_state_updater,
        nsweeps = nsweeps,
        kwargs...,
    )
end
