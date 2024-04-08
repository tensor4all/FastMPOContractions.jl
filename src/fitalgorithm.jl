# Only temporary, until the performance issune in ITensorNetworks.jl is fixed.
import ITensors.ITensorMPS: AbstractProjMPO, makeL!, makeR!, set_nsite!, OneITensor
import Base: copy

"""
Contract M1 and M2, and return the result as an MPO.
"""
function contract_fit(M1::MPO, M2::MPO; init = nothing, kwargs...)::MPO
    M2_ = MPS([M2[v] for v in eachindex(M2)])
    if init === nothing
        init = M2_
    else
        init = MPS([init[v] for v in eachindex(M2)])
    end
    M12_ = contract_fit(M1, M2_; init_mps = init, kwargs...)
    M12 = MPO([M12_[v] for v in eachindex(M1)])

    return M12
end

#==
function contract_fit(M1s::Vector{MPO}, M2s::Vector{MPO}; init=nothing,
    kwargs...)::Vector{MPO}
    @show init
    N = length(M1s[1])
    M2s_ = [MPS([M2[v] for v in eachindex(M2)]) for M2 in M2s]
    if init === nothing
        init = M2s_[1]
    else
        init = MPS([init[v] for v in eachindex(M2s[1])]) 
    end
    @show init
    M12s_ = contract_fit(M1s, M2s_; init_mps=init, kwargs...)
    return [MPO([M12_[v] for v in 1:N]) for M12_ in M12s_]
end
==#


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

    t = Inf
    reverse_step = false
    PH = ITensorTDVP.ProjMPOApply(psi0, A)
    psi = ITensorTDVP.alternating_update(
        ITensorTDVP.contractmpo_solver(; kwargs...),
        PH,
        t,
        init_mps;
        nsweeps,
        reverse_step,
        kwargs...,
    )

    return psi
end


#===
function contract_fit(A::Vector{MPO}, psi0::Vector{MPS}; init_mps=psi0[1], nsweeps=1, kwargs...)::MPS
    nterm = length(A)
    nterm == length(psi0) || error("Number of terms in MPO ($nterm) and MPS ($(length(psi0))) do not match")
    n = length(A[1])
    n != length(psi0[1]) &&
        throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi0[1]))) do not match"))
    if n == 1
        return MPS([[A[i][1] * psi0[i][1] for i in 1:nterm]])
    end

    for t in 1:nterm
        any(i -> isempty(i), siteinds(commoninds, A[t], psi0[t])) &&
            error("In `contract(A::MPO, x::MPS)`, `A` and `x` must share a set of site indices")
    end

    # In case A and psi0 have the same link indices
    for t in 1:nterm
        A[t] = sim(linkinds, A[t])
    end

    # Fix site and link inds of init_mps
    init_mps = deepcopy(init_mps)
    init_mps = sim(linkinds, init_mps)
    init_mpsi = siteinds(init_mps)
    Ai = siteinds(A[1])
    for j in 1:n
        ti = nothing
        for i in Ai[j]
            if !hasind(psi0[1][j], i)
                ti = i
                break
            end
        end
        if ti !== nothing
            ci = commoninds(init_mpsi[j], A[1][j])[1]
            replaceind!(init_mps[j], ci => ti)
        end
    end

    t = Inf
    reverse_step = false
    PH = ProjMPOApplySum(psi0, A)
    psi = ITensorTDVP.tdvp(contractmpo_solver(; kwargs...), PH, t, init_mps;
        nsweeps, reverse_step, kwargs...)

    return psi
end


function contractmpo_solver(; kwargs...)
    function solver(PH, t, psi; kws...)
        Hpsi0s = []
        for t in 1:length(PH.H)
            v = ITensor(true)
            for j in (PH.lpos + 1):(PH.rpos - 1)
                v *= PH.psi0[t][j]
            end
            push!(Hpsi0s, contract(PH, v, t))
        end
        Hpsi0 = sum(Hpsi0s)
        return Hpsi0, nothing
    end 
    return solver
 end

"""
# Original docstring
A ProjMPOApply represents the application of an
MPO `H` onto an MPS `psi0` but "projected" by
the basis of a different MPS `psi` (which
could be an approximation to H|psi>).

As an implementation of the AbstractProjMPO
type, it supports multiple `nsite` values for
one- and two-site algorithms.

```
     *--*--*-      -*--*--*--*--*--* <psi|
     |  |  |  |  |  |  |  |  |  |  |
     h--h--h--h--h--h--h--h--h--h--h H  
     |  |  |  |  |  |  |  |  |  |  |
     o--o--o-      -o--o--o--o--o--o |psi0>
```
"""
mutable struct ProjMPOApplySum <: AbstractProjMPO
    lpos::Int
    rpos::Int
    nsite::Int
    psi0::Vector{MPS}
    H::Vector{MPO}
    LR::Vector{Vector{ITensor}}
end

nterm(Prj::ProjMPOApplySum) = length(Prj.projs)

function lproj(P::ProjMPOApplySum, t::Int)
    (P.lpos <= 0) && return nothing
    return P.LR[P.lpos][t]
end

function lproj(P::ProjMPOApplySum)
    (P.lpos <= 0) && return nothing
    return P.LR[P.lpos]
end


function rproj(P::ProjMPOApplySum, t::Int)
    (P.rpos >= length(P) + 1) && return nothing
    return P.LR[P.rpos][t]
end

function rproj(P::ProjMPOApplySum)
    (P.rpos >= length(P) + 1) && return nothing
    return P.LR[P.rpos]
end

function contract(P::ProjMPOApplySum, v::ITensor, t::Int)::ITensor
    @show P.lpos
    @show P.LR[P.lpos]
   @show lproj(P)
    itensor_map = Union{ITensor,OneITensor}[lproj(P, t)]
    append!(itensor_map, P.H[t][site_range(P)])
    push!(itensor_map, rproj(P, t))

    # Reverse the contraction order of the map if
    # the first tensor is a scalar (for example we
    # are at the left edge of the system)
    if dim(first(itensor_map)) == 1
      reverse!(itensor_map)
    end

    # Apply the map
    Hv = v
    for it in itensor_map
      Hv *= it
    end
    return Hv
end

function ProjMPOApplySum(psi0::Vector{MPS}, H::Vector{MPO})
    nterm = length(psi0)
    N = length(psi0[1])
    LR = Vector{ITensor}[Vector{ITensor}(undef, nterm) for _ in 1:nterm]
    return ProjMPOApplySum(0, N + 1, 2, psi0, H, LR)
end

function copy(P::ProjMPOApplySum)
    return ProjMPOApplySum(P.lpos, P.rpos, P.nsite, copy(P.psi0), copy(P.H), deepcopy(P.LR))
end

function set_nsite!(P::ProjMPOApplySum, nsite)
    P.nsite = nsite
    return P
end

function makeL!(P::ProjMPOApplySum, psi::MPS, k::Int)
    # Save the last `L` that is made to help with caching
    # for DiskProjMPO
    ll = P.lpos
    nterm = length(P.H)
    if ll ≥ k
      # Special case when nothing has to be done.
      # Still need to change the position if lproj is
      # being moved backward.
      P.lpos = k
      return nothing
    end
    # Make sure ll is at least 0 for the generic logic below
    ll = max(ll, 0)
    L = lproj(P)
    while ll < k
        for t in 1:nterm
            L[t] = L[t] * P.psi0[t][ll + 1] * P.H[t][ll + 1] * dag(psi[ll + 1])
            P.LR[ll + 1][t] = L[t]
        end
        ll += 1
    end
    # Needed when moving lproj backward.
    P.lpos = k
    return P
end

function makeR!(P::ProjMPOApplySum, psi::MPS, k::Int)
  # Save the last `R` that is made to help with caching
  # for DiskProjMPO
    nterm = length(P.H)
  rl = P.rpos
  if rl ≤ k
    # Special case when nothing has to be done.
    # Still need to change the position if rproj is
    # being moved backward.
    P.rpos = k
    return nothing
  end
  N = length(P.H)
  # Make sure rl is no bigger than `N + 1` for the generic logic below
  rl = min(rl, N + 1)
  R = rproj(P)
  while rl > k
    for t in 1:nterm
        R[t] = R[t] * P.psi0[t][rl - 1] * P.H[t][rl - 1] * dag(psi[rl - 1])
        P.LR[rl - 1][t] = R[t]
    end
    rl -= 1
  end
  P.rpos = k
  return P
end
==#
