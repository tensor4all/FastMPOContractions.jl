using ITensorMPS: ITensorMPS, AbstractProjMPO, MPO, MPS
using ITensorMPS: linkinds, replaceinds
using ITensors: ITensors, OneITensor
import ITensorMPS: alternating_update, rproj, lproj

"""
A ReducedFitProblem represents the projection
of an MPS `input_state` onto the basis of a different MPS `state`.
`state` may be an approximation of `input_state`.
```
     *--*--*-      -*--*--*--*--*--* <state|
     |  |  |  |  |  |  |  |  |  |  |
     o--o--o-      -o--o--o--o--o--o |input_state>
```
"""
mutable struct ReducedFitProblem <: AbstractProjMPO
    lpos::Int
    rpos::Int
    nsite::Int
    input_state::MPS
    environments::Vector{ITensor}
end

function ReducedFitProblem(input_state::MPS)
    lpos = 0
    rpos = length(input_state) + 1
    nsite = 2
    environments = Vector{ITensor}(undef, length(input_state))
    return ReducedFitProblem(lpos, rpos, nsite, input_state, environments)
end

function lproj(P::ReducedFitProblem)::Union{ITensor,OneITensor}
    (P.lpos <= 0) && return OneITensor()
    return P.environments[P.lpos]
end

function rproj(P::ReducedFitProblem)::Union{ITensor,OneITensor}
    (P.rpos >= length(P) + 1) && return OneITensor()
    return P.environments[P.rpos]
end


function Base.copy(reduced_operator::ReducedFitProblem)
    return ReducedFitProblem(
        reduced_operator.lpos,
        reduced_operator.rpos,
        reduced_operator.nsite,
        copy(reduced_operator.input_state),
        copy(reduced_operator.environments),
    )
end

Base.length(reduced_operator::ReducedFitProblem) = length(reduced_operator.input_state)

function ITensorMPS.set_nsite!(reduced_operator::ReducedFitProblem, nsite)
    reduced_operator.nsite = nsite
    return reduced_operator
end

function ITensorMPS.makeL!(reduced_operator::ReducedFitProblem, state::MPS, k::Int)
    # Save the last `L` that is made to help with caching
    # for DiskProjMPO
    ll = reduced_operator.lpos
    if ll ≥ k
        # Special case when nothing has to be done.
        # Still need to change the position if lproj is
        # being moved backward.
        reduced_operator.lpos = k
        return nothing
    end
    # Make sure ll is at least 0 for the generic logic below
    ll = max(ll, 0)
    L = lproj(reduced_operator)
    while ll < k
        L = L * reduced_operator.input_state[ll+1] * dag(state[ll+1])
        reduced_operator.environments[ll+1] = L
        ll += 1
    end
    # Needed when moving lproj backward.
    reduced_operator.lpos = k
    return reduced_operator
end

function ITensorMPS.makeR!(reduced_operator::ReducedFitProblem, state::MPS, k::Int)
    # Save the last `R` that is made to help with caching
    # for DiskProjMPO
    rl = reduced_operator.rpos
    if rl ≤ k
        # Special case when nothing has to be done.
        # Still need to change the position if rproj is
        # being moved backward.
        reduced_operator.rpos = k
        return nothing
    end
    N = length(state)
    # Make sure rl is no bigger than `N + 1` for the generic logic below
    rl = min(rl, N + 1)
    R = rproj(reduced_operator)
    while rl > k
        R = R * reduced_operator.input_state[rl-1] * dag(state[rl-1])
        reduced_operator.environments[rl-1] = R
        rl -= 1
    end
    reduced_operator.rpos = k
    return reduced_operator
end


struct ReducedFitMPSsProblem <: AbstractProjMPO
    problems::Vector{ReducedFitProblem}
    coeffs::Vector{<:Number}
end

function ReducedFitMPSsProblem(
    input_states::AbstractVector{MPS},
    coeffs::AbstractVector{<:Number},
)
    ReducedFitMPSsProblem(ReducedFitProblem.(input_states), coeffs)
end

function Base.copy(reduced_operator::ReducedFitMPSsProblem)
    return ReducedFitMPSsProblem(reduced_operator.problems, reduced_operator.coeffs)
end

function Base.getproperty(reduced_operator::ReducedFitMPSsProblem, sym::Symbol)
    if sym === :nsite
        return getfield(reduced_operator, :problems)[1].nsite
    end
    return getfield(reduced_operator, sym)
end


Base.length(reduced_operator::ReducedFitMPSsProblem) = length(reduced_operator.problems[1])

function ITensorMPS.set_nsite!(reduced_operator::ReducedFitMPSsProblem, nsite)
    for p in reduced_operator.problems
        ITensorMPS.set_nsite!(p, nsite)
    end
    return reduced_operator
end

function ITensorMPS.makeL!(reduced_operator::ReducedFitMPSsProblem, state::MPS, k::Int)
    for p in reduced_operator.problems
        ITensorMPS.makeL!(p, state, k)
    end
    return reduced_operator
end


function ITensorMPS.makeR!(reduced_operator::ReducedFitMPSsProblem, state::MPS, k::Int)
    for p in reduced_operator.problems
        ITensorMPS.makeR!(p, state, k)
    end
    return reduced_operator
end



function _contract(P::ReducedFitProblem, v::ITensor)::ITensor
    itensor_map = Union{ITensor,OneITensor}[lproj(P)]
    push!(itensor_map, rproj(P))

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

function contract_operator_state_updater(operator::ReducedFitProblem, init; internal_kwargs)
    state = ITensor(true)
    for j = (operator.lpos+1):(operator.rpos-1)
        state *= operator.input_state[j]
    end
    state = _contract(operator, state)
    return state, (;)
end

function contract_operator_state_updater(
    operator::ReducedFitMPSsProblem,
    init;
    internal_kwargs,
)
    states = ITensor[]
    for (p, coeff) in zip(operator.problems, operator.coeffs)
        res = contract_operator_state_updater(p, init; internal_kwargs)
        push!(states, coeff * res[1])
    end
    return sum(states), (;)
end


function contract_fit(input_state::MPS, init::MPS; coeff::Number = 1, kwargs...)::MPS
    links = ITensors.sim.(linkinds(init))
    init = replaceinds(linkinds, init, links)
    reduced_operator = ReducedFitProblem(input_state)
    return alternating_update(
        reduced_operator,
        init;
        updater = contract_operator_state_updater,
        kwargs...,
    )
end


function fit(
    input_states::AbstractVector{MPS},
    init::MPS;
    coeffs::AbstractVector{<:Number} = ones(Int, length(input_states)),
    kwargs...,
)::MPS
    links = ITensors.sim.(linkinds(init))
    init = replaceinds(linkinds, init, links)
    reduced_operator = ReducedFitMPSsProblem(input_states, coeffs)
    return alternating_update(
        reduced_operator,
        init;
        updater = contract_operator_state_updater,
        kwargs...,
    )
end

function fit(
    input_states::AbstractVector{MPO},
    init::MPO;
    coeffs::AbstractVector{<:Number} = ones(Int, length(input_states)),
    kwargs...,
)::MPO
    to_mps(Ψ::MPO) = MPS([x for x in Ψ])

    res = fit(to_mps.(input_states), to_mps(init); coeffs = coeffs, kwargs...)
    return MPO([x for x in res])
end
