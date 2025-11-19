"""
Taken from from ITensors.jl
with some modifications.
* The dominator is replaced by <y|A|x> instead of <x|A|A|x> to reduce the computational cost.
* Specialized to MPO-MPO contraction.


    error_contract(y::MPS, A::MPO, x::MPS;
                   make_inds_match::Bool = true)
    error_contract(y::MPS, x::MPS, x::MPO;
                   make_inds_match::Bool = true)

Compute the distance between A|x> and an approximation MPS y:
`| |y> - A|x> |/| A|x> | = √(1 + (<y|y> - 2*real(<y|A|x>))/<y|A|x>)`.

If `make_inds_match = true`, the function attempts match the site
indices of `y` with the site indices of `A` that are not common
with `x`.
"""
function error_contract(y::MPO, A::MPO, x::MPO; kwargs...)
    N = length(A)
    if length(y) != N || length(x) != N
        throw(
            DimensionMismatch(
                "inner: mismatched lengths $N and $(length(x)) or $(length(y))",
            ),
        )
    end
    iyy = dot(y, y; kwargs...)
    iyax = _dot(y, A, x; kwargs...)
    return sqrt(abs(1.0 + (iyy - 2 * real(iyax)) / abs(iyax)))
end

function _dot(y::MPO, A::MPO, x::MPO; kwargs...)
    return _log_or_not_dot(y, A, x, false; kwargs...)
end

function _log_or_not_dot(y::MPO, A::MPO, x::MPO, loginner::Bool; kwargs...)::Number
    N = length(A)
    ydag = dag(y)
    sim!(linkinds, ydag)
    check_hascommoninds(siteinds, A, y)
    O = ydag[1] * A[1] * x[1]

    log_inner_tot = 0.0
    if loginner
        normO = norm(O)
        log_inner_tot = log(normO)
        O ./= normO
    end
    for j = 2:N
        O = O * ydag[j] * A[j] * x[j]
        if length(inds(O)) > 3
            error("Link inds are too many! Something may be wrong")
        end
        if loginner
            normO = norm(O)
            log_inner_tot += log(normO)
            O ./= normO
        end
    end
    if loginner
        if !isreal(O[]) || real(O[]) < 0
            log_inner_tot += log(complex(O[]))
        end
        return log_inner_tot
    else
        return O[]
    end
end

function flatten(A::AbstractVector{<:AbstractVector{T}})::Vector{T} where {T}
    return vcat(A...)
end
