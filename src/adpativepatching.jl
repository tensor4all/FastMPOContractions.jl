abstract type AbstractAdaptiveTCINode{C} end

struct AdaptiveTCILeaf{C} <: AbstractAdaptiveTCINode{C}
    data::C
    prefix::Vector{Int}
end

function Base.show(io::IO, obj::AdaptiveTCILeaf{C}) where {C}
    prefix = convert.(Int, obj.prefix)
    return println(
        io,
        "  "^length(prefix) * "Leaf $(prefix): rank=$(maximum(_linkdims(obj.data)))",
    )
end

_linkdims(tci::TensorCI2{T}) where {T} = TCI.linkdims(tci)

struct AdaptiveTCIInternalNode{C} <: AbstractAdaptiveTCINode{C}
    children::Dict{Int,AbstractAdaptiveTCINode{C}}
    prefix::Vector{Int}

    function AdaptiveTCIInternalNode{C}(
        children::Dict{Int,AbstractAdaptiveTCINode{C}},
        prefix::Vector{Int},
    ) where {C}
        return new{C}(children, prefix)
    end
end

"""
prefix is the common prefix of all children
"""
function AdaptiveTCIInternalNode{C}(
    children::Vector{AbstractAdaptiveTCINode{C}},
    prefix::Vector{Int},
) where {C}
    d = Dict{Int,AbstractAdaptiveTCINode{C}}()
    for child in children
        d[child.prefix[end]] = child
    end
    return AdaptiveTCIInternalNode{C}(d, prefix)
end

function Base.show(io::IO, obj::AdaptiveTCIInternalNode{C}) where {C}
    println(
        io,
        "  "^length(obj.prefix) *
        "InternalNode $(obj.prefix) with $(length(obj.children)) children",
    )
    for (k, v) in obj.children
        Base.show(io, v)
    end
end

"""
Evaluate the tree at given idx
"""
function evaluate(obj::AdaptiveTCIInternalNode{C}, idx::AbstractVector{Int}) where {C}
    child_key = idx[length(obj.prefix)+1]
    return evaluate(obj.children[child_key], idx)
end

function evaluate(obj::AdaptiveTCILeaf{C}, idx::AbstractVector{Int}) where {C}
    return _evaluate(obj.data, idx[(length(obj.prefix)+1):end])
end

_evaluate(obj::TensorCI2, idx) = TCI.evaluate(obj, idx)

"""
Convert a dictionary of patches to a tree
"""
function _to_tree(
    patches::Dict{Vector{Int},C};
    nprefix = 0,
)::AbstractAdaptiveTCINode{C} where {C}
    length(unique(k[1:nprefix] for (k, v) in patches)) == 1 ||
        error("Inconsistent prefixes")

    common_prefix = first(patches)[1][1:nprefix]

    # Return a leaf
    if nprefix == length(first(patches)[1])
        return AdaptiveTCILeaf{C}(first(patches)[2], common_prefix)
    end

    subgroups = Dict{Int,Dict{Vector{Int},C}}()

    # Look at the first index after nprefix skips
    # and group the patches by that index
    for (k, v) in patches
        idx = k[nprefix+1]
        if idx in keys(subgroups)
            subgroups[idx][k] = v
        else
            subgroups[idx] = Dict{Vector{Int},C}(k => v)
        end
    end

    # Recursively construct the tree
    children = AbstractAdaptiveTCINode{C}[]
    for (_, grp) in subgroups
        push!(children, _to_tree(grp; nprefix = nprefix + 1))
    end

    return AdaptiveTCIInternalNode{C}(children, common_prefix)
end

"""
T: Float64, ComplexF64, etc.
M: TensorCI2, MPS, etc.
"""
abstract type AbstractPatchCreator{T,M} end

mutable struct TCI2PatchCreator{T} <: AbstractPatchCreator{T,TensorCI2{T}}
    f::Any
    localdims::Vector{Int}
    rtol::Float64
    maxbonddim::Int
    verbosity::Int
    tcikwargs::Dict
    maxval::Float64
    atol::Float64
end


function TCI2PatchCreator(
    ::Type{T},
    f,
    localdims::AbstractVector{Int};
    rtol::Float64 = 1e-8,
    maxbonddim::Int = 100,
    verbosity::Int = 0,
    tcikwargs=Dict(),
)::TCI2PatchCreator{T} where {T}
    maxval, _ = _estimate_maxval(f, localdims; ntry=100)
    return TCI2PatchCreator{T}(
        f,
        localdims,
        rtol,
        maxbonddim,
        verbosity,
        tcikwargs,
        maxval,
        rtol * maxval
    )
end

#==
module ResultStatus
@enum EnumType begin
    WAITING = 1
    FETCHABLE = 2
    FETCHED = 3
end
end
==#

mutable struct PatchCreatorResult{T,M}
    data::M
    isconverged::Bool
end

#==
function getstatus(res::PatchCreatorResult{T,M})::ResultStatus.EnumType where {T,M}
    if res.data isa Future
        if !isready(res.spwanresult)
            return ResultStatus.WAITING
        else
            return ResultStatus.FETCHABLE
        end
    else
        return ResultStatus.FETCHED
    end
end
==#

function _estimate_maxval(f, localdims; ntry=100)
    pivot = fill(1, length(localdims))
    maxval::Float64 = abs(f(pivot))
    for i in 1:ntry
        pivot_ = [rand(1:localdims[i]) for i in eachindex(localdims)]
        pivot_ = TCI.optfirstpivot(f, localdims, pivot_)
        maxval_ = abs(f(pivot_))
        if maxval_ > maxval
            maxval = maxval_
            pivot .= pivot_
        end
    end
    return maxval, pivot
end


function _crossinterpolate2(
    ::Type{T},
    f,
    localdims::Vector{Int},
    initialpivots::Vector{MultiIndex},
    tolerance::Float64;
    maxbonddim::Int=typemax(Int),
    verbosity::Int=0
)  where {T}
    tci, _, _ = TCI.crossinterpolate2(
        T,
        f,
        localdims,
        initialpivots;
        tolerance = tolerance,
        maxbonddim = maxbonddim,
        verbosity = verbosity,
        normalizeerror = false
    )

    return PatchCreatorResult{T,TensorCI2{T}}(tci, TCI.maxbonderror(tci) < tolerance)
end

function createpatch(obj::TCI2PatchCreator{T}, prefix::AbstractVector{Int})::Future where {T}
    localdims_ = obj.localdims[(length(prefix)+1):end]
    f_ = x -> obj.f(vcat(prefix, x))
    firstpivot = TCI.optfirstpivot(f_, localdims_, fill(1, length(localdims_)))

    return @spawnat :any _crossinterpolate2(
        T,
        f_,
        localdims_,
        [firstpivot],
        obj.atol;
        maxbonddim = obj.maxbonddim,
        verbosity = obj.verbosity,
    )
end


#==
function fetch!(creator::AbstractPatchCreator{T,M}, res::PatchCreatorResult{T,M})::Nothing where {T,M}
    getstatus(res) == ResultStatus.FETCHABLE || error("Invalid state of the result")

    if res.spwanresult isa Future
        res.spwanresult, _, _ = fetch(res.spwanresult)
        res.isconverged = TCI.maxbonderror(res.spwanresult) < creator.atol
        res.status = ResultStatus.FETCHED
    elseif res.spwanresult isa RemoteException
        error("An exception occured: $(res)")
    else
        error("Something got wrong")
    end

    return nothing
end
==#
#==
function isconverged(res::TCI2PatchCreatorResult{T})::Bool where {T}
    res.spwanresult isa TensorCI2{T} || error("Not ready yet")

    tci = res.spwanresult 
    return TCI.maxbonderror(tci) < res.creator.atol
end
==#


function adaptivepatchtci2(
    creator::AbstractPatchCreator{T,M};
    sleep_time::Float64 = 1e-6,
    maxnleaves = 100,
    verbosity = 0
)::Union{AdaptiveTCILeaf{M},AdaptiveTCIInternalNode{M}} where {T,M}
    leaves = Dict{Vector{Int},Union{Future,PatchCreatorResult{T,M}}}()

    # Add root
    root = createpatch(creator, Int[])
    while !isready(root)
        sleep(sleep_time) # Not to run the loop too quickly
    end
    leaves[[]] = fetch(root)

    while true
        sleep(sleep_time) # Not to run the loop too quickly

        done = true
        for (prefix, leaf) in leaves

            # Fetch leaf if ready
            if leaf isa Future
                done = false
                if isready(leaf)
                    leaves[prefix] = fetch(leaf)
                end
            elseif !(leaf isa Future) && !leaf.isconverged && length(leaves) < maxnleaves
                done = false
                delete!(leaves, prefix)

                for ic = 1:creator.localdims[length(prefix)+1]
                    prefix_ = vcat(prefix, ic)
                    if verbosity > 0
                        println("Interpolating $(prefix_) ...")
                    end
                    leaves[prefix_] = createpatch(creator, prefix_)
                end
            end
        end
        if done
            break
        end
    end

    leaves_done = Dict{Vector{Int},M}()
    for (k, v) in leaves
        if v isa Future || !v.isconverged
            error("Something got wrong. All leaves must be fetched and converged!")
        end
        leaves_done[k] = v.data
    end

    return _to_tree(leaves_done)
end
