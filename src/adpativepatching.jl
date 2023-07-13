abstract type AbstractAdaptiveTCINode{C} end

struct AdaptiveLeaf{C} <: AbstractAdaptiveTCINode{C}
    data::C
    prefix::Vector{Int}
end

function Base.show(io::IO, obj::AdaptiveLeaf{C}) where {C}
    prefix = convert.(Int, obj.prefix)
    return println(
        io,
        "  "^length(prefix) * "Leaf $(prefix): rank=$(maximum(_linkdims(obj.data)))",
    )
end

_linkdims(tci::TensorCI2{T}) where {T} = TCI.linkdims(tci)

struct AdaptiveInternalNode{C} <: AbstractAdaptiveTCINode{C}
    children::Dict{Int,AbstractAdaptiveTCINode{C}}
    prefix::Vector{Int}

    function AdaptiveInternalNode{C}(
        children::Dict{Int,AbstractAdaptiveTCINode{C}},
        prefix::Vector{Int},
    ) where {C}
        return new{C}(children, prefix)
    end
end

"""
prefix is the common prefix of all children
"""
function AdaptiveInternalNode{C}(
    children::Vector{AbstractAdaptiveTCINode{C}},
    prefix::Vector{Int},
) where {C}
    d = Dict{Int,AbstractAdaptiveTCINode{C}}()
    for child in children
        d[child.prefix[end]] = child
    end
    return AdaptiveInternalNode{C}(d, prefix)
end

function Base.show(io::IO, obj::AdaptiveInternalNode{C}) where {C}
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
function evaluate(obj::AdaptiveInternalNode{C}, idx::AbstractVector{Int}) where {C}
    child_key = idx[length(obj.prefix)+1]
    return evaluate(obj.children[child_key], idx)
end

function evaluate(obj::AdaptiveLeaf{C}, idx::AbstractVector{Int}) where {C}
    return _evaluate(obj.data, idx[(length(obj.prefix)+1):end])
end


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
        return AdaptiveLeaf{C}(first(patches)[2], common_prefix)
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

    return AdaptiveInternalNode{C}(children, common_prefix)
end

"""
T: Float64, ComplexF64, etc.
M: TensorCI2, MPS, etc.
"""
abstract type AbstractPatchCreator{T,M} end


mutable struct PatchCreatorResult{T,M}
    data::M
    isconverged::Bool
end


function adaptivepatches(
    creator::AbstractPatchCreator{T,M};
    sleep_time::Float64 = 1e-6,
    maxnleaves = 100,
    verbosity = 0,
)::Union{AdaptiveLeaf{M},AdaptiveInternalNode{M}} where {T,M}
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
                        println("Creating a patch for $(prefix_) ...")
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
        #if v isa Future || !v.isconverged
        #error("Something got wrong. All leaves must be fetched and converged! $(v) $(v.isconverged)")
        #end
        leaves_done[k] = v.data
    end
    @show length(leaves_done)

    return _to_tree(leaves_done)
end


#======================================================================
   TCI2 Interpolation of a function
======================================================================#
_evaluate(obj::TensorCI2, idx) = TCI.evaluate(obj, idx)

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
    tcikwargs = Dict(),
    ntry = 100,
)::TCI2PatchCreator{T} where {T}
    maxval, _ = _estimate_maxval(f, localdims; ntry = ntry)
    return TCI2PatchCreator{T}(
        f,
        localdims,
        rtol,
        maxbonddim,
        verbosity,
        tcikwargs,
        maxval,
        rtol * maxval,
    )
end


function _crossinterpolate2(
    ::Type{T},
    f,
    localdims::Vector{Int},
    initialpivots::Vector{MultiIndex},
    tolerance::Float64;
    maxbonddim::Int = typemax(Int),
    verbosity::Int = 0,
) where {T}
    tci, others = TCI.crossinterpolate2(
        T,
        f,
        localdims,
        initialpivots;
        tolerance = 1e-1 * tolerance,
        maxbonddim = maxbonddim,
        verbosity = verbosity,
        normalizeerror = false,
    )

    err(x) = abs(TCI.evaluate(tci, x) - f(x))
    abserror = 0.0
    for _ = 1:10
        p = TCI.optfirstpivot(err, localdims, [rand(1:d) for d in localdims])
        newerr = abs(err(p))
        if abserror < newerr
            abserror = newerr
        end
    end

    true_error = max(TCI.maxbonderror(tci), abserror)

    return PatchCreatorResult{T,TensorCI2{T}}(
        tci,
        true_error < tolerance && maximum(TCI.linkdims(tci)) <= maxbonddim,
    )
end

function createpatch(
    obj::TCI2PatchCreator{T},
    prefix::AbstractVector{Int},
)::Future where {T}
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


function _estimate_maxval(f, localdims; ntry = 100)
    pivot = fill(1, length(localdims))
    maxval::Float64 = abs(f(pivot))
    for i = 1:ntry
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
