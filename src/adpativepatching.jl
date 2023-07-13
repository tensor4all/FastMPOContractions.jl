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
Construct QTTs using adaptive partitioning of the domain.

TODO

  - Allow arbitrary order of partitioning
"""
function adaptivepatchtci(
    ::Type{T},
    f,
    localdims::AbstractVector{Int};
    tolerance::Float64 = 1e-8,
    maxbonddim::Int = 100,
    firstpivot = ones(Int, length(localdims)),
    sleep_time::Float64 = 1e-6,
    verbosity::Int = 0,
    maxnleaves = 100,
    kwargs...,
)::Union{AdaptiveTCILeaf{TensorCI2{T}},AdaptiveTCIInternalNode{TensorCI2{T}}} where {T}
    R = length(localdims)
    leaves = Dict{Vector{Int},Union{TensorCI2{T},Future}}()

    # Add root node
    firstpivot = TCI.optfirstpivot(f, localdims, firstpivot)
    tci, ranks, errors = TCI.crossinterpolate2(
        T,
        f,
        localdims,
        [firstpivot];
        tolerance = tolerance,
        maxbonddim = maxbonddim,
        verbosity = verbosity,
        kwargs...,
    )
    leaves[[]] = tci
    maxsamplevalue = tci.maxsamplevalue

    while true
        sleep(sleep_time) # Not to run the loop too quickly

        done = true
        for (prefix, tci) in leaves
            if tci isa Future
                done = false
                if isready(tci)
                    res = try
                        fetch(tci)
                    catch ex
                        error("An exception occured: $ex")
                    end
                    if res isa RemoteException
                        error("An exception occured: $(res)")
                    end
                    tci = leaves[prefix] = res[1]
                    if verbosity > 0
                        println(
                            "Fetched, bond dimension = $(maximum(TCI.linkdims(leaves[prefix]))) $(TCI.maxbonderror(tci)) $(tci.maxsamplevalue) for $(prefix)",
                        )
                    end
                    maxsamplevalue = max(maxsamplevalue, leaves[prefix].maxsamplevalue)
                end
            elseif tci isa TensorCI2 &&
                   (TCI.maxbonderror(tci) > tolerance * maxsamplevalue) &&
                   length(leaves) < maxnleaves
                done = false
                delete!(leaves, prefix)
                for ic = 1:localdims[length(prefix)+1]
                    prefix_ = vcat(prefix, ic)
                    localdims_ = localdims[(length(prefix_)+1):end]
                    f_ = x -> f(vcat(prefix_, x))

                    firstpivot_ = ones(Int, R - length(prefix_))
                    maxval = abs(f_(firstpivot_))

                    for r = 1:10
                        firstpivot_rnd =
                            [rand(1:localdims_[r]) for r in eachindex(localdims_)]
                        firstpivot_rnd = TCI.optfirstpivot(f_, localdims_, firstpivot_rnd)
                        if abs(f_(firstpivot_rnd)) > maxval
                            firstpivot_ = firstpivot_rnd
                            maxval = abs(f_(firstpivot_))
                        end
                    end

                    if verbosity > 0
                        println("Interpolating $(prefix_) ...")
                    end
                    leaves[prefix_] = @spawnat :any TCI.crossinterpolate2(
                        T,
                        f_,
                        localdims_,
                        [firstpivot_];
                        tolerance = tolerance * maxsamplevalue,
                        maxbonddim = maxbonddim,
                        verbosity = verbosity,
                        normalizeerror = false,
                        kwargs...,
                    )
                end
            end
        end
        if done
            break
        end
    end

    leaves_done = Dict{Vector{Int},TensorCI2{T}}()
    for (k, v) in leaves
        if v isa Future
            error("Something got wrong. Not all leaves are fetched")
        end
        if TCI.maxbonderror(v) > tolerance * maxsamplevalue
            error(
                "TCI for k= $(k) has bond error $(TCI.maxbonderror(v)) larger than $(tolerance) * $(maxsamplevalue) = $(tolerance*maxsamplevalue)!",
            )
        end
        leaves_done[k] = v
    end

    return _to_tree(leaves_done)
end
