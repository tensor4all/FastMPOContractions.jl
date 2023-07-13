abstract type AbstractAdaptiveTCINode end

struct AdaptiveTCILeaf{T<:Number} <: AbstractAdaptiveTCINode
    tci::TensorCI2{T}
    prefix::Vector{Int}
end

function Base.sum(tci::AdaptiveTCILeaf{T})::T where {T}
    return sum(TCI.tensortrain(tci.tci))
end

function Base.show(io::IO, obj::AdaptiveTCILeaf{T}) where {T}
    prefix = convert.(Int, obj.prefix)
    println(io,
        "  "^length(prefix) *
        "Leaf $(prefix): rank=$(maximum(TCI.linkdims(obj.tci)))")
end

struct AdaptiveTCIInternalNode{T<:Number} <: AbstractAdaptiveTCINode
    children::Dict{Int,AbstractAdaptiveTCINode}
    prefix::Vector{Int}

    function AdaptiveTCIInternalNode{T}(children::Dict{Int,AbstractAdaptiveTCINode},
        prefix::Vector{Int}) where {T}
        return new{T}(children, prefix)
    end
end

function Base.sum(tci::AdaptiveTCIInternalNode{T})::T where {T}
    return sum((sum(child) for child in values(tci.children)))
end

"""
prefix is the common prefix of all children
"""
function AdaptiveTCIInternalNode{T}(children::Vector{AbstractAdaptiveTCINode},
    prefix::Vector{Int}) where {T}
    d = Dict{Int,AbstractAdaptiveTCINode}()
    for child in children
        d[child.prefix[end]] = child
    end
    return AdaptiveTCIInternalNode{T}(d, prefix)
end

function Base.show(io::IO, obj::AdaptiveTCIInternalNode{T}) where {T}
    println(io,
        "  "^length(obj.prefix) *
        "InternalNode $(obj.prefix) with $(length(obj.children)) children")
    for (k, v) in obj.children
        Base.show(io, v)
    end
end

"""
Evaluate the tree at given idx
"""
function evaluate(obj::AdaptiveTCIInternalNode{T}, idx::AbstractVector{Int})::T where {T}
    child_key = idx[length(obj.prefix) + 1]
    return evaluate(obj.children[child_key], idx)
end

function evaluate(obj::AdaptiveTCILeaf{T}, idx::AbstractVector{Int})::T where {T}
    return TCI.evaluate(obj.tci, idx[(length(obj.prefix) + 1):end])
end

"""
Convert a dictionary of patches to a tree
"""
function _to_tree(patches::Dict{Vector{Int},TensorCI2{T}};
    nprefix=0)::AbstractAdaptiveTCINode where {T}
    length(unique(k[1:nprefix] for (k, v) in patches)) == 1 ||
        error("Inconsistent prefixes")

    common_prefix = first(patches)[1][1:nprefix]

    # Return a leaf
    if nprefix == length(first(patches)[1])
        return AdaptiveTCILeaf{T}(first(patches)[2], common_prefix)
    end

    subgroups = Dict{Int,Dict{Vector{Int},TensorCI2{T}}}()

    # Look at the first index after nprefix skips
    # and group the patches by that index
    for (k, v) in patches
        idx = k[nprefix + 1]
        if idx in keys(subgroups)
            subgroups[idx][k] = v
        else
            subgroups[idx] = Dict{Vector{Int},TensorCI2{T}}(k => v)
        end
    end

    # Recursively construct the tree
    children = AbstractAdaptiveTCINode[]
    for (_, grp) in subgroups
        push!(children, _to_tree(grp; nprefix=nprefix + 1))
    end

    return AdaptiveTCIInternalNode{T}(children, common_prefix)
end

"""
Construct QTTs using adaptive partitioning of the domain.

TODO

  - Allow arbitrary order of partitioning
"""
function adaptivetci(::Type{T}, f, localdims::AbstractVector{Int};
    tolerance::Float64=1e-8, maxbonddim::Int=100,
    firstpivot=ones(Int, length(localdims)),
    sleep_time::Float64=1e-6, verbosity::Int=0, maxnleaves=100,
    kwargs...)::Union{AdaptiveTCILeaf{T},AdaptiveTCIInternalNode{T}
} where {T}
    R = length(localdims)
    leaves = Dict{Vector{Int},Union{TensorCI2{T},Future}}()

    # Add root node
    firstpivot = TCI.optfirstpivot(f, localdims, firstpivot)
    tci, ranks, errors = TCI.crossinterpolate2(T, f, localdims,
        [firstpivot];
        tolerance=tolerance,
        maxbonddim=maxbonddim,
        verbosity=verbosity,
        kwargs...)
    leaves[[]] = tci
    maxsamplevalue = tci.maxsamplevalue

    while true
        #if length(leaves) > 30
        #break
        #end
        sleep(sleep_time) # Not to run the loop too quickly

        #maxsamplevalue = max(
        #maxsamplevalue,
        #maximum(tci.maxsamplevalue for tci in values(leaves) if tci isa TensorCI2)
        #)
        done = true
        for (prefix, tci) in leaves
            if tci isa Future
                done = false
                if isready(tci)
                    #if verbosity > 0
                    #println("Fetching $(prefix) ...")
                    #end
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
                        println("Fetched, bond dimension = $(maximum(TCI.linkdims(leaves[prefix]))) $(TCI.maxbonderror(tci)) $(tci.maxsamplevalue) for $(prefix)")
                    end
                    maxsamplevalue = max(maxsamplevalue, leaves[prefix].maxsamplevalue)
                end
            elseif tci isa TensorCI2 &&
                   (TCI.maxbonderror(tci) > tolerance * maxsamplevalue) &&
                   length(leaves) < maxnleaves
                done = false
                delete!(leaves, prefix)
                for ic in 1:localdims[length(prefix) + 1]
                    prefix_ = vcat(prefix, ic)
                    localdims_ = localdims[(length(prefix_) + 1):end]
                    f_ = x -> f(vcat(prefix_, x))

                    firstpivot_ = ones(Int, R - length(prefix_))
                    maxval = abs(f_(firstpivot_))

                    for r in 1:10
                        firstpivot_rnd = [rand(1:localdims_[r])
                                          for r in eachindex(localdims_)]
                        firstpivot_rnd = TCI.optfirstpivot(f_, localdims_, firstpivot_rnd)
                        if abs(f_(firstpivot_rnd)) > maxval
                            firstpivot_ = firstpivot_rnd
                            maxval = abs(f_(firstpivot_))
                        end
                    end

                    if verbosity > 0
                        println("Interpolating $(prefix_) ...")
                    end
                    leaves[prefix_] = @spawnat :any TCI.crossinterpolate2(T,
                        f_,
                        localdims_,
                        [firstpivot_];
                        tolerance=tolerance *
                                  maxsamplevalue,
                        maxbonddim=maxbonddim,
                        verbosity=verbosity,
                        normalizeerror=false,
                        kwargs...)
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
            error("TCI for k= $(k) has bond error $(TCI.maxbonderror(v)) larger than $(tolerance) * $(maxsamplevalue) = $(tolerance*maxsamplevalue)!")
        end
        leaves_done[k] = v
    end

    return _to_tree(leaves_done)
end
