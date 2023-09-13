struct PatchedMPO
    mpos::Vector{MPO}
    sites::Vector{Vector{Index{Int}}}
    prefix::Vector{Vector{Int}}

    function PatchedMPO(
        mpos::Vector{MPO}, sites::Vector{Vector{Index{Int}}}, prefix::Vector{Vector{Int}})

        for M in mpos
            _check_sites(M, sites) || error("sites are not compatible with MPO: $(sites), $(siteinds(M))")
        end

        new(mpos, sites, prefix)
    end
end

function _check_sites(M::MPO, sites)::Bool
    for n in eachindex(M)
        length(sites[n]) == length(siteinds(M, n)) || return false
        for s in sites[n]
            hasind(M[n], s) || return false
        end
    end
    return true
end

function Base.show(io::IO, obj::PatchedMPO)
    print(io, "PatchedMPO with $(length(obj.mpos)) MPOs with siteinds: $(obj.sites)")
end

"""
Override * operactor for PatchedMPO
"""
function Base.:*(A::PatchedMPO, B::PatchedMPO)
    length(A.mpos) == length(B.mpos) || error("length of PatchedMPOs are not equal: $(length(A.mpos)), $(length(B.mpos))")
    length(A.sites) == length(B.sites) || error("length of PatchedMPOs are not equal: $(length(A.sites)), $(length(B.sites))")
    length(A.prefix) == length(B.prefix) || error("length of PatchedMPOs are not equal: $(length(A.prefix)), $(length(B.prefix))")

    mpos = Vector{MPO}(undef, length(A.mpos))
    sites = Vector{Vector{Index{Int}}}(undef, length(A.sites))
    prefix = Vector{Vector{Int}}(undef, length(A.prefix))

    for n in eachindex(A.mpos)
        mpos[n] = A.mpos[n] * B.mpos[n]
        sites[n] = A.sites[n]
        prefix[n] = A.prefix[n]
    end

    PatchedMPO(mpos, sites, prefix)
end

