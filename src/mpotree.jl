struct MPOTree{T}
    localdims::Vector{Tuple{Int,Int}}
    mpos::Vector{TTO}
    sites::Vector{Tuple{Index,Index}}
    prefix::Vector{Tuple{Int,Int}}
end