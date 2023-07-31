struct MPOTree
    localdims::Vector{Tuple{Int,Int}}
    mpos::Vector{MPO}
    sites::Vector{Tuple{Index,Index}}
    prefix::Vector{Tuple{Int,Int}}
end