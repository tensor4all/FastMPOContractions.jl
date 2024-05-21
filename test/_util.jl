using ITensors

function _random_mpo(sites::Vector{Vector{Index{T}}}; linkdims = 1) where {T}
    N = length(sites)
    links = [Index(linkdims, "Link,n=$n") for n = 1:N-1]
    M = MPO(N)
    M[1] = random_itensor(sites[1]..., links[1])
    M[N] = random_itensor(links[N-1], sites[N]...)
    for n = 2:N-1
        M[n] = random_itensor(links[n-1], sites[n]..., links[n])
    end
    return M
end
