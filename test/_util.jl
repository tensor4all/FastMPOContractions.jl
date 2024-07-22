using ITensors
using Random

function _random_mpo(sites::Vector{Vector{Index{T}}}; linkdims = 1) where {T}
    _random_mpo(Random.GLOBAL_RNG, sites; linkdims = linkdims)
end

function _random_mpo(rng, sites::Vector{Vector{Index{T}}}; linkdims = 1) where {T}
    N = length(sites)
    links = [Index(linkdims, "Link,n=$n") for n = 1:N-1]
    M = MPO(N)
    M[1] = random_itensor(rng, sites[1]..., links[1])
    M[N] = random_itensor(rng, links[N-1], sites[N]...)
    for n = 2:N-1
        M[n] = random_itensor(rng, links[n-1], sites[n]..., links[n])
    end
    return M
end
