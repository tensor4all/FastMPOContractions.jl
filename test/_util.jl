using ITensors

function _randomMPO(sites::Vector{Vector{Index{T}}}; linkdims = 1) where {T}
    N = length(sites)
    links = [Index(linkdims, "Link,n=$n") for n = 1:N-1]
    M = MPO(N)
    M[1] = randomITensor(sites[1]..., links[1])
    M[N] = randomITensor(links[N-1], sites[N]...)
    for n = 2:N-1
        M[n] = randomITensor(links[n-1], sites[n]..., links[n])
    end
    return M
end
