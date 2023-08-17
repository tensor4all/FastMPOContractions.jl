using Test

import FastMPOContractions as FMPOC
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

@testset "densitymatrix (x-y-z)" begin
    R = 3
    sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
    sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]

    sitesa = collect(collect.(zip(sitesx, sitesy)))
    sitesb = collect(collect.(zip(sitesy, sitesz)))
    a = _randomMPO(sitesa)
    b = _randomMPO(sitesb)
    ab_ref = contract(a, b; alg = "naive")
    ab = FMPOC.contract_densitymatrix(a, b)
    @test ab_ref ≈ ab
end

@testset "densitymatrix (xk-y-z)" begin
    R = 3
    sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
    sitesk = [Index(2, "Qubit,k=$n") for n = 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
    sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]

    sitesa = collect(collect.(zip(sitesx, sitesk, sitesy)))
    sitesb = collect(collect.(zip(sitesy, sitesz)))
    a = _randomMPO(sitesa)
    b = _randomMPO(sitesb)
    ab_ref = contract(a, b; alg = "naive")
    ab = FMPOC.contract_densitymatrix(a, b)
    @test ab_ref ≈ ab
end

@testset "densitymatrix (xk-y-zl)" begin
    R = 3
    sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
    sitesk = [Index(2, "Qubit,k=$n") for n = 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
    sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]
    sitesl = [Index(2, "Qubit,l=$n") for n = 1:R]

    sitesa = collect(collect.(zip(sitesx, sitesk, sitesy)))
    sitesb = collect(collect.(zip(sitesy, sitesz, sitesl)))
    a = _randomMPO(sitesa)
    b = _randomMPO(sitesb)
    ab_ref = contract(a, b; alg = "naive")
    ab = FMPOC.contract_densitymatrix(a, b)
    @test ab_ref ≈ ab
end

@testset "densitymatrix (xk-ym-zl)" begin
    R = 3
    sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
    sitesk = [Index(2, "Qubit,k=$n") for n = 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
    sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]
    sitesl = [Index(2, "Qubit,l=$n") for n = 1:R]
    sitesm = [Index(2, "Qubit,m=$n") for n = 1:R]

    sitesa = collect(collect.(zip(sitesx, sitesk, sitesm, sitesy)))
    sitesb = collect(collect.(zip(sitesy, sitesm, sitesz, sitesl)))
    a = _randomMPO(sitesa)
    b = _randomMPO(sitesb)
    ab_ref = contract(a, b; alg = "naive")
    ab = FMPOC.contract_densitymatrix(a, b)
    @test ab_ref ≈ ab
end
