using Test

import FastMPOContractions as FMPOC
using ITensors

@testset "MPO*MPO contraction (densitymatrix)" begin
    R = 3
    sites = [Index(2, "Qubit,n=$n") for n in 1:R]
    a = replaceprime(randomMPO(sites), 0 => 1, 1 => 2)
    b = randomMPO(sites)
    ab_ref = contract(a, b; alg="naive")

    # MPO-MPO contraction
    ab = FMPOC.contract_densitymatrix(a, b)
    @test ab_ref ≈ ab

    # MPO-MPS contraction
    ab_MPS = FMPOC.contract_densitymatrix(a, MPS([b[n] for n in eachindex(b)]))
    @test ab_ref ≈ MPO([ab_MPS[n] for n in eachindex(ab_MPS)])
end