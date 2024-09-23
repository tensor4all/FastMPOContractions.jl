using Test

import FastMPOContractions as FMPOC
using ITensors



@testset "naive (x-y-z)" begin
    R = 3
    sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
    sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]

    sitesa = collect(collect.(zip(sitesx, sitesy)))
    sitesb = collect(collect.(zip(sitesy, sitesz)))
    a = _random_mpo(sitesa)
    b = _random_mpo(sitesb)
    ab_ref = contract(a, b; alg = "naive")
    ab = FMPOC.contract_mpo_mpo(a, b; alg = "naive")
    @test ab_ref ≈ ab
end
