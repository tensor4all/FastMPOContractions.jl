using Test

import FastMPOContractions as FMPOC
using ITensors
using Random

algs = ["densitymatrix", "fit", "zipup"]

@testset "MPO-MPO contraction (x-y-z)" for alg in algs
    Random.seed!(1234)

    R = 3
    sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
    sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]

    sitesa = collect(collect.(zip(sitesx, sitesy)))
    sitesb = collect(collect.(zip(sitesy, sitesz)))
    a = _random_mpo(sitesa)
    b = _random_mpo(sitesb)
    ab_ref = contract(a, b; alg = "naive")
    ab = FMPOC.contract_mpo_mpo(a, b; alg=alg)
    @test ab_ref ≈ ab
end

@testset "MPO-MPO contraction (xk-y-z)" for alg in algs
    R = 3
    sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
    sitesk = [Index(2, "Qubit,k=$n") for n = 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
    sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]

    sitesa = collect(collect.(zip(sitesx, sitesk, sitesy)))
    sitesb = collect(collect.(zip(sitesy, sitesz)))
    a = _random_mpo(sitesa)
    b = _random_mpo(sitesb)
    ab_ref = contract(a, b; alg = "naive")
    ab = FMPOC.contract_mpo_mpo(a, b; alg=alg)
    @test ab_ref ≈ ab
end

@testset "MPO-MPO contraction (xk-y-zl)" for alg in algs
    R = 3
    sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
    sitesk = [Index(2, "Qubit,k=$n") for n = 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
    sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]
    sitesl = [Index(2, "Qubit,l=$n") for n = 1:R]

    sitesa = collect(collect.(zip(sitesx, sitesk, sitesy)))
    sitesb = collect(collect.(zip(sitesy, sitesz, sitesl)))
    a = _random_mpo(sitesa)
    b = _random_mpo(sitesb)
    ab_ref = contract(a, b; alg = "naive")
    ab = FMPOC.contract_mpo_mpo(a, b; alg=alg)
    @test ab_ref ≈ ab
end

@testset "MPO-MPO contraction (xk-ym-zl)" for alg in algs
    R = 3
    sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
    sitesk = [Index(2, "Qubit,k=$n") for n = 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
    sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]
    sitesl = [Index(2, "Qubit,l=$n") for n = 1:R]
    sitesm = [Index(2, "Qubit,m=$n") for n = 1:R]

    sitesa = collect(collect.(zip(sitesx, sitesk, sitesm, sitesy)))
    sitesb = collect(collect.(zip(sitesy, sitesm, sitesz, sitesl)))
    a = _random_mpo(sitesa)
    b = _random_mpo(sitesb)
    ab_ref = contract(a, b; alg = "naive")
    ab = FMPOC.contract_mpo_mpo(a, b; alg=alg)
    @test ab_ref ≈ ab
end
