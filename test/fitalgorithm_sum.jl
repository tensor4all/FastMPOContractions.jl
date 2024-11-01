import FastMPOContractions as FMPOC
using ITensors
using ITensorMPS
using ITensors: random_itensor, Algorithm
using Random
using StableRNGs

@testset "fitalgorithm.jl" begin
    @testset for elt in [Float64, ComplexF64]
        R = 10
        N = 10
        sites = [Index(2, "Qubit,x=$n") for n = 1:R]

        rng = StableRNG(1234)

        Ψs = [random_mps(rng, sites) for _ = 1:N]
        coeffs = rand(rng, elt, N)

        x0 =
            +(Ψs...; alg = "directsum") +
            elt(0.05) * random_mps(rng, elt, sites; linkdims = 2)
        ab_fit = FMPOC.fit(Ψs, x0; coeffs = coeffs, nsweeps = 2)

        Ψs_ = [Ψs[n] * coeffs[n] for n = 1:N]

        ab_ref = +(Ψs_...; alg = "directsum")
        @test isapprox(ab_fit, ab_ref; rtol = 1e-14)
    end

    @testset for elt in [Float64, ComplexF64]
        R = 3
        sitesx = [Index(2, "x=$n") for n = 1:R]
        sitesy = [Index(2, "y=$n") for n = 1:R]

        _to_mps(Ψ::MPO) = MPS([x for x in Ψ])

        sites = [[x, y] for (x, y) in zip(sitesx, sitesy)]

        rng = StableRNG(1234)
        a = _random_mpo(rng, sites)
        b = _random_mpo(rng, sites)
        coeffs = rand(rng, elt, 2)

        x0 = a + b + elt(0.05) * _random_mpo(rng, sites)
        ab_fit = FMPOC.fit([a, b], x0; coeffs = coeffs, nsweeps = 2)
        @test ab_fit ≈ +(coeffs[1] * a, coeffs[2] * b; alg = "directsum")
    end
end
