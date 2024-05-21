using Test
import FastMPOContractions as FMPOC
using ITensors

@testset "error_contract" begin
    L = 10
    sites = [Index(2, "Qubit,n=$n") for n = 1:L]
    phi = random_mpo(sites)
    K = prime(random_mpo(sites) + random_mpo(sites))

    # Apply K to phi and check that error_contract is close to 0.
    Kphi = contract(K, phi; alg = "naive", cutoff = 1E-8)
    @test FMPOC.error_contract(Kphi, K, phi) ≈ 0.0 atol = 1e-4
end
