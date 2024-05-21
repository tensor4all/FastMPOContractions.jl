using Test
import FastMPOContractions as FMPOC
using ITensors
using ITensorTDVP
using Random

@testset "fitalgorithm.jl" begin
    @testset "Contract MPO-MPO" begin
        Random.seed!(1234)
        nbit = 5
        sites = siteinds("Qubit", nbit)
        M1 = random_mpo(sites) + random_mpo(sites)
        M2 = random_mpo(sites) + random_mpo(sites)

        # The function `apply` does not work correctly with the mapping-MPO-to-MPS trick.
        M1 = replaceprime(M1, 1 => 2, 0 => 1)

        M2_ = MPS(length(sites))
        for n in eachindex(sites)
            M2_[n] = M2[n]
        end

        M12_ref = contract(M1, M2; alg = "naive")
        M12 = FMPOC.contract_fit(M1, M2_)
        t12_ref = Array(reduce(*, M12_ref), sites, setprime(sites, 2))
        t12 = Array(reduce(*, M12), sites, setprime(sites, 2))
        @test maximum(abs, t12 .- t12_ref) < 1e-12 * maximum(abs, t12_ref)

        M12_2 = FMPOC.contract_fit(M1, M2)
        t12_2 = Array(reduce(*, M12_2), sites, setprime(sites, 2))
        @test maximum(abs, t12_2 .- t12_ref) < 1e-12 * maximum(abs, t12_ref)
    end

    #==
    @testset "Contract MPO-MPO (sum)" begin
        Random.seed!(1234)
        nbit = 5
        nterm = 2
        sites = siteinds("Qubit", nbit)
        M1s = [random_mpo(sites) + random_mpo(sites) for _ in 1:nterm]
        M2s = [random_mpo(sites) + random_mpo(sites) for _ in 1:nterm]

        # The function `apply` does not work correctly with the mapping-MPO-to-MPS trick.
        for t in 1:nterm
            M1s[t] = replaceprime(M1s[t], 1 => 2, 0 => 1)
        end

        M2s_ = MPS[]
        for t in 1:nterm
            M2_ = MPS(length(sites))
            for n in eachindex(sites)
                M2_[n] = M2s[t][n]
            end
            push!(M2s_, M2_)
        end

        M12_ref = sum([contract(M1s[t], M2s[t]; alg="naive") for t in 1:nterm])
        M12 = FMPOC.contract_fit(M1s, M2s_)
        #==
        t12_ref = Array(reduce(*, M12_ref), sites, setprime(sites, 2))
        t12 = Array(reduce(*, M12), sites, setprime(sites, 2))
        @test maximum(abs, t12 .- t12_ref) < 1e-12 * maximum(abs, t12_ref)

        M12_2 = FMPOC.contract_fit(M1, M2)
        t12_2 = Array(reduce(*, M12_2), sites, setprime(sites, 2))
        @test maximum(abs, t12_2 .- t12_ref) < 1e-12 * maximum(abs, t12_ref)
        ==#
    end
    ==#
end
