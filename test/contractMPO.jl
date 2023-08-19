using Test

import TensorCrossInterpolation as TCI
using TCIITensorConversion
import FastMPOContractions
using ITensors

@testset "Contraction using TCI2" begin
    localdims = [4, 2, 7]

    sitesa = Index.(localdims, "a")
    sitescontract = Index.(localdims, "contract")
    sitesb = Index.(localdims, "b")

    a = sum(randomMPO(sitesa) for _ = 1:5)
    replaceind!.(a, sitesa', sitescontract)

    b = sum(randomMPO(sitesb) for _ = 1:5)
    replaceind!.(b, sitesb', sitescontract)

    c = FastMPOContractions.contract_tci2(a, b; maxbonddim = 100)
    @test siteinds(c) == collect(zip(sitesa, sitesb))

    cref = a * b

    for i in CartesianIndices(tuple(localdims...))
        for j in CartesianIndices(tuple(localdims...))
            aindices = Tuple.(zip(sitesa, Tuple(i)))
            bindices = Tuple.(zip(sitesb, Tuple(j)))
            @test evaluate_mps(c, aindices, bindices) ≈
                  evaluate_mps(cref, aindices, bindices)
        end
    end
end


@testset "contract_tci2 (xk-ym-zl)" begin
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
    b = im * _randomMPO(sitesb)
    ab_ref = contract(a, b; alg = "naive")
    ab = FastMPOContractions.contract_tci2(a, b; tolerance = 1e-10)
    @test ab_ref ≈ ab
end
