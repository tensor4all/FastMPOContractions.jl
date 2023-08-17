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

    a = sum(randomMPO(sitesa) for _ in 1:5)
    replaceind!.(a, sitesa', sitescontract)

    b = sum(randomMPO(sitesb) for _ in 1:5)
    replaceind!.(b, sitesb', sitescontract)

    c = FastMPOContractions.contract_tci2(a, b; maxbonddim=100)
    @test siteinds(c) == collect(zip(sitesa, sitesb))

    cref = a * b

    for i in CartesianIndices(tuple(localdims...))
        for j in CartesianIndices(tuple(localdims...))
            aindices = Tuple.(zip(sitesa, Tuple(i)))
            bindices = Tuple.(zip(sitesb, Tuple(j)))
            @test evaluate_mps(c, aindices, bindices) ≈ evaluate_mps(cref, aindices, bindices)
        end
    end
end
