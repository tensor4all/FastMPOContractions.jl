using Test

import TensorCrossInterpolation as TCI
using TCIITensorConversion
import FastMPOContractions as FMPOC
using ITensors

@testset "Contraction" begin
    R = 3
    sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
    sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]

    sitesa = collect(collect.(zip(sitesx, sitesy)))
    sitesb = collect(collect.(zip(sitesy, sitesz)))

    @show sitesa

    #a = TCI.TensorTrain{Float64,4}(_randomMPO(sitesa))
    #b = TCI.TensorTrain{Float64,4}(_randomMPO(sitesb))

    a = FMPOC.PartitionedTTO{Float64}(
        [TCI.TensorTrain{Float64,4}(_randomMPO(sitesa))],
        [fill(0, R)],
        [fill(0, R)]
    )

    b = FMPOC.PartitionedTTO{Float64}(
        [TCI.TensorTrain{Float64,4}(_randomMPO(sitesb))],
        [fill(0, R)],
        [fill(0, R)]
    )

    FMPOC.contract(a, b, fill(0, R), fill(0, R))
end