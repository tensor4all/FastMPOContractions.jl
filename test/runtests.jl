using FastMPOContractions
using Test

include("test_with_aqua.jl")
include("test_with_jet.jl")

include("_util.jl")

include("contractMPO.jl")
include("fitalgorithm.jl")
include("util.jl")
include("fitalgorithm_sum.jl")
include("naive.jl")
