module FastMPOContractions

using StaticArrays

using ITensors
import ITensorMPS:
    AbstractMPS, sim!, setleftlim!, setrightlim!, check_hascommoninds

using ITensorMPS

include("densitymatrix.jl")
include("fitalgorithm.jl")
include("util.jl")
include("contractMPO.jl")
include("fitalgorithm_sum.jl")

end
