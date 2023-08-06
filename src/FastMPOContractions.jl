module FastMPOContractions

using TensorCrossInterpolation
import TensorCrossInterpolation: TensorCI, CachedFunction, TensorCI2, MultiIndex, TensorTrain
import TensorCrossInterpolation as TCI
using StaticArrays
using Distributed

using ITensors
import ITensors: AbstractMPS, sim!, setleftlim!, setrightlim!

using ITensorTDVP

# Write your package code here.

const TTO{T} = TensorTrain{T,4}

include("adpativepatching.jl")
include("densitymatrix.jl")
include("fitalgorithm.jl")
#include("mpotree.jl")
#include("contraction.jl")

end
