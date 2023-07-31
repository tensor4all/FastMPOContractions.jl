module FastMPOContractions

using TensorCrossInterpolation
import TensorCrossInterpolation: TensorCI, CachedFunction, TensorCI2, MultiIndex, TensorTrain
import TensorCrossInterpolation as TCI
using StaticArrays
using Distributed

# Write your package code here.

include("adpativepatching.jl")
#include("mpotree.jl")
#include("contraction.jl")

end
