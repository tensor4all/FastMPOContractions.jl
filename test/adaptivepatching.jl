using Distributed

using Test

# Define the maximum number of worker processes.
const MAX_WORKERS = 4

# Add worker processes if necessary.
addprocs(max(0, MAX_WORKERS - nworkers()))

@everywhere include("_quanticsrepr.jl")
@everywhere include("_grid.jl")

@everywhere using TensorCrossInterpolation
@everywhere import TensorCrossInterpolation as TCI
@everywhere import FastMPOContractions as FMPOC
@everywhere using ITensors
@everywhere ITensors.disable_warn_order()

@testset "tci.jl" begin
#==
    @testset "2D fermi gk" begin
        ek(kx, ky) = 2 * cos(kx) + 2 * cos(ky) - 1.0

        function gk(kx, ky, β)
            iv = im * π / β
            return 1 / (iv - ek(kx, ky))
        end

        R = 20
        grid = DiscretizedGrid{2}(R, (0.0, 0.0), (2π, 2π))
        localdims = fill(4, R)

        β = 20.0
        f = x -> gk(originalcoordinate(grid, QuanticsInd{2}.(x))..., β)
        firstpivot = fill(4, R)
        firstpivot = TCI.optfirstpivot(f, localdims, firstpivot)
        absmaxval = abs(f(firstpivot))
        tol = 1e-5
        tree =
            FMPOC.adaptivepatchtci(ComplexF64, f, localdims; maxbonddim = 45, tolerance = tol)
        #@show tree

        for _ = 1:100
            pivot = rand(1:4, R)
            isapprox(FMPOC.evaluate(tree, pivot), f(pivot); atol = tol * absmaxval)
        end
    end
==#

@testset "2D fermi gk" begin
    ek(kx, ky) = 2 * cos(kx) + 2 * cos(ky) - 1.0

    function gk(kx, ky, β)
        iv = im * π / β
        return 1 / (iv - ek(kx, ky))
    end

    R = 20
    grid = DiscretizedGrid{2}(R, (0.0, 0.0), (2π, 2π))
    localdims = fill(4, R)

    β = 20.0
    f = x -> gk(originalcoordinate(grid, QuanticsInd{2}.(x))..., β)

    tol = 1e-5
    creator = FMPOC.TCI2PatchCreator(ComplexF64, f, localdims; maxbonddim = 45, rtol = tol, verbosity=1)

    tree = FMPOC.adaptivepatchtci2(ComplexF64, creator)
    @show tree

    for _ = 1:100
        pivot = rand(1:4, R)
        isapprox(FMPOC.evaluate(tree, pivot), f(pivot); atol = creator.atol)
    end
end
end

nothing
