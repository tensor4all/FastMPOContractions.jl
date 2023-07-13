using Distributed

using Test
using Random

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

@testset "adaptivepatching.jl" begin

    @testset "2D fermi gk" begin
        Random.seed!(1234)

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
        creator = FMPOC.TCI2PatchCreator(
            ComplexF64,
            f,
            localdims;
            maxbonddim = 150,
            rtol = tol,
            verbosity = 1,
            ntry = 10,
        )
        #@show creator.maxval

        tree = FMPOC.adaptivepatches(creator; verbosity = 1, maxnleaves = 1000)
        @show tree

        for _ = 1:100
            pivot = rand(1:4, R)
            error_func = x -> abs(f(x) - FMPOC.evaluate(tree, x))
            pivot = TCI.optfirstpivot(error_func, localdims, pivot)
            @test isapprox(FMPOC.evaluate(tree, pivot), f(pivot); atol = 10 * creator.atol)
        end

    end
end

nothing
