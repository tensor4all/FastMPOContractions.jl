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

    @testset "PatchOrdering" begin
        po = FMPOC.PatchOrdering([4, 3, 2, 1])
        @test FMPOC.maskactiveindices(po, 2) == [1, 1, 0, 0]
        @test FMPOC.maskactiveindices(po, 1) == [1, 1, 1, 0]
        @test FMPOC.fullindices(po, [1], [2, 3, 4]) == [2, 3, 4, 1]
        @test FMPOC.fullindices(po, [1, 2], [3, 4]) == [3, 4, 2, 1]
    end

    @testset "2D fermi gk" for _flipordering in [false, true]
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
        flipper = _flipordering ? x -> reverse(x) : x -> x
        f = x -> gk(originalcoordinate(grid, QuanticsInd{2}.(flipper(x)))..., β)

        tol = 1e-5

        pordering = FMPOC.PatchOrdering(flipper(collect(1:R)))

        creator = FMPOC.TCI2PatchCreator(
            ComplexF64,
            f,
            localdims;
            maxbonddim = 150,
            rtol = tol,
            verbosity = 1,
            ntry = 10,
        )

        tree = FMPOC.adaptivepatches(creator, pordering; verbosity = 1, maxnleaves = 1000)
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
