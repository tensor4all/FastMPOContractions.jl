using JET
import FastMPOContractions

@testset "JET" begin
    if VERSION ≥ v"1.9"
        JET.test_package(FastMPOContractions; target_defined_modules = true)
    end
end
