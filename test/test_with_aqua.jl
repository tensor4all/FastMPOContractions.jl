using Aqua
import FastMPOContractions

@testset "Aqua" begin
    Aqua.test_all(
        FastMPOContractions;
        ambiguities = false,
        unbound_args = false,
        deps_compat = false,
    )
end
