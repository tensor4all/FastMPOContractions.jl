function contract_mpo_mpo(M1::MPO, M2::MPO; alg::String="densitymatrix", cutoff::Real=1e-30, kwargs...)::MPO
    if alg == "densitymatrix"
        return contract_densitymatrix(M1, M2; cutoff, kwargs...)
    elseif alg == "fit"
        return contract_fit(M1, M2; cutoff, kwargs...)
    elseif alg == "zipup"
        return ITensors.contract(M1, M2; alg="zipup", cutoff, kwargs...)
    elseif alg == "naive"
        return ITensors.contract(M1, M2; alg="naive", cutoff, kwargs...)
    else
        error("Unknown algorithm: $alg")
    end

end

function apply(A::MPO, Ψ::MPO; alg::String="fit", cutoff::Real=1e-30, kwargs...)::MPO
    if :algorithm ∈ keys(kwargs)
        error("keyword argument :algorithm is not allowed")
    end
    if alg == "densitymatrix" && cutoff <= 1e-10
        @warn "cutoff is too small for densitymatrix algorithm. Use fit algorithm instead."
    end
    AΨ = replaceprime(
        contract_mpo_mpo(A', MPO(collect(Ψ)); alg, cutoff, kwargs...), 2 => 1)
    MPO(collect(AΨ))
end


function apply(A::MPO, Ψ::MPS; alg::String="fit", cutoff::Real=1e-30, kwargs...)::MPS
    if :algorithm ∈ keys(kwargs)
        error("keyword argument :algorithm is not allowed")
    end
    if alg == "densitymatrix" && cutoff <= 1e-10
        @warn "cutoff is too small for densitymatrix algorithm. Use fit algorithm instead."
    end
    AΨ = noprime.(contract_mpo_mpo(A, MPO(collect(Ψ)); alg, cutoff, kwargs...))
    MPS(collect(AΨ))
end
