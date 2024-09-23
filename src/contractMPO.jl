function contract_mpo_mpo(M1::MPO, M2::MPO; alg::String = "densitymatrix", kwargs...)::MPO
    if alg == "densitymatrix"
        return contract_densitymatrix(M1, M2; kwargs...)
    elseif alg == "fit"
        return contract_fit(M1, M2; kwargs...)
    elseif alg == "naive"
        return ITensors.contract(M1, M2; alg = "naive", kwargs...)
    else
        error("Unknown algorithm: $alg")
    end

end
