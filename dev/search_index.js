var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = FastMPOContractions","category":"page"},{"location":"#FastMPOContractions","page":"Home","title":"FastMPOContractions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for FastMPOContractions.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [FastMPOContractions]","category":"page"},{"location":"#FastMPOContractions.contract_fit-Tuple{ITensors.ITensorMPS.MPO, ITensors.ITensorMPS.MPO}","page":"Home","title":"FastMPOContractions.contract_fit","text":"Contract M1 and M2, and return the result as an MPO.\n\n\n\n\n\n","category":"method"},{"location":"#FastMPOContractions.error_contract-Tuple{ITensors.ITensorMPS.MPO, ITensors.ITensorMPS.MPO, ITensors.ITensorMPS.MPO}","page":"Home","title":"FastMPOContractions.error_contract","text":"Taken from from ITensors.jl with some modifications.\n\nThe dominator is replaced by <y|A|x> instead of <x|A|A|x> to reduce the computational cost.\nSpecialized to MPO-MPO contraction.\n\nerror_contract(y::MPS, A::MPO, x::MPS;\n               make_inds_match::Bool = true)\nerror_contract(y::MPS, x::MPS, x::MPO;\n               make_inds_match::Bool = true)\n\nCompute the distance between A|x> and an approximation MPS y: | |y> - A|x> |/| A|x> | = √(1 + (<y|y> - 2*real(<y|A|x>))/<y|A|x>).\n\nIf make_inds_match = true, the function attempts match the site indices of y with the site indices of A that are not common with x.\n\n\n\n\n\n","category":"method"}]
}
