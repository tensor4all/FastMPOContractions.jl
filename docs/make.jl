using FastMPOContractions
using Documenter

DocMeta.setdocmeta!(FastMPOContractions, :DocTestSetup, :(using FastMPOContractions); recursive=true)

makedocs(;
    modules=[FastMPOContractions],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    repo="https://gitlab.com/quanticstci/FastMPOContractions.jl/blob/{commit}{path}#{line}",
    sitename="FastMPOContractions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gitlab.com/quanticstci/FastMPOContractions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="https://gitlab.com/quanticstci/FastMPOContractions.jl",
    devbranch="main",
)
