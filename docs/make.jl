using FastMPOContractions
using Documenter

DocMeta.setdocmeta!(FastMPOContractions, :DocTestSetup, :(using FastMPOContractions); recursive=true)

makedocs(;
    modules=[FastMPOContractions],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    repo="https://github.com/shinaoka/FastMPOContractions.jl/blob/{commit}{path}#{line}",
    sitename="FastMPOContractions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://shinaoka.github.io/FastMPOContractions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/shinaoka/FastMPOContractions.jl",
    devbranch="main",
)
