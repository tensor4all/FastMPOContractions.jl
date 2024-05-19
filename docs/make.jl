using FastMPOContractions
using Documenter

DocMeta.setdocmeta!(FastMPOContractions, :DocTestSetup, :(using FastMPOContractions); recursive=true)

makedocs(;
    modules=[FastMPOContractions],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    sitename="FastMPOContractions.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/FastMPOContractions.jl",
        edit_link="main",
        assets=String[]),
    pages=[
        "Home" => "index.md",
        "API Reference" => "apireference.md",
    ])

deploydocs(;
    repo="github.com/tensor4all/FastMPOContractions.jl.git",
    devbranch="main",
)
