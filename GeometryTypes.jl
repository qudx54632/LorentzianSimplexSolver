module GeometryTypes

export GeometryDataset, GeometryCollection

mutable struct GeometryDataset
    solgsl2c::Vector{Matrix{ComplexF64}}
    solgso13::Vector{Matrix{ComplexF64}}

    bdyxi::Vector{Vector{Vector{Vector{ComplexF64}}}}
    nabout::Vector{Vector{Vector{Float64}}}
    nabfrombivec::Vector{Vector{Vector{Float64}}}

    bdysu::Vector{Vector}
    bdybivec4d55::Vector{Vector{Matrix{ComplexF64}}}
    bdybivec55::Vector{Vector{Matrix{ComplexF64}}}

    dihedrals::Vector{Vector{Float64}}
    areas::Vector{Vector{Float64}}
    kappa::Vector{Vector{Int}}
    tetareasign::Vector{Vector{Int}}
    tetn0sign::Vector{Vector{Int}}

    tetnormalvec::Vector{Vector{Float64}}
    sgndet::Vector{Int}

    zdataf::Vector{Vector{Vector{ComplexF64}}}
end

mutable struct GeometryCollection
    simplex::Vector{GeometryDataset}
    connectivity::Vector{Dict{String, Any}}

    # >>> ADD THESE <<<
    crit::Dict{Symbol, Any}
    varias::Dict{Symbol, Any}
end

GeometryCollection(simplex::Vector{GeometryDataset}) =
    GeometryCollection(simplex, Dict{String,Any}[], Dict{Symbol,Any}(), Dict{Symbol,Any}())

end # module