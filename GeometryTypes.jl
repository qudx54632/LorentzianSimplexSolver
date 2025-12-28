module GeometryTypes

export GeometryDataset, GeometryCollection

# ============================================================
# GeometryDataset: all data for one 4-simplex
# ============================================================
mutable struct GeometryDataset{T<:Real}

    # Group elements
    solgsl2c::Vector{Matrix{Complex{T}}}
    solgso13::Vector{Matrix{T}}

    # Boundary spinors and normals
    bdyxi::Vector{Vector{Vector{Vector{Complex{T}}}}}
    nabout::Vector{Vector{Vector{T}}}
    nabfrombivec::Vector{Vector{Vector{T}}}

    # Boundary group data
    bdysu::Vector{Vector{Matrix{Complex{T}}}}
    bdybivec4d55::Vector{Vector{Matrix{T}}}
    bdybivec55::Vector{Vector{Matrix{Complex{T}}}}

    # Geometry
    dihedrals::Vector{Vector{T}}
    areas::Vector{Vector{T}}
    kappa::Vector{Vector{Int}}
    tetareasign::Vector{Vector{Int}}
    tetn0sign::Vector{Vector{Int}}

    # Tetrahedron data
    tetnormalvec::Vector{Vector{T}}
    sgndet::Vector{Int}

    # Auxiliary / critical data
    zdataf::Vector{Vector{Vector{Complex{T}}}}
end

# ============================================================
# GeometryCollection: multiple simplices + global data
# ============================================================
mutable struct GeometryCollection{T<:Real}
    simplex::Vector{GeometryDataset{T}}
    connectivity::Vector{Dict{String, Any}}
    crit::Dict{Symbol, Any}
    varias::Dict{Symbol, Any}
end

# Convenience constructor
GeometryCollection(simplex::Vector{GeometryDataset{T}}) where {T<:Real} =
    GeometryCollection{T}(simplex, Dict{String,Any}[], Dict{Symbol,Any}(), Dict{Symbol,Any}())

end # module