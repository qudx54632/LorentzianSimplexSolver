module GeometryPipeline

using LinearAlgebra
using Combinatorics

using ..PrecisionUtils: get_tolerance
using ..GeometryTypes: GeometryDataset

using ..SimplexGeometry: eto_f, edge_or
using ..TetraNormals: compute_edgevec, get4dnormal
using ..Dihedral: theta_ab
using ..LorentzGroup: getso13, getsl2c
using ..ThreeDTetra: get3dtet, threetofour, getbivec, getbivec2d
using ..Volume: compute_all_areas, compute_area_signs
using ..Su2Su11FromBivector: getnabfrombivec, face_timelike_sign, su_from_bivectors
using ..XiFromSU: get_xi_from_su
using ..FaceNormals3D: getnabout
using ..KappaFromNormals: compute_kappa

export run_geometry_pipeline

"""
    run_geometry_pipeline(bdypoints) -> GeometryDataset{T}

`bdypoints` are the 5 boundary vertices of a 4-simplex, each a 4-vector.
They must all have the same scalar type `T` (Float64 or BigFloat).
"""
function run_geometry_pipeline(bdypoints::Vector{<:AbstractVector{T}}) where {T<:Real}

    tol = T(get_tolerance())

    # ------------------------------------------------------------
    # Basic counts: bdypoints are VERTICES (must be 5)
    # ------------------------------------------------------------
    Nverts = length(bdypoints)
    Nverts == 5 || error("run_geometry_pipeline: expected 5 vertices, got $Nverts")

    Ntet = 5

    # ------------------------------------------------------------
    # 1. edge vectors for each tetra (5 tets × 6 edges)
    # ------------------------------------------------------------
    edgevec = compute_edgevec(bdypoints)

    # ------------------------------------------------------------
    # 2. 4D tetrahedron normals (5 normals)
    # ------------------------------------------------------------
    tetnormalvec = get4dnormal(bdypoints)  # Vector{Vector{T}}

    # ------------------------------------------------------------
    # 3. SO(1,3) group elements for each tetra normal
    # ------------------------------------------------------------
    solgso13 = [getso13(n) for n in tetnormalvec]   # Vector{Matrix{T}}

    # ------------------------------------------------------------
    # 4. SL(2,C) group elements for each tetra normal
    # ------------------------------------------------------------
    solgsl2c = [getsl2c(n) for n in tetnormalvec]   # Vector{Matrix{Complex{T}}}

    # ------------------------------------------------------------
    # 5. 3D tetra data from 4D embedding
    # ------------------------------------------------------------
    tet3d = [get3dtet(edges, solgso13[i]) for (i, edges) in enumerate(edgevec)]
    threededgevec = [t[1] for t in tet3d]
    sgndet        = [t[3] for t in tet3d]   # Int ±1

    # ------------------------------------------------------------
    # 6. 3D -> 4D edge vectors (per tetra)
    # ------------------------------------------------------------
    threeto4dedgevec = [
        [threetofour(v, sgndet[i]) for v in threededgevec[i]]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 7. face bivectors (4×4 and 2×2), 4 faces per tet then insert identity
    # ------------------------------------------------------------
    bdybivec4d = [
        [getbivec(threeto4dedgevec[i][p[1]], threeto4dedgevec[i][p[2]]) for p in eto_f]
        for i in 1:Ntet
    ]

    Id4 = Matrix{T}(I, 4, 4)
    bdybivec4d55 = [insert!(copy(bdybivec4d[i]), i, Id4) for i in 1:Ntet]

    bdybivec54 = [
        [getbivec2d(threeto4dedgevec[i][p[1]], threeto4dedgevec[i][p[2]]) for p in eto_f]
        for i in 1:Ntet
    ]

    Id2 = Matrix{Complex{T}}(I, 2, 2)
    bdybivec55 = [insert!(copy(bdybivec54[i]), i, Id2) for i in 1:Ntet]

    # ------------------------------------------------------------
    # 8. areas and signs
    # ------------------------------------------------------------
    areas, areasqv = compute_all_areas(bdypoints, edge_or)          # returns T-consistent
    tetareasign = compute_area_signs(areasqv, sgndet)

    tetn0sign = [
        [i == j ? 0 :
         face_timelike_sign(bdybivec55[i][j], sgndet[i], tetareasign[i][j])
         for j in 1:Ntet]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 9. dihedral angles between tetra normals
    # ------------------------------------------------------------
    dihedrals = [
        [theta_ab(tetnormalvec[i], tetnormalvec[j]) for j in 1:Ntet]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 10. boundary SU(2)/SU(1,1) group elements
    # ------------------------------------------------------------
    Z2 = zeros(Complex{T}, 2, 2)
    bdysu = [
        [i == j ? Z2 :
         su_from_bivectors(bdybivec55[i][j], sgndet[i], tetareasign[i][j])
         for j in 1:Ntet]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 11. boundary spinors ξ
    # ------------------------------------------------------------
    zspin = zeros(Complex{T}, 2)
    bdyxi = [
        [i == j ? [copy(zspin), copy(zspin)] :
         get_xi_from_su(bdysu[i][j], sgndet[i], tetareasign[i][j], tetn0sign[i][j])
         for j in 1:Ntet]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 12. outward face normals in 3D
    # IMPORTANT: local indices only (1:5), not global vertex labels
    # ------------------------------------------------------------
    Tetbdypoints = [bdypoints[I] for I in combinations(eachindex(bdypoints), 4)]  # length 5

    nabout54 = [getnabout(Tetbdypoints[i], sgndet[i], solgso13[i]) for i in 1:Ntet]

    z3 = zeros(T, 3)
    nabout = [insert!(copy(nabout54[i]), i, copy(z3)) for i in 1:Ntet]

    # ------------------------------------------------------------
    # 13. normals from bivectors
    # ------------------------------------------------------------
    nabfrombivec = [
        [i == j ? copy(z3) :
         getnabfrombivec(bdybivec55[i][j], sgndet[i])
         for j in 1:Ntet]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 14. κ matrix
    # ------------------------------------------------------------
    kappa = compute_kappa(nabout, nabfrombivec, tetareasign)

    # ------------------------------------------------------------
    # 15. placeholder z-data (if needed later)
    # ------------------------------------------------------------
    zdataf = [[copy(zspin), copy(zspin)] for _ in 1:Ntet]

    # ------------------------------------------------------------
    # Return GeometryDataset{T}
    # ------------------------------------------------------------
    return GeometryDataset{T}(
        solgsl2c, solgso13,
        bdyxi, nabout, nabfrombivec,
        bdysu, bdybivec4d55, bdybivec55,
        dihedrals, areas, kappa, tetareasign, tetn0sign,
        tetnormalvec, sgndet, zdataf
    )
end

end # module GeometryPipeline