module GeometryPipeline

using LinearAlgebra
using Combinatorics

using ..GeometryTypes: GeometryDataset

# These modules / functions should already exist in your code base.
# Adjust the paths / names if needed.
using ..SpinAlgebra
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
# using ..write_geometry_data:save_geometry_data

export run_geometry_pipeline

"""
    run_geometry_pipeline(bdypoints) -> GeometryDataset

Given the boundary points `bdypoints::Vector{Vector{Float64}}` of a 4-simplex,
compute all geometric data (group elements, bivectors, spinors, normals,
areas, κ, etc.) and return a `GeometryDataset`.
"""

function run_geometry_pipeline(bdypoints::Vector{<:Vector{<:Real}})

    # ------------------------------------------------------------
    # Basic counts
    # ------------------------------------------------------------
    Ntet = length(bdypoints)   # for a 4-simplex: Ntet = 5

    # ------------------------------------------------------------
    # 1. edge vectors
    # ------------------------------------------------------------
    edgevec = compute_edgevec(bdypoints)

    # ------------------------------------------------------------
    # 2. 4D tetrahedron normals
    # ------------------------------------------------------------
    tetnormalvec = get4dnormal(bdypoints)

    # ------------------------------------------------------------
    # 3. SO(1,3) group elements for each tetra normal
    # ------------------------------------------------------------
    solgso13 = [getso13(n) for n in tetnormalvec]

    # ------------------------------------------------------------
    # 4. SL(2,C) group elements for each tetra normal
    # ------------------------------------------------------------
    solgsl2c = [getsl2c(n) for n in tetnormalvec]

    # ------------------------------------------------------------
    # 5. 3D tetra data from 4D embedding
    #    get3dtet(edgevec[i], solgso13[i]) → (3d_edges, zero_pos, sgndet)
    # ------------------------------------------------------------
    tet3d = [get3dtet(edges, solgso13[i]) for (i, edges) in enumerate(edgevec)]
    threededgevec = [t[1] for t in tet3d]
    sgndet        = [t[3] for t in tet3d]
    
    # ------------------------------------------------------------
    # 6. 3D → 4D edge vectors (per tetra)
    # ------------------------------------------------------------
    threeto4dedgevec = [
        [threetofour(v, sgndet[i]) for v in threededgevec[i]]
        for i in 1:Ntet
    ]
    

    # ------------------------------------------------------------
    # 7. face bivectors (4×4 and 2×2)
    # ------------------------------------------------------------

    # spin-1 rep (4×4): 4 faces per tet
    bdybivec4d = [
        [getbivec(threeto4dedgevec[i][p[1]], threeto4dedgevec[i][p[2]]) for p in eto_f]
        for i in 1:Ntet
    ]

    # Build an identity of the same element type as these bivectors
    T4 = eltype(first(first(bdybivec4d)))  # element type of 4×4 matrices
    Id4 = Matrix{T4}(I, 4, 4)

    # Insert identity at face i → 5 faces per tet
    bdybivec4d55 = [insert!(copy(bdybivec4d[i]), i, Id4) for i in 1:Ntet]

    # spin-½ rep (2×2): 4 faces per tet
    bdybivec54 = [
        [getbivec2d(threeto4dedgevec[i][p[1]], threeto4dedgevec[i][p[2]]) for p in eto_f]
        for i in 1:Ntet
    ]

    T2 = eltype(first(first(bdybivec54)))  # element type of 2×2 matrices
    Id2 = Matrix{T2}(I, 2, 2)

    # Insert identity at face i → 5 faces per tet (2×2 matrices)
    bdybivec55 = [insert!(copy(bdybivec54[i]), i, Id2) for i in 1:Ntet]

    # ------------------------------------------------------------
    # 9. areas and signs
    # ------------------------------------------------------------
    areas, areasqv = compute_all_areas(bdypoints, edge_or)
    tetareasign = compute_area_signs(areasqv, sgndet)

    tetn0sign = [
        [i == j ? 0 :
         face_timelike_sign(bdybivec55[i][j], sgndet[i], tetareasign[i][j])
         for j in 1:Ntet]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 10. dihedral angles: dihedrals[i][j] = θ_ab(N_i, N_j)
    # ------------------------------------------------------------
    dihedrals = [
        [theta_ab(tetnormalvec[i], tetnormalvec[j]) for j in 1:Ntet]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 11. boundary SU(2)/SU(1,1) group elements
    # ------------------------------------------------------------
    bdysu = [
        [i == j ? zeros(T2, 2, 2) :
         su_from_bivectors(bdybivec55[i][j], sgndet[i], tetareasign[i][j])
         for j in 1:Ntet]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 12. boundary spinors ξ from SU(2)/SU(1,1)
    #      bdyxi[i][j] = [ξ⁺, ξ⁻] (each length-2 vector)
    # ------------------------------------------------------------
    bdyxi = [
        [i == j ? [zeros(T2, 2), zeros(T2, 2)] :
         get_xi_from_su(bdysu[i][j], sgndet[i], tetareasign[i][j], tetn0sign[i][j])
         for j in 1:Ntet]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 13. outward face normals in 3D
    # ------------------------------------------------------------
    Tetbdypoints = [bdypoints[I] for I in combinations(1:Ntet, 4)]
    nabout54 = [getnabout(Tetbdypoints[i], sgndet[i], solgso13[i]) for i in 1:Ntet]
    nabout = [insert!(copy(nabout54[i]), i, [0.0, 0.0, 0.0]) for i in 1:Ntet]

    # ------------------------------------------------------------
    # 14. normals from bivectors (for closure)
    # ------------------------------------------------------------
    nabfrombivec = [
        [i == j ? [0.0, 0.0, 0.0] :
         getnabfrombivec(bdybivec55[i][j], sgndet[i])
         for j in 1:Ntet]
        for i in 1:Ntet
    ]

    # ------------------------------------------------------------
    # 15. orientation of each face (κ matrix)
    # ------------------------------------------------------------
    kappa = compute_kappa(nabout, nabfrombivec, tetareasign)

    # Placeholder z-data (spinor data if needed later)
    zdataf = [[zeros(T2, 2), zeros(T2, 2)] for _ in 1:Ntet]

    # ------------------------------------------------------------
    # Return a GeometryDataset
    # ------------------------------------------------------------
    return GeometryDataset(
        solgsl2c, solgso13,
        bdyxi, nabout, nabfrombivec,
        bdysu, bdybivec4d55, bdybivec55,
        dihedrals, areas, kappa, tetareasign, tetn0sign,
        tetnormalvec, sgndet, zdataf
    )
end

end # module GeometryPipeline