module FaceNormals3D

using LinearAlgebra

export getvertices3d, compute_unsigned_normal, getnabout

# ------------------------------------------------------------
# getvertices3d:
#   Compute 3D coordinates of tetrahedron vertices by applying
#   inverse SO(1,3) transformation and dropping 1 component.
# ------------------------------------------------------------
function getvertices3d(tetpoints, so13soln, sgndet::Int)

    invΛ = inv(so13soln)

    verts3d = Vector{Vector{Float64}}()

    for P4 in tetpoints
        P3 = invΛ * P4
        if sgndet == 1
            push!(verts3d, P3[2:4])   # drop time component
        else
            push!(verts3d, P3[1:3])   # drop last component
        end
    end

    return verts3d
end


# ------------------------------------------------------------
# metric(3D): diag(1,1,1) for spacelike tetra
#             diag(-1,1,1) for timelike tetra
# ------------------------------------------------------------
function face_metric(sgndet)
    sgndet == 1 ? Diagonal([1.0,1.0,1.0]) : Diagonal([-1.0,1.0,1.0])
end


# ------------------------------------------------------------
# Solve for face-normal of a triangular face
# Inputs:
#    v1,v2,v3 — 3D vertex coordinates of face
#    metric   — 3×3 metric diag(±1,1,1)
#
# Output:
#    unit 3-vector normal (unsigned)
# ------------------------------------------------------------
function compute_unsigned_normal(v1, v2, v3, metric)
    # Edge vectors
    e1 = v2 .- v1
    e2 = v3 .- v1

    # Euclidean cross product first (produces covector orthogonal under δ_ij)
    c = cross(e1, e2)

    if norm(c) < 1e-14
        error("compute_unsigned_normal: degenerate triangle.")
    end

    # Convert to vector orthogonal under metric η : nᵀ η e_i = 0
    n_raw = metric * c

    # Normalize under metric
    normsq = abs(n_raw' * metric * n_raw)
    if normsq < 1e-20
        error("compute_unsigned_normal: metric-normalized norm too small.")
    end

    return n_raw / sqrt(normsq)
end


# ------------------------------------------------------------
# Compute whether a point lies inside the tetrahedron
# Using barycentric coordinates
# ------------------------------------------------------------
function point_inside_tetra(point, verts3d)

    A = hcat(verts3d...)       # 3×4 matrix
    A = vcat(A, ones(1,4))     # 4×4 barycentric matrix
    b = vcat(point, 1.0)

    α = A \ b
    return all(α .>= -1e-12)
end


# ------------------------------------------------------------
# Main function: compute OUTGOING face normals
# tetpoints: 4 points in 4D
# sgndet: signature of tetra
# so13soln: SO(1,3) matrix Λ
#
# Returns:
#    Vector{Vector{Float64}} length 4 → one normal per face
# ------------------------------------------------------------
function getnabout(tetpoints, sgndet::Int, so13soln)

    # Step 1: 3D embedded vertices
    verts3D = getvertices3d(tetpoints, so13soln, sgndet)

    metric = face_metric(sgndet)

    # Faces of tetrahedron (indices)
    faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]

    unsigned_normals = Vector{Vector{Float64}}()
    outgoing_normals = Vector{Vector{Float64}}()

    centers = Vector{Vector{Float64}}()

    # Step 2: compute unsigned normal & face centers
    for f in faces
        i,j,k = f
        v1,v2,v3 = verts3D[i], verts3D[j], verts3D[k]

        n_unsigned = compute_unsigned_normal(v1,v2,v3,metric)
        push!(unsigned_normals, n_unsigned)

        # face barycenter
        push!(centers, (v1 .+ v2 .+ v3) ./ 3)
    end

    # Step 3: decide outgoing orientation
    for (n, c) in zip(unsigned_normals, centers)

        ε = 1e-2

        p_out = c .+ ε*n
        p_in  = c .- ε*n

        # if p_out is OUTSIDE tetra → n is outgoing
        if point_inside_tetra(p_out, verts3D)
            # wrong direction → flip
            push!(outgoing_normals, -n)
        else
            push!(outgoing_normals, n)
        end
    end

    return outgoing_normals
end


end # module