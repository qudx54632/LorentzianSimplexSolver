module FaceNormals3D

using LinearAlgebra
using ..PrecisionUtils: get_tolerance

export getvertices3d, compute_unsigned_normal, getnabout

# ------------------------------------------------------------
# getvertices3d:
#   Compute 3D coordinates of tetrahedron vertices by applying
#   inverse SO(1,3) transformation and dropping 1 component.
# ------------------------------------------------------------
function getvertices3d(tetpoints::Vector{<:AbstractVector{T}},
                       so13soln::AbstractMatrix{T},
                       sgndet::Int) where {T<:Real}

    invΛ = inv(so13soln)
    verts3d = Vector{Vector{T}}()

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
function face_metric(::Type{T}, sgndet::Int) where {T<:Real}
    z = zero(T)
    o = one(T)
    sgndet == 1 ? Diagonal(T[o, o, o]) : Diagonal(T[-o, o, o])
end

# ------------------------------------------------------------
# Solve for face-normal of a triangular face (unsigned)
# ------------------------------------------------------------
function compute_unsigned_normal(v1::AbstractVector{T},
                                 v2::AbstractVector{T},
                                 v3::AbstractVector{T},
                                 metric::AbstractMatrix{T}) where {T<:Real}

    tol = T(get_tolerance())

    e1 = v2 .- v1
    e2 = v3 .- v1

    # Euclidean cross product
    c = cross(e1, e2)

    norm_c = norm(c)
    norm_c > tol || error("compute_unsigned_normal: degenerate triangle.")

    # Convert covector to vector under metric
    n_raw = metric * c

    normsq = abs((n_raw' * metric * n_raw)[1])
    normsq > tol || error("compute_unsigned_normal: metric norm too small.")

    return n_raw / sqrt(normsq)
end

# ------------------------------------------------------------
# Barycentric test: point inside tetrahedron
# ------------------------------------------------------------
function point_inside_tetra(point::AbstractVector{T},
                            verts3d::Vector{<:AbstractVector{T}}) where {T<:Real}

    tol = T(get_tolerance())

    A = hcat(verts3d...)                    # 3×4
    A = vcat(A, fill(one(T), 1, 4))         # 4×4
    b = vcat(point, one(T))

    α = A \ b
    return all(α .>= -tol)
end

# ------------------------------------------------------------
# Main function: compute OUTGOING face normals
# ------------------------------------------------------------
function getnabout(tetpoints::Vector{<:AbstractVector{T}},
                   sgndet::Int,
                   so13soln::AbstractMatrix{T}) where {T<:Real}

    tol = T(get_tolerance())

    # Step 1: embedded 3D vertices
    verts3D = getvertices3d(tetpoints, so13soln, sgndet)

    metric = face_metric(T, sgndet)

    faces = ((1,2,3), (1,2,4), (1,3,4), (2,3,4))

    unsigned_normals = Vector{Vector{T}}()
    centers = Vector{Vector{T}}()

    # Step 2: unsigned normals and face centers
    for (i,j,k) in faces
        v1, v2, v3 = verts3D[i], verts3D[j], verts3D[k]

        n_unsigned = compute_unsigned_normal(v1, v2, v3, metric)
        push!(unsigned_normals, n_unsigned)

        push!(centers, (v1 .+ v2 .+ v3) ./ T(3))
    end

    outgoing_normals = Vector{Vector{T}}()
    ε = sqrt(tol)

    # Step 3: outgoing orientation
    for (n, c) in zip(unsigned_normals, centers)
        p_out = c .+ ε * n

        if point_inside_tetra(p_out, verts3D)
            push!(outgoing_normals, -n)
        else
            push!(outgoing_normals, n)
        end
    end

    return outgoing_normals
end

end # module