module TetraNormals

using LinearAlgebra
using Combinatorics
using ..SpinAlgebra: Params
using ..SimplexGeometry: edge_or
using ..PrecisionUtils: get_tolerance

export minkowski_norm, tet_normal, get4dnormal, compute_edgevec

η(::Type{T}) where {T<:Real} = Params{T}().eta

# -------------------------------------------------------------
# Minkowski inner product  (-,+,+,+)
# -------------------------------------------------------------
minkowski_norm(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:Real} =
    (a' * η(T) * b)[1]

# -----------------------------------------------------------
#  list of edge vectors for each tetrahedron in the 4-simplex
# -----------------------------------------------------------
function compute_edgevec(bdypoints::Vector{<:AbstractVector{T}}) where {T<:Real}
    ntet = length(edge_or)   # for a 4-simplex this should be 5
    edgevec = Vector{Vector{Vector{T}}}(undef, ntet)

    for j in 1:ntet
        edges_j = Vector{Vector{T}}()
        for pair in edge_or[j]
            v1 = bdypoints[pair[1]]
            v2 = bdypoints[pair[2]]
            push!(edges_j, v1 .- v2)
        end
        edgevec[j] = edges_j
    end

    return edgevec
end

# -------------------------------------------------------------
# Compute normal vector to a tetrahedron given its 3 edge vectors
# edgetet = [e1, e2, e3]   each ei is a 4-vector
# -------------------------------------------------------------
function tet_normal(edgetet::Vector{Vector{T}}) where {T<:Real}

    # Build covariant edge rows: η * e
    E = hcat((η(T) * e for e in edgetet[1:3])...)

    # Compute normal via Levi-Civita (exact)
    n = Vector{T}(undef, 4)

    for μ in 1:4
        inds = setdiff(1:4, μ)
        n[μ] = (isodd(μ + 1) ? one(T) : -one(T)) * det(E[inds, :])
    end

    # Minkowski norm
    normsq = (n' * η(T) * n)[1]
    # abs(normsq) > get_tolerance() || error("Degenerate tetrahedron")

    # Normalize (keep sign!)
    n ./= sqrt(abs(normsq))

    return n
end

# -------------------------------------------------------------
# Barycentric test for whether a point lies inside the 4-simplex
#
# Returns true if point is inside the convex hull of bdypoints.
#
# A * α = b,   where
#   A = [P1 P2 P3 P4 P5; 1 1 1 1 1]
#   α = barycentric coordinates
#   b = [point; 1]
# -------------------------------------------------------------
function is_inside_simplex(point::AbstractVector{T},
                           bdypoints::Vector{<:AbstractVector{T}}) where {T<:Real}
    # Build coefficient matrix A
    # A is 5×5:
    # [P1 P2 P3 P4 P5
    #   1  1  1  1  1]
    nv = length(bdypoints) 
    A = vcat(hcat(bdypoints...), fill(one(T), 1, nv))
    
    # Right-hand side b = [point; 1]
    b = vcat(point, one(T))

    # Solve for barycentric coordinates α
    α = A \ b

    tol = T(get_tolerance())
    return all(α .>= -tol)
end


function get4dnormal(bdypoints::Vector{<:AbstractVector{T}}) where {T<:Real}

    edgetet_all = compute_edgevec(bdypoints)
    # Step 1: unsigned normals
    normals_unsigned = [tet_normal(edges) for edges in edgetet_all]

    # Step 2: tetrahedron barycenters
    tets = collect(combinations(bdypoints, 4))
    centers = [sum(tet) / T(4) for tet in tets]

    # Step 3: shifted test points
    eps = sqrt(T(get_tolerance()))
    pert_plus = [centers[i] .+ eps * normals_unsigned[i] for i in 1:5]

    normals = Vector{Vector{T}}(undef, 5)

    # Step 4: determine sign by barycentric test
    for i in 1:5
        if is_inside_simplex(pert_plus[i], bdypoints)
            # (center + eps n) inside ⇒ n points inward ⇒ flip
            normals[i] = -normals_unsigned[i]
        else
            # already outward
            normals[i] = normals_unsigned[i]
        end
    end

    return normals
end

end # module