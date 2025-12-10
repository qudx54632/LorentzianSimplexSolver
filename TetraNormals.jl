module TetraNormals

using LinearAlgebra
using Combinatorics
using ..SpinAlgebra: eta   # reuse Minkowski metric
using ..SimplexGeometry: edge_or, tet_order

export minkowski_norm, tet_normal, get4dnormal, chop, compute_edgevec

# -------------------------------------------------------------
# Minkowski inner product  (-,+,+,+)
# -------------------------------------------------------------
minkowski_norm(a, b) = a' * eta * b

# Optional numerical chop
# -----------------------------------------------------------
# Real number chop
chop(x::Real; tol=1e-12) = abs(x) < tol ? 0.0 : x

# Complex number chop
function chop(z::Complex; tol=1e-12)
    re = abs(real(z)) < tol ? 0.0 : real(z)
    im = abs(imag(z)) < tol ? 0.0 : imag(z)
    return complex(re, im)
end

# Recursive chop for arrays
chop(A::AbstractArray; tol=1e-12) = map(x -> chop(x; tol=tol), A)

# -----------------------------------------------------------
#  list of edge vectors for each tetrahedron in the 4-simplex
# -----------------------------------------------------------
function compute_edgevec(bdypoints::Vector{<:AbstractVector})
    ntet = length(edge_or)   # for a 4-simplex this should be 5
    edgevec = Vector{Vector{Vector{Float64}}}(undef, ntet)

    for j in 1:ntet
        edges_j = Vector{Vector{Float64}}()
        for pair in edge_or[j]
            v1 = bdypoints[pair[1]]
            v2 = bdypoints[pair[2]]
            push!(edges_j, chop(v1 .- v2))
        end
        edgevec[j] = edges_j
    end

    return edgevec
end

# -------------------------------------------------------------
# Compute normal vector to a tetrahedron given its 3 edge vectors
# edgetet = [e1, e2, e3]   each ei is a 4-vector
# -------------------------------------------------------------
function tet_normal(edgetet::Vector{Vector{Float64}})

    # Step 1 — Build M
    M = zeros(Float64, 3, 4)
    for i in 1:3
        e = edgetet[i]
        # (eᵗ η) is the row
        M[i, :] = (e' * eta)
    end

    # Step 2 — Nullspace (exact linear-algebra solution)
    N = nullspace(M)   # size (4,1) for 1D nullspace

    n_raw = vec(N[:, 1])   # extract as a 4-vector

    # Step 3 — Normalize using Minkowski metric
    normsq = minkowski_norm(n_raw, n_raw)
    n = n_raw / sqrt(abs(normsq))

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
function is_inside_simplex(point::AbstractVector{<:Real},
                           bdypoints::Vector{<:AbstractVector})
    # Build coefficient matrix A
    # A is 5×5:
    # [P1 P2 P3 P4 P5
    #   1  1  1  1  1]
    A = vcat(hcat(bdypoints...), ones(1, length(bdypoints)))

    # Right-hand side b = [point; 1]
    b = vcat(point, 1.0)

    # Solve for barycentric coordinates α
    α = A \ b

    # Inside iff all α_i >= 0 (small negative tolerance allowed)
    return all(α .>= -1e-12)
end


function get4dnormal(bdypoints)

    edgetet_all = compute_edgevec(bdypoints)
    # Step 1: unsigned normals
    normals_unsigned = [tet_normal(edges) for edges in edgetet_all]

    # Step 2: tetrahedron barycenters
    tets = collect(combinations(bdypoints, 4))
    centers = [sum(tet) / 4 for tet in tets]

    # Step 3: shifted test points
    eps = 1e-4
    pert_plus  = [centers[i] .+ eps * normals_unsigned[i] for i in 1:5]

    normals = Vector{Vector{Float64}}(undef, 5)

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

    # if you want to chop them too:
    normals = [chop(n) for n in normals]

    return normals
end

end # module