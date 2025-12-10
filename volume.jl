module Volume

using LinearAlgebra
using ..SpinAlgebra: eta   # reuse Minkowski metric

export distance_sq, V2sq, V3sq, V4sq,
       CM2D, CM3D, CM4D, compute_all_areas, compute_area_signs

# -------------------------------------------------------------
# Minkowski squared distance between points P1, P2
# -------------------------------------------------------------
distance_sq(P1::AbstractVector{<:Real},
            P2::AbstractVector{<:Real}) =
    (P1 - P2)' * eta * (P1 - P2)

# =============================================================
# 2D AREA SQUARE (triangle)
# =============================================================

# Cayley–Menger matrix for triangle
function CM2D(ls12, ls13, ls23)
    return [
        0.0   ls12  ls13  1.0;
        ls12  0.0   ls23  1.0;
        ls13  ls23  0.0   1.0;
        1.0   1.0   1.0   0.0
    ]
end

# Triangle area squared
function V2sq(ls12, ls13, ls23)
    cm = CM2D(ls12, ls13, ls23)
    return (-1)^(2+1) * det(cm) / (2^2 * factorial(2)^2)
end

# =============================================================
# 3D VOLUME SQUARE (tetrahedron)
# =============================================================

# Cayley–Menger matrix for tetrahedron
function CM3D(ls12, ls13, ls14, ls23, ls24, ls34)
    return [
        0.0   ls12  ls13  ls14  1.0;
        ls12  0.0   ls23  ls24  1.0;
        ls13  ls23  0.0   ls34  1.0;
        ls14  ls24  ls34  0.0   1.0;
        1.0   1.0   1.0   1.0   0.0
    ]
end

# Tetrahedron volume squared
function V3sq(ls12, ls13, ls14, ls23, ls24, ls34)
    cm = CM3D(ls12, ls13, ls14, ls23, ls24, ls34)
    return (-1)^(3+1) * det(cm) / (2^3 * factorial(3)^2)
end

# =============================================================
# 4D VOLUME SQUARE (4-simplex)
# =============================================================

# Cayley–Menger matrix for 4-simplex
function CM4D(ls12, ls13, ls14, ls15,
              ls23, ls24, ls25,
              ls34, ls35,
              ls45)
    return [
        0.0   ls12  ls13  ls14  ls15  1.0;
        ls12  0.0   ls23  ls24  ls25  1.0;
        ls13  ls23  0.0   ls34  ls35  1.0;
        ls14  ls24  ls34  0.0   ls45  1.0;
        ls15  ls25  ls35  ls45  0.0   1.0;
        1.0   1.0   1.0   1.0   1.0   0.0
    ]
end

# 4-simplex volume squared
function V4sq(ls12, ls13, ls14, ls15,
              ls23, ls24, ls25,
              ls34, ls35,
              ls45)
    cm = CM4D(ls12, ls13, ls14, ls15,
              ls23, ls24, ls25,
              ls34, ls35,
              ls45)

    return (-1)^(4+1) * det(cm) / (2^4 * factorial(4)^2)
end

# =============================================================
# AREA MATRIX FOR ALL TETRAHEDRA
# =============================================================
"""
    compute_all_areas(bdypoints, edge_or)

Return a matrix `A[i,j] = sqrt(abs(V2sq))` for each shared triangle
of boundary tetrahedra indexed by `edge_or`.
"""
function compute_all_areas(bdypoints, edge_or)
    N = length(bdypoints)
    areasq = zeros(Float64, N, N)

    for i in 1:N, j in 1:N
        if i == j
            areasq[i, j] = 0
            continue
        end

        # intersection of edges = the triangle (3 edges)
        triangle = intersect(edge_or[i], edge_or[j])

        # compute squared lengths
        ls = [distance_sq(bdypoints[e[1]], bdypoints[e[2]]) for e in triangle]

        if length(ls) == 3
            areasq[i, j] = V2sq(ls[1], ls[2], ls[3])
        else
            areasq[i, j] = 0
        end
    end

    # Convert from NxN matrix -> Vector of N vectors of length N
    areas  = [ [ sqrt(abs(areasq[i,j])) for j in 1:N ] for i in 1:N ]
    areasqv = [ [ areasq[i,j]          for j in 1:N ] for i in 1:N ]

    return areas, areasqv
end

# =============================================================
"""
    compute_area_signs(face_areasq, sgndet)

Return signs[i][j] with same indexing meaning as face_areasq:
    diagonal = 0
    off-diagonal = sign rule
"""
function compute_area_signs(face_areasq, sgndet)
    N = length(sgndet)

    signs = [
        [ i == j ? 0 :
          (sgndet[i] == 1 ?
              1 :
              (face_areasq[i][j] < 0 ? -1 : 1))
        for j in 1:N ]
        for i in 1:N
    ]

    return signs
end

end # module