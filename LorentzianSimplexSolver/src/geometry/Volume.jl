module Volume

using LinearAlgebra
using ..PrecisionUtils: get_tolerance
using ..SpinAlgebra: Params

export distance_sq, V2sq, V3sq, V4sq,
       CM2D, CM3D, CM4D,
       compute_all_areas, compute_area_signs

# -------------------------------------------------------------
# Minkowski metric (-,+,+,+) for scalar type T
# -------------------------------------------------------------
η(::Type{T}) where {T<:Real} = Params{T}().eta

# -------------------------------------------------------------
# Minkowski squared distance between points P1, P2
# -------------------------------------------------------------
function distance_sq(P1::AbstractVector{T},
                     P2::AbstractVector{T}) where {T<:Real}
    ((P1 - P2)' * η(T) * (P1 - P2))[1]
end

# =============================================================
# 2D AREA SQUARE (triangle)
# =============================================================

# Cayley–Menger matrix for triangle
function CM2D(ls12::T, ls13::T, ls23::T) where {T<:Real}
    T[
        zero(T) ls12    ls13    one(T);
        ls12    zero(T) ls23    one(T);
        ls13    ls23    zero(T) one(T);
        one(T)  one(T)  one(T)  zero(T)
    ]
end

# Triangle area squared
V2sq(ls12::T, ls13::T, ls23::T) where {T<:Real} =
    (-one(T))^(2+1) * det(CM2D(ls12, ls13, ls23)) /
    (T(2)^2 * factorial(2)^2)

# =============================================================
# 3D VOLUME SQUARE (tetrahedron)
# =============================================================

# Cayley–Menger matrix for tetrahedron
function CM3D(ls12::T, ls13::T, ls14::T,
              ls23::T, ls24::T, ls34::T) where {T<:Real}
    T[
        zero(T) ls12    ls13    ls14    one(T);
        ls12    zero(T) ls23    ls24    one(T);
        ls13    ls23    zero(T) ls34    one(T);
        ls14    ls24    ls34    zero(T) one(T);
        one(T)  one(T)  one(T)  one(T)  zero(T)
    ]
end

# Tetrahedron volume squared
V3sq(ls12::T, ls13::T, ls14::T,
     ls23::T, ls24::T, ls34::T) where {T<:Real} =
    (-one(T))^(3+1) * det(CM3D(ls12, ls13, ls14, ls23, ls24, ls34)) /
    (T(2)^3 * factorial(3)^2)

# =============================================================
# 4D VOLUME SQUARE (4-simplex)
# =============================================================

# Cayley–Menger matrix for 4-simplex
function CM4D(ls12::T, ls13::T, ls14::T, ls15::T,
              ls23::T, ls24::T, ls25::T,
              ls34::T, ls35::T,
              ls45::T) where {T<:Real}
    T[
        zero(T) ls12    ls13    ls14    ls15    one(T);
        ls12    zero(T) ls23    ls24    ls25    one(T);
        ls13    ls23    zero(T) ls34    ls35    one(T);
        ls14    ls24    ls34    zero(T) ls45    one(T);
        ls15    ls25    ls35    ls45    zero(T) one(T);
        one(T)  one(T)  one(T)  one(T)  one(T)  zero(T)
    ]
end

# 4-simplex volume squared
V4sq(ls12::T, ls13::T, ls14::T, ls15::T,
     ls23::T, ls24::T, ls25::T,
     ls34::T, ls35::T,
     ls45::T) where {T<:Real} =
    (-one(T))^(4+1) * det(CM4D(ls12, ls13, ls14, ls15,
                               ls23, ls24, ls25,
                               ls34, ls35, ls45)) /
    (T(2)^4 * factorial(4)^2)

# =============================================================
# AREA MATRIX FOR ALL TETRAHEDRA
# =============================================================
function compute_all_areas(bdypoints::Vector{<:AbstractVector{T}},
                           edge_or) where {T<:Real}

    N = length(bdypoints)
    areasq = zeros(T, N, N)

    for i in 1:N, j in 1:N
        i == j && continue

        triangle = intersect(edge_or[i], edge_or[j])
        ls = [distance_sq(bdypoints[e[1]], bdypoints[e[2]]) for e in triangle]

        if length(ls) == 3
            areasq[i, j] = V2sq(ls[1], ls[2], ls[3])
        else
            areasq[i, j] = zero(T)
        end
    end

    areas   = [[sqrt(abs(areasq[i,j])) for j in 1:N] for i in 1:N]
    areasqv = [[areasq[i,j]            for j in 1:N] for i in 1:N]

    return areas, areasqv
end

# =============================================================
# AREA SIGN RULES
# =============================================================
function compute_area_signs(face_areasq, sgndet)
    N = length(sgndet)
    tol = get_tolerance()

    signs = [
        [ i == j ? 0 :
          (sgndet[i] == 1 ?
              1 :
              (face_areasq[i][j] < -tol ? -1 : 1))
        for j in 1:N ]
        for i in 1:N
    ]

    return signs
end

end # module