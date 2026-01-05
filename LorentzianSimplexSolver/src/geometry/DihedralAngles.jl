module Dihedral

using LinearAlgebra
using ..PrecisionUtils: get_tolerance
using ..SpinAlgebra: Params

export theta_ab

η(::Type{T}) where {T<:Real} = Params{T}().eta

# Minkowski inner product
minkowski_dot(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:Real} =
    (a' * η(T) * b)[1]

# Minkowski squared norm
minkowski_norm2(a::AbstractVector{T}) where {T<:Real} =
    minkowski_dot(a, a)

# -------------------------------------------------------------
# Lorentzian / Euclidean dihedral angle between normals Na, Nb
# -------------------------------------------------------------
function theta_ab(Na::AbstractVector{T},
                  Nb::AbstractVector{T}) where {T<:Real}
    tol = T(get_tolerance())
    # squared norms
    Na2 = minkowski_norm2(Na)
    Nb2 = minkowski_norm2(Nb)
    Nab = minkowski_dot(Na, Nb)

    oneT = one(T)
    infT = T(Inf)
    # ---------------------------------------------------------
    # CASE 1: Mixed signature  (spacelike × timelike)
    # ---------------------------------------------------------
    # Na2 * Nb2 < 0
    if Na2 * Nb2 < -tol
        # -ArcSinh[Na . η . Nb]
        return -asinh(Nab)
    end

    # ---------------------------------------------------------
    # CASE 2: Both timelike  (Na2 < 0 && Nb2 < 0)
    # ---------------------------------------------------------
    if Na2 < -tol && Nb2 < -tol
        if Nab < -tol
            x = clamp(-Nab, oneT, infT)
            # -ArcCosh[-Na η Nb]
            return -acosh(x)
        else
            x = clamp(Nab, oneT, infT)
            # ArcCosh[Na η Nb]
            return acosh(x)
        end
    end

    # ---------------------------------------------------------
    # CASE 3: Euclidean-like (both spacelike) → ArcCos(Na ⋅ Nb)
    # ---------------------------------------------------------
    if Na2 > tol && Nb2 > tol
        x = clamp(dot(Na, Nb), -oneT, oneT)
        return acos(x)
    end

    # ---------------------------------------------------------
    # If no branch matches, throw descriptive error
    # ---------------------------------------------------------
    error("theta_ab: undefined signature combination Na2=$Na2, Nb2=$Nb2")
end

end # module