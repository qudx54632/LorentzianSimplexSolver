module Dihedral

using LinearAlgebra
using ..SpinAlgebra: eta   # reuse Minkowski metric

export theta_ab


# Minkowski inner product
minkowski_dot(a, b) = a' * eta * b

# Minkowski squared norm
minkowski_norm2(a) = minkowski_dot(a, a)

# -------------------------------------------------------------
# Lorentzian / Euclidean dihedral angle between normals Na, Nb
# -------------------------------------------------------------
function theta_ab(Na::AbstractVector, Nb::AbstractVector)

    # squared norms
    Na2 = minkowski_norm2(Na)
    Nb2 = minkowski_norm2(Nb)

    # Minkowski product
    Nab = minkowski_dot(Na, Nb)

    # ---------------------------------------------------------
    # CASE 1: Mixed signature  (spacelike × timelike)
    # ---------------------------------------------------------
    # Na2 * Nb2 < 0
    if Na2 * Nb2 < 0
        # -ArcSinh[Na . η . Nb]
        return -asinh(Nab)
    end

    # ---------------------------------------------------------
    # CASE 2: Both timelike  (Na2 < 0 && Nb2 < 0)
    # ---------------------------------------------------------
    if Na2 < 0 && Nb2 < 0
        if Nab < 0
            x = clamp(-Nab, 1.0, Inf)
            # -ArcCosh[-Na η Nb]
            return -acosh(x)
        else
            Nab = clamp(Nab, 1.0, Inf)
            # ArcCosh[Na η Nb]
            return acosh(Nab)
        end
    end

    # ---------------------------------------------------------
    # CASE 3: Euclidean-like (both spacelike) → ArcCos(Na ⋅ Nb)
    # ---------------------------------------------------------
    if Na2 > 0 && Nb2 > 0
        Nab = clamp(dot(Na, Nb), -1.0, 1.0)
        return acos(Nab)
    end

    # ---------------------------------------------------------
    # If no branch matches, throw descriptive error
    # ---------------------------------------------------------
    error("theta_ab: undefined signature combination Na2=$Na2, Nb2=$Nb2")
end

end # module