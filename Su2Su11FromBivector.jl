module Su2Su11FromBivector

using LinearAlgebra
using ..PrecisionUtils: get_tolerance
using ..SpinAlgebra: σ1, σ2, σ3, imag_unit
using ..Dihedral: theta_ab
using ..LorentzGroup: wedge, bivec1tohalf, exp2x2_from_square

export getnabfrombivec, nab3d_to_nab4d, face_timelike_sign,
       choose_Nref, su_from_bivectors

# ----------------------------------------------
# Local trace helper
# ----------------------------------------------
tr(A) = LinearAlgebra.tr(A)

# ----------------------------------------------
# Pauli matrices promoted to Complex{T}
# ----------------------------------------------
σ1C(::Type{T}) where {T<:Real} = Complex{T}.(σ1(T))
σ2C(::Type{T}) where {T<:Real} = σ2(T)                  # already Complex{T}
σ3C(::Type{T}) where {T<:Real} = Complex{T}.(σ3(T))

# ---------------------------------------------------------
# 1. Extract 3-vector n_ab from 2×2 bivector B_1/2
# ---------------------------------------------------------
function getnabfrombivec(B12::AbstractMatrix{Complex{T}}, sgndet::Int) where {T<:Real}
    if sgndet == 1
        E = real(tr(B12 * σ1C(T)))
        F = real(tr(B12 * σ2C(T)))
        G = real(tr(B12 * σ3C(T)))
        return [E, F, G]
    else
        i = imag_unit(T)
        E = real(tr(B12 * σ3C(T)))
        F = real(tr(B12 * (-i * σ1C(T))))
        G = real(tr(B12 * (-i * σ2C(T))))
        return [E, -G, F]
    end
end

# ---------------------------------------------------------
# 2. Convert 3-vector n → 4-vector depending on sgndet
# ---------------------------------------------------------
function nab3d_to_nab4d(n3::AbstractVector{T}, sgndet::Int) where {T<:Real}
    z = zero(T)
    sgndet > 0 ? vcat(T[z], n3) : vcat(n3, T[z])
end

# ---------------------------------------------------------
# Helper: normalize 4D bivector (4×4 real matrix)
# ---------------------------------------------------------
tr4(A) = real(tr(A * A))

function _normalize_bivec(B::AbstractMatrix{T}) where {T<:Real}
    val = abs(tr4(B) / T(2))
    # val > T(get_tolerance()) || error("normalize_bivec: degenerate bivector")
    return B / sqrt(val)
end

# ---------------------------------------------------------
# Decide a spacelike face in timelike tetra is future/past pointing
# ---------------------------------------------------------
function face_timelike_sign(B12::AbstractMatrix{Complex{T}},
                            sgndet::Int,
                            tetareasign::Int) where {T<:Real}

    if sgndet == -1 && tetareasign == 1
        t = real(tr(B12 * σ3C(T)))
        return t > zero(T) ? 1 : (t < zero(T) ? -1 : 0)
    end

    return 0
end

# ---------------------------------------------------------
# Choose Nref for SU(2) or SU(1,1), with tetareasign
# ---------------------------------------------------------
function choose_Nref(B12::AbstractMatrix{Complex{T}},
                     sgndet::Int,
                     tetareasign::Int) where {T<:Real}

    z = zero(T)
    o = one(T)

    if sgndet == 1
        return T[z, z, z, o]                 # (0,0,0,1)
    else
        if tetareasign == -1
            return T[z, z, o, z]             # (0,0,1,0)
        else
            s = face_timelike_sign(B12, sgndet, tetareasign)
            return s == 1 ? T[o, z, z, z] : T[-o, z, z, z]
        end
    end
end

# ---------------------------------------------------------
# Main SU(2)/SU(1,1) group element
# ---------------------------------------------------------
function su_from_bivectors(B12::AbstractMatrix{Complex{T}},
                           sgndet::Int,
                           tetareasign::Int) where {T<:Real}

    atol = T(get_tolerance())
    z = zero(T)
    o = one(T)

    # 1. algebra vector
    n3 = getnabfrombivec(B12, sgndet)

    # 2. embed into 4D
    n4 = nab3d_to_nab4d(n3, sgndet)

    # 3. pick reference normal
    Nref_vec = choose_Nref(B12, sgndet, tetareasign)

    i = imag_unit(T)

    # ---------------------------
    # SU(2) special case
    # ---------------------------
    if sgndet == 1
        if norm(n4 .- T[z, z, z, -o]) < atol
            return i * σ2C(T)
        elseif norm(n4 .- T[z, z, z, o]) < atol
            return Matrix{Complex{T}}(I, 2, 2)
        end
    end

    # ---------------------------
    # SU(1,1) special case
    # ---------------------------
    if sgndet == -1
        if tetareasign == 1
            if norm(n4 .- Nref_vec) < atol
                return Matrix{Complex{T}}(I, 2, 2)
            end
        else
            if norm(n4 .- T[z, z, -o, z]) < atol
                return i * σ3C(T)
            elseif norm(n4 .- T[z, z, o, z]) < atol
                return Matrix{Complex{T}}(I, 2, 2)
            end
        end
    end

    # 4. bivector
    B4 = wedge(n4, Nref_vec)                 # 4×4 real matrix (T)

    # 5. normalize
    Bnorm = _normalize_bivec(B4)

    # 6. dihedral angle
    θ = theta_ab(Nref_vec, n4)               # θ::T

    # 7. lie algebra → 2×2 matrix
    X = bivec1tohalf(Bnorm)                  # Matrix{Complex{T}}

    # 8. exponentiate
    g = exp2x2_from_square(θ, X)

    return g
end

end # module