module Su2Su11FromBivector

using LinearAlgebra
using ..SpinAlgebra: σ1, σ2, σ3
using ..Dihedral: theta_ab
using ..LorentzGroup: wedge, bivec1tohalf
using ..TetraNormals: chop

export getnabfrombivec, nab3d_to_nab4d, face_timelike_sign,
       choose_Nref, su_from_bivectors

# ----------------------------------------------
# Local trace helper
# ----------------------------------------------
tr(A) = LinearAlgebra.tr(A)

# ---------------------------------------------------------
# 1. Extract 3-vector n_ab from 2×2 bivector B_1/2
#
# sgndet = +1 → su(2)
# sgndet = -1 → su(1,1)
#
# Returns a real 3-vector (n1, n2, n3)
# ---------------------------------------------------------
function getnabfrombivec(B12::AbstractMatrix, sgndet::Int)
    if sgndet == 1
        # SU(2): (E, F, G)
        E = real(tr(B12 * σ1))
        F = real(tr(B12 * σ2))
        G = real(tr(B12 * σ3))
        return [E, F, G]
    else
        # SU(1,1): (E, -G, F)
        E = real(tr(B12 * σ3))
        F = real(tr(B12 * (-1im * σ1)))
        G = real(tr(B12 * (-1im * σ2)))
        return [E, -G, F]
    end
end

# ---------------------------------------------------------
# 2. Convert 3-vector n → 4-vector depending on sgndet:
#
# sgndet > 0 → (0, n1, n2, n3)
# sgndet < 0 → (n1, n2, n3, 0)
# ---------------------------------------------------------
function nab3d_to_nab4d(n3, sgndet)
    if sgndet > 0
        return vcat(0.0, n3)
    else
        return vcat(n3, 0.0)
    end
end

# ---------------------------------------------------------
# Helper: normalize 4D bivector
# ---------------------------------------------------------
tr4(A) = real(tr(A * A))
_normalize_bivec(B) = B / sqrt(abs(0.5 * tr4(B)))

# ---------------------------------------------------------
# Decide a spacelike face in timelike tetra is future pointing 
# or past pointing
# ---------------------------------------------------------
function face_timelike_sign(B12::AbstractMatrix, 
                            sgndet::Int, 
                            tetareasign::Int)

    # Only evaluate in the required case:
    if sgndet == -1 && tetareasign == 1
        t = real(tr(B12 * σ3))
        return t > 0 ? 1 : (t < 0 ? -1 : 0)
    end

    # All other cases → 0
    return 0
end

# ---------------------------------------------------------
# Choose Nref for SU(2) or SU(1,1), with tetareasign
#
# sgndet =  1  → SU(2)  → Nref = (0,0,0,1)
# sgndet = -1  → SU(1,1)
#    if tetareasign == -1 → Nref = (0,0,1,0)
#    else                 → Nref from sign of Tr(B σ3)
# ---------------------------------------------------------
function choose_Nref(B12::AbstractMatrix, sgndet::Int, tetareasign::Int)
    if sgndet == 1
        # spacelike tetra → SU(2) stabilizing spacelike normal
        return [0.0, 0.0, 0.0, 1.0]
    else
        # timelike tetra → SU(1,1) sector
        if tetareasign == -1
            # timelike face case: spacelike normal in x^2 direction
            return [0.0, 0.0, 1.0, 0.0]
        else
            # spacelike face in timelike tet: timelike normal ±(1,0,0,0)
            s = face_timelike_sign(B12, sgndet, tetareasign)    
            return s == 1 ? [1.0, 0.0, 0.0, 0.0] : [-1.0, 0.0, 0.0, 0.0]
        end
    end
end

# ---------------------------------------------------------
# Main SU(2)/SU(1,1) group element
# ---------------------------------------------------------
function su_from_bivectors(B12::AbstractMatrix, sgndet::Int, tetareasign::Int)

    atol = 1e-12

    # 1. algebra vector
    n3 = getnabfrombivec(B12, sgndet)

    # 2. embed into 4D
    n4 = nab3d_to_nab4d(n3, sgndet)

    # 3. pick reference normal
    Nref_vec = choose_Nref(B12, sgndet, tetareasign)

    # ---------------------------
    # SU(2) special case
    # ---------------------------
    if sgndet == 1
        if norm(n4 .- [0.0,0.0,0.0,-1.0]) < atol
            return 1im * σ2
        elseif norm(n4 .- [0.0,0.0,0.0,1.0]) < atol
            return Matrix{ComplexF64}(I,2,2)
        end
    end

    # ---------------------------
    # SU(1,1) special case
    # ---------------------------
    if sgndet == -1
        if tetareasign == 1
            if norm(n4 .- Nref_vec) < atol
                return Matrix{ComplexF64}(I,2,2)
            end
        else
            if norm(n4 .- [0.0,0.0,-1.0,0.0]) < atol 
                return 1im * σ3
            elseif norm(n4 .- [0.0,0.0,1.0,0.0]) < atol
                return Matrix{ComplexF64}(I,2,2)
            end
        end
    end

    # 4. bivector
    B4 = wedge(n4, Nref_vec)

    # 5. normalize
    Bnorm = _normalize_bivec(B4)

    # 6. dihedral angle
    θ = theta_ab(Nref_vec, n4)

    # 7. lie algebra → 2×2 matrix
    X = bivec1tohalf(Bnorm)

    # 8. exponentiate
    g = chop(exp(θ * X))

    return chop(g)
end

end # module