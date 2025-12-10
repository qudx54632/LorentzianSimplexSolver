module GeometryConsistency

using LinearAlgebra
using Combinatorics
using ..Volume: distance_sq, V3sq
using ..TetraNormals: minkowski_norm, chop

export check_sl2c_parallel_transport, check_so13_parallel_transport, check_closure_bivectors

# -------------------------------------------------------------
# 1. SL(2,C) / SU(2) parallel transport check on 2×2 bivectors
#
# Input:
#   sl2c   :: Vector{Matrix{ComplexF64}}         (length Ntet)
#   Bspin  :: Vector{Vector{Matrix{ComplexF64}}} (Ntet×Ntet, bdybivec55stest)
#
# Returns:
#   residuals[i][j] = sl2c[i] * Bspin[i][j] * inv(sl2c[i])
#                     - sl2c[j] * Bspin[j][i] * inv(sl2c[j])
#   for j > i. Each entry is a 2×2 matrix (chopped).
# -------------------------------------------------------------
function check_sl2c_parallel_transport(sl2c, Bspin; tol=1e-12)
    Ntet = length(sl2c)
    residuals = Vector{Vector{Matrix{ComplexF64}}}(undef, Ntet)

    maxnorm = 0.0

    for i in 1:Ntet
        row = Matrix{ComplexF64}[]
        gi = sl2c[i]
        gi_inv = inv(gi)

        for j in i+1:Ntet
            gj = sl2c[j]
            gj_inv = inv(gj)

            Rij = gi * Bspin[i][j] * gi_inv - gj * Bspin[j][i] * gj_inv
            Rchop = chop(Rij; tol=tol)

            # update maximum norm found
            maxnorm = max(maxnorm, norm(Rchop))

            push!(row, Rchop)
        end

        residuals[i] = row
    end

    # ----------- Print diagnostic -----------
    if maxnorm < tol
        println("✓ SL(2,C) parallel transport satisfied (max residual = $maxnorm).")
    else
        println("✗ SL(2,C) parallel transport violated (max residual = $maxnorm).")
    end

    return residuals
end

function check_so13_parallel_transport(so13, B4; tol=1e-12)
    Ntet = length(so13)
    residuals = Vector{Vector{Matrix{Float64}}}(undef, Ntet)

    η = Diagonal([-1.0, 1.0, 1.0, 1.0])  # Minkowski metric
    maxnorm = 0.0

    for i in 1:Ntet
        row = Matrix{Float64}[]
        Li = so13[i]
        LiT = transpose(Li)

        for j in i+1:Ntet
            Lj = so13[j]
            LjT = transpose(Lj)

            Rij = Li * B4[i][j] * η * LiT - Lj * B4[j][i] * η * LjT

            # chop small entries
            Rij_chop = map(x -> abs(x) < tol ? 0.0 : x, Rij)

            maxnorm = max(maxnorm, norm(Rij_chop))
            push!(row, Rij_chop)
        end

        residuals[i] = row
    end

    # Diagnostic
    if maxnorm < tol
        println("✓ SO(1,3) parallel transport satisfied (max residual = $maxnorm)")
    else
        println("✗ SO(1,3) parallel transport violated (max residual = $maxnorm)")
    end

    return residuals
end

# ============================================================
# Closure condition from κ, areas, and 2D bivectors B_ij
# C[i] = Σ_j κ[i,j] * areas[i,j] * B[i,j]
# ============================================================
function check_closure_bivectors(kappa, areas, bdybivec55; tol=1e-12)
    Ntet = length(kappa)
    closure = Vector{Matrix{ComplexF64}}(undef, Ntet)

    maxnorm = 0.0

    for i in 1:Ntet
        Ci = zeros(ComplexF64, 2, 2)

        for j in 1:Ntet
            Ci .+= kappa[i][j] * areas[i][j] * bdybivec55[i][j]
        end

        # chop numerically
        Ci_chop = map(x -> abs(x) < tol ? 0.0 : x, Ci)
        closure[i] = Ci_chop

        maxnorm = max(maxnorm, norm(Ci_chop))
    end

    # diagnostic message
    if maxnorm < tol
        println("✓ Closure condition satisfied for bivectors (max residual = $maxnorm)")
    else
        println("✗ Closure condition violated for bivectors (max residual = $maxnorm)")
    end

    return closure
end



end # module