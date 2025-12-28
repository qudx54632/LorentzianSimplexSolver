module GeometryConsistency

using LinearAlgebra
using ..PrecisionUtils: get_tolerance
using ..SpinAlgebra: eta

export check_sl2c_parallel_transport,
       check_so13_parallel_transport,
       check_closure_bivectors

# -------------------------------------------------------------
# 1) SL(2,C) parallel transport check on 2×2 bivectors
# -------------------------------------------------------------
function check_sl2c_parallel_transport(sl2c, Bspin)

    Ntet = length(sl2c)

    T = real(eltype(sl2c[1]))      # underlying real type
    tol = T(get_tolerance())

    residuals = Vector{Vector{Matrix{Complex{T}}}}(undef, Ntet)
    maxnorm = zero(T)

    for i in 1:Ntet
        row = Matrix{Complex{T}}[]
        gi = sl2c[i]
        gi_inv = inv(gi)

        for j in i+1:Ntet
            gj = sl2c[j]
            gj_inv = inv(gj)

            Rij = gi * Bspin[i][j] * gi_inv -
                  gj * Bspin[j][i] * gj_inv

            maxnorm = max(maxnorm, norm(Rij))
            push!(row, Rij)
        end

        residuals[i] = row
    end

    if maxnorm < tol
        println("✓ SL(2,C) parallel transport satisfied (max residual = $maxnorm).")
    else
        println("✗ SL(2,C) parallel transport violated (max residual = $maxnorm).")
    end

    return residuals
end

# -------------------------------------------------------------
# 2) SO(1,3) parallel transport check on 4×4 bivectors
# -------------------------------------------------------------
function check_so13_parallel_transport(so13, B4)

    Ntet = length(so13)

    T = eltype(so13[1])                  # real scalar type
    tol = T(get_tolerance())

    residuals = Vector{Vector{Matrix{T}}}(undef, Ntet)
    maxnorm = zero(T)

    η = eta(T)                           # Minkowski metric with type T

    for i in 1:Ntet
        row = Matrix{T}[]
        Li = so13[i]
        LiT = transpose(Li)

        for j in i+1:Ntet
            Lj = so13[j]
            LjT = transpose(Lj)

            Rij = Li * B4[i][j] * η * LiT -
                  Lj * B4[j][i] * η * LjT

            maxnorm = max(maxnorm, norm(Rij))
            push!(row, Rij)
        end

        residuals[i] = row
    end

    if maxnorm < tol
        println("✓ SO(1,3) parallel transport satisfied (max residual = $maxnorm)")
    else
        println("✗ SO(1,3) parallel transport violated (max residual = $maxnorm)")
    end

    return residuals
end

# -------------------------------------------------------------
# 3) Closure check: C[i] = Σ_j κ[i,j] * A[i,j] * B[i,j]
# -------------------------------------------------------------
function check_closure_bivectors(kappa, areas, bdybivec55)

    Ntet = length(kappa)

    T = real(eltype(bdybivec55[1][1]))
    tol = T(get_tolerance())

    closure = Vector{Matrix{Complex{T}}}(undef, Ntet)
    maxnorm = zero(T)

    for i in 1:Ntet
        Ci = zeros(Complex{T}, 2, 2)
        for j in 1:Ntet
            Ci .+= (kappa[i][j] * areas[i][j]) * bdybivec55[i][j]
        end

        closure[i] = Ci
        maxnorm = max(maxnorm, norm(Ci))
    end

    if maxnorm < tol
        println("✓ Closure condition satisfied for bivectors (max residual = $maxnorm)")
    else
        println("✗ Closure condition violated for bivectors (max residual = $maxnorm)")
    end

    return closure
end

end # module