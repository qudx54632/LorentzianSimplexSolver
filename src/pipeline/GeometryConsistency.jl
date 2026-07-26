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
function check_sl2c_parallel_transport(sl2c, Bspin; verbose::Bool=true)

    Ntet = length(sl2c)

    T = real(eltype(sl2c[1]))      # underlying real type
    tol = T(get_tolerance())

    residuals = Vector{Vector{Matrix{Complex{T}}}}(undef, Ntet)
    maxnorm = zero(T)
    sl2c_inv = inv.(sl2c)

    for i in 1:Ntet
        row = Matrix{Complex{T}}[]
        gi = sl2c[i]
        gi_inv = sl2c_inv[i]

        for j in i+1:Ntet
            gj = sl2c[j]
            gj_inv = sl2c_inv[j]

            Rij = gi * Bspin[i][j] * gi_inv -
                  gj * Bspin[j][i] * gj_inv

            maxnorm = max(maxnorm, norm(Rij))
            push!(row, Rij)
        end

        residuals[i] = row
    end

    if verbose
        if maxnorm < tol
            println("✓ SL(2,C) parallel transport satisfied (max residual = $maxnorm).")
        else
            println("✗ SL(2,C) parallel transport violated (max residual = $maxnorm).")
        end
    end

    return residuals
end

# -------------------------------------------------------------
# 2) SO(1,3) parallel transport check on 4×4 bivectors
# -------------------------------------------------------------
function check_so13_parallel_transport(so13, B4; verbose::Bool=true)

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

    if verbose
        if maxnorm < tol
            println("✓ SO(1,3) parallel transport satisfied (max residual = $maxnorm)")
        else
            println("✗ SO(1,3) parallel transport violated (max residual = $maxnorm)")
        end
    end

    return residuals
end

# -------------------------------------------------------------
# 3) Closure check: C[i] = Σ_j κ[i,j] * A[i,j] * B[i,j]
# -------------------------------------------------------------
function check_closure_bivectors(kappa, areas, bdybivec55; verbose::Bool=true)

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

    if verbose
        if maxnorm < tol
            println("✓ Closure condition satisfied for bivectors (max residual = $maxnorm)")
        else
            println("✗ Closure condition violated for bivectors (max residual = $maxnorm)")
        end
    end

    return closure
end

function critical_point_equations_satisfied(simplex)
    sl2c_residuals = check_sl2c_parallel_transport(
        simplex.solgsl2c,
        simplex.bdybivec55;
        verbose=false,
    )
    so13_residuals = check_so13_parallel_transport(
        simplex.solgso13,
        simplex.bdybivec4d55;
        verbose=false,
    )
    closure_residuals = check_closure_bivectors(
        simplex.kappa,
        simplex.areas,
        simplex.bdybivec55;
        verbose=false,
    )

    T = real(eltype(simplex.solgsl2c[1]))
    tol = T(get_tolerance())

    sl2c_ok = all(norm(residual) < tol for row in sl2c_residuals for residual in row)
    so13_ok = all(norm(residual) < tol for row in so13_residuals for residual in row)
    closure_ok = all(norm(residual) < tol for residual in closure_residuals)

    return sl2c_ok && so13_ok && closure_ok
end

end # module
