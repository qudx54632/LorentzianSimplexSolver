module FaceXiMatching

using LinearAlgebra
using LeastSquaresOptim
using ..ThreeDTetra: threetofour
using ..SpinAlgebra: sigmabar4, σ3
using ..XiFromSU: get_xi_from_su
using ..Su2Su11FromBivector: su_from_bivectors

export run_face_xi_matching

# ============================================================
# 1. Compute Tetchange
# ============================================================
compute_tetchange(sharedTetsPos) =
    [sharedTetsPos[i][2] for i in 1:length(sharedTetsPos)]

# ============================================================
# 2. SO(3) from 4 normals
# ============================================================
function so3_from_normals(nabchange, nabref; tol=1e-10)
    Y = hcat(nabchange[1], nabchange[2], nabchange[3])
    Z = hcat(nabref[1],    nabref[2],    nabref[3])
    M = Z * inv(Y)

    I3 = Matrix{Float64}(I,3,3)

    if maximum(abs.(M' * M .- I3)) > tol
        error("SO3d: Not orthogonal")
    end
    if maximum(abs.(M * nabchange[4] .- nabref[4])) > tol
        error("SO3d: does not map 4th normal")
    end
    return M
end

# ============================================================
# 3. SO(1,2) version
# ============================================================
function so12_from_normals(nabchange, nabref; tol=1e-10)
    η = Diagonal([-1.0, 1.0, 1.0])
    A_row = zeros(4,3)
    for i in 1:4
        A_row[i,:] = nabchange[i]'
    end

    M = zeros(3,3)
    for row in 1:3
        b = [nabref[i][row] for i in 1:4]
        M[row,:] = (A_row \ b)'
    end

    cond = M' * η * M .- η
    if norm(cond) > tol
        error("Not SO(1,2)")
    end
    return M
end

# ============================================================
# 4. Build SO(3)/SO(1,2)
# ============================================================
function build_SO_matrix(nabtest, sharedTetsPos, Tetchange, sgndet)

    ns = length(nabtest)
    ntets = length(nabtest[1])

    SOmat = [ [Matrix{Float64}(I,3,3) for _ in 1:ntets]
              for _ in 1:ns ]

    base = [1,2,3,4,5]

    for i in 1:ns, j in 1:ntets
        pos = findfirst(k ->
            Tetchange[k][1] == i && Tetchange[k][2] == j, 1:length(Tetchange)
        )
        pos === nothing && continue

        iref = sharedTetsPos[pos][1][1]
        jref = sharedTetsPos[pos][1][2]

        idxC   = filter(x -> x ≠ j, base)
        idxRef = filter(x -> x ≠ jref, base)

        nabC = nabtest[i][j][idxC]
        nabR = nabtest[iref][jref][idxRef]

        SOmat[i][j] = sgndet[i][j] == 1 ?
            so3_from_normals(nabC, nabR) :
            so12_from_normals(nabC, nabR)
    end

    return SOmat
end

# ============================================================
# 5. Find reflections
# ============================================================
function find_reflecting_tets(SO)
    out = Vector{Vector{Int}}()
    for i in eachindex(SO)
        for j in 1:length(SO[i])
            if abs(det(SO[i][j]) + 1) < 1e-10
                push!(out, [i,j])
            end
        end
    end
    return out
end

# ============================================================
# 6. Build 4d normals
# ============================================================
function build_nabtest4d_from_3d(nabtest0, sgndet)
    nbdy = length(nabtest0)
    ntets = length(nabtest0[1])

    out = Vector{Vector{Vector{Vector{Float64}}}}(undef, nbdy)
    for i in 1:nbdy
        out[i] = Vector{Vector{Vector{Float64}}}(undef, ntets)
        for j in 1:ntets
            out[i][j] = [threetofour(n, sgndet[i][j]) for n in nabtest0[i][j]]
        end
    end
    return out
end

# ============================================================
# 7. Flip 4d normals
# ============================================================
function flip_4d_normals(nab4, TetsReflection)
    nbdy = length(nab4)
    ntets = length(nab4[1])
    R = Set(TetsReflection)

    out = Vector{Vector{Vector{Vector{Float64}}}}(undef, nbdy)
    for i in 1:nbdy
        out[i] = Vector{Vector{Vector{Float64}}}(undef, ntets)
        for j in 1:ntets
            out[i][j] = ([i,j] in R ?
                [-v for v in nab4[i][j]] :
                copy(nab4[i][j]))
        end
    end
    return out
end

# ============================================================
# 8. four_to_three
# ============================================================
four_to_three(n,s) = s > 0 ? n[2:4] : n[1:3]

# ============================================================
# 9. Convert back to 3d normals
# ============================================================
function build_nabtest1(nab4, sgndet)
    nbdy = length(nab4)
    ntets = length(nab4[1])
    out = Vector{Vector{Vector{Vector{Float64}}}}(undef, nbdy)

    for i in 1:nbdy
        out[i] = Vector{Vector{Vector{Float64}}}(undef, ntets)
        for j in 1:ntets
            out[i][j] = [four_to_three(n, sgndet[i][j]) for n in nab4[i][j]]
        end
    end
    return out
end

# ============================================================
# 10. Build SO(1,3)
# ============================================================
function so4_from_so3(SO, sgndet)
    A = zeros(4,4)
    if sgndet > 0
        A[1,1] = 1
        A[2:4,2:4] .= SO
    else
        A[1:3,1:3] .= SO
        A[4,4] = -1
    end
    if A[1,1] <= 0
        A .= -A
    end
    return A
end

function build_SO4_all(SO3, sgndet)
    nbdy = length(SO3)
    ntets = length(SO3[1])
    out = Vector{Vector{Matrix{Float64}}}(undef, nbdy)
    for i in 1:nbdy
        out[i] = Vector{Matrix{Float64}}(undef, ntets)
        for j in 1:ntets
            out[i][j] = so4_from_so3(SO3[i][j], sgndet[i][j])
        end
    end
    return out
end

# ============================================================
# 11. Correct solgso13
# ============================================================
function build_SO13_corrected(solgso13, TetsReflection, Tetchange, SO4)

    nbdy = length(solgso13)
    ntets = length(solgso13[1])
    out = [ [zeros(4,4) for _ in 1:ntets] for _ in 1:nbdy ]

    R = Set(TetsReflection)
    C = Set(Tetchange)

    for i in 1:nbdy
        for j in 1:ntets
            M = solgso13[i][j]
            if [i,j] in R
                M = -M
            end
            if [i,j] in C
                M = M * SO4[i][j]
            end
            out[i][j] = M
        end
    end
    return out
end

# ============================================================
# 12. SL(2,C) constraints
# ============================================================
function so_from_sl(X)
    M = zeros(4,4)
    Xdag = X'
    for μ in 1:4, ν in 1:4
        M[μ,ν] = 0.5 * real(tr(X * sigmabar4[μ] * Xdag * sigmabar4[ν]))
    end
    return M
end

function eq18(v, SO, sgndet)
    a = v[1] + im*v[2]
    b = v[3] + im*v[4]
    c = v[5] + im*v[6]
    d = v[7] + im*v[8]
    X = ComplexF64[a b; c d]

    SOx   = so_from_sl(X)
    SOinv = inv(SO)

    res = zeros(18)
    k = 1

    for i in 1:4, j in 1:4
        res[k] = sgndet * SOx[i,j] - SOinv[i,j]
        k += 1
    end

    detX = det(X)
    res[17] = real(detX) - 1
    res[18] = imag(detX)
    return res
end

# ============================================================
# 13. Solve SL(2,C)
# ============================================================
function get_sl2c_MMA_style(SO, sgndet)

    v0 = 0.01 .* randn(8)
    v0[1] = 1.0

    f(v) = eq18(v,SO,sgndet)

    result = optimize(
        f,
        v0,
        Dogleg();
        x_tol=1e-12,
        f_tol=1e-12,
        iterations=3000
    )

    v = result.minimizer

    a = v[1] + im*v[2]
    b = v[3] + im*v[4]
    c = v[5] + im*v[6]
    d = v[7] + im*v[8]
    X = ComplexF64[a b; c d]

    X ./= sqrt(det(X))

    return sgndet * X
end

# ============================================================
# 14. Build SL(2,C) new
# ============================================================
function build_SL2C_all(sl2ctest, solso13_new, sgndet, Tetchange)
    nbdy = length(sl2ctest)
    ntets = length(sl2ctest[1])
    Tset = Set(Tetchange)

    out = [ [zeros(ComplexF64,2,2) for _ in 1:ntets] for _ in 1:nbdy ]

    for i in 1:nbdy
        for j in 1:ntets
            if [i,j] in Tset
                out[i][j] = get_sl2c_MMA_style(solso13_new[i][j], sgndet[i][j])
            else
                out[i][j] = sl2ctest[i][j]
            end
        end
    end
    return out
end

# ============================================================
# 15. bivec changes
# ============================================================
build_sl2cchange(sl2ctest, sl2c_new) =
    [ [inv(sl2ctest[i][j]) * sl2c_new[i][j] for j in 1:length(sl2c_new[i])]
      for i in 1:length(sl2c_new) ]

transform_bivec(bdybivec55stest, sl2cchange) =
    [ [ [inv(sl2cchange[i][j]) * B * sl2cchange[i][j] for B in bdybivec55stest[i][j]]
         for j in 1:length(bdybivec55stest[i])]
      for i in 1:length(bdybivec55stest) ]

# ============================================================
# 16. update SL2C
# ============================================================
function update_sl2ctest(sl2c_new, sgndet)
    nbdy = length(sl2c_new)
    ntets = length(sl2c_new[1])

    return [
        [ sgndet[i][j] == 1 ?
            Matrix(inv(sl2c_new[i][j]')) :
            Matrix(im * inv(sl2c_new[i][j]') * σ3)
          for j in 1:ntets ]
        for i in 1:nbdy
    ]
end

# ============================================================
# 17. compute tet n0 signs
# ============================================================
function compute_tetn0signtest(bdybivec55stest2, sgndet, tetareasign)
    nbdy = length(bdybivec55stest2)
    ntets = length(bdybivec55stest2[1])

    return [
        [ [ sgndet[k][i] == 1 ? 1 :
            tetareasign[k][i][j] == -1 ? 1 :
            sign(real(tr(bdybivec55stest2[k][i][j] * σ3)))
          for j in 1:length(bdybivec55stest2[k][i])]
        for i in 1:ntets ]
    for k in 1:nbdy ]
end

# ============================================================
# 18. SU from bivec
# ============================================================
function build_su_from_bivec(bdybivec55stest2, sgndet, tetareasign, tetn0)
    nbdy = length(bdybivec55stest2)
    ntets = length(bdybivec55stest2[1])
    I2 = Matrix{ComplexF64}(I,2,2)

    return [
        [ [ (j == i) ?
              I2 :
              su_from_bivectors(
                  bdybivec55stest2[k][i][j],
                  sgndet[k][i],
                  tetareasign[k][i][j]
              )
          for j in 1:ntets ]
        for i in 1:ntets ]
    for k in 1:nbdy ]
end

# ============================================================
# 19. xi from su
# ============================================================
function build_xi_from_su(su, sgndet, tetareasign, tetn0)
    nbdy = length(su)
    ntets = length(su[1])

    return [
        [ [ get_xi_from_su(
                su[k][i][j],
                sgndet[k][i],
                tetareasign[k][i][j],
                tetn0[k][i][j]
            )
          for j in 1:ntets ]
        for i in 1:ntets ]
    for k in 1:nbdy ]
end

# ============================================================
# 20. FULL PIPELINE MAIN FUNCTION
# ============================================================
function run_face_xi_matching(geom)

    ns = length(geom.simplex)

    # --------------------------------------------------------
    # Extract base data
    # --------------------------------------------------------
    nabtest         = [geom.simplex[i].nabout      for i in 1:ns]
    sharedTetsPos   = geom.connectivity[1]["sharedTetsPos"]
    sgndet          = [geom.simplex[i].sgndet      for i in 1:ns]
    solgso13        = [geom.simplex[i].solgso13    for i in 1:ns]
    sl2ctest        = [geom.simplex[i].solgsl2c    for i in 1:ns]
    tetareasign     = [geom.simplex[i].tetareasign for i in 1:ns]
    bdybivec55stest = [geom.simplex[i].bdybivec55  for i in 1:ns]

    # --------------------------------------------------------
    # Step 1–9: SO(3), SO(1,2), normals, SO(1,3)
    # --------------------------------------------------------
    Tetchange      = compute_tetchange(sharedTetsPos)
    SO3            = build_SO_matrix(nabtest, sharedTetsPos, Tetchange, sgndet)
    TetsReflection = find_reflecting_tets(SO3)
    nab4           = build_nabtest4d_from_3d(nabtest, sgndet)
    nab4_flip      = flip_4d_normals(nab4, TetsReflection)
    nab1           = build_nabtest1(nab4_flip, sgndet)
    SO4            = build_SO4_all(SO3, sgndet)
    solso13_new    = build_SO13_corrected(solgso13, TetsReflection, Tetchange, SO4)

    # --------------------------------------------------------
    # Step 10: First SL(2,C) solve
    # --------------------------------------------------------
    sl2c_new = build_SL2C_all(sl2ctest, solso13_new, sgndet, Tetchange)

    # --------------------------------------------------------
    # Step 11: Compute sl2cchange = g_old^{-1} g_new
    # --------------------------------------------------------
    sl2cchange = build_sl2cchange(sl2ctest, sl2c_new)

    # --------------------------------------------------------
    # Step 12: Transform bivectors
    # --------------------------------------------------------
    bdybivec55stest_new = transform_bivec(bdybivec55stest, sl2cchange)

    # --------------------------------------------------------
    # Step 13: Compute new tet-n0 signs
    # --------------------------------------------------------
    tetn0_new = compute_tetn0signtest(bdybivec55stest_new, sgndet, tetareasign)

    # --------------------------------------------------------
    # Step 14: Build new SU(2)/SU(1,1)
    # --------------------------------------------------------
    bdysu_new = build_su_from_bivec(bdybivec55stest_new, sgndet, tetareasign, tetn0_new)

    # --------------------------------------------------------
    # Step 15: Build xi
    # --------------------------------------------------------
    xi_new = build_xi_from_su(bdysu_new, sgndet, tetareasign, tetn0_new)

    # --------------------------------------------------------
    # Step 16: Update SL(2,C) again
    # --------------------------------------------------------
    sl2c_new_new = update_sl2ctest(sl2c_new, sgndet)

    # --------------------------------------------------------
    # UPDATE GEOMETRY IN-PLACE
    # --------------------------------------------------------
    for i in 1:ns
        geom.simplex[i].solgsl2c   = sl2c_new_new[i]
        geom.simplex[i].bdybivec55 = bdybivec55stest_new[i]
        geom.simplex[i].bdyxi         = xi_new[i]
        geom.simplex[i].bdysu         = bdysu_new[i]
        geom.simplex[i].solgso13   = solso13_new[i]
    end
    println("  SL(2,C) matrices updated:      ✓")
    println("  Boundary bivectors updated:    ✓")
    println("  boundary ξ variables updated:  ✓")
    println("  SU(2)/SU(1,1) elements updated: ✓")
    println("  SO(1,3) frames corrected:       ✓\n")
    # --------------------------------------------------------
    # FINAL return 
    # --------------------------------------------------------
    # return (Tetchange, SO3, TetsReflection, nab4, nab4_flip, nab1, SO4, solso13_new, sl2c_new, sl2cchange, bdybivec55stest_new, tetn0_new, bdysu_new, xi_new, sl2c_new_new)
    return nothing
end

end # module