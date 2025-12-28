module FaceMatchingChecks

using LinearAlgebra
using ..PrecisionUtils: get_tolerance

export check_face_matching_bivec,
       check_parallel_transport,
       check_closure,
       check_all

# ============================================================
# 1. Face-matching check
# ============================================================
"""
    check_face_matching_bivec(bdybivec55, sharedTetsPos)

Checks whether bivectors on two matching tetrahedra agree after deleting
the corresponding internal face.

Returns `(ok, maxnorm)` and prints diagnostics.
"""
function check_face_matching_bivec(bdybivec55, sharedTetsPos)

    T = real(eltype(bdybivec55[1][1][1]))
    tol = T(get_tolerance())
    maxnorm = zero(T)

    for pair in sharedTetsPos
        s1, t1 = pair[1]
        s2, t2 = pair[2]

        nf = length(bdybivec55[s1][t1])

        list1 = [bdybivec55[s1][t1][i] for i in 1:nf if i != t1]
        list2 = [bdybivec55[s2][t2][i] for i in 1:nf if i != t2]

        for (B1, B2) in zip(list1, list2)
            maxnorm = max(maxnorm, norm(B1 - B2))
        end
    end

    ok = maxnorm <= tol

    if ok
        println("✓ Face matching satisfied (max residual = $maxnorm).")
    else
        println("✗ Face matching FAILED (max residual = $maxnorm).")
        println("  → Hint: incorrect SO(3) mapping or incorrect Tetchange.")
    end

    return ok, maxnorm
end


# ============================================================
# 2. Parallel transport (SL(2,C))
# ============================================================
"""
    check_parallel_transport(sl2c, bivec)

Checks:
    gᵢ Bᵢⱼ gᵢ⁻¹ ≈ gⱼ Bⱼᵢ gⱼ⁻¹
"""
function check_parallel_transport(sl2c, bivec)

    T = real(eltype(sl2c[1][1]))
    tol = T(get_tolerance())
    maxnorm = zero(T)

    nbdy = length(sl2c)
    ntet = length(sl2c[1])

    for k in 1:nbdy
        for i in 1:ntet
            gi = sl2c[k][i]
            giinv = inv(gi)

            for j in 1:ntet
                gj = sl2c[k][j]
                gjinv = inv(gj)

                Bij = bivec[k][i][j]
                Bji = bivec[k][j][i]

                diff = gi * Bij * giinv - gj * Bji * gjinv
                maxnorm = max(maxnorm, norm(diff))
            end
        end
    end

    ok = maxnorm <= tol

    if ok
        println("✓ SL(2,C) parallel transport satisfied (max residual = $maxnorm).")
    else
        println("✗ SL(2,C) parallel transport FAILED (max residual = $maxnorm).")
        println("  → Hint: check sl2cchange construction or tet orientation.")
    end

    return ok, maxnorm
end


# ============================================================
# 3. Closure condition
# ============================================================
"""
    check_closure(bivec, kappa, area)

Checks closure:
    Σⱼ κᵢⱼ Aᵢⱼ Bᵢⱼ ≈ 0
"""
function check_closure(bivec, kappa, area)

    T = real(eltype(bivec[1][1][1]))
    tol = T(get_tolerance())
    maxnorm = zero(T)

    nbdy = length(bivec)
    ntet = length(bivec[1])

    for k in 1:nbdy
        for i in 1:ntet
            S = zero(bivec[k][i][1])
            nf = length(bivec[k][i])

            for j in 1:nf
                S += kappa[k][i][j] * area[k][i][j] * bivec[k][i][j]
            end

            maxnorm = max(maxnorm, norm(S))
        end
    end

    ok = maxnorm <= tol

    if ok
        println("✓ Closure condition satisfied (max residual = $maxnorm).")
    else
        println("✗ Closure condition FAILED (max residual = $maxnorm).")
        println("  → Hint: check tetareasign or n0 sign propagation.")
    end

    return ok, maxnorm
end


# ============================================================
# 4. Full diagnostic wrapper
# ============================================================
"""
    check_all(geom)

Runs:
  1. Face matching
  2. SL(2,C) parallel transport
  3. Closure
"""
function check_all(geom)

    println("\n==============================")
    println("   FACE–XI GEOMETRY CHECKS")
    println("==============================\n")

    ns = length(geom.simplex)

    bivec = [geom.simplex[i].bdybivec55 for i in 1:ns]
    sharedTetsPos = geom.connectivity[1]["sharedTetsPos"]
    sl2c  = [geom.simplex[i].solgsl2c for i in 1:ns]
    kappa = [geom.simplex[i].kappa for i in 1:ns]
    area  = [geom.simplex[i].areas for i in 1:ns]

    ok_face,  _ = check_face_matching_bivec(bivec, sharedTetsPos)
    ok_para,  _ = check_parallel_transport(sl2c, bivec)
    ok_close, _ = check_closure(bivec, kappa, area)

    println("\n------------------------------")
    if ok_face && ok_para && ok_close
        println("✓ All geometric consistency checks PASSED.")
    else
        println("✗ Some checks FAILED.")
        println("  → Review output above for hints.")
    end
    println("------------------------------\n")

    return nothing
end

end # module