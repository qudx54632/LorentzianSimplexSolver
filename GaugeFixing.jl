module GaugeFixingSU

using LinearAlgebra
using ..SpinAlgebra: σ3
using ..FaceXiMatching: compute_tetn0signtest, build_su_from_bivec, build_xi_from_su

export sl2c_to_su2,
       run_su2_su11_gauge_fix

# ============================================================
# 1. SL(2,C) → SU(2) map (Sl2cToSu2)
# ============================================================
function sl2c_to_su2(g::Matrix{ComplexF64})
    @assert size(g) == (2,2)

    a = g[1,1]
    c = g[1,2]
    b = g[2,1]
    d = g[2,2]

    λ = sqrt(abs2(b) + abs2(d))
    if λ == 0
        error("sl2c_to_su2: λ = 0, cannot normalize.")
    end

    u = b / λ
    v = d / λ

    # upper = [inv(λ) (a*conj(u) + c*conj(v)); 0 λ]   # not used here
    su2 = ComplexF64[
        conj(v)   -conj(u);
        u         v
    ]

    return su2
end

# ============================================================
# 2. Build gauge-fix sets for SU(2) gauge fixing
# ============================================================
function build_gauge_fix_sets(sharedTetsPos, sgndet)
    GaugeFixUpperTriangle = Tuple{Int,Int}[]
    oppositesl2c          = Tuple{Int,Int}[]

    for pair in sharedTetsPos
        s1, t1 = pair[1]
        s2, t2 = pair[2]

        if sgndet[s1][t1] == 1
            push!(GaugeFixUpperTriangle, (s1, t1))
            push!(oppositesl2c, (s2, t2))
        end
    end

    return GaugeFixUpperTriangle, oppositesl2c
end

# ============================================================
# 3. Build su2triangle (SU(2) matrices that fix the "triangle gauge")
# ============================================================
function build_su2_triangle(sl2ctest2, GaugeFixUpperTriangle, oppositesl2c)
    ns   = length(sl2ctest2)
    ntet = length(sl2ctest2[1])

    I2 = Matrix{ComplexF64}(I, 2, 2)

    su2triangle = [
        [ I2 for _ in 1:ntet ]
        for _ in 1:ns
    ]

    Gset = Set(GaugeFixUpperTriangle)
    Oset = Set(oppositesl2c)

    # map opposite → gauge-fixed tet (Mathematica Position trick)
    opp_to_gauge = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    for (gt, op) in zip(GaugeFixUpperTriangle, oppositesl2c)
        opp_to_gauge[op] = gt
    end

    for i in 1:ns
        for j in 1:ntet
            key = (i,j)
            if key in Gset
                su2triangle[i][j] = sl2c_to_su2(sl2ctest2[i][j])
            elseif key in Oset
                s0, t0 = opp_to_gauge[key]
                su2triangle[i][j] = sl2c_to_su2(sl2ctest2[s0][t0])
            else
                su2triangle[i][j] = I2
            end
        end
    end

    return su2triangle
end

# ============================================================
# 4. Normalize xi for timelike faces:
#
# For faces with tetareasign == -1:
#   ξ → ξ̃ = {{1, z}, conj(ξ[1,1])*ξ[2,:]}
#   with z = ξ[1,2]/ξ[1,1].
#
# Otherwise keep ξ as is.
# ============================================================
function normalize_timelike_xi(bdyxi, tetareasign)
    ns   = length(bdyxi)
    ntet = length(bdyxi[1])

    out = Vector{Vector{Vector{Matrix{ComplexF64}}}}(undef, ns)

    for k in 1:ns
        out[k] = Vector{Vector{Matrix{ComplexF64}}}(undef, ntet)
        for i in 1:ntet
            nf = length(bdyxi[k][i])
            out[k][i] = Vector{Matrix{ComplexF64}}(undef, nf)

            for j in 1:nf
                X = bdyxi[k][i][j]        # 2×2
                if tetareasign[k][i][j] == -1
                    a11 = X[1,1]
                    z   = X[1,2] / a11
                    r21 = conj(a11) * X[2,1]
                    r22 = conj(a11) * X[2,2]

                    Y = Matrix{ComplexF64}(undef, 2, 2)
                    Y[1,1] = 1
                    Y[1,2] = z
                    Y[2,1] = r21
                    Y[2,2] = r22

                    out[k][i][j] = Y
                else
                    out[k][i][j] = X
                end
            end
        end
    end

    return out
end

# ============================================================
# 5. Build timelike pairs and gauge triples for SU(1,1) fixing
#
# We mirror the Mathematica constructions:
#   timelikeTetsPos, Gaugespacelike, Gaugetimelike
#
# - timelike_pairs[p] = ((s1,t1),(s2,t2)) where sgndet[s1][t1] == -1
# - gaugespacelike[p] = (s1,t1,j_sp) with tetareasign[s1][t1][j_sp] == +1
# - gaugetimelike[p]  = (s1,t1,j_tm) with tetareasign[s1][t1][j_tm] == -1
#
# lookup[(k,i)] = p if (k,i) belongs to pair p (either first or second tet)
# ============================================================
function build_timelike_data(sharedTetsPos, sgndet, tetareasign)
    timelike_pairs = Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
    gaugespacelike = Tuple{Int,Int,Int}[]
    gaugetimelike  = Tuple{Int,Int,Int}[]

    for pair in sharedTetsPos
        s1, t1 = pair[1]
        s2, t2 = pair[2]

        if sgndet[s1][t1] == -1
            # spacelike face (area > 0) on the first tet, different from matched face
            j_sp = findfirst(1:5) do j
                tetareasign[s1][t1][j] == 1 && j != t1
            end

            # timelike face (area < 0) on the first tet, different from matched face
            j_tm = findfirst(1:5) do j
                tetareasign[s1][t1][j] == -1 && j != t1
            end

            if j_sp === nothing || j_tm === nothing
                continue
            end

            push!(timelike_pairs, ((s1,t1), (s2,t2)))
            push!(gaugespacelike, (s1,t1,j_sp))
            push!(gaugetimelike,  (s1,t1,j_tm))
        end
    end

    # lookup (k,i) → index p in timelike_pairs
    lookup = Dict{Tuple{Int,Int},Int}()
    for (p,(p1,p2)) in enumerate(timelike_pairs)
        lookup[p1] = p
        lookup[p2] = p
    end

    return timelike_pairs, gaugespacelike, gaugetimelike, lookup
end

# ============================================================
# 6. Build Uinverse[k,i] (SU(1,1) gauge from spacelike edge)
#
# If (k,i) is timelike, we pick a "spacelike" face on the first tet in
# the corresponding pair and set
#
#   Uinverse[k,i] = inv( bdysutest3[s,t,j_sp] )
#
# Otherwise Uinverse[k,i] = I₂.
# ============================================================
function build_Uinverse(ns, ntet, timelike_lookup, gaugespacelike, bdysutest3)
    I2 = Matrix{ComplexF64}(I, 2, 2)

    Uinverse = [
        [ I2 for _ in 1:ntet ]
        for _ in 1:ns
    ]

    for k in 1:ns, i in 1:ntet
        key = (k,i)
        if haskey(timelike_lookup, key)
            p           = timelike_lookup[key]
            s, t, j_sp  = gaugespacelike[p]
            Uinverse[k][i] = inv(bdysutest3[s][t][j_sp])
        end
    end

    return Uinverse
end

# ============================================================
# 7. Left-multiply a family of ξ’s by U[k,i]
#
# bdyxi[k][i][j] is a 2×2 matrix of spinors; we apply:
#   ξ → U[k,i] * ξ
# ============================================================
function apply_left_on_xi(U, bdyxi)
    ns   = length(bdyxi)
    ntet = length(bdyxi[1])

    out = Vector{Vector{Vector{Matrix{ComplexF64}}}}(undef, ns)

    for k in 1:ns
        out[k] = Vector{Vector{Matrix{ComplexF64}}}(undef, ntet)
        for i in 1:ntet
            nf = length(bdyxi[k][i])
            out[k][i] = Vector{Matrix{ComplexF64}}(undef, nf)

            for j in 1:nf
                X = bdyxi[k][i][j]
                out[k][i][j] = U[k][i] * X
            end
        end
    end

    return out
end

# ============================================================
# 8. Build U2[k,i]: phase gauge on timelike faces
#
# For each timelike tet (k,i), we look at a timelike face picked by
# gaugetimelike[p] = (s,t,j_tm), take z = ξ₆[s,t,j_tm][1,2], and define
#
#   φ = Arg(z)
#   U2 = diag( e^{iφ/2}, e^{-iφ/2} )
#
# Else U2 = I₂.
# ============================================================
function build_U2(ns, ntet, timelike_lookup, gaugetimelike, bdyxi6)
    I2 = Matrix{ComplexF64}(I, 2, 2)

    U2 = [
        [ I2 for _ in 1:ntet ]
        for _ in 1:ns
    ]

    for k in 1:ns, i in 1:ntet
        key = (k,i)
        if haskey(timelike_lookup, key)
            p           = timelike_lookup[key]
            s, t, j_tm  = gaugetimelike[p]
            X = bdyxi6[s][t][j_tm]
            z = X[1,2]
            φ = angle(z)

            U2[k][i] = ComplexF64[
                exp(im*φ/2)  0;
                0            exp(-im*φ/2)
            ]
        end
    end

    return U2
end

# ============================================================
# 9. Full SU(2) + SU(1,1) gauge fixing pipeline
#
# Inputs:
#   sl2ctest2          :: ns×ntet   (your sl2ctest2)
#   bdybivec55stest2   :: ns×ntet×ntet (your bdybivec55stest2)
#   sgndet             :: ns×ntet
#   tetareasign        :: ns×ntet×ntet
#   sharedTetsPos      :: list of [[s1,t1],[s2,t2]] (face identifications)
#
# Output (named tuple):
#   (su2triangle,
#    sl2c3,
#    bivec3,
#    tetn0_3,
#    su3,
#    xi3,
#    xi_final,
#    Uinverse,
#    U2,
#    sl2c4)
#
# sl2c4 is the final SL(2,C) connection after all gauge fixings.
# xi_final is the final, gauge-fixed boundary spinor data.
# ============================================================
function run_su2_su11_gauge_fix(sl2ctest2, bdybivec55stest2, sgndet, tetareasign, sharedTetsPos)
    ns   = length(sl2ctest2)
    ntet = length(sl2ctest2[1])

    # --- SU(2) gauge fixing on spacelike tets ---
    GaugeFixUpperTriangle, oppositesl2c = build_gauge_fix_sets(sharedTetsPos, sgndet)
    su2triangle = build_su2_triangle(sl2ctest2, GaugeFixUpperTriangle, oppositesl2c)

    # sl2ctest3 = sl2ctest2 ⋅ su2triangle^{-1}
    sl2c3 = [
        [ sl2ctest2[i][j] * inv(su2triangle[i][j]) for j in 1:ntet ]
        for i in 1:ns
    ]

    # bdybivec55stest3 = su2triangle⋅B⋅su2triangle^{-1}
    bivec3 = [
        [ [ su2triangle[i][j] * B * inv(su2triangle[i][j])
            for B in bdybivec55stest2[i][j] ]
          for j in 1:ntet ]
        for i in 1:ns
    ]

    # tetn0 signs, SU(2)/SU(1,1) from bivectors, then ξ
    tetn0_3 = compute_tetn0signtest(bivec3, sgndet, tetareasign)
    su3     = build_su_from_bivec(bivec3, sgndet, tetareasign, tetn0_3)
    xi3     = build_xi_from_su(su3, sgndet, tetareasign, tetn0_3)

    # --- SU(1,1) gauge fixing on timelike tets ---
    xi4 = normalize_timelike_xi(xi3, tetareasign)

    timelike_pairs, gaugespacelike, gaugetimelike, lookup =
        build_timelike_data(sharedTetsPos, sgndet, tetareasign)

    Uinverse = build_Uinverse(ns, ntet, lookup, gaugespacelike, su3)
    xi5      = apply_left_on_xi(Uinverse, xi4)
    xi6      = normalize_timelike_xi(xi5, tetareasign)
    U2       = build_U2(ns, ntet, lookup, gaugetimelike, xi6)
    xi7      = apply_left_on_xi(U2, xi6)
    xi_final = normalize_timelike_xi(xi7, tetareasign)

    # Final SL(2,C) after both Uinverse and U2
    sl2c4 = [
        [ sl2c3[i][j] * inv(Uinverse[i][j]) * inv(U2[i][j]) for j in 1:ntet ]
        for i in 1:ns
    ]

    return (
        su2triangle,  # SU(2) gauge transform for each tet
        sl2c3,        # after SU(2) gauge fixing
        bivec3,       # bivectors after SU(2) gauge fixing
        tetn0_3,      # new n0 signs
        su3,          # SU(2)/SU(1,1) from bivectors (gauge-fixed)
        xi3,          # ξ from su3 (before SU(1,1) gauge)
        xi_final,     # final ξ after SU(1,1) gauge fixing
        Uinverse,     # first SU(1,1) gauge
        U2,           # second SU(1,1) phase gauge
        sl2c4         # final SL(2,C) connection
    )
end

end # module