module GaugeFixingSU

using LinearAlgebra
using ..SpinAlgebra: σ3
using ..FaceXiMatching: compute_tetn0signtest, build_su_from_bivec, build_xi_from_su

export run_su2_su11_gauge_fix

# ------------------------------------------------------------
# SL(2,C) → SU(2)
# ------------------------------------------------------------
function sl2c_to_su2(g::Matrix{ComplexF64})
    @assert size(g) == (2,2)

    b, d = g[2,1], g[2,2]
    λ = sqrt(abs2(b) + abs2(d))
    λ == 0 && error("sl2c_to_su2: λ = 0")

    u = b / λ
    v = d / λ

    return ComplexF64[
        conj(v)   -conj(u);
        u          v
    ]
end

# # ------------------------------------------------------------
# # SU(2) gauge-fix selection
# # ------------------------------------------------------------
# function build_gauge_fix_sets(sharedTetsPos, sgndet)
#     GaugeFixUpperTriangle = Tuple{Int,Int}[]
#     oppositesl2c          = Tuple{Int,Int}[]

#     for pair in sharedTetsPos
#         s1, t1 = pair[1]
#         s2, t2 = pair[2]

#         if sgndet[s1][t1] == 1
#             push!(GaugeFixUpperTriangle, (s1, t1))
#             push!(oppositesl2c, (s2, t2))
#         end
#     end
#     return GaugeFixUpperTriangle, oppositesl2c
# end

# ------------------------------------------------------------
# Build SU(2) triangle gauge matrices
# ------------------------------------------------------------
function build_su2_triangle(sl2ctest2, GaugeFixUpperTriangle, oppositesl2c)
    ns   = length(sl2ctest2)
    ntet = length(sl2ctest2[1])
    I2 = Matrix{ComplexF64}(I, 2, 2)

    su2triangle = [ [I2 for _ in 1:ntet] for _ in 1:ns ]
    Gset = Set(GaugeFixUpperTriangle)
    Oset = Set(oppositesl2c)

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
                s0,t0 = opp_to_gauge[key]
                su2triangle[i][j] = sl2c_to_su2(sl2ctest2[s0][t0])
            end
        end
    end

    return su2triangle
end

# ------------------------------------------------------------
# Normalize ξ for timelike faces
# ------------------------------------------------------------
function normalize_timelike_xi(bdyxi, tetareasign)
    ns   = length(bdyxi)
    ntet = length(bdyxi[1])

    out = Vector{Vector{Vector{Vector{Vector{ComplexF64}}}}}(undef, ns)

    for k in 1:ns
        out[k] = Vector{Vector{Vector{Vector{ComplexF64}}}}(undef, ntet)
        for i in 1:ntet
            nf = length(bdyxi[k][i])
            out[k][i] = Vector{Vector{Vector{ComplexF64}}}(undef, nf)

            for j in 1:nf
                ξ1 = bdyxi[k][i][j][1]
                ξ2 = bdyxi[k][i][j][2]

                if tetareasign[k][i][j] == -1
                    a11 = ξ1[1]
                    newξ1 = ξ1 / a11
                    newξ2 = conj(a11) .* ξ2
                    out[k][i][j] = [newξ1, newξ2]
                else
                    out[k][i][j] = [ξ1, ξ2]
                end
            end
        end
    end
    return out
end

# # ------------------------------------------------------------
# # Build SU(1,1) gauge-fix triple sets
# # ------------------------------------------------------------
# function build_timelike_data(sharedTetsPos, sgndet, tetareasign)
#     timelike_pairs = Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
#     gaugespacelike = Tuple{Int,Int,Int}[]
#     gaugetimelike  = Tuple{Int,Int,Int}[]

#     for pair in sharedTetsPos
#         s1,t1 = pair[1]
#         s2,t2 = pair[2]

#         if sgndet[s1][t1] == -1
#             j_sp = findfirst(j -> tetareasign[s1][t1][j] == 1  && j != t1, 1:5)
#             j_tm = findfirst(j -> tetareasign[s1][t1][j] == -1 && j != t1, 1:5)
#             (j_sp === nothing || j_tm === nothing) && continue

#             push!(timelike_pairs, ((s1,t1),(s2,t2)))
#             push!(gaugespacelike, (s1,t1,j_sp))
#             push!(gaugetimelike,  (s1,t1,j_tm))
#         end
#     end

#     lookup = Dict{Tuple{Int,Int},Int}()
#     for (p,(p1,p2)) in enumerate(timelike_pairs)
#         lookup[p1] = p
#         lookup[p2] = p
#     end

#     return timelike_pairs, gaugespacelike, gaugetimelike, lookup
# end

# ------------------------------------------------------------
# Build U^{-1} for SU(1,1)
# ------------------------------------------------------------
function build_Uinverse(ns, ntet, lookup, gaugespacelike, su)
    I2 = Matrix{ComplexF64}(I,2,2)
    Uinverse = [ [I2 for _ in 1:ntet] for _ in 1:ns ]

    for k in 1:ns, i in 1:ntet
        key = (k,i)
        if haskey(lookup,key)
            p = lookup[key]
            s,t,j_sp = gaugespacelike[p]
            Uinverse[k][i] = inv(su[s][t][j_sp])
        end
    end
    return Uinverse
end

# ------------------------------------------------------------
# Left-multiply ξ by gauge U
# ------------------------------------------------------------
function apply_left_on_xi(U, bdyxi)
    ns   = length(bdyxi)
    ntet = length(bdyxi[1])

    out = Vector{Vector{Vector{Vector{Vector{ComplexF64}}}}}(undef, ns)

    for k in 1:ns
        out[k] = Vector{Vector{Vector{Vector{ComplexF64}}}}(undef, ntet)
        for i in 1:ntet
            nf = length(bdyxi[k][i])
            out[k][i] = Vector{Vector{Vector{ComplexF64}}}(undef, nf)
            for j in 1:nf
                Umat = U[k][i]
                ξ1 = Umat * bdyxi[k][i][j][1]
                ξ2 = Umat * bdyxi[k][i][j][2]
                out[k][i][j] = [ξ1, ξ2]
            end
        end
    end
    return out
end

# ------------------------------------------------------------
# Build U2 (phase fixing)
# ------------------------------------------------------------
function build_U2(ns, ntet, lookup, gaugetimelike, xi)
    I2 = Matrix{ComplexF64}(I,2,2)
    U2 = [ [I2 for _ in 1:ntet] for _ in 1:ns ]

    for k in 1:ns, i in 1:ntet
        key = (k,i)
        if haskey(lookup,key)
            p = lookup[key]
            s,t,j_tm = gaugetimelike[p]
            X = xi[s][t][j_tm]
            z = X[1][2]
            φ = angle(z)
            U2[k][i] = ComplexF64[
                exp(im*φ/2)  0;
                0            exp(-im*φ/2)
            ]
        end
    end
    return U2
end

# ------------------------------------------------------------
# Full SU(2) + SU(1,1) gauge fixing pipeline
# ------------------------------------------------------------
function run_su2_su11_gauge_fix(geom)

    ns = length(geom.simplex)
    sl2c  = [geom.simplex[i].solgsl2c   for i in 1:ns]
    bivec = [geom.simplex[i].bdybivec55 for i in 1:ns]
    sgnd  = [geom.simplex[i].sgndet     for i in 1:ns]
    area  = [geom.simplex[i].tetareasign for i in 1:ns]

    ntet = length(sl2c[1])

    # SU(2) gauge fix
    Gfix, Ofix = geom.connectivity[1]["GaugeFixUpperTriangle"], geom.connectivity[1]["oppositesl2c"]
    su2triangle = build_su2_triangle(sl2c, Gfix, Ofix)

    sl2c3 = [
        [ sl2c[i][j] * inv(su2triangle[i][j]) for j in 1:ntet ]
        for i in 1:ns
    ]

    bivec3 = [
        [ [ su2triangle[i][j] * B * inv(su2triangle[i][j]) for B in bivec[i][j] ]
          for j in 1:ntet ]
        for i in 1:ns
    ]

    n0   = compute_tetn0signtest(bivec3, sgnd, area)
    su3  = build_su_from_bivec(bivec3, sgnd, area, n0)
    xi3  = build_xi_from_su(su3, sgnd, area, n0)

    # SU(1,1) gauge fix
    xi4 = normalize_timelike_xi(xi3, area)
    gsp, gtm, lookup = geom.connectivity[1]["gaugespacelike"], geom.connectivity[1]["gaugetimelike"], geom.connectivity[1]["lookup"]

    Uinv = build_Uinverse(ns, ntet, lookup, gsp, su3)
    xi5  = apply_left_on_xi(Uinv, xi4)
    xi6  = normalize_timelike_xi(xi5, area)

    U2   = build_U2(ns, ntet, lookup, gtm, xi6)
    xi7  = apply_left_on_xi(U2, xi6)
    xi_final = normalize_timelike_xi(xi7, area)

    sl2c4 = [
        [ sl2c3[i][j] * inv(Uinv[i][j]) * inv(U2[i][j]) for j in 1:ntet ]
        for i in 1:ns
    ]

    # Store final results back into geom
    for s in 1:ns
        geom.simplex[s].bdyxi     = xi_final[s]
        geom.simplex[s].solgsl2c  = sl2c4[s]
        geom.simplex[s].tetn0sign = n0[s]
    end

    return nothing
end

end # module