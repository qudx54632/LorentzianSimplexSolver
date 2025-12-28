module GaugeFixingSU

using LinearAlgebra

using ..PrecisionUtils: get_tolerance
using ..SpinAlgebra: σ3, imag_unit
using ..FaceXiMatching: compute_tetn0signtest, build_su_from_bivec, build_xi_from_su

export run_su2_su11_gauge_fix

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
# Get underlying real scalar type from a Complex matrix element type
_real_scalar_type(::Type{Complex{T}}) where {T<:Real} = T
_real_scalar_type(::Type{T}) where {T<:Real} = T

function _scalarT_from_geom(geom)
    # solgsl2c :: Vector{Matrix{Complex{T}}}
    elT = eltype(geom.simplex[1].solgsl2c[1])
    return _real_scalar_type(elT)
end

# ------------------------------------------------------------
# SL(2,C) → SU(2)
# ------------------------------------------------------------
function sl2c_to_su2(g::AbstractMatrix{Complex{T}}) where {T<:Real}
    @assert size(g) == (2,2)

    b = g[2,1]
    d = g[2,2]

    λ = sqrt(abs2(b) + abs2(d))
    iszero(λ) && error("sl2c_to_su2: λ = 0")

    u = b / λ
    v = d / λ

    return Matrix{Complex{T}}([
        conj(v)   -conj(u);
        u          v
    ])
end

# ------------------------------------------------------------
# Build SU(2) triangle gauge matrices
# ------------------------------------------------------------
function build_su2_triangle(sl2ctest2, GaugeFixUpperTriangle, oppositesl2c)
    ns   = length(sl2ctest2)
    ntet = length(sl2ctest2[1])

    T = _real_scalar_type(eltype(sl2ctest2[1][1]))
    I2 = Matrix{Complex{T}}(I, 2, 2)

    su2triangle = [[I2 for _ in 1:ntet] for _ in 1:ns]

    # convert [s,t] -> (s,t)
    gauge_keys = Set{Tuple{Int,Int}}((p[1], p[2]) for p in GaugeFixUpperTriangle)
    opp_keys   = Set{Tuple{Int,Int}}((p[1], p[2]) for p in oppositesl2c)

    # map opposite -> gauge
    opp_to_gauge = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    for (gt, op) in zip(GaugeFixUpperTriangle, oppositesl2c)
        opp_to_gauge[(op[1], op[2])] = (gt[1], gt[2])
    end

    for s in 1:ns
        for t in 1:ntet
            key = (s, t)

            if key in gauge_keys
                # println("gauge key detected: ", key)
                su2triangle[s][t] = sl2c_to_su2(sl2ctest2[s][t])

            elseif key in opp_keys
                # println("opposite key detected: ", key)
                (s0, t0) = opp_to_gauge[key]
                su2triangle[s][t] = sl2c_to_su2(sl2ctest2[s0][t0])
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

    # infer scalar type
    T = _real_scalar_type(eltype(bdyxi[1][1][1][1]))
    CT = Complex{T}

    out = Vector{Any}(undef, ns)  # keep structure flexible but values typed

    for k in 1:ns
        out_k = Vector{Any}(undef, ntet)
        for i in 1:ntet
            nf = length(bdyxi[k][i])
            out_i = Vector{Any}(undef, nf)

            for j in 1:nf
                ξ1 = bdyxi[k][i][j][1]
                ξ2 = bdyxi[k][i][j][2]

                if tetareasign[k][i][j] == -1
                    a11 = ξ1[1]
                    # avoid accidental Float64 literal
                    newξ1 = ξ1 / a11
                    newξ2 = conj(a11) .* ξ2
                    out_i[j] = Vector{Vector{CT}}([newξ1, newξ2])
                else
                    out_i[j] = Vector{Vector{CT}}([ξ1, ξ2])
                end
            end

            out_k[i] = out_i
        end
        out[k] = out_k
    end

    return out
end

# ------------------------------------------------------------
# Build U^{-1} using spacelike gauge face indices
# ------------------------------------------------------------
function build_Uinverse(ns::Int, ntet::Int, lookup, gaugespacelike, su)
    T = _real_scalar_type(eltype(su[1][1][1]))
    I2 = Matrix{Complex{T}}(I, 2, 2)

    Uinverse = [[I2 for _ in 1:ntet] for _ in 1:ns]

    for k in 1:ns, i in 1:ntet
        key = (k, i)
        if haskey(lookup, key)
            p = lookup[key]
            s, t, j_sp = gaugespacelike[p][1]
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

    T = _real_scalar_type(eltype(U[1][1]))
    CT = Complex{T}

    out = Vector{Any}(undef, ns)

    for k in 1:ns
        out_k = Vector{Any}(undef, ntet)
        for i in 1:ntet
            nf = length(bdyxi[k][i])
            out_i = Vector{Any}(undef, nf)
            for j in 1:nf
                Umat = U[k][i]
                ξ1 = Umat * bdyxi[k][i][j][1]
                ξ2 = Umat * bdyxi[k][i][j][2]
                out_i[j] = Vector{Vector{CT}}([ξ1, ξ2])
            end
            out_k[i] = out_i
        end
        out[k] = out_k
    end

    return out
end

# ------------------------------------------------------------
# Build U2 (phase fixing) for timelike gauge
# Supports lookup keys as String "k_i" (your current connectivity format).
# ------------------------------------------------------------
function build_U2(ns::Int, ntet::Int, lookup, gaugetimelike, xi)
    # infer scalar type from xi
    T = _real_scalar_type(eltype(xi[1][1][1][1]))
    iT = imag_unit(T)

    I2 = Matrix{Complex{T}}(I, 2, 2)
    U2 = [[I2 for _ in 1:ntet] for _ in 1:ns]

    tol = T(get_tolerance())  # not strictly needed here, but consistent

    for k in 1:ns, i in 1:ntet
        key = string(k, "_", i)
        haskey(lookup, key) || continue

        pos = lookup[key]
        v, t, f = gaugetimelike[pos][1]

        # use xi[k][i][f][1][2] as in your MMA mapping
        z = xi[k][i][f][1][2]
        ϕ = angle(z)

        # phase matrix
        U2[k][i] = Matrix{Complex{T}}([
            exp(iT * (ϕ / T(2)))      zero(Complex{T});
            zero(Complex{T})          exp(-iT * (ϕ / T(2)))
        ])
    end

    return U2
end

# ------------------------------------------------------------
# Full SU(2) + SU(1,1) gauge fixing pipeline
# ------------------------------------------------------------
function run_su2_su11_gauge_fix(geom)
    isempty(geom.connectivity) && error("run_su2_su11_gauge_fix: geom.connectivity is empty")

    ns = length(geom.simplex)

    sl2c  = [geom.simplex[i].solgsl2c     for i in 1:ns]
    bivec = [geom.simplex[i].bdybivec55   for i in 1:ns]
    sgnd  = [geom.simplex[i].sgndet       for i in 1:ns]
    area  = [geom.simplex[i].tetareasign  for i in 1:ns]

    ntet = length(sl2c[1])

    conn = geom.connectivity[1]

    # SU(2) gauge fix
    Gfix = conn["GaugeFixUpperTriangle"]
    Ofix = conn["oppositesl2c"]

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

    gsp    = conn["gaugespacelike"]
    gtm    = conn["gaugetimelike"]
    lookup = conn["lookup"]

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