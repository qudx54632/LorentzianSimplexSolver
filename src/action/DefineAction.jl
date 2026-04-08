module DefineAction

using LinearAlgebra
using SymEngine
using ..SpinAlgebra: σ3

export γsym, compute_action

const _I = Ref{Union{Basic,Nothing}}(nothing)
const _γ = Ref{Union{Basic,Nothing}}(nothing)

function __init__()
    _I[] = SymEngine.Basic(im)
    _γ[] = symbols("gamma")
end

@inline γsym() = _γ[]

@inline function sconj_real(expr)
    return SymEngine.subs(expr, Dict(_I[] => -_I[]))
end

@inline col(v) = reshape(v, :, 1)
@inline herm(v) = transpose(map(sconj_real, col(v)))  

@inline slog(x) = SymEngine.log(x)
@inline ssqrt(x) = SymEngine.sqrt(x)

@inline symzero(x) = x - x
@inline symone(x) = symzero(x) + 1

# ============================================================
# small helpers
# ============================================================
@inline function meta_det(meta)::Int
    return Int(round(real(det(meta))))
end

@inline function scalar_A(xi, z, g, meta; signzz::Int=1)
    xiC = col(xi)
    zC  = col(z)
    gt  = transpose(g)
    Z   = gt * zC
    return (signzz * (herm(xiC) * meta * Z))[1,1]
end

@inline function scalar_B(xi, z, g, meta; signzz::Int=1)
    xiC = col(xi)
    zC  = col(z)
    gt  = transpose(g)
    Z   = gt * zC
    return (signzz * (herm(Z) * meta * xiC))[1,1]
end

@inline function scalar_C(z, g, meta; signzz::Int=1)
    zC = col(z)
    gt = transpose(g)
    Z  = gt * zC
    return (signzz * (herm(Z) * meta * Z))[1,1]
end

# ============================================================
# Spacelike Xi-Z contribution
# xiZ = \[\left(m_{ef}\,\langle \xi_{ef}, Z_{vef} \rangle \right)^{\frac{\kappa_{ef} + \det \eta_e}{2}}\]
# Zxi = \[\left(m_{ef}\,\langle Z_{vef} , \xi_{ef} \rangle \right)^{\frac{-\kappa_{ef} + \det \eta_e}{2}}\]
# ============================================================
function halfedgeactionXiZ(xi, z, g, κ::Int, meta; signzz::Int=1)
    detm = meta_det(meta)

    A = scalar_A(xi, z, g, meta; signzz=signzz)
    B = scalar_B(xi, z, g, meta; signzz=signzz)

    e1 = (κ + detm) ÷ 2
    e2 = (-κ + detm) ÷ 2

    return A^e1 * B^e2
end

# ============================================================
# Spacelike Z-Z contribution
# ZZ = (i\gamma\kappa - \det \eta_e)\[\Symbolics.Symbolics.log(\langle Z_{vef}, Z_{vef} \rangle)\]
# ============================================================
function halfedgeactionZZ(xi, z, g, κ::Int, meta; signzz::Int=1, γ=γsym())
    detm = meta_det(meta)
    C = scalar_C(z, g, meta; signzz=signzz)
    return (_I[] * γ * κ - detm) * slog(C)
end

# ============================================================
# Timelike Xi–Z contribution
# XZ_t = ( ssqrt( <xi , Z> / <Z , xi> ) )^κ
# where Z = g^T z
# ============================================================
function halfedgeactiontXiZ(xi, z, g, κ::Int, meta; signzz::Int=1)
    A = scalar_A(xi, z, g, meta; signzz=signzz)
    B = scalar_B(xi, z, g, meta; signzz=signzz)
    return (ssqrt(A / B))^κ
end

# ============================================================
# Timelike Z–Z contribution
# ZZ_t = -(i/γ) κ Symbolics.log( <xi , Z> <Z , xi> )
# where Z = g^T z
# ============================================================
function halfedgeactiontZZ(xi, z, g, κ::Int, meta; signzz::Int=1, γ=γsym())
    A = scalar_A(xi, z, g, meta; signzz=1)
    B = scalar_B(xi, z, g, meta; signzz=1)
    return -(_I[] / γ) * κ * slog(A * B)
end

# ============================================================
# Bulk edge action ZZ (type 1)
#
# Only defined for orientation (κ2, κ3) = (-1, +1)
# Returns <Z_2 , Z_3>
# ============================================================
function edgebulkactionssZZ1(zs, gs, κs)
    z1, z3 = zs
    g2, g3 = gs
    κ2, κ3 = κs

    if κ2 == -1 && κ3 == 1
        Z2 = transpose(g2) * col(z1)
        Z3 = transpose(g3) * col(z3)
        return (herm(vec(Z2)) * Z3)[1,1]
    else
        error("Wrong orientation")
    end
end

# ============================================================
# Bulk edge action ZZ (type 2)
#
# Only defined for orientation (κ2, κ3) = (-1, +1)
#
# 2 (iγ - 1) log ssqrt(<Z3,Z3>) - 2 (iγ + 1) log ssqrt(<Z2,Z2>)
# ============================================================
function edgebulkactionssZZ2(zs, gs, κs; γ=γsym())
    z1, z3 = zs
    g2, g3 = gs
    κ2, κ3 = κs

    if κ2 == -1 && κ3 == 1
        Z3 = transpose(g3) * col(z3)
        Z2 = transpose(g2) * col(z1)

        term3 = 2 * (_I[] * γ - 1) * slog(ssqrt((herm(vec(Z3)) * Z3)[1,1]))
        term2 = 2 * (_I[] * γ + 1) * slog(ssqrt((herm(vec(Z2)) * Z2)[1,1]))

        return term3 - term2
    else
        error("Wrong orientation")
    end
end

# ============================================================
# Face action: Xi–Z sector
#
# facesign > 0 : spacelike face (product structure)
# facesign ≤ 0 : timelike face (product of timelike half-edges)
# ============================================================
function faceactionXiZ(xilist, zlist, glist, κlist, metalist, sgndetlist; facesign::Int=1)
    n = length(glist)

    if facesign > 0
        res = halfedgeactionXiZ(xilist[1], zlist[1], glist[1], κlist[1], metalist[1,1];
                                signzz=metalist[1,2])

        for i in 2:2:(n-1)
            if sgndetlist[i] == 1
                res *= edgebulkactionssZZ1((zlist[i-1], zlist[i+1]),
                                           (glist[i], glist[i+1]),
                                           (κlist[i], κlist[i+1]))
            else
                res *= halfedgeactionXiZ(xilist[i], zlist[i-1], glist[i], κlist[i], metalist[i,1];
                                         signzz=metalist[i,2]) *
                       halfedgeactionXiZ(xilist[i+1], zlist[i+1], glist[i+1], κlist[i+1], metalist[i+1,1];
                                         signzz=metalist[i+1,2])
            end
        end

        res *= halfedgeactionXiZ(xilist[end], zlist[end-1], glist[end], κlist[end], metalist[end,1];
                                 signzz=metalist[end,2])

        return res
    else
        res = symone(xilist[1][1])
        for i in eachindex(xilist)
            zidx = isodd(i) ? i : i - 1
            res *= halfedgeactiontXiZ(xilist[i], zlist[zidx], glist[i], κlist[i], metalist[i,1];
                                      signzz=metalist[i,2])
        end
        return res
    end
end

# ============================================================
# Face action: Boundary ZZ sector
#
# facesign > 0 : spacelike face (sum structure)
# facesign ≤ 0 : timelike face (sum of timelike ZZ terms)
# ============================================================
function faceactionBDZZ(xilist, zlist, glist, κlist, metalist, sgndetlist; facesign::Int=1, γ=γsym())
    n = length(glist)

    if facesign > 0
        res = halfedgeactionZZ(xilist[1], zlist[1], glist[1], κlist[1], metalist[1,1];
                               signzz=metalist[1,2], γ=γ)

        for i in 2:2:(n-1)
            if sgndetlist[i] == 1
                res += edgebulkactionssZZ2((zlist[i-1], zlist[i+1]),
                                           (glist[i], glist[i+1]),
                                           (κlist[i], κlist[i+1]);
                                           γ=γ)
            else
                res += halfedgeactionZZ(xilist[i], zlist[i-1], glist[i], κlist[i], metalist[i,1];
                                        signzz=metalist[i,2], γ=γ) +
                       halfedgeactionZZ(xilist[i+1], zlist[i+1], glist[i+1], κlist[i+1], metalist[i+1,1];
                                        signzz=metalist[i+1,2], γ=γ)
            end
        end

        res += halfedgeactionZZ(xilist[end], zlist[end-1], glist[end], κlist[end], metalist[end,1];
                                signzz=metalist[end,2], γ=γ)

        return res
    else
        res = symzero(xilist[1][1])
        for i in eachindex(xilist)
            zidx = isodd(i) ? i : i - 1
            res += halfedgeactiontZZ(xilist[i], zlist[zidx], glist[i], κlist[i], metalist[i,1];
                                     signzz=metalist[i,2], γ=γ)
        end
        return res
    end
end

function bulkfaceactionttXiZ(xilist, zlist, glist, κlist, metalist)
    res = symone(xilist[1][1])
    for i in eachindex(xilist)
        zidx = iseven(i) ? i : (i == firstindex(xilist) ? lastindex(zlist) : i - 1)
        res *= halfedgeactiontXiZ(xilist[i], zlist[zidx], glist[i], κlist[i], metalist[i,1];
                                  signzz=metalist[i,2])
    end
    return res
end

function bulkfaceactionttZZ(xilist, zlist, glist, κlist, metalist; γ=γsym())
    res = symzero(xilist[1][1])
    for i in eachindex(xilist)
        zidx = iseven(i) ? i : (i == firstindex(xilist) ? lastindex(zlist) : i - 1)
        res += halfedgeactiontZZ(xilist[i], zlist[zidx], glist[i], κlist[i], metalist[i,1];
                                 signzz=metalist[i,2], γ=γ)
    end
    return res
end

function bulkfaceactionsXiZ(xilist, zlist, glist, κlist, metalist, sgndetlist)
    res = symone(xilist[1][1])
    I0 = firstindex(glist)
    Iend = lastindex(glist)

    for i in I0:2:Iend
        zim1 = (i == I0) ? lastindex(zlist) : i - 1

        if sgndetlist[i] == 1
            res *= edgebulkactionssZZ1((zlist[zim1], zlist[i+1]),
                                       (glist[i], glist[i+1]),
                                       (κlist[i], κlist[i+1]))
        else
            res *= halfedgeactionXiZ(xilist[i], zlist[zim1], glist[i], κlist[i], metalist[i,1];
                                     signzz=metalist[i,2]) *
                   halfedgeactionXiZ(xilist[i+1], zlist[i+1], glist[i+1], κlist[i+1], metalist[i+1,1];
                                     signzz=metalist[i+1,2])
        end
    end

    return res
end

function bulkfaceactionsZZ(xilist, zlist, glist, κlist, metalist, sgndetlist; γ=γsym())
    res = symzero(xilist[1][1])
    I0 = firstindex(glist)
    Iend = lastindex(glist)

    for i in I0:2:Iend
        zim1 = (i == I0) ? lastindex(zlist) : i - 1

        if sgndetlist[i] == 1
            res += edgebulkactionssZZ2((zlist[zim1], zlist[i+1]),
                                       (glist[i], glist[i+1]),
                                       (κlist[i], κlist[i+1]);
                                       γ=γ)
        else
            res += halfedgeactionZZ(xilist[i], zlist[zim1], glist[i], κlist[i], metalist[i,1];
                                    signzz=metalist[i,2], γ=γ) +
                   halfedgeactionZZ(xilist[i+1], zlist[i+1], glist[i+1], κlist[i+1], metalist[i+1,1];
                                    signzz=metalist[i+1,2], γ=γ)
        end
    end

    return res
end

@inline function chain_in_list(chain, list_of_chains)
    for c in list_of_chains
        c == chain && return true
    end
    return false
end

function ActionComplex(jvariablesall, gvariablesall, zvariablesall, bdyxikappafa,
                       OrderBDryFaces, OrderBulkFaces, metaxikappaf,
                       kappa, sgndet, tetareasign; γ=γsym())

    seed = jvariablesall[1][1][2]
    totalXiZ = symzero(seed)
    totalZZ  = symzero(seed)

    allfaces = vcat(OrderBDryFaces, OrderBulkFaces)

    for faces in allfaces
        nfaces = length(faces)

        xilist      = Vector{Any}(undef, nfaces)
        zlist       = Vector{Any}(undef, nfaces)
        κlist       = Vector{Int}(undef, nfaces)
        metalist    = Matrix{Any}(undef, nfaces, 2)
        glist       = Vector{Any}(undef, nfaces)
        sgndetlist  = Vector{Int}(undef, nfaces)

        for r in 1:nfaces
            k, i, j = faces[r]
            xilist[r] = bdyxikappafa[k][i][j]
            zlist[r]  = zvariablesall[k][i][j]
            glist[r]  = gvariablesall[k][i]
            κlist[r]  = kappa[k][i][j]

            m = metaxikappaf[k][i][j]
            metalist[r,1] = m[1]
            metalist[r,2] = m[2]

            sgndetlist[r] = sgndet[k][i]
        end

        k1, i1, j1 = faces[1]
        areasign = tetareasign[k1][i1][j1]
        jvalue   = jvariablesall[k1][i1][j1]

        is_bdry = chain_in_list(faces, OrderBDryFaces)

        XiZprod = if is_bdry
            faceactionXiZ(xilist, zlist, glist, κlist, metalist, sgndetlist; facesign=areasign)
        else
            areasign == -1 ?
                bulkfaceactionttXiZ(xilist, zlist, glist, κlist, metalist) :
                bulkfaceactionsXiZ(xilist, zlist, glist, κlist, metalist, sgndetlist)
        end

        ZZsum = if is_bdry
            faceactionBDZZ(xilist, zlist, glist, κlist, metalist, sgndetlist; facesign=areasign, γ=γ)
        else
            areasign == -1 ?
                bulkfaceactionttZZ(xilist, zlist, glist, κlist, metalist; γ=γ) :
                bulkfaceactionsZZ(xilist, zlist, glist, κlist, metalist, sgndetlist; γ=γ)
        end

        totalXiZ += 2 * jvalue * slog(XiZprod)
        totalZZ  += jvalue * ZZsum
    end

    return totalXiZ + totalZZ
end


function build_metaxikappaf(sgndet, tetareasign, tetn0sign)
    ns = length(sgndet)
    ntet = 5

    Id2 = ComplexF64[1 0; 0 1]
    σ3mat = ComplexF64.(σ3(Int))

    metaxikappaf = Vector{Vector{Vector{Tuple{Matrix{ComplexF64},Int}}}}(undef, ns)

    for k in 1:ns
        metaxikappaf[k] = Vector{Vector{Tuple{Matrix{ComplexF64},Int}}}(undef, ntet)

        for i in 1:ntet
            metaxikappaf[k][i] = Vector{Tuple{Matrix{ComplexF64},Int}}(undef, ntet)

            if sgndet[k][i] > 0
                for j in 1:ntet
                    metaxikappaf[k][i][j] = (Id2, 1)
                end
            else
                for j in 1:ntet
                    if tetareasign[k][i][j] < 0
                        metaxikappaf[k][i][j] = (σ3mat, 1)
                    else
                        metaxikappaf[k][i][j] = (σ3mat, tetn0sign[k][i][j])
                    end
                end
            end
        end
    end

    return metaxikappaf
end

# ============================================================
# Half-edge action (spacelike face)
# ============================================================
function halfedgeaction(xi, z, g, κ::Int, meta; signzz::Int=1, γ=γsym())
    detm = meta_det(meta)

    A = scalar_A(xi, z, g, meta; signzz=signzz)
    B = scalar_B(xi, z, g, meta; signzz=signzz)
    C = scalar_C(z, g, meta; signzz=signzz)

    term1 = 2 * slog(A^((κ + detm) ÷ 2) * B^((-κ + detm) ÷ 2))
    term2 = (_I[] * γ * κ - detm) * slog(C)

    return term1 + term2
end

function halfedgeactiont(xi, z, g, κ::Int, meta; signzz::Int=1, γ=γsym())
    A = scalar_A(xi, z, g, meta; signzz=1)
    B = scalar_B(xi, z, g, meta; signzz=1)

    term1 = 2 * κ * slog(ssqrt((signzz * A) / (signzz * B)))
    term2 = -(_I[] / γ) * κ * slog(A * B)

    return term1 + term2
end


# ============================================================
# Vertex action (single 4-simplex)
# ============================================================
function vertexaction(j_mat1, xi_mat1, z_mat1, g_mat1, κdata, sgndet, tetn0sign, tetareasign; γ=γsym())
    ntet = 5

    Id2 = ComplexF64[1 0; 0 1]
    σ3mat = ComplexF64.(σ3(Int))

    seed = j_mat1[1][2]
    act = symzero(seed)

    for i in 1:ntet, j in 1:ntet
        i == j && continue

        jf = j_mat1[i][j]

        κ = κdata[i][j]
        z = (κ == 1) ? z_mat1[i][j] : z_mat1[j][i]

        signzz_s = (tetn0sign[i][j] < 0) ? tetn0sign[i][j] : 1
        meta = (sgndet[i] == 1 ? Id2 : σ3mat)

        he = tetareasign[i][j] > 0 ?
            halfedgeaction(xi_mat1[i][j], z, g_mat1[i], κ, meta; signzz=signzz_s, γ=γ) :
            halfedgeactiont(xi_mat1[i][j], z, g_mat1[i], κ, σ3mat; signzz=1, γ=γ)

        act += jf * he
    end

    return act
end


function compute_action(geom)

    ns = length(geom.simplex)

    xi_mat = geom.varias[:xi_mat]
    z_mat  = geom.varias[:z_mat]
    g_mat  = geom.varias[:g_mat]
    j_mat  = geom.varias[:j_mat]

    kappa = [geom.simplex[i].kappa for i in 1:ns]
    sgndet = [geom.simplex[i].sgndet for i in 1:ns]
    tetn0sign = [geom.simplex[i].tetn0sign for i in 1:ns]
    tetareasign = [geom.simplex[i].tetareasign for i in 1:ns];

    if ns == 1
        # single 4-simplex case
        return vertexaction(j_mat[1], xi_mat[1], z_mat[1], g_mat[1], kappa[1], sgndet[1], tetn0sign[1], tetareasign[1]; γ = γsym())
    else
        OrderBDryFaces = geom.connectivity[1]["OrderBDryFaces"]
        OrderBulkFaces = geom.connectivity[1]["OrderBulkFaces"]  
        metaxikappaf = build_metaxikappaf(sgndet, tetareasign, tetn0sign)
        return ActionComplex(j_mat, g_mat, z_mat, xi_mat, OrderBDryFaces, OrderBulkFaces, metaxikappaf, kappa, sgndet, tetareasign; γ = γsym())
    end

end

end # module