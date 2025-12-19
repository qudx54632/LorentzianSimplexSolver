module DefineAction

using ..SpinAlgebra: σ3
# using ..SafeSqrtLog: safe_sqrt, safe_log

using PythonCall
sympy = pyimport("sympy")

export vertexaction, γsym, run_action

I      = sympy.I
spMatrix = sympy.Matrix   # SymPy matrix constructor
safe_log = sympy.log
safe_sqrt = sympy.sqrt

# Make gamma a SymPy symbol at module scope
const γsym = sympy.symbols("gamma", real=true)

# ============================================================
# Spacelike Xi-Z contribution
# xiZ = \[\left(m_{ef}\,\langle \xi_{ef}, Z_{vef} \rangle \right)^{\frac{\kappa_{ef} + \det \eta_e}{2}}\]
# Zxi = \[\left(m_{ef}\,\langle Z_{vef} , \xi_{ef} \rangle \right)^{\frac{-\kappa_{ef} + \det \eta_e}{2}}\]
# ============================================================
function halfedgeactionXiZ(xi::Py, z::Py, g::Py, κ::Int, meta::Py; signzz::Int=1)
    κpy  = Py(κ)
    half = sympy.Rational(1, 2)

    detm = meta.det()
    gt   = g.T

    # scalars (1×1 matrices -> take [0])
    A = (signzz * (xi.conjugate().T * meta * gt * z))[0]
    B = (signzz * ((gt * z).conjugate().T * meta * xi))[0]

    e1 = (κpy + detm) * half
    e2 = (-κpy + detm) * half

    expr = A^e1 * B^e2

    return expr
end

# ============================================================
# Spacelike Z-Z contribution
# ZZ = (i\gamma\kappa - \det \eta_e)\[\log(\langle Z_{vef}, Z_{vef} \rangle)\]
# ============================================================
function halfedgeactionZZ(xi::Py, z::Py, g::Py, κ::Int, meta::Py;
                          signzz::Int = 1, γ::Py)

    # Transpose of g
    gt = g.T

    # scalar contraction (take [0] to extract scalar)
    C = (signzz * ((gt * z).conjugate().T * meta * gt * z))[0]

    return (sympy.I * γ * κ  - meta.det()) * safe_log(C)
end

# ============================================================
# Timelike Xi–Z contribution
# XZ_t = ( sqrt( <xi , Z> / <Z , xi> ) )^κ
# where Z = g^T z
# ============================================================
function halfedgeactiontXiZ(xi::Py, z::Py, g::Py, κ::Int, meta::Py;
                            signzz::Int = 1)

    gt = g.T

    A = (signzz * (xi.conjugate().T * meta * gt * z))[0]
    B = (signzz * ((gt * z).conjugate().T * meta * xi))[0]

    return (safe_sqrt(A / B))^κ
end

# ============================================================
# Timelike Z–Z contribution
# ZZ_t = -(i/γ) κ log( <xi , Z> <Z , xi> )
# where Z = g^T z
# ============================================================
function halfedgeactiontZZ(xi::Py, z::Py, g::Py, κ::Int, meta::Py;
                           signzz::Int = 1, γ::Py)

    gt = g.T

    A = (xi.conjugate().T * meta * gt * z)[0]
    B = ((gt * z).conjugate().T * meta * xi)[0]

    return -(sympy.I / γ) * κ * safe_log(A * B)
end

# ============================================================
# Bulk edge action ZZ (type 1)
#
# Only defined for orientation (κ2, κ3) = (-1, +1)
# Returns <Z_2 , Z_3>
# ============================================================
function edgebulkactionssZZ1(zs::Tuple{Py,Py}, gs::Tuple{Py,Py}, κs::Tuple{Int,Int})

    z1, z3 = zs
    g2, g3 = gs
    κ2, κ3 = κs

    if κ2 == -1 && κ3 == 1
        return ((g2.T * z1).conjugate().T * (g3.T * z3))[0]
    else
        error("Wrong orientation")
    end
end

# ============================================================
# Bulk edge action ZZ (type 2)
#
# Only defined for orientation (κ2, κ3) = (-1, +1)
#
# 2 (iγ - 1) log sqrt(<Z3,Z3>) - 2 (iγ + 1) log sqrt(<Z2,Z2>)
# ============================================================
function edgebulkactionssZZ2(zs::Tuple{Py,Py}, gs::Tuple{Py,Py}, κs::Tuple{Int,Int}; γ::Py)

    z1, z3 = zs
    g2, g3 = gs
    κ2, κ3 = κs

    if κ2 == -1 && κ3 == 1
        Z3 = g3.T * z3
        Z2 = g2.T * z1

        term3 = 2 * (sympy.I * γ - 1) * safe_log(safe_sqrt((Z3.conjugate().T * Z3)[0]))

        term2 = 2 * (sympy.I * γ + 1) * safe_log(safe_sqrt((Z2.conjugate().T * Z2)[0]))

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
function faceactionXiZ(xilist, zlist, glist,
                        κlist, metalist, sgndetlist;
                        facesign::Int = 1)

    n = length(glist)

    if facesign > 0
        # --- first half-edge ---
        res = halfedgeactionXiZ(xilist[1], zlist[1], glist[1], κlist[1], metalist[1,1]; signzz=metalist[1,2])

        # --- bulk contributions ---
        for i in 2:2:(n-1)
            if sgndetlist[i] == 1
                res *= edgebulkactionssZZ1((zlist[i-1], zlist[i+1]),(glist[i], glist[i+1]), (κlist[i], κlist[i+1]))
            else
                res *= halfedgeactionXiZ(xilist[i], zlist[i-1], glist[i],κlist[i], metalist[i,1]; signzz=metalist[i,2]) *
                       halfedgeactionXiZ(xilist[i+1], zlist[i+1], glist[i+1],κlist[i+1], metalist[i+1,1]; signzz=metalist[i+1,2])
            end
        end

        # --- last half-edge ---
        res *= halfedgeactionXiZ(xilist[end], zlist[end-1], glist[end],κlist[end], metalist[end,1]; signzz=metalist[end,2])

        return res

    else
        # --- timelike face ---
        res = one(Py)
        for i in eachindex(xilist)
            zidx = isodd(i) ? i : i-1
            res *= halfedgeactiontXiZ(xilist[i], zlist[zidx], glist[i],κlist[i], metalist[i,1];signzz=metalist[i,2])
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
function faceactionBDZZ(xilist, zlist, glist, κlist, metalist, sgndetlist;
                        facesign::Int = 1, γ::Py = γsym)

    n = length(glist)

    if facesign > 0
        res = halfedgeactionZZ(xilist[1], zlist[1], glist[1], κlist[1], metalist[1,1];
                               signzz = metalist[1,2], γ = γ)

        for i in 2:2:(n-1)
            if sgndetlist[i] == 1
                res += edgebulkactionssZZ2((zlist[i-1], zlist[i+1]),
                                          (glist[i], glist[i+1]),
                                          (κlist[i], κlist[i+1]);
                                          γ = γ)
            else
                res += halfedgeactionZZ(xilist[i], zlist[i-1], glist[i], κlist[i], metalist[i,1];
                                        signzz = metalist[i,2], γ = γ) +
                       halfedgeactionZZ(xilist[i+1], zlist[i+1], glist[i+1], κlist[i+1], metalist[i+1,1];
                                        signzz = metalist[i+1,2], γ = γ)
            end
        end

        res += halfedgeactionZZ(xilist[end], zlist[end-1], glist[end], κlist[end], metalist[end,1];
                                signzz = metalist[end,2], γ = γ)

        return res
    else
        res = Py(0)
        for i in eachindex(xilist)
            zidx = isodd(i) ? i : i-1
            res += halfedgeactiontZZ(xilist[i], zlist[zidx], glist[i], κlist[i], metalist[i,1];
                                     signzz = metalist[i,2], γ = γ)
        end
        return res
    end
end

function bulkfaceactionttXiZ(xilist, zlist, glist, κlist, metalist)
    res = one(Py) # multiplicative identity

    for i in eachindex(xilist)
        zidx = iseven(i) ? i : (i == firstindex(xilist) ? lastindex(zlist) : i - 1)

        res *= halfedgeactiontXiZ(xilist[i],zlist[zidx],glist[i],κlist[i],metalist[i, 1];signzz=metalist[i, 2])
    end

    return res
end

function bulkfaceactionttZZ(xilist, zlist, glist, κlist, metalist; γ::Py = γsym)
    res = Py(0)
    for i in eachindex(xilist)
        zidx = iseven(i) ? i : (i == firstindex(xilist) ? lastindex(zlist) : i - 1)
        res += halfedgeactiontZZ(xilist[i], zlist[zidx], glist[i], κlist[i], metalist[i,1];
                                 signzz = metalist[i,2], γ = γ)
    end
    return res
end

function bulkfaceactionsXiZ(xilist, zlist, glist, κlist, metalist, sgndetlist)
    res = one(Py)
    I0  = firstindex(glist)
    Iend = lastindex(glist)

    for i in I0:2:Iend
        zim1 = (i == I0) ? lastindex(zlist) : i - 1

        if sgndetlist[i] == 1
            res *= edgebulkactionssZZ1(
                (zlist[zim1], zlist[i + 1]),
                (glist[i], glist[i + 1]),
                (κlist[i], κlist[i + 1])
            )
        else
            res *= halfedgeactionXiZ(xilist[i], zlist[zim1], glist[i], κlist[i],metalist[i, 1]; signzz=metalist[i, 2]) *
            halfedgeactionXiZ(xilist[i + 1],zlist[i + 1],glist[i + 1],κlist[i + 1],metalist[i + 1, 1]; signzz=metalist[i + 1, 2])
        end
    end

    return res
end

function bulkfaceactionsZZ(xilist, zlist, glist, κlist, metalist, sgndetlist; γ::Py = γsym)
    res  = Py(0)
    I0   = firstindex(glist)
    Iend = lastindex(glist)

    for i in I0:2:Iend
        zim1 = (i == I0) ? lastindex(zlist) : i - 1

        if sgndetlist[i] == 1
            res += edgebulkactionssZZ2((zlist[zim1], zlist[i+1]),
                                      (glist[i], glist[i+1]),
                                      (κlist[i], κlist[i+1]);
                                      γ = γ)
        else
            res += halfedgeactionZZ(xilist[i], zlist[zim1], glist[i], κlist[i], metalist[i,1];
                                    signzz = metalist[i,2], γ = γ) +
                   halfedgeactionZZ(xilist[i+1], zlist[i+1], glist[i+1], κlist[i+1], metalist[i+1,1];
                                    signzz = metalist[i+1,2], γ = γ)
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
                       OrderBDryFaces, OrderBulkFaces,
                       metaxikappaf,
                       kappa, sgndet, tetareasign;
                       γ::Py = γsym)

    allfaces = vcat(OrderBDryFaces, OrderBulkFaces)

    BDActionsXiZ = Vector{Py}(undef, length(allfaces))
    BDActionsZZ  = Vector{Py}(undef, length(allfaces))

    for (idx, faces) in pairs(allfaces)
        # faces is a chain: Vector{Vector{Int}} with elements [k,i,j]

        # build lists along the chain
        nfaces = length(faces)

        xilist = Vector{Any}(undef, nfaces)
        zlist  = Vector{Any}(undef, nfaces)
        κlist  = Vector{Int}(undef, nfaces)

        # metalist[r,1]=meta, metalist[r,2]=signzz
        metalist  = Matrix{Any}(undef, nfaces, 2)

        # glist/sgndetlist depend on (k,i) only
        glist     = Vector{Any}(undef, nfaces)
        sgndetlist = Vector{Int}(undef, nfaces)

        for r in 1:nfaces
            k,i,j = faces[r][1], faces[r][2], faces[r][3]

            # MMA: bdyxikappafa[[k,i,j]][[1]]
            xilist[r] = bdyxikappafa[k][i][j]

            zlist[r]  = zvariablesall[k][i][j]
            glist[r]  = gvariablesall[k][i]
            κlist[r]  = kappa[k][i][j]

            m = metaxikappaf[k][i][j]
            metalist[r,1] = m[1]
            metalist[r,2] = m[2]

            sgndetlist[r] = sgndet[k][i]
        end

        # per-chain face sign and j-value taken from first face, like MMA
        k1,i1,j1 = faces[1][1], faces[1][2], faces[1][3]
        areasign = tetareasign[k1][i1][j1]
        jvalue   = jvariablesall[k1][i1][j1]

        is_bdry = DefineAction.chain_in_list(faces, OrderBDryFaces)

        # -------------------------
        # XiZ part: 2*j*log(…)
        # -------------------------
        XiZprod = if is_bdry
            DefineAction.faceactionXiZ(xilist, zlist, glist, κlist, metalist, sgndetlist;
                          facesign = areasign)
        else
            if areasign == -1
                DefineAction.bulkfaceactionttXiZ(xilist, zlist, glist, κlist, metalist)
            else
                DefineAction.bulkfaceactionsXiZ(xilist, zlist, glist, κlist, metalist, sgndetlist)
            end
        end

        BDActionsXiZ[idx] = 2 * jvalue * safe_log(XiZprod)

        # -------------------------
        # ZZ part: j * (…)
        # -------------------------
        ZZsum = if is_bdry
            faceactionBDZZ(xilist, zlist, glist, κlist, metalist, sgndetlist;
                           facesign = areasign, γ = γ)
        else
            if areasign == -1
                bulkfaceactionttZZ(xilist, zlist, glist, κlist, metalist; γ = γ)
            else
                bulkfaceactionsZZ(xilist, zlist, glist, κlist, metalist, sgndetlist; γ = γ)
            end
        end

        BDActionsZZ[idx] = jvalue * ZZsum
    end

    totalXiZ = Py(0)
    totalZZ  = Py(0)

    for i in eachindex(BDActionsXiZ)
        totalXiZ += BDActionsXiZ[i]
        totalZZ  += BDActionsZZ[i]
    end

    return totalXiZ + totalZZ
end


function build_metaxikappaf(sgndet, tetareasign, tetn0signtest3)
    ns   = length(sgndet)
    ntet = 5

    Id2  = sympy.eye(2)
    σ3py = spMatrix(σ3)

    # metaxikappaf[k][i][j] = (meta, signzz)
    metaxikappaf = Vector{Vector{Vector{Tuple{Py,Int}}}}(undef, ns)

    for k in 1:ns
        metaxikappaf[k] = Vector{Vector{Tuple{Py,Int}}}(undef, ntet)

        for i in 1:ntet
            metaxikappaf[k][i] = Vector{Tuple{Py,Int}}(undef, ntet)

            if sgndet[k][i] > 0
                # spacelike tetra: always (Id, +1)
                for j in 1:ntet
                    metaxikappaf[k][i][j] = (Id2, 1)
                end
            else
                # timelike tetra
                for j in 1:ntet
                    if tetareasign[k][i][j] < 0
                        metaxikappaf[k][i][j] = (σ3py, 1)
                    else
                        metaxikappaf[k][i][j] = (σ3py, tetn0signtest3[k][i][j])
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
function halfedgeaction(xi::Py, z::Py, g::Py, κ::Int, meta::Py; signzz::Int=1, γ=γsym)
    gt   = g.T
    detm = meta.det()

    A = (signzz * (xi.conjugate().T * meta * gt * z))[0]
    B = (signzz * ((gt * z).conjugate().T * meta * xi))[0]
    C = (signzz * ((gt * z).conjugate().T * meta * gt * z))[0]

    term1 = 2 * safe_log(A^((κ + detm)/2) * B^((-κ + detm)/2))
    term2 = (I * γ * κ - detm) * safe_log(C)

    return term1 + term2
end

# ============================================================
# Half-edge action (timelike face)
# ============================================================
function halfedgeactiont(xi::Py, z::Py, g::Py, κ::Int, meta::Py; signzz::Int=1, γ=γsym)
    gt = g.T

    A = (xi.conjugate().T * meta * gt * z)[0]
    B = ((gt * z).conjugate().T * meta * xi)[0]

    term1 = 2 * κ * safe_log(safe_sqrt((signzz * A) / (signzz * B)))
    term2 = -(I / γ) * κ * safe_log(A * B)

    return term1 + term2
end
# ============================================================
# Vertex action (single 4-simplex)
# ============================================================
function vertexaction(j_mat1, xi_mat1, z_mat1, g_mat1,
                            κdata, sgndet, tetn0sign, tetareasign; γ=γsym)

    ntet = 5
    Id2  = sympy.eye(2)
    σ3py = spMatrix(σ3)

    # terms = Vector{Tuple{Int,Int,Py}}()  # (i, j, jf*he)
    act   = Py(0)

    for i in 1:ntet, j in 1:ntet
        i == j && continue

        jf = j_mat1[i][j]
        jf === Py(0) && continue

        κ = κdata[i][j]
        z = (κ == 1) ? z_mat1[i][j] : z_mat1[j][i]

        signzz_s = (tetn0sign[i][j] < 0) ? tetn0sign[i][j] : 1
        meta     = (sgndet[i] == 1 ? Id2 : σ3py)

        he = (tetareasign[i][j] > 0) ?
            halfedgeaction( xi_mat1[i][j], z, g_mat1[i], κ, meta;
                            signzz=signzz_s, γ=γ) :
            halfedgeactiont(xi_mat1[i][j], z, g_mat1[i], κ, σ3py;
                            signzz=1, γ=γ)

        # push!(terms, (i, j, jf * he))
        act += jf * he
    end

    return act
end


function run_action(geom)

    ns = length(geom.simplex)

    xi_mat = geom.varias[:xi_mat]
    z_mat = geom.varias[:z_mat]
    g_mat = geom.varias[:g_mat]
    j_mat = geom.varias[:j_mat]

    kappa = [geom.simplex[i].kappa for i in 1:ns]
    sgndet = [geom.simplex[i].sgndet for i in 1:ns]
    tetn0sign = [geom.simplex[i].tetn0sign for i in 1:ns]
    tetareasign = [geom.simplex[i].tetareasign for i in 1:ns];

    if ns == 1
        # single 4-simplex case
        return vertexaction(j_mat[1], xi_mat[1], z_mat[1], g_mat[1], kappa[1], sgndet[1], tetn0sign[1], tetareasign[1]; γ = γsym)
    else
        OrderBDryFaces = geom.connectivity[1]["OrderBDryFaces"]
        OrderBulkFaces = geom.connectivity[1]["OrderBulkFaces"]  
        metaxikappaf = build_metaxikappaf(sgndet, tetareasign, tetn0sign)
        return ActionComplex(j_mat, g_mat, z_mat, xi_mat, OrderBDryFaces, OrderBulkFaces, metaxikappaf, kappa, sgndet, tetareasign; γ = γsym)
    end

end

end # module