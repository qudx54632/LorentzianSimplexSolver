module DefineAction

using LinearAlgebra
using SymEngine

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

function Eehss(zs, gs, κs; γ=γsym())
    z1, z3 = zs
    g2, g3 = gs
    κ2, κ3 = κs

    if κ2 == -1 && κ3 == 1
        Z2 = transpose(g2) * col(z1)
        Z3 = transpose(g3) * col(z3)

        num  = (herm(Z2) * Z3)[1,1]
        den1 = ((herm(Z2) * Z2)[1,1])^((1 + _I[]*γ)/2)
        den2 = ((herm(Z3) * Z3)[1,1])^((1 - _I[]*γ)/2)

        return num / (den1 * den2)
    else
        error("Wrong orientation")
    end
end

function EebssSource(g, z, xi, κ; γ=γsym())
    if κ == 1
        Z   = transpose(g) * col(z)
        xiC = col(xi)
        num = (herm(xiC) * Z)[1,1]
        den = ((herm(Z) * Z)[1,1])^((1 - _I[]*γ)/2)
        return num / den
    else
        error("Wrong orientation")
    end
end

function EebssTarget(g, z, xi, κ; γ=γsym())
    if κ == -1
        Z   = transpose(g) * col(z)
        xiC = col(xi)
        num = (herm(Z) * xiC)[1,1]
        den = ((herm(Z) * Z)[1,1])^((1 + _I[]*γ)/2)
        return num / den
    else
        error("Wrong orientation")
    end
end

function boundary_action(gvariablesall, zvariablesall, ηlabelsMat, zetabdryall, kappaMat, OrderBDryFaces; γ=γsym())
    seed = ηlabelsMat[1][1][2]
    Sb = symzero(seed)

    for faces in OrderBDryFaces
        xilist = [zetabdryall[k][i][j] for (k,i,j) in faces]
        zlist  = [zvariablesall[k][i][j] for (k,i,j) in faces]
        glist  = [gvariablesall[k][i] for (k,i,j) in faces]
        κlist  = [kappaMat[k][i][j] for (k,i,j) in faces]

        k1, i1, j1 = faces[1]
        ηval = ηlabelsMat[k1][i1][j1]

        prodEh = symone(ηval)
        for i in 2:2:(length(faces)-1)
            prodEh *= Eehss((zlist[i-1], zlist[i+1]),
                            (glist[i], glist[i+1]),
                            (κlist[i], κlist[i+1]); γ=γ)
        end

        term = EebssSource(glist[1], zlist[1], xilist[1], κlist[1]; γ=γ) *
               prodEh *
               EebssTarget(glist[end], zlist[end-1], xilist[end], κlist[end]; γ=γ)

        Sb += ηval * slog(term)
    end

    return Sb
end

function bulk_action(gvariablesall, zvariablesall, ηlabelsMat, kappaMat, OrderBulkFaces; γ=γsym())
    seed = ηlabelsMat[1][1][2]
    Sh = symzero(seed)

    for faces in OrderBulkFaces
        zlist = [zvariablesall[k][i][j] for (k,i,j) in faces]
        glist = [gvariablesall[k][i] for (k,i,j) in faces]
        κlist = [kappaMat[k][i][j] for (k,i,j) in faces]

        k1, i1, j1 = faces[1]
        ηval = ηlabelsMat[k1][i1][j1]

        prodEh = symone(ηval)
        for i in 1:2:length(faces)
            zim1 = (i == 1) ? length(zlist) : i - 1
            prodEh *= Eehss((zlist[zim1], zlist[i+1]),
                            (glist[i], glist[i+1]),
                            (κlist[i], κlist[i+1]); γ=γ)
        end

        # logEh = slog(prodEh)
        Sh += -ηval * (1 - prodEh)
    end

    return Sh
end

function compute_action(geom; γ=γsym())
    gvariablesall = geom.varias[:g_mat]
    zvariablesall = geom.varias[:z_mat]
    zetabdryall   = geom.varias[:xi_mat]
    ηlabelsMat    = geom.varias[:η_mat]
    kappaMat      = [geom.simplex[i].kappa for i in 1:length(geom.simplex)]

    ns = length(geom.simplex)
    if ns > 1
        OrderBDryFaces = geom.connectivity[1]["OrderBDryFaces"]
        OrderBulkFaces = geom.connectivity[1]["OrderBulkFaces"]

        Sb = boundary_action(gvariablesall, zvariablesall, ηlabelsMat, zetabdryall, kappaMat, OrderBDryFaces; γ=γ)
        Sh = bulk_action(gvariablesall, zvariablesall, ηlabelsMat, kappaMat, OrderBulkFaces; γ=γ)
        return Sb + Sh
    else 
        OrderBDryFaces = [kappaMat[1][i][j]== 1 ? [[1, i, j], [1, j, i]] : [[1, j, i], [1, i, j]] for i in 1:5 for j in 1:5 if i < j]
        Sb = boundary_action(gvariablesall, zvariablesall, ηlabelsMat, zetabdryall, kappaMat, OrderBDryFaces; γ=γ)

        return Sb
    end

end

end