module DefineAction

using ..SpinAlgebra: σ3

using PythonCall
sympy = pyimport("sympy")

export vertexaction, γsym

const I      = sympy.I
const Matrix = sympy.Matrix

# Make gamma a SymPy symbol at module scope
const γsym = sympy.symbols("gamma", real=true)

# ============================================================
# Half-edge action (spacelike face)
# ============================================================
function halfedgeaction(xi::Py, z::Py, g::Py, κ::Int, meta::Py; signzz::Int=1, γ=γsym)
    gt   = g.T
    detm = meta.det()

    A = (signzz * (xi.conjugate().T * meta * gt * z))[0]
    B = (signzz * ((gt * z).conjugate().T * meta * xi))[0]
    C = (signzz * ((gt * z).conjugate().T * meta * gt * z))[0]

    term1 = 2 * sympy.log(A^((κ + detm)/2) * B^((-κ + detm)/2))
    term2 = (I * γ * κ - detm) * sympy.log(C)

    return term1 + term2
end

# ============================================================
# Half-edge action (timelike face)
# ============================================================
function halfedgeactiont(xi::Py, z::Py, g::Py, κ::Int, meta::Py; signzz::Int=1, γ=γsym)
    gt = g.T

    A = (xi.conjugate().T * meta * gt * z)[0]
    B = ((gt * z).conjugate().T * meta * xi)[0]

    term1 = 2 * κ * sympy.log(sympy.sqrt((signzz * A) / (signzz * B)))
    term2 = -(I / γ) * κ * sympy.log(A * B)

    return term1 + term2
end

# ============================================================
# Vertex action (single 4-simplex)
# ============================================================
function vertexaction(j_mat1, xi_mat1, z_mat1, g_mat1,
                            κdata, sgndet, tetn0sign, tetareasign; γ=γsym)

    ntet = 5
    Id2  = sympy.eye(2)
    σ3py = Matrix(σ3)

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

end # module