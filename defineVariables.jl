module DefineVariables

using ..SpinAlgebra: σ3
using ..CriticalPoints: compute_bdy_critical_data

using PythonCall

sympy = pyimport("sympy")

const I        = sympy.I
const symbols  = sympy.symbols
const Matrix   = sympy.Matrix
const simplify = sympy.simplify

export run_define_variables

# ------------------------------------------------------------
# g variables (flat var list + per-(simplex,tet) 2x2 matrix list)
# g_mat is length ns*5, ordered by a=1..ns, b=1..5
# ------------------------------------------------------------
function build_g_variables(num_vertex::Int)
    var_g = Vector{Py}()                        # flat list of symbols
    g_mat = Vector{Vector{Py}}(undef, num_vertex)  # g_mat[a][b] is 2x2 SymPy Matrix

    for a in 1:num_vertex
        g_mat[a] = Vector{Py}(undef, 5)

        for b in 1:5
            g1 = collect(symbols("g_$(a)$(b)_1:7", real=true))
            @assert length(g1) == 6
            append!(var_g, g1)

            g_mat[a][b] = simplify(Matrix([
                1 + g1[1] + g1[2]*I      g1[3] + g1[4]*I;
                g1[5] + g1[6]*I          (1 + g1[3]*g1[5]
                                           + I*g1[4]*g1[5]
                                           + I*g1[3]*g1[6]
                                           - g1[4]*g1[6]) /
                                          (1 + g1[1] + g1[2]*I)
            ]))
        end
    end

    return var_g, g_mat
end

# ------------------------------------------------------------
# z variables
# kappa_all is a Vector of 5x5 (or nested vectors) per simplex: kappa_all[a][i][j]
# zspecialPos entries like [a,i,j] (simplex index included)
# z_mat[a][i][j] is either Py[] (inactive) or Py[*,*] length 2
# ------------------------------------------------------------
function build_z_variables(num_vertex::Int,
                           kappa_all,
                           zspecialPos::Vector{Vector{Int}})

    ntet = 5
    special_set = Set((p[1], p[2], p[3]) for p in zspecialPos)

    var_z = Vector{Py}()
    z_mat = Vector{Vector{Vector{Py}}}(undef, num_vertex)

    for a in 1:num_vertex
        z_mat[a] = Vector{Vector{Py}}(undef, ntet)

        for i in 1:ntet
            z_mat[a][i] = Vector{Py}(undef, ntet)

            for j in 1:ntet
                if i == j || kappa_all[a][i][j] != 1
                    z_mat[a][i][j] = Matrix([0; 0])
                    continue
                end

                ii, jj = i < j ? (i, j) : (j, i)

                z  = symbols("z_$(a)$(ii)$(jj)",  real=true)
                zc = symbols("zc_$(a)$(ii)$(jj)", real=true)

                push!(var_z, z)
                push!(var_z, zc)

                zcplx = 1 + z + I*zc

                if (a, i, j) in special_set
                    z_mat[a][i][j] = simplify(Matrix([zcplx; 1]))
                else
                    z_mat[a][i][j] = simplify(Matrix([1; zcplx]))
                end
            end
        end
    end

    return var_z, z_mat
end

# ------------------------------------------------------------
# xi variables (zeta)
# sgndet[a][i], tetareasign[a][i][j], tetn0sign[a][i][j]
# xi_mat[a][i][j] is always Py[*,*] length 2, diagonal is [0,0]
# ------------------------------------------------------------
function build_xi_variables(num_vertex::Int,
                            sgndet::Vector{Vector{Int}},
                            tetareasign::Vector{Vector{Vector{Int}}},
                            tetn0sign::Vector{Vector{Vector{Int}}})

    ntet = 5
    var_xi = Vector{Py}()
    xi_mat = Vector{Vector{Vector{Py}}}(undef, num_vertex)

    for a in 1:num_vertex
        xi_mat[a] = Vector{Vector{Py}}(undef, ntet)

        for i in 1:ntet
            xi_mat[a][i] = Vector{Py}(undef, ntet)

            for j in 1:ntet
                if i == j
                    xi_mat[a][i][j] = Matrix([0; 0])
                    continue
                end

                if sgndet[a][i] == 1
                    za = symbols("zeta_$(a)_$(i)_$(j)_a", real=true)
                    zb = symbols("zeta_$(a)_$(i)_$(j)_b", real=true)

                    xi_mat[a][i][j] = simplify(Matrix([
                        sympy.sin(za);
                        sympy.cos(za) * sympy.exp(I * zb)
                    ]))

                    push!(var_xi, za)
                    push!(var_xi, zb)

                elseif tetareasign[a][i][j] == 1
                    za = symbols("zeta_$(a)_$(i)_$(j)_a", real=true)
                    zb = symbols("zeta_$(a)_$(i)_$(j)_b", real=true)

                    if tetn0sign[a][i][j] == 1
                        xi_mat[a][i][j] = simplify(Matrix([
                            sympy.cosh(za);
                            sympy.exp(-I * zb) * sympy.sinh(za)
                        ]))
                    else
                        xi_mat[a][i][j] = simplify(Matrix([
                            sympy.sinh(za) * sympy.exp(I * zb);
                            sympy.cosh(za)
                        ]))
                    end

                    push!(var_xi, za)
                    push!(var_xi, zb)

                else
                    zb = symbols("zeta_$(a)_$(i)_$(j)_b", real=true)

                    xi_mat[a][i][j] = simplify(Matrix([
                        1;
                        sympy.exp(I * zb)
                    ]))

                    push!(var_xi, zb)
                end
            end
        end
    end

    return var_xi, xi_mat
end

@inline function find_chain_index(key::Vector{Int}, chains)
    for idx in 1:length(chains)
        for v in chains[idx]
            if v[1] == key[1] && v[2] == key[2] && v[3] == key[3]
                return idx
            end
        end
    end
    return nothing
end

# ------------------------------------------------------------
# j variables
# returns (j_var, j_mat) where j_var is unique flat symbol list
# ------------------------------------------------------------
function build_j_variables(num_vertex::Int,
                           OrderBulkFaces,
                           OrderBDryFaces;
                           ntet=5)

    j_mat = Vector{Vector{Vector{Py}}}(undef, num_vertex)
    j_var = Vector{Py}()  # flat list, unique later

    for k in 1:num_vertex
        j_mat[k] = Vector{Vector{Py}}(undef, ntet)

        for i in 1:ntet
            j_mat[k][i] = Vector{Py}(undef, ntet)

            for j in 1:ntet
                if i == j
                    j_mat[k][i][j] = Py(0)
                    continue
                end

                key = [k, i, j]

                pos = find_chain_index(key, OrderBulkFaces)
                if pos !== nothing
                    a, b, c = OrderBulkFaces[pos][1]
                    jsym = symbols("j_$(a)_$(b)_$(c)", real=true)
                    j_mat[k][i][j] = jsym
                    push!(j_var, jsym)
                    continue
                end

                pos = find_chain_index(key, OrderBDryFaces)
                @assert pos !== nothing

                a, b, c = OrderBDryFaces[pos][1]
                jsym = symbols("j_$(a)_$(b)_$(c)", real=true)
                j_mat[k][i][j] = jsym
                push!(j_var, jsym)
            end
        end
    end

    j_var = collect(Set(j_var))
    return j_var, j_mat
end

# ------------------------------------------------------------
# main
# expects geom.varias exists and is a Dict{Symbol,Any}
# ------------------------------------------------------------
function run_define_variables(geom)
    ns   = length(geom.simplex)
    ntet = length(geom.simplex[1].bdyxi)

    @assert ntet == 5

    kappa_all    = [geom.simplex[s].kappa      for s in 1:ns]
    sgndet       = [geom.simplex[s].sgndet     for s in 1:ns]
    tetareasign  = [geom.simplex[s].tetareasign for s in 1:ns]
    tetn0sign    = [geom.simplex[s].tetn0sign   for s in 1:ns]

    var_xi, xi_mat = build_xi_variables(ns, sgndet, tetareasign, tetn0sign)

    zspecialPos = compute_bdy_critical_data(geom).zspecialpos

    if ns > 1
        sharedTetsPos   = geom.connectivity[1]["sharedTetsPos"]
        OrderBDryFaces  = geom.connectivity[1]["OrderBDryFaces"]
        OrderBulkFaces  = geom.connectivity[1]["OrderBulkFaces"]
        apply_shared_tets_to_xi!(xi_mat, sharedTetsPos)
    else
        OrderBulkFaces = Vector{Vector{Vector{Int}}}()
        OrderBDryFaces = Vector{Vector{Vector{Int}}}()

        for i in 1:ntet
            for j in i+1:ntet
                fwd = [1, i, j]
                bwd = [1, j, i]
                if kappa_all[1][i][j] == 1
                    push!(OrderBDryFaces, [fwd, bwd])
                else
                    push!(OrderBDryFaces, [bwd, fwd])
                end
            end
        end
    end

    g_var, g_mat = build_g_variables(ns)
    z_var, z_mat = build_z_variables(ns, kappa_all, zspecialPos)
    j_var, j_mat = build_j_variables(ns, OrderBulkFaces, OrderBDryFaces)

    geom.varias[:xi_var] = var_xi
    geom.varias[:xi_mat] = xi_mat

    geom.varias[:g_var]  = g_var
    geom.varias[:g_mat]  = g_mat

    geom.varias[:z_var]  = z_var
    geom.varias[:z_mat]  = z_mat

    geom.varias[:j_var]  = j_var
    geom.varias[:j_mat]  = j_mat

    return nothing
end

end # module