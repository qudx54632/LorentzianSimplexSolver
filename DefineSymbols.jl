module DefineSymbols

using ..SpinAlgebra: σ3
using ..CriticalPoints: compute_bdy_critical_data
using ..PrecisionUtils: get_tolerance

using PythonCall
sympy = pyimport("sympy")

const I        = sympy.I
const symbols  = sympy.symbols
const Matrix   = sympy.Matrix
const simplify = sympy.simplify

export run_define_variables, collect_bdry_symbols, collect_varias_symbols

# ------------------------------------------------------------
# g variables (flat var list + per-(simplex,tet) 2x2 matrix list)
# g_mat is length ns*5, ordered by a=1..ns, b=1..5
# ------------------------------------------------------------
@inline function find_position_in_chain(key::AbstractVector, chain)
    for (idx, v) in pairs(chain)
        v == key && return idx
    end
    return nothing
end

"""
    compute_gspecialpos(gdataof, GaugeTet; tol=1e-12)

Return list [k,i] where gdataof[k][i][1,1] ≈ 0 and not in GaugeTet.
"""
function compute_gspecialpos(gdataof, GaugeTet)
    tol = get_tolerance()
    ns   = length(gdataof)
    ntet = length(gdataof[1])

    gauge_set = Set(Tuple(p) for p in GaugeTet)
    gspecialpos = Vector{Vector{Int}}()

    for k in 1:ns, i in 1:ntet
        key = (k, i)
        key ∈ gauge_set && continue
        if abs(gdataof[k][i][1,1]) < tol
            push!(gspecialpos, [k, i])
        end
    end

    return gspecialpos
end

function build_g_variables(num_vertex::Int, GaugeTet::Vector{Vector{Int}}, gspecialpos::Vector{Vector{Int}}, GaugeFixUpperTriangle::Vector{Vector{Int}})
    g_var = Vector{Py}()                        # flat list of symbols
    g_bdry = Vector{Py}()     
    g_mat = Vector{Vector{Py}}(undef, num_vertex)  # g_mat[a][b] is 2x2 SymPy Matrix

    for a in 1:num_vertex
        g_mat[a] = Vector{Py}(undef, 5)

        for b in 1:5
            # append!(symbols_g, g1)
            pos_gauge    = find_position_in_chain([a,b], GaugeTet)
            pos_gspecial = find_position_in_chain([a,b], gspecialpos)
            pos_gupper   = find_position_in_chain([a,b], GaugeFixUpperTriangle)

            if pos_gauge === nothing && pos_gspecial === nothing && pos_gupper === nothing
                g1 = collect(symbols("g_$(a)$(b)_1:7", real=true))
                g_mat[a][b] = simplify(Matrix([1 + g1[1] + g1[2]*I      g1[3] + g1[4]*I;
                        g1[5] + g1[6]*I          (1 + g1[3]*g1[5] + I*g1[4]*g1[5] + I*g1[3]*g1[6] - g1[4]*g1[6]) /(1 + g1[1] + g1[2]*I)]))
                append!(g_var, g1)
            elseif pos_gauge !== nothing && pos_gspecial === nothing && pos_gupper === nothing
                g1 = collect(symbols("g_$(a)$(b)_1:7", real=true))
                g_mat[a][b] = simplify(Matrix([1 + g1[1] + g1[2]*I      g1[3] + g1[4]*I;
                        g1[5] + g1[6]*I          (1 + g1[3]*g1[5] + I*g1[4]*g1[5] + I*g1[3]*g1[6] - g1[4]*g1[6]) /(1 + g1[1] + g1[2]*I)]))
                append!(g_bdry, g1)
            elseif pos_gauge === nothing && pos_gspecial !== nothing && pos_gupper === nothing
                g1 = collect(symbols("g_$(a)$(b)_1:7", real=true))
                g_mat[a][b] = simplify(Matrix([1 + g1[1] + g1[2]*I      (-1 + g1[5] + g1[1]*g1[5] + I*g1[2]*g1[5] + I*g1[6] + I*g1[1]*g1[6] - g1[2]*g1[6]) /(g1[3] + g1[4]*I);
                        g1[3] + g1[4]*I          g1[5] + g1[6]*I]))
                append!(g_var, g1)
            elseif pos_gauge === nothing && pos_gspecial === nothing && pos_gupper !== nothing
                g1 = collect(symbols("g_$(a)$(b)_1:4", real=true))
                g_mat[a][b] = simplify(Matrix([1 + g1[1]      0;
                        g1[2] + g1[3]*I          1/(1 + g1[1])]))
                append!(g_var, g1)
            else 
                error("something wrong when define sl2c group variables!")
            end
        end
    end

    return g_var, g_bdry, g_mat
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

function compute_zspecialpos(zdataf, kappa)
    tol = get_tolerance()
    ns   = length(zdataf)
    ntet = length(zdataf[1])

    out = Vector{Vector{Int}}()

    for k in 1:ns, i in 1:ntet, j in 1:ntet
        if kappa[k][i][j] == 1 && i != j
            if abs(zdataf[k][i][j][1] - 1) > tol
                push!(out, [k, i, j])
            end
        end
    end

    return out
end

# ------------------------------------------------------------
# xi variables (zeta)
# sgndet[a][i], tetareasign[a][i][j], tetn0sign[a][i][j]
# xi_mat[a][i][j] is always Py[*,*] length 2, diagonal is [0,0]
# ------------------------------------------------------------
function apply_shared_tets_to_xi!(xi_expr, sharedTetsPos)
    ntet = length(xi_expr[1][1])  # should be 5

    for pair in sharedTetsPos
        # unpack ((s1,t1),(s2,t2))
        s1, t1 = pair[1]
        s2, t2 = pair[2]

        row_src = xi_expr[s1][t1]  # Vector{Py} of length ntet

        # remove the self-entry at t1
        row_wo_self = Vector{Py}()
        for j in 1:ntet
            j == t1 && continue
            push!(row_wo_self, row_src[j])
        end

        @assert length(row_wo_self) == ntet - 1

        # build destination row with [1,0] inserted at t2
        row_dst = Vector{Py}(undef, ntet)
        k = 1
        for j in 1:ntet
            if j == t2
                row_dst[j] = Matrix([0; 0])
            else
                row_dst[j] = row_wo_self[k]
                k += 1
            end
        end

        xi_expr[s2][t2] = row_dst
    end

    return xi_expr
end

function build_xi_variables(num_vertex::Int,
                            sgndet::Vector{Vector{Int}},
                            tetareasign::Vector{Vector{Vector{Int}}},
                            tetn0sign::Vector{Vector{Vector{Int}}}, sharedTetsPos)

    ntet = 5
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

                else
                    zb = symbols("zeta_$(a)_$(i)_$(j)_b", real=true)

                    xi_mat[a][i][j] = simplify(Matrix([
                        1;
                        sympy.exp(I * zb)
                    ]))
                end
            end
        end
    end
    if num_vertex > 1 && !isempty(sharedTetsPos)
        apply_shared_tets_to_xi!(xi_mat, sharedTetsPos)
    end
 
    return xi_mat
end

function extract_symbols(expr::Py)
    collect(sympy.Matrix(expr).free_symbols)
end

function split_xi_variables(xi_mat, timelikeTetsPos, Gaugespacelike, Gaugetimelike)

    # normalize lookup sets
    tl_set = Set(Tuple(p) for p in timelikeTetsPos)
    gauge_set = Set(Tuple(p) for p in Iterators.flatten(vcat(Gaugespacelike, Gaugetimelike)))

    var_xi    = Set{Py}()
    var_bdry  = Set{Py}()

    ns   = length(xi_mat)
    ntet = length(xi_mat[1])

    for k in 1:ns
        for i in 1:ntet
            # only shared timelike tetrahedra, some of the faces have the variables
            if (k,i) ∈ tl_set
                for j in 1:ntet
                    i == j && continue
                    syms = extract_symbols(xi_mat[k][i][j])
                    isempty(syms) && continue

                    if (k,i,j) ∈ gauge_set
                        union!(var_bdry, syms)
                    else
                        union!(var_xi, syms)
                    end
                end
            else
                for j in 1:ntet
                    i == j && continue

                    syms = extract_symbols(xi_mat[k][i][j])
                    isempty(syms) && continue
                    union!(var_bdry, syms)
                end
            end
        end
    end

    return collect(var_xi), collect(var_bdry)
end

# ------------------------------------------------------------
# j variables
# returns (j_var, j_mat) where j_var is unique flat symbol list
# ------------------------------------------------------------
@inline function find_chain_index(key::Vector{Int}, chains)
    for idx in eachindex(chains)
        for v in chains[idx]
            if v[1] == key[1] && v[2] == key[2] && v[3] == key[3]
                return idx
            end
        end
    end
    return nothing
end

function build_j_variables(num_vertex::Int,
                           OrderBulkFaces,
                           OrderBDryFaces;
                           ntet=5)

    j_mat = Vector{Vector{Vector{Py}}}(undef, num_vertex)
    j_var = Vector{Py}()  # flat list, unique later
    j_bdry = Vector{Py}() 

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
                push!(j_bdry, jsym)
            end
        end
    end

    j_var = collect(Set(j_var))
    j_bdry = collect(Set(j_bdry))
    return j_var, j_bdry, j_mat
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

    critical_data = compute_bdy_critical_data(geom)
    gdataof = critical_data.gdataof
    zdataf  = critical_data.zdataf

    if ns > 1
        sharedTetsPos = geom.connectivity[1]["sharedTetsPos"]
        # connectivity inputs for g/z/j variable-building
        GaugeTet              = geom.connectivity[1]["GaugeTet"]
        GaugeFixUpperTriangle = geom.connectivity[1]["GaugeFixUpperTriangle"]
        OrderBDryFaces        = geom.connectivity[1]["OrderBDryFaces"]
        OrderBulkFaces        = geom.connectivity[1]["OrderBulkFaces"]
        Gaugetimelike         = geom.connectivity[1]["gaugetimelike"]
        Gaugespacelike        = geom.connectivity[1]["gaugespacelike"]
        timelike_pairs       = geom.connectivity[1]["timelike_pairs"]

        timelikeTetsSharingPos = collect(Iterators.flatten(timelike_pairs))
    else
    # ---------- single simplex case ----------
        GaugeTet = [[1,1]]

        # IMPORTANT: typed empty vectors
        GaugeFixUpperTriangle = Vector{Vector{Int}}()
        OrderBulkFaces        = Vector{Vector{Vector{Int}}}()
        sharedTetsPos         = Vector{Vector{Vector{Int}}}()
        # every (1,i,j), i≠j is a boundary face with a trivial chain
        OrderBDryFaces = Vector{Vector{Vector{Int}}}()
        Gaugetimelike = Vector{Vector{Vector{Int}}}()
        Gaugespacelike = Vector{Vector{Vector{Int}}}()
        timelikeTetsSharingPos = Vector{Vector{Vector{Int}}}()

        for i in 1:ntet, j in i+1:ntet
            # two oriented faces
            fwd = [1, i, j]
            bwd = [1, j, i]

            # kappa-positive one goes first
            if kappa_all[1][i][j] == 1
                push!(OrderBDryFaces, [fwd, bwd])
            else
                push!(OrderBDryFaces, [bwd, fwd])
            end
        end
    end

    gspecialPos = compute_gspecialpos(gdataof, GaugeTet)
    zspecialPos = compute_zspecialpos(zdataf, kappa_all)

    xi_mat = build_xi_variables(ns, sgndet, tetareasign, tetn0sign, sharedTetsPos)
    if ns > 1
        apply_shared_tets_to_xi!(xi_mat, sharedTetsPos)
    end 
    xi_var, xi_bdry = split_xi_variables(xi_mat, timelikeTetsSharingPos, Gaugespacelike, Gaugetimelike)

    g_var, g_bdry, g_mat = build_g_variables(ns, GaugeTet, gspecialPos, GaugeFixUpperTriangle)
    z_var, z_mat = build_z_variables(ns, kappa_all, zspecialPos)
    j_var, j_bdry, j_mat = build_j_variables(ns, OrderBulkFaces, OrderBDryFaces)

    geom.varias[:xi_var] = xi_var
    geom.varias[:xi_bdry] = xi_bdry
    geom.varias[:xi_mat] = xi_mat

    geom.varias[:g_var]  = g_var
    geom.varias[:g_bdry]  = g_bdry
    geom.varias[:g_mat]  = g_mat

    geom.varias[:z_var]  = z_var
    geom.varias[:z_mat]  = z_mat

    geom.varias[:j_var]  = j_var
    geom.varias[:j_mat]  = j_mat
    geom.varias[:j_bdry] = j_bdry

    geom.varias[:gspecialPos] = gspecialPos
    geom.varias[:zspecialPos] = zspecialPos

    return nothing
end

# ------------------------------------------------------------
# symbol collectors
# ------------------------------------------------------------
function collect_bdry_symbols(geom)
    bdry_syms = Set{Py}()
    for key in (:xi_bdry, :g_bdry, :j_bdry)
        haskey(geom.varias, key) || continue
        union!(bdry_syms, geom.varias[key])
    end
    return bdry_syms
end

function collect_varias_symbols(geom)
    all_syms = Set{Py}()
    for key in (:xi_var, :g_var, :z_var, :j_var)
        haskey(geom.varias, key) || continue
        union!(all_syms, geom.varias[key])
    end
    return all_syms
end

end # module