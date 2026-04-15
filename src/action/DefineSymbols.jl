module DefineSymbols

using ..CriticalPoints: compute_bdy_critical_data
using ..PrecisionUtils: get_tolerance

using SymEngine

export run_define_variables, collect_bdry_symbols, collect_varias_symbols

# ------------------------------------------------------------
# Symbol helpers (Julia native)
# ------------------------------------------------------------
@inline function make_symbol(name::String)
    return symbols(name)
end

@inline function make_symbols(prefix::String, n::Int)
    return [make_symbol("$(prefix)_$i") for i in 1:n]
end

const _im = Ref{Union{Basic,Nothing}}(nothing)

function __init__()
    _im[] = SymEngine.Basic("I")
end

@inline sim(x) = _im[] * x

# ------------------------------------------------------------
# g variables (flat var list + per-(simplex,tet) 2x2 matrix list)
# g_mat is length ns*5, ordered by a=1..ns, b=1..5
# ------------------------------------------------------------
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

function build_g_variables(
    num_vertex::Int,
    GaugeTet::Vector{Vector{Int}},
    gspecialpos::Vector{Vector{Int}},
    GaugeFixUpperTriangle::Vector{Vector{Int}}
)
    gauge_set    = Set((x[1], x[2]) for x in GaugeTet)
    gspecial_set = Set((x[1], x[2]) for x in gspecialpos)
    gupper_set   = Set((x[1], x[2]) for x in GaugeFixUpperTriangle)

    g_var = Basic[]
    g_bdry = Basic[]
    g_mat = [Vector{Matrix{Basic}}(undef, 5) for _ in 1:num_vertex]

    for a in 1:num_vertex
        for b in 1:5
            key = (a, b)

            in_gauge    = key in gauge_set
            in_gspecial = key in gspecial_set
            in_gupper   = key in gupper_set

            if !in_gauge && !in_gspecial && !in_gupper
                g1 = make_symbols("g_$(a)$(b)", 6)
                g_mat[a][b] = Basic[
                    1 + g1[1] + sim(g1[2])     g1[3] + sim(g1[4]);
                    g1[5] + sim(g1[6])         (1 + g1[3]*g1[5] + sim(g1[4]*g1[5]) + sim(g1[3]*g1[6]) - g1[4]*g1[6])/(1 + g1[1] + sim(g1[2]))
                ]
                append!(g_var, g1)

            elseif in_gauge && !in_gspecial && !in_gupper
                g1 = make_symbols("g_$(a)$(b)", 8)
                g_mat[a][b] = Basic[
                    g1[1] + sim(g1[2])   g1[3] + sim(g1[4]);
                    g1[5] + sim(g1[6])   g1[7] + sim(g1[8])
                ]
                append!(g_bdry, g1)

            elseif !in_gauge && in_gspecial && !in_gupper
                g1 = make_symbols("g_$(a)$(b)", 6)
                g_mat[a][b] = Basic[
                    1 + g1[1] + sim(g1[2])   (-1 + g1[5] + g1[1]*g1[5] + sim(g1[2]*g1[5]) + sim(g1[6]) + sim(g1[1]*g1[6]) - g1[2]*g1[6])/(g1[3] + sim(g1[4]));
                    g1[3] + sim(g1[4])       g1[5] + sim(g1[6])
                ]
                append!(g_var, g1)

            elseif !in_gauge && !in_gspecial && in_gupper
                g1 = make_symbols("g_$(a)$(b)", 3)
                g_mat[a][b] = Basic[
                    1 + g1[1]    0;
                    g1[2] + sim(g1[3])   1/(1 + g1[1])
                ]
                append!(g_var, g1)

            else
                error("something wrong when define sl2c group variables! key = $key")
            end
        end
    end

    return g_var, g_bdry, g_mat
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

function build_z_variables(
    num_vertex::Int,
    kappa_all,
    zspecialPos::Vector{Vector{Int}}
)
    ntet = 5

    # fast lookup set
    special_set = Set((p[1], p[2], p[3]) for p in zspecialPos)

    var_z = Basic[]

    # allocate structure
    z_mat = [ [ Vector{Vector{Basic}}(undef, ntet) for _ in 1:ntet ]
              for _ in 1:num_vertex ]

    # define zero once (avoid repeated calls)
    z0 = zero(Basic)

    for a in 1:num_vertex
        for i in 1:ntet
            for j in 1:ntet

                # skip invalid entries
                if i == j || kappa_all[a][i][j] != 1
                    z_mat[a][i][j] = Basic[z0, z0]
                    continue
                end

                # canonical ordering
                ii, jj = i < j ? (i, j) : (j, i)

                # create symbols
                z  = make_symbol("z_$(a)$(ii)$(jj)")
                zc = make_symbol("zc_$(a)$(ii)$(jj)")

                push!(var_z, z)
                push!(var_z, zc)

                # complex combination
                zcplx = 1 + z + sim(zc)

                key = (a, i, j)

                if key in special_set
                    z_mat[a][i][j] = Basic[zcplx, 1]
                else
                    z_mat[a][i][j] = Basic[1, zcplx]
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
function apply_shared_tets_to_xi!(xi_expr, sharedTetsPos)
    ntet = length(xi_expr[1][1])  # should be 5

    for pair in sharedTetsPos
        # unpack ((s1,t1),(s2,t2))
        s1, t1 = pair[1]
        s2, t2 = pair[2]

        row_src = xi_expr[s1][t1]  

        # remove the self-entry at t1
        row_wo_self = Vector{Vector{Basic}}()
        for j in 1:ntet
            j == t1 && continue
            push!(row_wo_self, row_src[j])
        end

        @assert length(row_wo_self) == ntet - 1

        # build destination row with [1,0] inserted at t2
        row_dst = Vector{Vector{Basic}}(undef, ntet)
        k = 1
        for j in 1:ntet
            if j == t2
                row_dst[j] = [zero(Basic), zero(Basic)]
            else
                row_dst[j] = row_wo_self[k]
                k += 1
            end
        end

        xi_expr[s2][t2] = row_dst
    end

    return xi_expr
end

function extract_symbols(expr)
    vars = Set{Basic}()

    if isa(expr, Basic)
        union!(vars, free_symbols(expr))
    elseif isa(expr, AbstractArray)
        for e in expr
            isa(e, Basic) || continue
            union!(vars, free_symbols(e))
        end
    end

    return vars
end

function build_xi_variables(
    num_vertex::Int,
    sgndet::Vector{Vector{Int}},
    tetareasign::Vector{Vector{Vector{Int}}},
    tetn0sign::Vector{Vector{Vector{Int}}},
    sharedTetsPos
)
    ntet = 5

    xi_mat = [ [ Vector{Vector{Basic}}(undef, ntet) for _ in 1:ntet ]
               for _ in 1:num_vertex ]

    z0 = zero(Basic)

    # cache symbols to avoid duplication
    symbol_cache = Dict{Tuple{Int,Int,Int,Symbol}, Basic}()

    function get_symbol(a,i,j,tag::Symbol)
        key = (a,i,j,tag)
        if haskey(symbol_cache, key)
            return symbol_cache[key]
        else
            s = make_symbol("zeta_$(a)$(i)$(j)$(tag)")
            symbol_cache[key] = s
            return s
        end
    end

    for a in 1:num_vertex
        for i in 1:ntet
            for j in 1:ntet

                if i == j
                    xi_mat[a][i][j] = Basic[z0, z0]
                    continue
                end

                if sgndet[a][i] == 1
                    za = get_symbol(a,i,j,:a)
                    zb = get_symbol(a,i,j,:b)

                    xi_mat[a][i][j] = Basic[
                        sin(za),
                        cos(za) * exp(sim(zb))
                    ]

                elseif tetareasign[a][i][j] == 1
                    za = get_symbol(a,i,j,:a)
                    zb = get_symbol(a,i,j,:b)

                    if tetn0sign[a][i][j] == 1
                        xi_mat[a][i][j] = Basic[
                            cosh(za),
                            exp(-sim(zb)) * sinh(za)
                        ]
                    else
                        xi_mat[a][i][j] = Basic[
                            sinh(za) * exp(sim(zb)),
                            cosh(za)
                        ]
                    end

                else
                    zb = get_symbol(a,i,j,:b)

                    xi_mat[a][i][j] = Basic[
                        1,
                        exp(sim(zb))
                    ]
                end
            end
        end
    end

    if num_vertex > 1 && !isempty(sharedTetsPos)
        apply_shared_tets_to_xi!(xi_mat, sharedTetsPos)
    end

    return xi_mat
end

function split_xi_variables(
    xi_mat,
    timelikeTetsPos,
    Gaugespacelike,
    Gaugetimelike
)

    tl_set = Set(Tuple(p) for p in timelikeTetsPos)

    gauge_set = Set(
        Tuple(p) for p in Iterators.flatten(vcat(Gaugespacelike, Gaugetimelike))
    )

    var_xi   = Set{Basic}()
    var_bdry = Set{Basic}()

    ns   = length(xi_mat)
    ntet = length(xi_mat[1])

    for k in 1:ns
        for i in 1:ntet

            is_timelike = (k,i) in tl_set

            for j in 1:ntet
                i == j && continue

                syms = extract_symbols(xi_mat[k][i][j])
                isempty(syms) && continue

                if is_timelike
                    if (k,i,j) in gauge_set
                        union!(var_bdry, syms)
                    else
                        union!(var_xi, syms)
                    end
                else
                    union!(var_bdry, syms)
                end
            end
        end
    end

    return sort!(collect(var_xi), by=string),
           sort!(collect(var_bdry), by=string)
end

# ------------------------------------------------------------
# j variables
# returns (j_var, j_mat) where j_var is unique flat symbol list
# ------------------------------------------------------------
function build_j_variables(
    num_vertex::Int,
    OrderBulkFaces,
    OrderBDryFaces;
    ntet=5
)

    # --------------------------------------------------
    # Precompute lookup dictionaries
    # --------------------------------------------------
    bulk_dict = Dict{Tuple{Int,Int,Int}, Tuple{Int,Int,Int}}()
    for chain in OrderBulkFaces
        rep = Tuple(chain[1])
        for v in chain
            bulk_dict[Tuple(v)] = rep
        end
    end

    bdry_dict = Dict{Tuple{Int,Int,Int}, Tuple{Int,Int,Int}}()
    for chain in OrderBDryFaces
        rep = Tuple(chain[1])
        for v in chain
            bdry_dict[Tuple(v)] = rep
        end
    end

    # --------------------------------------------------
    # Symbol cache
    # --------------------------------------------------
    symbol_cache = Dict{Tuple{Int,Int,Int}, Basic}()

    function get_symbol(a,b,c)
        key = (a,b,c)
        if haskey(symbol_cache, key)
            return symbol_cache[key]
        else
            s = make_symbol("j_$(a)$(b)$(c)")
            symbol_cache[key] = s
            return s
        end
    end

    # --------------------------------------------------
    # Build ordered bulk j_var directly from OrderBulkFaces
    # --------------------------------------------------
    j_var = Basic[]
    for chain in OrderBulkFaces
        a,b,c = chain[1]
        push!(j_var, get_symbol(a,b,c))
    end

    # --------------------------------------------------
    # Boundary j's still collected as set
    # --------------------------------------------------
    j_bdry_set = Set{Basic}()

    # --------------------------------------------------
    # Allocate j_mat
    # --------------------------------------------------
    j_mat = [ [ Vector{Basic}(undef, ntet) for _ in 1:ntet ]
              for _ in 1:num_vertex ]

    z0 = zero(Basic)

    # --------------------------------------------------
    # Fill j_mat
    # --------------------------------------------------
    for k in 1:num_vertex
        for i in 1:ntet
            for j in 1:ntet

                if i == j
                    j_mat[k][i][j] = z0
                    continue
                end

                key = (k,i,j)

                if haskey(bulk_dict, key)
                    a,b,c = bulk_dict[key]
                    j_mat[k][i][j] = get_symbol(a,b,c)

                else
                    # @assert haskey(bdry_dict, key)
                    if !haskey(bdry_dict, key)
                        println("❌ Missing key detected:")
                        println("  simplex k = ", k)
                        println("  tetrahedra pair (i,j) = ", (i,j))
                        println("  full key = ", key)

                        println("  In bulk_dict? ", haskey(bulk_dict, key))
                        println("  In bdry_dict? ", haskey(bdry_dict, key))

                        error("Key not found in either bulk_dict or bdry_dict")
                    end

                    a,b,c = bdry_dict[key]
                    jsym = get_symbol(a,b,c)

                    j_mat[k][i][j] = jsym
                    push!(j_bdry_set, jsym)
                end
            end
        end
    end

    j_bdry = sort!(collect(j_bdry_set), by=string)

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
    bdry_syms = Set{Basic}()

    for key in (:xi_bdry, :g_bdry, :j_bdry)
        haskey(geom.varias, key) || continue

        vals = geom.varias[key]
        @inbounds for v in vals
            push!(bdry_syms, v)
        end
    end

    return bdry_syms
end

function collect_varias_symbols(geom)
    all_syms = Set{Basic}()

    for key in (:xi_var, :g_var, :z_var, :j_var)
        haskey(geom.varias, key) || continue

        vals = geom.varias[key]
        @inbounds for v in vals
            push!(all_syms, v)
        end
    end

    return all_syms
end

end # module