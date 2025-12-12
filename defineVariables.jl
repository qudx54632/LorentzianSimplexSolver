module DefineVariables

using LinearAlgebra
using ..CriticalPoints: compute_critical_data

# ============================================================
# 1) Build symbolic ξ-expressions (as VECTORS, not Tuples)
# ============================================================
#
# xi_expr[k][i][j] is ALWAYS a Vector{Any} of length 2:
#     xi_expr[k][i][j] = [expr1, expr2]
#
# Diagonal i == j is [0,0].
#
function build_xi_expressions(sgndet, tetareasign, tetn0, ns)

    # Allocate xi_expr[ns][5][5]
    xi_expr = Vector{Vector{Vector{Any}}}(undef, ns)
    for k in 1:ns
        xi_expr[k] = Vector{Vector{Any}}(undef, 5)
        for i in 1:5
            xi_expr[k][i] = Vector{Any}(undef, 5)
        end
    end

    # Fill entries
    for k in 1:ns, i in 1:5, j in 1:5

        # diagonal
        if i == j
            xi_expr[k][i][j] = Any[0, 0]     # VECTOR (not tuple)
            continue
        end

        # symbolic names ζ_{k,i,j,a}, ζ_{k,i,j,b}
        a = Symbol("zeta_$(k)_$(i)_$(j)_a")
        b = Symbol("zeta_$(k)_$(i)_$(j)_b")

        # spacelike simplex: SU(2)-type parametrization
        if sgndet[k][i] == 1
            xi_expr[k][i][j] = Any[
                :(sin($a)),
                :(cos($a) * exp(im*$b))
            ]

        # timelike face: SU(1,1)-type parametrization
        elseif tetareasign[k][i][j] == 1
            if tetn0[k][i][j] == 1
                xi_expr[k][i][j] = Any[
                    :(cosh($a)),
                    :(exp(-im*$b) * sinh($a))
                ]
            else
                xi_expr[k][i][j] = Any[
                    :(sinh($a) * exp(im*$b)),
                    :(cosh($a))
                ]
            end

        # null/timelike default
        else
            xi_expr[k][i][j] = Any[
                1,
                :(exp(im*$b))
            ]
        end
    end

    return xi_expr
end

# ============================================================
# 2) Impose shared-tetrahedron identification on ξ
# ============================================================
#
# For each shared pair ((s1,t1),(s2,t2)):
#   xi_expr[s2][t2] = Insert( Delete(xi_expr[s1][t1], t1), [1,0], t2 )
#
function apply_shared_tets_to_xi!(xi_expr, sharedTetsPos)
    ns   = length(xi_expr)
    ntet = length(xi_expr[1])   # expected = 5

    for pair in sharedTetsPos
        s1 = pair[1][1]
        t1 = pair[1][2]
        s2 = pair[2][1]
        t2 = pair[2][2]

        @assert 1 ≤ s1 ≤ ns && 1 ≤ s2 ≤ ns
        @assert 1 ≤ t1 ≤ ntet && 1 ≤ t2 ≤ ntet

        # row_src is length-ntet vector: row_src[j] is a 2-vector [..,..]
        row_src = xi_expr[s1][t1]

        # Delete at position t1
        row_wo_self = Any[]
        for j in 1:ntet
            j == t1 && continue
            push!(row_wo_self, row_src[j])
        end

        # Insert [1,0] at position t2
        row_dst = Any[]
        for j in 1:ntet
            if j == t2
                push!(row_dst, Any[1, 0])   # VECTOR (not tuple)
            elseif j < t2
                push!(row_dst, row_wo_self[j])
            else
                push!(row_dst, row_wo_self[j-1])
            end
        end

        xi_expr[s2][t2] = row_dst
    end

    return xi_expr
end

# ============================================================
# 3) Collect symbols from nested Expr / Arrays
# ============================================================
#
# This is used to build the "varia" lists like Mathematica's Variables[...]
#
function _collect_syms!(vars::Vector{Symbol}, x)
    if x isa Symbol
        # ignore builtins/constants
        if !(x in (:sin, :cos, :sinh, :cosh, :exp, :im, :*, :^, :/, :+, :-, :π, :E))
            push!(vars, x)
        end
    elseif x isa Expr
        for arg in x.args
            _collect_syms!(vars, arg)
        end
    elseif x isa Tuple || x isa AbstractArray
        for elem in x
            _collect_syms!(vars, elem)
        end
    end
end

"""
    build_zetavariables(xi_expr)

Given xi_expr[k][i][j] = [expr1, expr2], return zetavars[k][i][j] = sorted unique zeta symbols.
"""
function build_zetavariables(xi_expr)
    ns   = length(xi_expr)
    ntet = length(xi_expr[1])

    zetavars = Vector{Vector{Vector{Vector{Symbol}}}}(undef, ns)

    for k in 1:ns
        zetavars[k] = Vector{Vector{Vector{Symbol}}}(undef, ntet)
        for i in 1:ntet
            zetavars[k][i] = Vector{Vector{Symbol}}(undef, ntet)
            for j in 1:ntet
                vars = Symbol[]
                x = xi_expr[k][i][j]
                _collect_syms!(vars, x)
                zetavars[k][i][j] = sort(unique(vars))
            end
        end
    end

    return zetavars
end

"""
    build_zetadata(xisoln, zetavars, tetareasign)

Return Dict{Symbol,Float64} giving ζ-values at the critical point.
- If tetareasign==1: use both entries of xisoln[k][i][j]
- Else: use only the first entry (like MMA logic)
"""
function build_zetadata(xisoln, zetavars, tetareasign)
    ns   = length(xisoln)
    ntet = length(xisoln[1])

    zeta_data = Dict{Symbol,Float64}()

    for k in 1:ns
        for i in 1:ntet
            for j in 1:ntet
                if i == j
                    continue
                end

                vars = zetavars[k][i][j]
                isempty(vars) && continue

                if tetareasign[k][i][j] == 1
                    vals = xisoln[k][i][j]
                else
                    vals = [xisoln[k][i][j][1]]
                end

                @assert length(vars) == length(vals) "Mismatch vars vs vals at (k,i,j) = ($k,$i,$j)"

                for (sym, val) in zip(vars, vals)
                    zeta_data[sym] = val
                end
            end
        end
    end

    return zeta_data
end

# ============================================================
# 4) g-special and z-special positions (Mathematica parity logic)
# ============================================================

"""
    compute_gspecialpos(gdataof, GaugeTet; tol=1e-12)

Return list [k,i] where gdataof[k][i][1,1] ≈ 0 and not in GaugeTet.
"""
function compute_gspecialpos(gdataof, GaugeTet; tol=1e-12)
    ns   = length(gdataof)
    ntet = length(gdataof[1])

    gauge_set = Set{Vector{Int}}()
    for p in GaugeTet
        push!(gauge_set, [p[1], p[2]])
    end

    gspecialpos = Vector{Vector{Int}}()

    for k in 1:ns
        for i in 1:ntet
            key = [k, i]
            if key ∈ gauge_set
                continue
            end
            if abs(gdataof[k][i][1,1]) < tol
                push!(gspecialpos, key)
            end
        end
    end

    return gspecialpos
end

"""
    compute_zspecialpos(zdataf, kappa; tol=1e-12)

Return list [k,i,j] where kappa==1, i≠j, and zdataf[k][i][j][1] is not ≈ 1.
"""
function compute_zspecialpos(zdataf, kappa; tol=1e-12)
    ns   = length(zdataf)
    ntet = length(zdataf[1])

    out = Vector{Vector{Int}}()

    for k in 1:ns
        for i in 1:ntet
            for j in 1:ntet
                if kappa[k][i][j] == 1 && i != j
                    z1 = zdataf[k][i][j][1]
                    if !(abs(z1 - 1) < tol)
                        push!(out, [k, i, j])
                    end
                end
            end
        end
    end

    return out
end

# ============================================================
# 5) g-variables
# ============================================================

# det([[a b];[c d]])==1 => d = (1 + b*c)/a
solve_det_for_d(a, b, c) = :((1 + $b*$c) / $a)

@inline function in_pos2(pos::Vector{Int}, lst::Vector{Vector{Int}})
    @inbounds for q in lst
        if q[1] == pos[1] && q[2] == pos[2]
            return true
        end
    end
    return false
end

"""
    build_gvariablesall(ns, ntet, GaugeTet, gspecialpos, GaugeFixUpperTriangle)

Build symbolic 2x2 matrices for g-variables (each entry is Expr/Number).
Matches your Mathematica branching:
1) GaugeTet: full 8 real params (Ga,Gac,Gb,Gbc,Gc,Gcc,Gd,Gdc)
2) gspecialpos: det=1 solved for d
3) GaugeFixUpperTriangle: triangular + det=1 solve
4) generic: det=1 solved for d
"""
function build_gvariablesall(ns::Int, ntet::Int,
                             GaugeTet::Vector{Vector{Int}},
                             gspecialpos::Vector{Vector{Int}},
                             GaugeFixUpperTriangle::Vector{Vector{Int}})

    gvars = Vector{Vector{Matrix{Any}}}(undef, ns)

    for k in 1:ns
        gvars[k] = Vector{Matrix{Any}}(undef, ntet)

        for i in 1:ntet
            pos = [k, i]

            if in_pos2(pos, GaugeTet)
                Ga  = Symbol("Ga$(k)_$(i)")
                Gac = Symbol("Gac$(k)_$(i)")
                Gb  = Symbol("Gb$(k)_$(i)")
                Gbc = Symbol("Gbc$(k)_$(i)")
                Gc  = Symbol("Gc$(k)_$(i)")
                Gcc = Symbol("Gcc$(k)_$(i)")
                Gd  = Symbol("Gd$(k)_$(i)")
                Gdc = Symbol("Gdc$(k)_$(i)")

                gvars[k][i] = Any[
                    :($Ga + im*$Gac)   :($Gb + im*$Gbc);
                    :($Gc + im*$Gcc)   :($Gd + im*$Gdc)
                ]

            elseif in_pos2(pos, gspecialpos)
                ga  = Symbol("ga$(k)_$(i)")
                gac = Symbol("gac$(k)_$(i)")
                gb  = Symbol("gb$(k)_$(i)")
                gbc = Symbol("gbc$(k)_$(i)")
                gc  = Symbol("gc$(k)_$(i)")
                gcc = Symbol("gcc$(k)_$(i)")

                a = :(1 + $ga + im*$gac)
                b = :($gb + im*$gbc)
                c = :($gc + im*$gcc)
                d = solve_det_for_d(a, b, c)

                gvars[k][i] = Any[a b; c d]

            elseif in_pos2(pos, GaugeFixUpperTriangle)
                ga  = Symbol("ga$(k)_$(i)")
                gc  = Symbol("gc$(k)_$(i)")
                gcc = Symbol("gcc$(k)_$(i)")

                a = :(1 + $ga)
                b = 0
                c = :($gc + im*$gcc)
                d = solve_det_for_d(a, b, c)

                gvars[k][i] = Any[a 0; c d]

            else
                ga  = Symbol("ga$(k)_$(i)")
                gac = Symbol("gac$(k)_$(i)")
                gb  = Symbol("gb$(k)_$(i)")
                gbc = Symbol("gbc$(k)_$(i)")
                gc  = Symbol("gc$(k)_$(i)")
                gcc = Symbol("gcc$(k)_$(i)")

                a = :(1 + $ga + im*$gac)
                b = :($gb + im*$gbc)
                c = :($gc + im*$gcc)
                d = solve_det_for_d(a, b, c)

                gvars[k][i] = Any[a b; c d]
            end
        end
    end

    return gvars
end

# ============================================================
# 6) z-variables
# ============================================================

"""
    build_zvariablesall(ns, ntet, kappa, zspecialPos)

Each zvars[k][i][j] is a Vector{Any} of length 2:
- [0,0] if not active
- [1, :(z + im*zc)] or [:(z + im*zc), 1] depending on special_set
"""
function build_zvariablesall(ns, ntet, kappa, zspecialPos)
    special_set = Set{NTuple{3,Int}}((p[1],p[2],p[3]) for p in zspecialPos)

    zvars = Vector{Vector{Vector{Vector{Any}}}}(undef, ns)

    for k in 1:ns
        zvars[k] = Vector{Vector{Vector{Any}}}(undef, ntet)
        for i in 1:ntet
            zvars[k][i] = Vector{Vector{Any}}(undef, ntet)

            for j in 1:ntet
                if kappa[k][i][j] == 1 && i != j
                    z  = Symbol("z$(k)_$(i)$(j)")
                    zc = Symbol("zc$(k)_$(i)$(j)")
                    key = (k,i,j)

                    if key in special_set
                        zvars[k][i][j] = Any[:($z + im*$zc), 1]
                    else
                        zvars[k][i][j] = Any[1, :($z + im*$zc)]
                    end
                else
                    zvars[k][i][j] = Any[0,0]
                end
            end
        end
    end

    return zvars
end

# ============================================================
# 7) j-labels (use FIRST element of the chain as label)
# ============================================================

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

"""
    build_jlabels(OrderBulkFaces, OrderBDryFaces, ns; ntet=5)

Build jlabels[k][i][j] where each non-diagonal face uses a unique label
defined by the FIRST element in its chain (bulk chain or bdry chain).
"""
function build_jlabels(OrderBulkFaces, OrderBDryFaces, ns; ntet=5)

    jlabels = Vector{Vector{Vector{Any}}}(undef, ns)

    for k in 1:ns
        jlabels[k] = Vector{Vector{Any}}(undef, ntet)

        for i in 1:ntet
            jlabels[k][i] = Vector{Any}(undef, ntet)

            for j in 1:ntet
                if i == j
                    jlabels[k][i][j] = 0
                    continue
                end

                key = [k,i,j]

                # Bulk faces: use FIRST element of its chain
                pos = find_chain_index(key, OrderBulkFaces)
                if pos !== nothing
                    a,b,c = OrderBulkFaces[pos][1]
                    jlabels[k][i][j] = Symbol("j$(a)_$(b)_$(c)")
                    continue
                end

                # Boundary faces: same rule
                pos = find_chain_index(key, OrderBDryFaces)
                @assert pos !== nothing

                a,b,c = OrderBDryFaces[pos][1]
                jlabels[k][i][j] = Symbol("j$(a)_$(b)_$(c)")
            end
        end
    end

    return jlabels
end

# ============================================================
# 8) Collect symbols wrapper
# ============================================================

function collect_symbols_any(x)
    vars = Symbol[]
    _collect_syms!(vars, x)
    return unique(vars)
end

# ============================================================
# 9) g-solution extraction helpers
# ============================================================

# Flatten nested x + y + z into a vector of terms
function _plus_terms(ex)
    if ex isa Expr && ex.head == :call && ex.args[1] == :+
        out = Any[]
        for t in ex.args[2:end]
            append!(out, _plus_terms(t))
        end
        return out
    else
        return Any[ex]
    end
end

# Return sym::Symbol if ex is im*sym or sym*im, else nothing
function _im_times_symbol(ex)
    if ex isa Expr && ex.head == :call && ex.args[1] == :*
        a, b = ex.args[2], ex.args[3]
        if a == :im && b isa Symbol
            return b
        elseif b == :im && a isa Symbol
            return a
        end
    end
    return nothing
end

"""
    extract_from_entry!(sol, expr, val)

Given one symbolic entry `expr` (Symbol/Expr/Number) and one complex value `val`,
extract the corresponding real variables:
- Symbol => assign real(val)
- (const + sym + im*sym) => solve sym and imag sym
- im*sym => assign imag(val)
Other derived expressions are ignored.
"""
function extract_from_entry!(sol::Dict{Symbol,Float64}, expr, val::ComplexF64)
    expr isa Number && return

    # pure symbol -> real part
    if expr isa Symbol
        sol[expr] = real(val)
        return
    end

    # sums like 1 + ga + im*gac (order-independent)
    if expr isa Expr && expr.head == :call && expr.args[1] == :+
        terms = _plus_terms(expr)

        real_const = 0.0
        real_syms  = Symbol[]
        imag_syms  = Symbol[]

        for t in terms
            if t isa Number
                real_const += float(t)
            elseif t isa Symbol
                push!(real_syms, t)
            else
                s = _im_times_symbol(t)
                s === nothing || push!(imag_syms, s)
            end
        end

        # common case: const + 1 real symbol + 1 imag symbol
        if length(real_syms) == 1 && length(imag_syms) == 1
            sol[real_syms[1]] = real(val) - real_const
            sol[imag_syms[1]] = imag(val)
        end
        return
    end

    # pure im*sym
    if expr isa Expr
        s = _im_times_symbol(expr)
        if s !== nothing
            sol[s] = imag(val)
        end
    end

    return
end

"""
    extract_gsoln(gvariablesall, gdataof)

Traverse all 2x2 entries and extract solvable variable assignments into Dict{Symbol,Float64}.
"""
function extract_gsoln(gvariablesall, gdataof)
    sol = Dict{Symbol,Float64}()

    ns   = length(gvariablesall)
    ntet = length(gvariablesall[1])

    for k in 1:ns
        for i in 1:ntet
            Gsym = gvariablesall[k][i]
            Gnum = gdataof[k][i]

            for a in 1:2, b in 1:2
                extract_from_entry!(sol, Gsym[a,b], Gnum[a,b])
            end
        end
    end

    return sol
end

# ============================================================
# 10) z-solution extraction (direct, no Solve)
# ============================================================

"""
    extract_zsoln(zvariablesall, zdataf)

For each active z-variable:
- identify which slot is the complex variable (the other slot is 1)
- extract real(z) and imag(z) into Dict{Symbol,Float64}
"""
function extract_zsoln(zvariablesall, zdataf)
    ns   = length(zvariablesall)
    ntet = length(zvariablesall[1])

    zsoln = Dict{Symbol,Float64}()

    for k in 1:ns, i in 1:ntet, j in 1:ntet
        v = zvariablesall[k][i][j]

        # skip [0,0]
        (v[1] == 0 && v[2] == 0) && continue

        # pick complex value and symbolic expression
        if v[1] == 1
            zcplx = zdataf[k][i][j][2]
            ex    = v[2]     # :(z + im*zc)
        else
            zcplx = zdataf[k][i][j][1]
            ex    = v[1]     # :(z + im*zc)
        end

        # ex = :(zsym + im*zcsym)
        zsym  = ex.args[2]
        zcsym = ex.args[3].args[3]

        zsoln[zsym]  = real(zcplx)
        zsoln[zcsym] = imag(zcplx)
    end

    return zsoln
end

# ============================================================
# 11) j-solution extraction
# ============================================================

"""
    extract_jsoln(jlabels, jdataf)

Assign jsoln[label] = jdataf[k][i][j] at first appearance only.
This matches the idea that all equal labels must share one value.
"""
function extract_jsoln(jlabels, jdataf)

    jsoln = Dict{Symbol,Float64}()

    ns   = length(jdataf)
    ntet = length(jdataf[1])

    for k in 1:ns, i in 1:ntet, j in 1:ntet
        lbl = jlabels[k][i][j]

        # skip diagonal / zero entries
        lbl == 0 && continue

        # first appearance fixes the variable
        if !haskey(jsoln, lbl)
            jsoln[lbl] = jdataf[k][i][j]
        end
    end

    return jsoln
end

# ============================================================
# 12) Main runner: build everything + compute critical solutions
# ============================================================
#
# Returns a NamedTuple containing EVERYTHING useful later:
# - symbolic objects: xi_expr, zetavars, gvariablesall, zvariablesall, jvariablesall
# - variable lists:   xivaria, zetavaria, gvaria, zvaria, jvaria
# - numeric critical data: xisoln, gdataof, zdataf, jdataf
# - extracted solutions: gsoln, zsoln, jsoln
# - zeta_data (Dict for ζ)
#
function run_define_variables(geom)

    ns   = length(geom.simplex)
    ntet = length(geom.simplex[1].bdyxi)

    # per-simplex kappa sign table
    kappa = [geom.simplex[s].kappa for s in 1:ns]

    # geometry inputs used to build ξ-expressions
    sgndet      = [geom.simplex[s].sgndet      for s in 1:ns]
    tetareasign = [geom.simplex[s].tetareasign for s in 1:ns]
    tetn0       = [geom.simplex[s].tetn0sign   for s in 1:ns]

    # 1) symbolic ξ(zeta)
    xi_expr = build_xi_expressions(sgndet, tetareasign, tetn0, ns)

    if ns > 1
        sharedTetsPos = geom.connectivity[1]["sharedTetsPos"]
        apply_shared_tets_to_xi!(xi_expr, sharedTetsPos)
        # connectivity inputs for g/z/j variable-building
        GaugeTet              = geom.connectivity[1]["GaugeTet"]
        GaugeFixUpperTriangle = geom.connectivity[1]["GaugeFixUpperTriangle"]
        OrderBDryFaces        = geom.connectivity[1]["OrderBDryFaces"]
        OrderBulkFaces        = geom.connectivity[1]["OrderBulkFaces"]
    else
    # ---------- single simplex case ----------
        GaugeTet = [[1,1]]

        # IMPORTANT: typed empty vectors
        GaugeFixUpperTriangle = Vector{Vector{Int}}()
        OrderBulkFaces        = Vector{Vector{Vector{Int}}}()

        # every (1,i,j), i≠j is a boundary face with a trivial chain
        OrderBDryFaces = Vector{Vector{Vector{Int}}}()

        for i in 1:ntet
            for j in i+1:ntet
                # two oriented faces
                fwd = [1, i, j]
                bwd = [1, j, i]

                # kappa-positive one goes first
                if kappa[1][i][j] == 1
                    push!(OrderBDryFaces, [fwd, bwd])
                else
                    push!(OrderBDryFaces, [bwd, fwd])
                end
            end
        end
    end

    # 2) ζ variables extracted from ξ
    zetavars = build_zetavariables(xi_expr)

    # 3) critical data (numerical)
    crit    = compute_critical_data(geom; gamma = 1)
    xisoln  = crit.xisoln
    gdataof = crit.gdataof
    zdataf  = crit.zdataf
    jdataf  = crit.jdataf

    # 4) ζ numeric solution from xisoln
    zeta_data = build_zetadata(xisoln, zetavars, tetareasign)

    # 5) find special positions
    gspecialpos = compute_gspecialpos(gdataof, GaugeTet)
    zspecialPos = compute_zspecialpos(zdataf, kappa)

    # 6) build all symbolic variables
    gvariablesall = build_gvariablesall(ns, ntet, GaugeTet, gspecialpos, GaugeFixUpperTriangle)
    zvariablesall = build_zvariablesall(ns, ntet, kappa, zspecialPos)
    jvariablesall = build_jlabels(OrderBulkFaces, OrderBDryFaces, ns; ntet=5)

    # 7) variable lists (Mathematica-style "varia")
    xivaria   = collect_symbols_any(xi_expr)
    zetavaria = collect_symbols_any(zetavars)
    gvaria    = collect_symbols_any(gvariablesall)
    zvaria    = collect_symbols_any(zvariablesall)
    jvaria    = collect_symbols_any(jvariablesall)

    # 8) extracted solutions (Dict{Symbol,Float64})
    gsoln = extract_gsoln(gvariablesall, gdataof)
    zsoln = extract_zsoln(zvariablesall, zdataf)
    jsoln = extract_jsoln(jvariablesall, jdataf)

    # Return EVERYTHING you will likely use later.
    return (
        # symbolic objects
        xi_expr       = xi_expr,
        zetavars      = zetavars,
        gvariablesall = gvariablesall,
        zvariablesall = zvariablesall,
        jvariablesall = jvariablesall,

        # variable lists
        xivaria       = xivaria,
        zetavaria     = zetavaria,
        gvaria        = gvaria,
        zvaria        = zvaria,
        jvaria        = jvaria,

        # numeric critical data (raw)
        xisoln        = xisoln,
        gdataof       = gdataof,
        zdataf        = zdataf,
        jdataf        = jdataf,

        # extracted solutions
        zeta_data     = zeta_data,
        gsoln         = gsoln,
        zsoln         = zsoln,
        jsoln         = jsoln,

        # special positions (often needed later)
        gspecialpos   = gspecialpos,
        zspecialPos   = zspecialPos
    )
end

end