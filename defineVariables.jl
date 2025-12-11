module DefineVariables
   
using LinearAlgebra
using ..CriticalPoints: compute_critical_data

export xivaria, gvaria, zvaria, jvaria,
       build_xi_expressions, build_g_variables,
       build_z_variables, build_j_variables

function build_xi_expressions(sgndet, tetareasign, tetn0, ns)

    xi_expr = Vector{Any}(undef, ns)
    for k in 1:ns
        xi_expr[k] = Vector{Any}(undef, 5)
        for i in 1:5
            xi_expr[k][i] = Vector{Any}(undef, 5)
        end
    end

    for k in 1:ns
        for i in 1:5
            for j in 1:5

                if i == j
                    xi_expr[k][i][j] = (0, 0)
                    continue
                end

                # symbolic names
                a = Symbol("zeta_$(k)_$(i)_$(j)_a")
                b = Symbol("zeta_$(k)_$(i)_$(j)_b")

                if sgndet[k][i] == 1
                    # ( sin(a) , cos(a)*exp(i b) )
                    xi_expr[k][i][j] = (
                        :(sin($a)),
                        :(cos($a) * exp(im*$b))
                    )

                elseif tetareasign[k][i][j] == 1
                    if tetn0[k][i][j] == 1
                        xi_expr[k][i][j] = (
                            :(cosh($a)),
                            :(exp(-im*$b) * sinh($a))
                        )
                    else
                        xi_expr[k][i][j] = (
                            :(sinh($a) * exp(im*$b)),
                            :(cosh($a))
                        )
                    end

                else
                    # null/timelike case (1 , exp(i b))
                    xi_expr[k][i][j] = (
                        1,
                        :(exp(im*$b))
                    )
                end
            end
        end
    end

    return xi_expr
end

# helper: collect all symbols inside an Expr / Tuple / Array
function _collect_syms!(vars::Vector{Symbol}, x)
    if x isa Symbol
        # ignore standard builtins and constants
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

Given xi_expr[k][i][j] = (expr1, expr2) from `build_xi_expressions`,
return `zetavars[k][i][j]::Vector{Symbol}` listing the zeta symbols
for that (k,i,j), in a deterministic order.
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

                # x is either (expr1, expr2) or (0,0)
                _collect_syms!(vars, x)

                # remove duplicates and sort to fix order (like MMA Sort)
                zetavars[k][i][j] = sort(unique(vars))
            end
        end
    end

    return zetavars
end

"""
    build_zetadata(xisoln, zetavars, tetareasign)

Return Dict{Symbol,Float64} giving the numerical value of each ζ–variable
at the critical point.

- If tetareasign[k][i][j] == 1:  both ζₐ, ζ_b get values from xisoln[k][i][j]
- Else: only ζ_b gets value xisoln[k][i][j][1]
"""
function build_zetadata(xisoln, zetavars, tetareasign)
    ns   = length(xisoln)
    ntet = length(xisoln[1])

    zeta_data = Dict{Symbol,Float64}()

    for k in 1:ns
        for i in 1:ntet
            for j in 1:ntet
                # skip diagonal if you like (i == j)
                if i == j
                    continue
                end

                vars = zetavars[k][i][j]
                isempty(vars) && continue

                if tetareasign[k][i][j] == 1
                    vals = xisoln[k][i][j]             # [θ, φ]
                else
                    vals = [xisoln[k][i][j][1]]        # only θ (like MMA)
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

"""
    compute_gspecialpos(gdataof, GaugeTet; tol=1e-12)

Return the list of (k,i) where gdataof[k][i][1,1] ≈ 0
and (k,i) is *not* in GaugeTet.
GaugeTet is taken from geom.connectivity[1]["GaugeTet"].
"""
function compute_gspecialpos(gdataof, GaugeTet; tol=1e-12)
    ns   = length(gdataof)
    ntet = length(gdataof[1])

    # Allow GaugeTet to be either Vector{Tuple} or Vector{Vector}
    gauge_set = Set{Tuple{Int,Int}}()
    for p in GaugeTet
        gauge_set |= Set([(p[1], p[2])])
    end

    pos = Tuple{Int,Int}[]

    for k in 1:ns, i in 1:ntet
        key = (k,i)
        key in gauge_set && continue

        g = gdataof[k][i]
        if abs(g[1,1]) < tol
            push!(pos, key)
        end
    end

    return pos
end


function run_define_variables(geom)
    xi_final     = [geom.simplex[s].bdyxi       for s in 1:ns]
    sgndet       = [geom.simplex[s].sgndet      for s in 1:ns]
    tetareasign  = [geom.simplex[s].tetareasign for s in 1:ns]
    tetn0        = [geom.simplex[s].tetn0sign   for s in 1:ns]

    # 1) build symbolic ξ(zeta) expressions
    xi_expr   = build_xi_expressions(sgndet, tetareasign, tetn0, ns)

    # 2) per-face zeta variables
    zetavars  = build_zetavariables(xi_expr)

    # 3) numeric ξ–parameters at the critical point
    crit = compute_critical_data(geom)
    xisoln = crit.xisoln

    # 4) final ζ–data dict
    zeta_data = build_zetadata(xisoln, zetavars, tetareasign)
    
end

end
