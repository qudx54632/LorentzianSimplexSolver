module SolveVars

using ..CriticalPoints: compute_bdy_critical_data
using ..DefineSymbols: collect_bdry_symbols, collect_varias_symbols
using ..DefineAction: γsym

using SymEngine

export run_solver, SolveData

# ============================================================
# Container for solver output (Julia-only values)
# ============================================================
struct SolveData{T<:Real}
    labels_vars :: Vector{Basic}
    values_vars :: Vector{T}
    flags_vars  :: BitVector

    labels_bdry :: Vector{Basic}
    values_bdry :: Vector{T}
    flags_bdry  :: BitVector

    labels_η    :: Vector{Basic}
    values_η    :: Vector{T}
    flags_η     :: BitVector
end

# ============================================================
# Helper: extract unique symbol from a SymPy expr
# ============================================================
@inline symkey(x::Basic) = string(x)

function vars_in_expr(expr)
    if isa(expr, Basic)
        return sort(collect(free_symbols(expr)), by=string)
    elseif isa(expr, AbstractArray)
        vars = Basic[]
        for e in expr
            isa(e, Basic) || continue
            append!(vars, free_symbols(e))
        end
        return sort(unique(vars), by=string)
    else
        return Basic[]
    end
end

function get_sym(expr)
    syms = vars_in_expr(expr)

    if isempty(syms)
        error("No free symbols in $expr")
    elseif length(syms) == 1
        return first(syms)
    else
        error("Expected one symbol, got $(syms) in $expr")
    end
end

function get_single_var(expr)
    vars = vars_in_expr(expr)
    length(vars) == 1 || error("Expected 1 variable in $expr, got $vars")
    return vars[1]
end

# ============================================================
# Helper: distribute labels/values/flags into vars or bdry
# ============================================================
function distribute!(labels_vars::Vector{Basic}, values_vars::Vector{T}, flags_vars::BitVector,
                     labels_bdry::Vector{Basic}, values_bdry::Vector{T}, flags_bdry::BitVector,
                     seen_vars::Set{String}, seen_bdry::Set{String},
                     L::Vector{Basic}, V::Vector{T}, F::BitVector,
                     bdry_keys::Set{String}, var_keys::Set{String}) where {T<:Real}

    @assert length(L) == length(V) == length(F)

    for k in eachindex(L)
        key = symkey(L[k])

        if key in bdry_keys
            key in seen_bdry && continue
            push!(seen_bdry, key)
            push!(labels_bdry, L[k])
            push!(values_bdry, V[k])
            push!(flags_bdry,  F[k])

        elseif key in var_keys
            key in seen_vars && continue
            push!(seen_vars, key)
            push!(labels_vars, L[k])
            push!(values_vars, V[k])
            push!(flags_vars,  F[k])

        else
            @warn "Symbol not classified as bdry or vars; defaulting to vars" symbol=key
            key in seen_vars && continue
            push!(seen_vars, key)
            push!(labels_vars, L[k])
            push!(values_vars, V[k])
            push!(flags_vars,  F[k])
        end
    end

    return nothing
end

# ============================================================
# g variables
# ============================================================
function solve_g_var(g_sym::AbstractMatrix{<:Basic}, g_num::AbstractMatrix{Complex{T}}) where {T<:Real}
    labels = Basic[]
    values = T[]
    flags  = BitVector()

    vars = vars_in_expr(g_sym[1,1] - 1)
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[1,1]) - one(T)); push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[1,1]));          push!(flags, false)

    vars = vars_in_expr(g_sym[1,2])
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[1,2]));          push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[1,2]));          push!(flags, false)

    vars = vars_in_expr(g_sym[2,1])
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[2,1]));          push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[2,1]));          push!(flags, false)

    return labels, values, flags
end

function solve_g_gauge(g_sym::AbstractMatrix{<:Basic}, g_num::AbstractMatrix{Complex{T}}) where {T<:Real}
    
    labels = Basic[]
    values = T[]
    flags  = BitVector()

    # (1,1)
    vars = vars_in_expr(g_sym[1,1])
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[1,1])); push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[1,1])); push!(flags, false)

    # (1,2)
    vars = vars_in_expr(g_sym[1,2])
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[1,2])); push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[1,2])); push!(flags, false)

    # (2,1)
    vars = vars_in_expr(g_sym[2,1])
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[2,1])); push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[2,1])); push!(flags, false)

    # (2,2)
    vars = vars_in_expr(g_sym[2,2])
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[2,2])); push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[2,2])); push!(flags, false)

    return labels, values, flags
end

function solve_g_special(g_sym::AbstractMatrix{<:Basic}, g_num::AbstractMatrix{Complex{T}}) where {T<:Real}

    labels = Basic[]
    values = T[]
    flags  = BitVector()

    # (1,1) → subtract 1
    vars = vars_in_expr(g_sym[1,1] - 1)
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[1,1]) - one(T)); push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[1,1]));          push!(flags, false)

    # (2,1)
    vars = vars_in_expr(g_sym[2,1])
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[2,1])); push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[2,1])); push!(flags, false)

    # (2,2)
    vars = vars_in_expr(g_sym[2,2])
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[2,2])); push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[2,2])); push!(flags, false)

    return labels, values, flags
end

function solve_g_upper(g_sym::AbstractMatrix{<:Basic}, g_num::AbstractMatrix{Complex{T}}) where {T<:Real}

    labels = Basic[]
    values = T[]
    flags  = BitVector()

    # (1,1) → subtract 1 (only ONE real variable here)
    vars = vars_in_expr(g_sym[1,1] - 1)
    @assert length(vars) == 1
    push!(labels, vars[1])
    push!(values, real(g_num[1,1]) - one(T))
    push!(flags, false)

    # (2,1) → complex (two variables)
    vars = vars_in_expr(g_sym[2,1])
    @assert length(vars) == 2
    push!(labels, vars[1]); push!(values, real(g_num[2,1])); push!(flags, false)
    push!(labels, vars[2]); push!(values, imag(g_num[2,1])); push!(flags, false)

    return labels, values, flags
end

# ============================================================
# z variables
# ============================================================
function solve_z_var(z_sym::Vector{<:Basic}, z_num::Vector{Complex{T}}) where {T<:Real}
    labels = Basic[]
    values = T[]
    flags  = BitVector()

    # check first component
    vars = vars_in_expr(z_sym[1])
    if !isempty(vars)
        @assert length(vars) == 2
        zval = z_num[1] - one(T)

        push!(labels, vars[1]); push!(values, real(zval)); push!(flags, false)
        push!(labels, vars[2]); push!(values, imag(zval)); push!(flags, false)

        return labels, values, flags
    end

    # check second component
    vars = vars_in_expr(z_sym[2])
    if !isempty(vars)
        @assert length(vars) == 2
        zval = z_num[2] - one(T)

        push!(labels, vars[1]); push!(values, real(zval)); push!(flags, false)
        push!(labels, vars[2]); push!(values, imag(zval)); push!(flags, false)

        return labels, values, flags
    end

    return labels, values, flags
end

# ============================================================
# η variables
# ============================================================
function solve_η_var(η_sym::Basic, area::T, tetareasign::Int) where {T<:Real}
    if η_sym == Basic(0)
        return Basic[], T[], BitVector()
    end

    return Basic[η_sym], T[2 * area], BitVector([tetareasign == 1])
end

# ============================================================
# xi variables
# ============================================================
function solve_xi_var(xi_sym::Vector{<:Basic}, xi_sol::Vector{T}) where {T<:Real}
    isempty(xi_sol) && return Basic[], T[], BitVector()

    # collect unique symbols
    vars = Basic[]
    seen = Set{Basic}()

    for e in xi_sym
        syms = vars_in_expr(e)
        for v in syms
            if !(v in seen)
                push!(seen, v)
                push!(vars, v)
            end
        end
    end

    isempty(vars) && return Basic[], T[], BitVector()

    labels = Basic[]
    values = T[]
    flags  = BitVector()

    if length(vars) == 1
        push!(labels, vars[1])
        push!(values, xi_sol[1])
        push!(flags, false)

    elseif length(vars) == 2
        for v in vars
            name = string(v)
            if endswith(name, "a")
                push!(labels, v); push!(values, xi_sol[1]); push!(flags, false)
            elseif endswith(name, "b")
                push!(labels, v); push!(values, xi_sol[2]); push!(flags, false)
            else
                error("Unexpected xi symbol name: $name")
            end
        end
    end

    return labels, values, flags
end

# ============================================================
# Main driver
# ============================================================
function run_solver(geom)

    g_mat  = geom.varias[:g_mat]
    z_mat  = geom.varias[:z_mat]
    η_mat  = geom.varias[:η_mat]
    xi_mat = geom.varias[:xi_mat]

    ns, ntet = length(g_mat), 5
    T = eltype(eltype(eltype(geom.simplex[1].areas)))

    # -------------------------------
    # OUTPUT CONTAINERS
    # -------------------------------
    labels_vars = Basic[]
    values_vars = T[]
    flags_vars  = BitVector()

    labels_bdry = Basic[]
    values_bdry = T[]
    flags_bdry  = BitVector()

    labels_η = Basic[]
    values_η = T[]
    flags_η  = BitVector()

    # -------------------------------
    # classification sets (NO string!)
    # -------------------------------
    bdry_set = Set(collect_bdry_symbols(geom))

    seen_vars = Set{Basic}()
    seen_bdry = Set{Basic}()
    seen_η    = Set{Basic}()

    # -------------------------------
    # numerical data
    # -------------------------------
    data = compute_bdy_critical_data(geom)
    gdataof   = data.gdataof
    zdataf    = data.zdataf
    areadataf = data.areadataf
    xisoln    = data.xisoln

    # -------------------------------
    # precompute lookup sets (FAST!)
    # -------------------------------
    gspecial_set = Set(Tuple(p) for p in geom.varias[:gspecialPos])
    gupper_set   = ns > 1 ? Set(Tuple(p) for p in geom.connectivity[1]["GaugeFixUpperTriangle"]) :
                            Set{Tuple{Int,Int}}()
    gauge_set = ns > 1 ? Set(Tuple(p) for p in geom.connectivity[1]["GaugeTet"]) : Set([(1,1)])

    kappa       = [geom.simplex[a].kappa       for a in 1:ns]
    tetareasign = [geom.simplex[a].tetareasign for a in 1:ns]

    # ============================================================
    # MAIN LOOP
    # ============================================================
    for a in 1:ns, i in 1:ntet

        key_ai = (a,i)

        # -------- g --------
        L, V, F =
            key_ai in gauge_set    ? solve_g_gauge(g_mat[a][i], gdataof[a][i]) :
            key_ai in gspecial_set ? solve_g_special(g_mat[a][i], gdataof[a][i]) :
            key_ai in gupper_set   ? solve_g_upper(g_mat[a][i], gdataof[a][i]) :
                                     solve_g_var(g_mat[a][i], gdataof[a][i])

        for k in eachindex(L)
            sym = L[k]

            if sym in bdry_set
                sym in seen_bdry && continue
                push!(seen_bdry, sym)
                push!(labels_bdry, sym)
                push!(values_bdry, V[k])
                push!(flags_bdry,  F[k])

            else
                sym in seen_vars && continue
                push!(seen_vars, sym)
                push!(labels_vars, sym)
                push!(values_vars, V[k])
                push!(flags_vars,  F[k])
            end
        end

        # -------- faces --------
        for j in 1:ntet
            i == j && continue

            # xi
            L, V, F = solve_xi_var(xi_mat[a][i][j], xisoln[a][i][j])
            for k in eachindex(L)
                sym = L[k]

                if sym in bdry_set
                    sym in seen_bdry && continue
                    push!(seen_bdry, sym)
                    push!(labels_bdry, sym)
                    push!(values_bdry, V[k])
                    push!(flags_bdry,  F[k])
                else
                    sym in seen_vars && continue
                    push!(seen_vars, sym)
                    push!(labels_vars, sym)
                    push!(values_vars, V[k])
                    push!(flags_vars,  F[k])
                end
            end

            # z
            if kappa[a][i][j] == 1
                L, V, F = solve_z_var(z_mat[a][i][j], zdataf[a][i][j])
                for k in eachindex(L)
                    sym = L[k]

                    if sym in bdry_set
                        sym in seen_bdry && continue
                        push!(seen_bdry, sym)
                        push!(labels_bdry, sym)
                        push!(values_bdry, V[k])
                        push!(flags_bdry,  F[k])
                    else
                        sym in seen_vars && continue
                        push!(seen_vars, sym)
                        push!(labels_vars, sym)
                        push!(values_vars, V[k])
                        push!(flags_vars,  F[k])
                    end
                end
            end

            # η (separate)
            L, V, F = solve_η_var(η_mat[a][i][j], areadataf[a][i][j], tetareasign[a][i][j])
            # @show a, i, j, L, V, F

            for k in eachindex(L)
                sym = L[k]
                if !(sym in seen_η)
                    push!(seen_η, sym)
                    push!(labels_η, sym)
                    push!(values_η, V[k])
                    push!(flags_η,  F[k])
                end

                # -------------------------
                # ALSO add to vars (NEW)
                # -------------------------
                if sym in bdry_set
                    if !(sym in seen_bdry)
                        push!(seen_bdry, sym)
                        push!(labels_bdry, sym)
                        push!(values_bdry, V[k])
                        push!(flags_bdry,  F[k])
                    end
                else
                    if !(sym in seen_vars)
                        push!(seen_vars, sym)
                        push!(labels_vars, sym)
                        push!(values_vars, V[k])
                        push!(flags_vars,  F[k])
                    end
                end
            end
        end
    end

    return SolveData(
        labels_vars, values_vars, flags_vars,
        labels_bdry, values_bdry, flags_bdry,
        labels_η,    values_η,    flags_η
    ), γsym()
end

end # module