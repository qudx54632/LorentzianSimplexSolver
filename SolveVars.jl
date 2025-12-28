module SolveVars

using ..CriticalPoints: compute_bdy_critical_data
using ..DefineSymbols: find_position_in_chain,
                       collect_bdry_symbols,
                       collect_varias_symbols
using ..DefineAction: γsym
using PythonCall

sympy = pyimport("sympy")

export run_solver, SolveData

# ============================================================
# Container for solver output (Julia-only values)
# ============================================================
struct SolveData{T<:Real}
    labels_vars :: Vector{Py}
    values_vars :: Vector{T}
    flags_vars  :: BitVector   # true => divide by γ

    labels_bdry :: Vector{Py}
    values_bdry :: Vector{T}
    flags_bdry  :: BitVector
end

# ============================================================
# Helper: extract unique symbol from a SymPy expr
# ============================================================
@inline get_sym(expr::Py) = only(collect(expr.free_symbols))

# ============================================================
# Helper: stable key for membership tests (avoid String(::Py) issues)
# ============================================================
@inline function symkey(x::Py)::String
    # sympy.sstr gives a canonical string form
    return pyconvert(String, sympy.sstr(x))
end

# ============================================================
# Helper: distribute labels/values/flags into vars or bdry
# ============================================================
function distribute!(labels_vars::Vector{Py}, values_vars::Vector{T}, flags_vars::BitVector,
                     labels_bdry::Vector{Py}, values_bdry::Vector{T}, flags_bdry::BitVector,
                     seen_vars::Set{String}, seen_bdry::Set{String},
                     L::Vector{Py}, V::Vector{T}, F::BitVector,
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
function solve_g_var(g_sym::Py, g_num::Matrix{Complex{T}}) where {T<:Real}
    re, im = sympy.re, sympy.im
    labels = Py[]; values = T[]; flags = BitVector()

    push!(labels, get_sym(re(g_sym[0,0] - 1))); push!(values, real(g_num[1,1]) - one(T)); push!(flags, false)
    push!(labels, get_sym(im(g_sym[0,0] - 1))); push!(values, imag(g_num[1,1]));          push!(flags, false)

    push!(labels, get_sym(re(g_sym[0,1])));     push!(values, real(g_num[1,2]));          push!(flags, false)
    push!(labels, get_sym(im(g_sym[0,1])));     push!(values, imag(g_num[1,2]));          push!(flags, false)

    push!(labels, get_sym(re(g_sym[1,0])));     push!(values, real(g_num[2,1]));          push!(flags, false)
    push!(labels, get_sym(im(g_sym[1,0])));     push!(values, imag(g_num[2,1]));          push!(flags, false)

    return labels, values, flags
end

function solve_g_special(g_sym::Py, g_num::Matrix{Complex{T}}) where {T<:Real}
    re, im = sympy.re, sympy.im
    labels = Py[]; values = T[]; flags = BitVector()

    push!(labels, get_sym(re(g_sym[0,0] - 1))); push!(values, real(g_num[1,1]) - one(T)); push!(flags, false)
    push!(labels, get_sym(im(g_sym[0,0] - 1))); push!(values, imag(g_num[1,1]));          push!(flags, false)

    push!(labels, get_sym(re(g_sym[1,0])));     push!(values, real(g_num[2,1]));          push!(flags, false)
    push!(labels, get_sym(im(g_sym[1,0])));     push!(values, imag(g_num[2,1]));          push!(flags, false)

    push!(labels, get_sym(re(g_sym[1,1])));     push!(values, real(g_num[2,2]));          push!(flags, false)
    push!(labels, get_sym(im(g_sym[1,1])));     push!(values, imag(g_num[2,2]));          push!(flags, false)

    return labels, values, flags
end

function solve_g_upper(g_sym::Py, g_num::Matrix{Complex{T}}) where {T<:Real}
    re, im = sympy.re, sympy.im
    labels = Py[]; values = T[]; flags = BitVector()

    push!(labels, get_sym(re(g_sym[0,0] - 1))); push!(values, real(g_num[1,1]) - one(T)); push!(flags, false)

    push!(labels, get_sym(re(g_sym[1,0])));     push!(values, real(g_num[2,1]));          push!(flags, false)
    push!(labels, get_sym(im(g_sym[1,0])));     push!(values, imag(g_num[2,1]));          push!(flags, false)

    return labels, values, flags
end

# ============================================================
# z variables
# ============================================================
function solve_z_var(z_sym::Py, z_num::Vector{Complex{T}}) where {T<:Real}
    re, im = sympy.re, sympy.im
    labels = Py[]; values = T[]; flags = BitVector()

    if !isempty(z_sym[0].free_symbols)
        expr = z_sym[0]
        zval = z_num[1] - one(T)
        push!(labels, get_sym(re(expr))); push!(values, real(zval)); push!(flags, false)
        push!(labels, get_sym(im(expr))); push!(values, imag(zval)); push!(flags, false)
        return labels, values, flags
    elseif !isempty(z_sym[1].free_symbols)
        expr = z_sym[1]
        zval = z_num[2] - one(T)
        push!(labels, get_sym(re(expr))); push!(values, real(zval)); push!(flags, false)
        push!(labels, get_sym(im(expr))); push!(values, imag(zval)); push!(flags, false)
        return labels, values, flags
    else
        return labels, values, flags
    end
end

# ============================================================
# j variables
# ============================================================
function solve_j_var(j_sym::Py, area::T, tetareasign::Int) where {T<:Real}
    j_sym === Py(0) && return Py[], T[], falses(0)
    labels = Py[j_sym]
    values = T[area]
    flags  = BitVector([tetareasign == 1])  # true => divide by γ
    return labels, values, flags
end

# ============================================================
# xi variables
# ============================================================
function solve_xi_var(xi_sym::Py, xi_sol::Vector{T}) where {T<:Real}
    isempty(xi_sol) && return Py[], T[], falses(0)

    vars = collect(sympy.Matrix(xi_sym).free_symbols)
    isempty(vars) && return Py[], T[], falses(0)

    labels = Py[]; values = T[]; flags = BitVector()

    if length(vars) == 1
        push!(labels, vars[1]); push!(values, xi_sol[1]); push!(flags, false)
    elseif length(vars) == 2
        for v in vars
            name = symkey(v)
            push!(labels, v)
            push!(values,
                  endswith(name, "_a") ? xi_sol[1] :
                  endswith(name, "_b") ? xi_sol[2] :
                  error("Unknown xi variable $name"))
            push!(flags, false)
        end
    else
        error("Unexpected number of xi symbols")
    end

    return labels, values, flags
end

# ============================================================
# Main driver
# ============================================================
function run_solver(geom)

    g_mat  = geom.varias[:g_mat]
    z_mat  = geom.varias[:z_mat]
    j_mat  = geom.varias[:j_mat]
    xi_mat = geom.varias[:xi_mat]

    ns, ntet = length(g_mat), 5
    T = eltype(eltype(eltype(geom.simplex[1].areas)))

    labels_vars = Py[]; values_vars = T[]; flags_vars = BitVector()
    labels_bdry = Py[]; values_bdry = T[]; flags_bdry = BitVector()

    # classification sets (string keys)
    bdry_keys = Set(symkey(s) for s in collect_bdry_symbols(geom))
    var_keys  = Set(symkey(s) for s in collect_varias_symbols(geom))

    seen_vars = Set{String}()
    seen_bdry = Set{String}()

    data = compute_bdy_critical_data(geom)
    gdataof, zdataf = data.gdataof, data.zdataf
    areadataf, xisoln = data.areadataf, data.xisoln

    gspecialpos = geom.varias[:gspecialPos]
    GaugeFixUpperTriangle = ns > 1 ? geom.connectivity[1]["GaugeFixUpperTriangle"] :
                                     Vector{Vector{Int}}()

    kappa       = [geom.simplex[a].kappa       for a in 1:ns]
    tetareasign = [geom.simplex[a].tetareasign for a in 1:ns]

    for a in 1:ns, i in 1:ntet
        pos_gspecial = find_position_in_chain([a,i], gspecialpos)
        pos_gupper   = find_position_in_chain([a,i], GaugeFixUpperTriangle)

        L,V,F = pos_gspecial !== nothing ? solve_g_special(g_mat[a][i], gdataof[a][i]) :
                pos_gupper   !== nothing ? solve_g_upper(g_mat[a][i], gdataof[a][i]) :
                                           solve_g_var(g_mat[a][i], gdataof[a][i])

        distribute!(labels_vars, values_vars, flags_vars, labels_bdry, values_bdry, flags_bdry, seen_vars, seen_bdry, L, V, F, bdry_keys, var_keys)

        for j in 1:ntet
            i == j && continue

            L,V,F = solve_xi_var(xi_mat[a][i][j], xisoln[a][i][j])
            distribute!(labels_vars, values_vars, flags_vars, labels_bdry, values_bdry, flags_bdry, seen_vars, seen_bdry, L, V, F, bdry_keys, var_keys)

            if kappa[a][i][j] == 1
                L,V,F = solve_z_var(z_mat[a][i][j], zdataf[a][i][j])
                distribute!(labels_vars, values_vars, flags_vars, labels_bdry, values_bdry, flags_bdry, seen_vars, seen_bdry, L, V, F, bdry_keys, var_keys)
            end

            L,V,F = solve_j_var(j_mat[a][i][j], areadataf[a][i][j], tetareasign[a][i][j])
            distribute!(labels_vars, values_vars, flags_vars, labels_bdry, values_bdry, flags_bdry, seen_vars, seen_bdry, L, V, F, bdry_keys, var_keys)
        end
    end

    return SolveData(labels_vars, values_vars, flags_vars, labels_bdry, values_bdry, flags_bdry), γsym
end

end # module