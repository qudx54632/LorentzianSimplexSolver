module SolveVars

using ..CriticalPoints: compute_bdy_critical_data
using ..DefineSymbols: find_position_in_chain, collect_bdry_symbols, collect_varias_symbols
using PythonCall

sympy = pyimport("sympy")

export run_solver


function solve_g_var(g_sym::Py, g_num::Matrix{ComplexF64})
    sol = Dict{Py,Py}()

    re = sympy.re
    im = sympy.im

    # helper: assign value to the unique symbol in expr
    function assign(expr, value)
        sym = only(collect(expr.free_symbols))   # the symbol (Py)
        sol[sym] = Py(value)                     # IMPORTANT FIX
    end

    # (0,0): 1 + g1 + i g2
    assign(re(g_sym[0,0] - 1), real(g_num[1,1]) - 1)
    assign(im(g_sym[0,0] - 1), imag(g_num[1,1]))

    # (0,1): g3 + i g4
    assign(re(g_sym[0,1]), real(g_num[1,2]))
    assign(im(g_sym[0,1]), imag(g_num[1,2]))

    # (1,0): g5 + i g6
    assign(re(g_sym[1,0]), real(g_num[2,1]))
    assign(im(g_sym[1,0]), imag(g_num[2,1]))

    return sol
end

function solve_g_special(g_sym::Py, g_num::Matrix{ComplexF64})
    sol = Dict{Py,Py}()

    re = sympy.re
    im = sympy.im

    # helper: assign value to the unique symbol in expr
    function assign(expr, value)
        sym = only(collect(expr.free_symbols))   # the symbol (Py)
        sol[sym] = Py(value)                     # IMPORTANT FIX
    end

    # (0,0): 1 + g1 + i g2
    assign(re(g_sym[0,0] - 1), real(g_num[1,1]) - 1)
    assign(im(g_sym[0,0] - 1), imag(g_num[1,1]) - 1)

    # (1,0): g3 + i g4
    assign(re(g_sym[1,0]), real(g_num[2,1]))
    assign(im(g_sym[1,0]), imag(g_num[2,1]))

    # (1,1): g5 + i g6
    assign(re(g_sym[1,1]), real(g_num[2,2]))
    assign(im(g_sym[1,2]), imag(g_num[2,2]))

    return sol
end

function solve_g_upper(g_sym::Py, g_num::Matrix{ComplexF64})
    sol = Dict{Py,Py}()

    re = sympy.re
    im = sympy.im

    # helper: assign value to the unique symbol in expr
    function assign(expr, value)
        sym = only(collect(expr.free_symbols))   # the symbol (Py)
        sol[sym] = Py(value)                     # IMPORTANT FIX
    end

    # (0,0): 1 + g1
    assign(re(g_sym[0,0] - 1), real(g_num[1,1]) - 1)

    # (1,0): g2 + i g3
    assign(re(g_sym[1,0]), real(g_num[2,1]))
    assign(im(g_sym[1,0]), imag(g_num[2,1]))

    return sol
end


function solve_z_var(z_sym::Py, z_num::Vector{ComplexF64})
    sol = Dict{Py,Py}()

    re = sympy.re
    im = sympy.im

    # detect which entry is nontrivial
    if !isempty(z_sym[0].free_symbols)
        expr = z_sym[0]     # IMPORTANT
        zval = z_num[1] - 1
    elseif !isempty(z_sym[1].free_symbols)
        expr = z_sym[1]      # IMPORTANT
        zval = z_num[2] - 1
    else
        error("solve_z_var: no symbolic entry found in z_sym")
    end

    # extract symbols from expr = za + i zb
    za = only(collect(re(expr).free_symbols))
    zb = only(collect(im(expr).free_symbols))

    sol[za] = Py(real(zval))
    sol[zb] = Py(imag(zval))

    return sol
end


function solve_j_var(j_sym::Py, area::Real, tetareasign::Int; γ::Py)
    sol = Dict{Py,Py}()

    # skip inactive entries
    j_sym === Py(0) && return sol

    if tetareasign == 1
        # spacelike: j = A / γ
        sol[j_sym] = Py(area) / γ
    else
        # timelike or others: j = A
        sol[j_sym] = Py(area)
    end

    return sol
end

function solve_xi_var(xi_sym::Py, xi_sol::Vector{Float64})
    sol = Dict{Py,Py}()

    isempty(xi_sol) && return sol
    any(isnan, xi_sol) && return sol

    vars = collect(sympy.Matrix(xi_sym).free_symbols)
    n = length(vars)

    if n == 0
        return sol

    elseif n == 1
        # only one angle (θ)
        sol[vars[1]] = Py(xi_sol[1])
        return sol

    elseif n == 2
        # two angles: *_a, *_b
        for v in vars
            name = string(v)

            if endswith(name, "_a")
                sol[v] = Py(xi_sol[1])
            elseif endswith(name, "_b")
                sol[v] = Py(xi_sol[2])
            else
                error("solve_xi_var: unknown xi symbol $name")
            end
        end
        return sol

    else
        error("solve_xi_var: unexpected number of xi symbols ($n)")
    end
end

function split_solution(sol_all::Dict{Py,Py}, geom)
    bdry_syms   = collect_bdry_symbols(geom)
    varias_syms = collect_varias_symbols(geom)

    sol_bdry = Dict{Py,Py}()
    sol_vars = Dict{Py,Py}()

    for (k, v) in sol_all
        if k in bdry_syms
            sol_bdry[k] = v
        elseif k in varias_syms
            sol_vars[k] = v
        else
            @warn "Symbol not classified as bdry or var" symbol=k
        end
    end

    return sol_vars, sol_bdry
end

function run_solver(geom)
    g_mat  = geom.varias[:g_mat]
    z_mat  = geom.varias[:z_mat]
    j_mat  = geom.varias[:j_mat]
    xi_mat = geom.varias[:xi_mat]

    ns   = length(g_mat)
    ntet = 5

    sol_all = Dict{Py,Py}()

    geom_data = compute_bdy_critical_data(geom)
    gdataof   = geom_data.gdataof
    zdataf    = geom_data.zdataf
    areadataf = geom_data.areadataf
    xisoln    = geom_data.xisoln
    gspecialpos = geom.varias[:gspecialPos]

    if ns > 1
        GaugeFixUpperTriangle = geom.connectivity[1]["GaugeFixUpperTriangle"]
    else 
        GaugeFixUpperTriangle = Vector{Vector{Int}}()
    end 

    kappa       = [geom.simplex[a].kappa for a in 1:ns]
    tetareasign = [geom.simplex[a].tetareasign for a in 1:ns]

    γsym = sympy.symbols("gamma", real=true)

    for a in 1:ns
        for i in 1:ntet
            # g variables

            pos_gspecial = find_position_in_chain([a,i], gspecialpos)
            pos_gupper   = find_position_in_chain([a,i], GaugeFixUpperTriangle)
            if pos_gspecial !== nothing 
                merge!(sol_all, solve_g_special(g_mat[a][i], gdataof[a][i]))
            elseif pos_gupper !== nothing
                merge!(sol_all, solve_g_upper(g_mat[a][i], gdataof[a][i]))
            else
                merge!(sol_all, solve_g_var(g_mat[a][i], gdataof[a][i]))
            end
            
            for j in 1:ntet
                i == j && continue

                # xi variables
                merge!(sol_all, solve_xi_var(xi_mat[a][i][j], xisoln[a][i][j]))

                # z variables
                if kappa[a][i][j] == 1
                    merge!(sol_all, solve_z_var(z_mat[a][i][j], zdataf[a][i][j]))
                end

                # j variables
                merge!(sol_all, solve_j_var(j_mat[a][i][j], areadataf[a][i][j], tetareasign[a][i][j]; γ=γsym))
            end
        end
    end

    sol_vars, sol_bdry = split_solution(sol_all, geom)

    return sol_vars, sol_bdry, γsym
end

end