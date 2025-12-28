module SymbolicToJulia

using PythonCall
using ..SolveVars: SolveData

export build_action_function,
       build_gradient_functions,
       build_hessian_functions,
       build_argument_vector
       
sympy      = pyimport("sympy")
juliaprint = pyimport("sympy.printing.julia")

@inline function sympy_to_expr(expr::Py)
    code = pyconvert(String, juliaprint.julia_code(expr))
    return Meta.parse(code)
end

# ------------------------------------------------------------
# Action S(x)
# ------------------------------------------------------------
function build_action_function(S::Py, sd::SolveData)

    body = sympy_to_expr(S)

    labels = vcat(sd.labels_vars, sd.labels_bdry)
    args   = Symbol.(string.(labels))
    push!(args, :gamma)   # γ always appears as an argument

    fname = gensym(:S)

    S_fn = eval(Expr(
        :function,
        Expr(:call, fname, args...),
        body
    ))

    return S_fn, labels
end

@inline is_numeric_gamma(γ) = γ isa Real

function build_argument_vector(sd::SolveData{T}, labels::Vector{Py}, γ) where {T<:Real}

    n = length(labels)
    vals = Vector{Any}(undef, n + 1)   # Any allows symbol or number

    var_index  = Dict(pyconvert(String, sympy.sstr(k)) => i
                      for (i,k) in enumerate(sd.labels_vars))
    bdry_index = Dict(pyconvert(String, sympy.sstr(k)) => i
                      for (i,k) in enumerate(sd.labels_bdry))

    numeric = is_numeric_gamma(γ)

    for (i, lab) in enumerate(labels)
        key = pyconvert(String, sympy.sstr(lab))

        if haskey(var_index, key)
            j = var_index[key]
            v = sd.values_vars[j]
            vals[i] = (numeric && sd.flags_vars[j]) ? (v / γ) : v

        elseif haskey(bdry_index, key)
            j = bdry_index[key]
            v = sd.values_bdry[j]
            vals[i] = (numeric && sd.flags_bdry[j]) ? (v / γ) : v

        else
            error("No value found for symbol $key")
        end
    end

    # γ goes in untouched (symbol or number)
    vals[end] = γ

    return vals
end

# ------------------------------------------------------------
# Gradient ∂S/∂xᵢ
# ------------------------------------------------------------
function build_gradient_functions(dS::Dict{Py,Py}, sd::SolveData)

    labels_all = vcat(sd.labels_vars, sd.labels_bdry)

    grad_fns = Dict{Py,Function}()

    for (v, expr) in dS
        body = sympy_to_expr(expr)

        args = Symbol.(string.(labels_all))
        push!(args, :gamma)

        fname = gensym(:dS)

        grad_fns[v] = eval(Expr(
            :function,
            Expr(:call, fname, args...),
            body
        ))
    end

    return grad_fns
end

# ------------------------------------------------------------
# Hessian ∂²S/∂xᵢ∂xⱼ
# ------------------------------------------------------------
function build_hessian_functions(H::Dict{Tuple{Py,Py},Py}, sd::SolveData)

    labels_all = vcat(sd.labels_vars, sd.labels_bdry)

    hess_fns = Dict{Tuple{Py,Py},Function}()

    for ((v, w), expr) in H
        body = sympy_to_expr(expr)

        args = Symbol.(string.(labels_all))
        push!(args, :gamma)

        fname = gensym(:H)

        hess_fns[(v,w)] = eval(Expr(
            :function,
            Expr(:call, fname, args...),
            body
        ))
    end

    return hess_fns
end

end # module