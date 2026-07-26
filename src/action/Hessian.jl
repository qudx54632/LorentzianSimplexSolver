module Hessian

using SymEngine
using ..SolveVars: SolveData
using ..ActionEvaluation: build_value_dict, eval_symbolic, to_complex_T

export compute_Hessian_block,
       compute_Hessian_block_half,
       evaluate_hessian_block

function first_derivatives(S::Basic, variables, precomputed)
    return [
        precomputed !== nothing && haskey(precomputed, variable) ?
            precomputed[variable] : SymEngine.diff(S, variable)
        for variable in variables
    ]
end

function compute_Hessian_block(S::Basic, variables; eom=nothing)
    return compute_hessian_symbols(S, variables; symmetric=true, eom=eom)
end

function compute_Hessian_block_half(S::Basic, variables; eom=nothing)
    return compute_hessian_symbols(S, variables; symmetric=false, eom=eom)
end

function compute_hessian_symbols(S::Basic, variables; symmetric::Bool, eom)
    n = length(variables)
    dS = first_derivatives(S, variables, eom)
    dependencies = [Set(free_symbols(expression)) for expression in dS]
    zero_symbol = Basic(0)
    H = fill(zero_symbol, n, n)

    # Only the upper triangle is differentiated. If dS_i does not contain
    # x_j, the corresponding second derivative is exactly zero and SymEngine
    # does not need to process it.
    for i in 1:n
        for j in i:n
            variables[j] in dependencies[i] || continue
            hij = SymEngine.diff(dS[i], variables[j])
            H[i, j] = hij
            symmetric && (H[j, i] = hij)
        end
    end

    return H
end


function evaluate_hessian_block(Hsym::AbstractMatrix{Basic},
                                sd::SolveData{T};
                                γ=one(T)) where {T<:Real}
    size(Hsym, 1) == size(Hsym, 2) || error("Hessian block must be square.")

    γsym = symbols("gamma")
    vals = build_value_dict(sd, γsym; γval=γ)
    n = size(Hsym, 1)
    H = zeros(Complex{T}, n, n)

    for i in 1:n
        for j in i:n
            hij = Hsym[i, j]
            hij == 0 && continue

            value = to_complex_T(eval_symbolic(hij, vals), T)
            H[i, j] = value
            H[j, i] = value
        end
    end

    return H
end

end
