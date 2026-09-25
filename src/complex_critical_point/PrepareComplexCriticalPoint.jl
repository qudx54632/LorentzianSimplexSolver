function complex_values_at(fixed, variables, x)
    values = copy(fixed)
    for i in eachindex(variables)
        values[variables[i]] = Basic(x[i])
    end
    return values
end

function evaluate_complex(expressions, fixed, variables, x, ::Type{T}) where {T<:Real}
    values = complex_values_at(fixed, variables, x)
    return Complex{T}[
        to_complex_T(eval_symbolic(expression, values), T)
        for expression in expressions
    ]
end

"""
    prepare_complex_critical_point(action, solve_data, gamma_symbol, gamma_value)

Fix the boundary variables and construct the equations of motion, Hessian,
and real pseudo-critical starting point.
"""
function prepare_complex_critical_point(
    action,
    solve_data,
    gamma_symbol,
    gamma_value,
)
    T = promote_type(typeof(float(gamma_value)), eltype(solve_data.values_vars))
    variables = solve_data.labels_vars

    fixed = Dict{Basic,Basic}(gamma_symbol => Basic(gamma_value))
    for i in eachindex(solve_data.labels_bdry)
        value = solve_data.flags_bdry[i] ?
                solve_data.values_bdry[i] / gamma_value :
                solve_data.values_bdry[i]
        fixed[solve_data.labels_bdry[i]] = Basic(value)
    end

    action_fixed = eval_symbolic(action, fixed)
    eom_dict = compute_EOMs(action_fixed, variables)
    eom = [eom_dict[x] for x in variables]
    hessian = compute_Hessian_block(action_fixed, variables; eom=eom_dict)

    initial = Complex{T}[
        solve_data.flags_vars[i] ?
        solve_data.values_vars[i] / gamma_value :
        solve_data.values_vars[i]
        for i in eachindex(variables)
    ]

    return (
        variables=variables,
        fixed=fixed,
        action=action_fixed,
        eom=eom,
        hessian=hessian,
        initial=initial,
        number_type=T,
    )
end
