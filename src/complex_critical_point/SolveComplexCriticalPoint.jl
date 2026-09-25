"""Result of the complex Newton iteration."""
struct ComplexCriticalPointResult{T<:Real}
    variables::Vector{Basic}
    initial::Vector{Complex{T}}
    solution::Vector{Complex{T}}
    action::Complex{T}
    residual::Vector{Complex{T}}
    iterations::Int
    converged::Bool
end

"""
    solve_complex_critical_point(action, solve_data, gamma_symbol, gamma_value)

Start from the real pseudo-critical point and solve the complex equations
`d(action)/d(variable) = 0` by Newton's method.
"""
function solve_complex_critical_point(
    action,
    solve_data,
    gamma_symbol,
    gamma_value;
    tol=1e-11,
    maxiter=30,
    verbose=true,
)
    data = prepare_complex_critical_point(
        action, solve_data, gamma_symbol, gamma_value,
    )

    x = copy(data.initial)
    residual = evaluate_complex(
        data.eom, data.fixed, data.variables, x, data.number_type,
    )
    iterations = 0

    for iteration in 1:maxiter
        iterations = iteration
        size_residual = maximum(abs.(residual); init=zero(data.number_type))
        verbose && println("complex Newton $iteration: max |dS| = $size_residual")
        size_residual < tol && break

        hessian = evaluate_complex(
            data.hessian, data.fixed, data.variables, x, data.number_type,
        )
        x -= hessian \ residual
        residual = evaluate_complex(
            data.eom, data.fixed, data.variables, x, data.number_type,
        )
    end

    converged = maximum(abs.(residual); init=zero(data.number_type)) < tol
    action_value = evaluate_complex(
        [data.action], data.fixed, data.variables, x, data.number_type,
    )[1]

    return ComplexCriticalPointResult(
        data.variables,
        data.initial,
        x,
        action_value,
        residual,
        iterations,
        converged,
    )
end
