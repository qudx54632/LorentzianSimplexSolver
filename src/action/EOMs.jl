module EOMs

using SymEngine
using ..SolveVars: SolveData
using ..PrecisionUtils: get_tolerance
using ..ActionEvaluation: build_value_dict, eval_symbolic, to_complex_T

export compute_EOMs, check_EOMs

function compute_EOMs(S::Basic, variables)
    dS = Dict{Basic,Basic}()

    # Avoid differentiating twice if a variable occurs more than once in the
    # requested list. SolveData is normally already unique.
    for variable in variables
        haskey(dS, variable) && continue
        dS[variable] = SymEngine.diff(S, variable)
    end

    return dS
end

compute_EOMs(S::Basic, sd::SolveData) = compute_EOMs(S, sd.labels_vars)

function check_EOMs(dS::AbstractDict{Basic,Basic}, sd::SolveData{T}; γ=one(T)) where {T<:Real}
    tol = T(get_tolerance())
    γsym = symbols("gamma")
    vals = build_value_dict(sd, γsym; γval=γ)
    all_zero = true

    for (variable, expression) in dS
        value = to_complex_T(eval_symbolic(expression, vals), T)
        re_value = abs(real(value))
        im_value = abs(imag(value))

        if re_value > tol || im_value > tol
            println("✘ dS/d$(variable) ≠ 0")
            println("|Re| = $re_value, |Im| = $im_value")
            all_zero = false
        end
    end

    if all_zero
        println("✔ All equations of motion satisfied (γ = $γ, tol = $tol).")
    else
        println("✘ Some equations of motion are NOT satisfied.")
    end

    return nothing
end

end
