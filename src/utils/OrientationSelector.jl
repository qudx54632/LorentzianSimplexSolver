module OrientationSelector

using Symbolics
using ..PrecisionUtils: get_tolerance

export select_orientation

@inline function small(x, tol)
    try
        if tol isa BigFloat
            return abs(BigFloat(x)) ≤ tol
        else
            return abs(Float64(x)) ≤ tol
        end
    catch
        return false
    end
end

function split_gamma(S, γ::Num)

    # Separate numerator / denominator
    num = Symbolics.numerator(S)
    den = Symbolics.denominator(S)

    # If there is no γ in denominator, nothing to do
    if !isequal(den, γ)
        return S
    end

    # Expand numerator
    num = Symbolics.expand(num)

    # Extract coefficients
    c2 = Symbolics.coeff(num, γ^2)
    c1 = Symbolics.coeff(num, γ)
    c0 = Symbolics.expand(num - c1*γ - c2*γ^2)

    return c0/γ + c1 + c2*γ
end

function select_orientation(
    S_ref::Complex{Num},
    S_parity::Complex{Num},
    S_regge::Num,
    γ::Num
)
    tol = get_tolerance()
    # phase
    phase = expand(simplify((S_ref + S_parity) // 2))

    expr_ref    = expand(simplify(S_ref    - phase))
    expr_parity = expand(simplify(S_parity - phase))

    # variables from Regge
    F_regge = Symbolics.coeff(S_regge, γ)
    vars = Symbolics.get_variables(F_regge)

    orientation = nothing

    for v in vars
        cg = Symbolics.coeff(F_regge, v)

        if abs(cg) > tol
            cr0 = Symbolics.coeff(imag(expr_ref),    v)
            cr =  Symbolics.coeff(split_gamma(cr0, γ), γ)
            cp0 = Symbolics.coeff(imag(expr_parity), v)
            cp = Symbolics.coeff(split_gamma(cp0, γ), γ)
            

            if small(cr - cg, tol)
                orientation = :ref_pos
            elseif small(cr + cg, tol)
                orientation = :ref_neg
            elseif small(cp - cg, tol)
                orientation = :parity_pos
            elseif small(cp + cg, tol)
                orientation = :parity_neg
            else
                error("No orientation match for variable $v")
            end
        end
    end
    return orientation
end

end # module