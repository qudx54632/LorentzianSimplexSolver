module ActionEvaluation

using SymEngine
using DoubleFloats: Double64

export build_value_dict, eval_symbolic, to_complex_T


# ------------------------------------------------------------
# Safe conversion to SymEngine.Basic
# ------------------------------------------------------------
@inline val_basic(x) = x isa Double64 ? Basic(string(x)) : Basic(x)

function add_value_group!(values_dict, labels, values, flags, gamma_denominator)
    @assert length(labels) == length(values) == length(flags)

    for i in eachindex(labels)
        value = val_basic(values[i])
        values_dict[labels[i]] = flags[i] ? value / gamma_denominator : value
    end

    return values_dict
end

function build_value_dict(sd, γsym::Basic; γval=nothing)
    d = Dict{Basic,Basic}()
    gamma_denominator = γval === nothing ? γsym : val_basic(γval)

    add_value_group!(d, sd.labels_vars, sd.values_vars, sd.flags_vars, gamma_denominator)
    add_value_group!(d, sd.labels_bdry, sd.values_bdry, sd.flags_bdry, gamma_denominator)
    add_value_group!(d, sd.labels_η, sd.values_η, sd.flags_η, gamma_denominator)

    d[γsym] = gamma_denominator

    return d
end

# ------------------------------------------------------------
# Evaluate symbolic expression
# ------------------------------------------------------------
function eval_symbolic(expr::Basic, vals::Dict{Basic,Basic})
    return subs(expr, vals)
end

eval_symbolic(x::Number, vals) = x

function eval_symbolic(A::AbstractArray, vals)
    return map(x -> eval_symbolic(x, vals), A)
end

# SymEngine numbers need an explicit conversion so Float64 and BigFloat
# workflows keep the scalar type selected in PrecisionUtils.
@inline function to_T(x, ::Type{T}) where {T<:Real}
    x isa Real && return T(x)

    sx = string(x)
    try
        return parse(T, sx)
    catch
        error("Cannot convert to $T: $sx (type = $(typeof(x)))")
    end
end

@inline function to_complex_T(x, ::Type{T}) where {T<:Real}
    return complex(to_T(real(x), T), to_T(imag(x), T))
end

end
