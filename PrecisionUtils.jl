module PrecisionUtils

export set_big_precision!,
       get_tolerance,
       set_tolerance!,
       parse_numeric_line

# ----------------------------
# Global scalar type + tolerance
# ----------------------------
const _TOLERANCE = Ref{Real}(1e-10)

# ----------------------------
# Precision control
# ----------------------------
"""
    set_big_precision!(p)

Enable BigFloat arithmetic with precision `p`.
"""
function set_big_precision!(p::Integer; tol=nothing)
    setprecision(p)
    _TOLERANCE[] = isnothing(tol) ? sqrt(eps(BigFloat)) : tol
    return nothing
end

# ----------------------------
# Tolerance API
# ----------------------------
get_tolerance() = _TOLERANCE[]

function set_tolerance!(x::Real)
    _TOLERANCE[] = x
end

# ----------------------------
# Parsing utilities
# ----------------------------
"""
    parse_numeric_line(line)

Parse a line like "0, 1, 2, 3" into Vector{ScalarT}.
"""
# in PrecisionUtils.jl
function parse_numeric_line(line::String, ::Type{T}) where {T<:Real}
    expr = Meta.parse("[$line]")
    vals = eval(expr)
    all(x -> x isa Number, vals) || error("Non-numeric input detected: $line")
    return T.(vals)
end

end # module