module PrecisionUtils

export set_big_precision!,
       get_tolerance,
       set_tolerance!,
       parse_numeric_line

const _TOLERANCE = Ref{Real}(1e-10)

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

"""
    parse_numeric_line(line, T)

Parse a comma- or whitespace-separated line into `Vector{T}`.
"""
function parse_numeric_line(line::AbstractString, ::Type{T}) where {T<:Real}
    fields = split(strip(line), r"[,\s]+"; keepempty=false)
    isempty(fields) && error("No numeric input detected.")

    try
        return parse.(T, fields)
    catch err
        error("Could not parse numeric input \"$(String(line))\" as $T:\n$err")
    end
end

end # module
