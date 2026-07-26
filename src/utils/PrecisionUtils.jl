module PrecisionUtils

export get_tolerance,
       parse_numeric_line

# The three numerical settings used by the whole package.
const ACTIVE_TYPE = Ref(Float64)
const BIGFLOAT_BITS = Ref(0)
const TOLERANCE = Ref{Real}(1e-10)

function configure_precision!(T; precision=100)
    if T === Float64
        TOLERANCE[] = 1e-10
    elseif T === BigFloat
        precision > 0 || error("BigFloat precision must be positive, got $precision.")
        setprecision(BigFloat, precision)
        BIGFLOAT_BITS[] = precision
        TOLERANCE[] = BigFloat("1e-12")
    else
        error("Supported scalar types are Float64 and BigFloat, got $T.")
    end

    ACTIVE_TYPE[] = T
    return TOLERANCE[]
end

get_tolerance() = TOLERANCE[]

function validate_active_precision(T)
    T === ACTIVE_TYPE[] || error("Expected $(ACTIVE_TYPE[]) numbers, got $T.")
    if T === BigFloat
        Base.precision(BigFloat) == BIGFLOAT_BITS[] ||
            error("BigFloat precision changed during the calculation.")
    end

    return nothing
end

function validate_precision(T, coordinates)
    validate_active_precision(T)
    all(eltype(point) === T for point in coordinates) ||
        error("All coordinates must use $T.")

    if T === BigFloat
        all(Base.precision(x) == BIGFLOAT_BITS[] for point in coordinates for x in point) ||
            error("Configure BigFloat precision before creating coordinates.")
    end

    return nothing
end

function validate_number_precision(value, T, name)
    validate_active_precision(T)
    value === nothing && return nothing
    (value isa Integer || value isa Rational) && return nothing
    value isa T || error("$name must use $T.")
    if T === BigFloat
        Base.precision(value) == BIGFLOAT_BITS[] ||
            error("$name uses a different BigFloat precision.")
    end

    return nothing
end

function parse_numeric_line(line, T)
    fields = split(strip(line), r"[,\s]+"; keepempty=false)
    isempty(fields) && error("No numeric input detected.")
    return parse.(T, fields)
end

end # module
