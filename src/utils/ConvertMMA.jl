using JSON

function to_mathematica(z::ComplexF64; tol=1e-12)
    re = real(z)
    im = imag(z)

    # clean numerical noise
    re = abs(re) < tol ? 0.0 : re
    im = abs(im) < tol ? 0.0 : im

    if im == 0.0
        return string(re)
    elseif re == 0.0
        return im > 0 ? "$(im) I" : "-$(-im) I"
    elseif im > 0
        return "$(re) + $(im) I"
    else
        return "$(re) - $(-im) I"
    end
end

function to_mathematica(z::ComplexF64; tol=1e-12)
    re = real(z)
    im = imag(z)

    # clean numerical noise
    re = abs(re) < tol ? 0.0 : re
    im = abs(im) < tol ? 0.0 : im

    if im == 0.0
        return string(re)
    elseif re == 0.0
        return im > 0 ? "$(im) I" : "-$(-im) I"
    elseif im > 0
        return "$(re) + $(im) I"
    else
        return "$(re) - $(-im) I"
    end
end

function matrix_to_mathematica(mat)
    rows = [
        "{" * join([to_mathematica(x) for x in row], ", ") * "}"
        for row in eachrow(mat)
    ]
    return "{" * join(rows, ", ") * "}"
end

function nested_to_mathematica(x)
    if x isa Matrix
        return matrix_to_mathematica(x)
    elseif x isa AbstractArray
        return "{" * join(nested_to_mathematica.(x), ", ") * "}"
    else
        error("Unexpected type: $(typeof(x))")
    end
end

function full_to_mathematica(data)
    return "{" * join([
        "{" * join([matrix_to_mathematica(mat) for mat in block], ", ") * "}"
        for block in data
    ], ", ") * "}"
end


function matrix_to_mathematica(mat)
    rows = [
        "{" * join([to_mathematica(x) for x in row], ", ") * "}"
        for row in eachrow(mat)
    ]
    return "{" * join(rows, ", ") * "}"
end

function nested_to_mathematica(x)
    if x isa Matrix
        return matrix_to_mathematica(x)
    elseif x isa AbstractArray
        return "{" * join(nested_to_mathematica.(x), ", ") * "}"
    else
        error("Unexpected type: $(typeof(x))")
    end
end

function full_to_mathematica(x)
    if x isa ComplexF64
        return to_mathematica(x)
    elseif x isa AbstractArray
        return "{" * join(full_to_mathematica.(x), ", ") * "}"
    else
        error("Unexpected type: $(typeof(x))")
    end
end


open("bdyxi.m", "w") do io
    write(io, "bdyxijulia = " * full_to_mathematica(bdyxi) * ";")
end