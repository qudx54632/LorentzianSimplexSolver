module SafeSqrtLog

using PythonCall
const sympy = pyimport("sympy")

export safe_sqrt, safe_log

function sympy_chop(z::Py; tol=1e-12)
    re = sympy.re(z)
    im = sympy.im(z)

    return sympy.Piecewise((re, sympy.Abs(im) < tol),
        (sympy.I * im, sympy.Abs(re) < tol),
        (z, true))
end

@inline function safe_sqrt(z::Py; tol=1e-12)
    return sympy.sqrt(sympy_chop(z; tol=tol))
end

@inline function safe_log(z::Py; tol=1e-12)
    return sympy.log(sympy_chop(z; tol=tol))
end
    
end