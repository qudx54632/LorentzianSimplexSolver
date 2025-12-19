module EOMs

using PythonCall
sympy = pyimport("sympy")

export compute_EOMs, zero_or_not

γsym = sympy.symbols("gamma", real=true)

function compute_EOMs(S::Py, sol_vars::Dict{Py,Py}; γ::Union{Py,Nothing}=γsym,γval=1)

    dS_eval = Dict{Py,Py}()

    for v in keys(sol_vars)
        # 1. symbolic derivative
        expr = sympy.diff(S, v)

        # 2. substitute solution
        expr = expr.subs(sol_vars)

        # 3. substitute gamma if requested
         if γ !== nothing
            expr = expr.subs(γ, γval)
        end

        # 4. numeric evaluation
        expr = expr.evalf(chop=true)

        dS_eval[v] = expr
    end

    return dS_eval
end

function zero_or_not(dS_eval; tol = 1e-10)

    all_zero = true

    for (v, val) in dS_eval
        # extract real / imaginary parts via SymPy
        re_part = sympy.re(val).evalf()
        im_part = sympy.im(val).evalf()

        # convert Python → Julia safely
        re_val = abs(pyconvert(Float64, re_part))
        im_val = abs(pyconvert(Float64, im_part))

        if re_val > tol || im_val > tol
            println("dS/d$(v) = $(val)  ≠ 0  ✘")
            all_zero = false
        end
    end

    if all_zero
        println("✔ All equations of motion are satisfied (within tol = $tol).")
    end

    return nothing
end

end # module