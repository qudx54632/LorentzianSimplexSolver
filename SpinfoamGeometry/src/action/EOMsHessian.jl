module EOMsHessian

using PythonCall
using ..SolveVars: SolveData
using ..PrecisionUtils: get_tolerance
using ..SymbolicToJulia: build_argument_vector


export compute_EOMs, compute_Hessian, check_EOMs, evaluate_hessian

const _sympy_ref = Ref{Union{Py,Nothing}}(nothing)

@inline function _sympy()
    s = _sympy_ref[]
    if s === nothing
        s = pyimport("sympy")
        _sympy_ref[] = s
    end
    return s
end
# ============================================================
# Equations of motion
#
# Input:
#   S  :: Py                 (SymPy expression)
#   sd :: SolveVars.SolveData
#
# Output:
#   Dict{Py,Py}   v ↦ ∂S/∂v
# ============================================================
function compute_EOMs(S::Py, sd::SolveData)
    sp = _sympy()
    dS = Dict{Py,Py}()

    for v in sd.labels_vars
        dS[v] = sp.diff(S, v)
    end

    return dS
end


# ============================================================
# Hessian
#
# Input:
#   S  :: Py
#   sd :: SolveVars.SolveData
#
# Output:
#   Dict{Tuple{Py,Py},Py}   (v,w) ↦ ∂²S/∂v∂w
# ============================================================
function compute_Hessian(S::Py, sd::SolveData)
    sp = _sympy()
    vars = sd.labels_vars
    H = Dict{Tuple{Py,Py},Py}()

    for v1 in vars
        dS_v1 = sp.diff(S, v1)
        for v2 in vars
            H[(v1, v2)] = sp.diff(dS_v1, v2)
        end
    end

    return H
end

# ------------------------------------------------------------
# Evaluate all gradient functions at γ = 1 and check EOMs
# ------------------------------------------------------------
function check_EOMs(grad_fns::Dict{Py,Function}, sd::SolveData; γ = 1)
    sp = _sympy()
    tol = get_tolerance()
    all_zero = true
    args = build_argument_vector(sd, vcat(sd.labels_vars, sd.labels_bdry), γ)

    for (v, dS_fn) in grad_fns

        val = dS_fn(args...)

        # ----------------------------------------------------
        # Extract real / imaginary parts
        # ----------------------------------------------------
        if val isa Real
            re_val = abs(val)
            im_val = 0.0

        elseif val isa Complex
            re_val = abs(real(val))
            im_val = abs(imag(val))
        else
            error("Unsupported value type: $(typeof(val))")
        end

        # ----------------------------------------------------
        # Threshold test
        # ----------------------------------------------------
        if re_val > tol || im_val > tol
            println("✘ dS/d$(sp.sstr(v)) ≠ 0")
            println("    |Re| = $re_val, |Im| = $im_val")
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


function evaluate_hessian(hess_fns::Dict{Tuple{Py,Py}, Function},
                          sd::SolveData{T};
                          γ = one(T)) where {T<:Real}
    sp = _sympy()
    # Hessian is only over variables used in differentiation
    labels = sd.labels_vars
    n = length(labels)

    # symbol -> index (vars only)
    index = Dict(pyconvert(String, sp.sstr(v)) => i
                 for (i, v) in enumerate(labels))

    # allocate
    H = Matrix{Complex{T}}(undef, n, n)
    fill!(H, zero(Complex{T}))

    # NOTE: your generated functions expect (vars..., bdry..., gamma)
    # so argument vector must still include BOTH vars and bdry.
    full_labels = vcat(sd.labels_vars, sd.labels_bdry)
    args = build_argument_vector(sd, full_labels, γ)

    # loop upper triangle only
    for ((v1, v2), h_fn) in hess_fns
        i = index[pyconvert(String, sp.sstr(v1))]
        j = index[pyconvert(String, sp.sstr(v2))]
        j < i && continue

        val = h_fn(args...)

        hij = if val isa Real
            Complex{T}(T(val), zero(T))
        elseif val isa Complex
            Complex{T}(T(real(val)), T(imag(val)))
        else
            error("Unsupported Hessian entry type: $(typeof(val))")
        end

        H[i, j] = hij
        H[j, i] = hij
    end

    return H, labels
end


end # module