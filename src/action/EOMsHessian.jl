module EOMsHessian

using SymEngine
using ..SolveVars: SolveData
using ..PrecisionUtils: get_tolerance
using ..ActionEvaluation: build_value_dict, eval_symbolic

export compute_EOMs, compute_Hessian, check_EOMs, evaluate_hessian

# ============================================================
# Equations of motion
# ============================================================
function compute_EOMs(S::Basic, sd::SolveData)
    dS = Dict{Basic,Basic}()

    for v in sd.labels_vars
        dS[v] = SymEngine.diff(S, v)
    end

    return dS
end

# ============================================================
# Hessian
# ============================================================
function compute_Hessian(S::Basic, sd::SolveData)
    vars = sd.labels_vars
    H = Dict{Tuple{Basic,Basic},Basic}()

    for v1 in vars
        dS_v1 = SymEngine.diff(S, v1)
        for v2 in vars
            H[(v1, v2)] = SymEngine.diff(dS_v1, v2)
        end
    end

    return H
end

# ============================================================
# Evaluate EOMs numerically
# ============================================================
function check_EOMs(dS::Dict{Basic,Basic}, sd::SolveData; γ=1)

    tol = get_tolerance()
    γsym = symbols("gamma")

    vals = build_value_dict(sd, γsym; γval=γ)

    all_zero = true

    for (v, expr) in dS
        val_sym = eval_symbolic(expr, vals)

        # convert to numeric
        val = complex(Float64(N(real(val_sym))),
                      Float64(N(imag(val_sym))))

        re_val = abs(real(val))
        im_val = abs(imag(val))

        if re_val > tol || im_val > tol
            println("✘ dS/d$(v) ≠ 0")
            println("|Re| = $re_val, |Im| = $im_val")
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

# ============================================================
# Evaluate Hessian numerically
# ============================================================
function evaluate_hessian(Hsym::Dict{Tuple{Basic,Basic},Basic},
                          sd::SolveData{T};
                          γ = one(T)) where {T<:Real}

    γsym = symbols("gamma")
    vals = build_value_dict(sd, γsym; γval=γ)

    labels = sd.labels_vars
    n = length(labels)

    index = Dict(string(v) => i for (i, v) in enumerate(labels))

    H = Matrix{Complex{T}}(undef, n, n)
    fill!(H, zero(Complex{T}))

    for ((v1, v2), expr) in Hsym
        i = index[string(v1)]
        j = index[string(v2)]
        j < i && continue

        val_sym = eval_symbolic(expr, vals)

        val = complex(Float64(N(real(val_sym))),
                      Float64(N(imag(val_sym))))

        H[i, j] = val
        H[j, i] = val
    end

    return H, labels
end

end # module