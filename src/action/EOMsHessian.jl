module EOMsHessian

using SymEngine
using ..SolveVars: SolveData
using ..PrecisionUtils: get_tolerance
using ..ActionEvaluation: build_value_dict, eval_symbolic

export compute_EOMs,
       compute_Hessian_block,
       compute_Hessian_block_half,
       check_EOMs,
       evaluate_hessian_block,
       evaluate_hessian_from_dS,
       evaluate_hessian_ondemand

# ============================================================
# helper function to convert SymEngine object to numeric type T
# ============================================================
@inline function to_T(x, ::Type{T}) where {T<:Real}
    # If already a Julia real number
    if x isa Real
        return T(x)
    end

    # Otherwise force string conversion
    sx = string(x)

    try
        return parse(T, sx)
    catch
        error("Cannot convert to $T: $sx (type = $(typeof(x)))")
    end
end

@inline function to_complex_T(val_sym, ::Type{T}) where {T<:Real}
    re = to_T(real(val_sym), T)
    im = to_T(imag(val_sym), T)
    return complex(re, im)
end

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
function compute_Hessian_block(S::Basic, vars)
    n = length(vars)
    dS = [SymEngine.diff(S, v) for v in vars]
    H = Matrix{Basic}(undef, n, n)

    for i in 1:n
        H[i, i] = SymEngine.diff(dS[i], vars[i])
        for j in i+1:n
            hij = SymEngine.diff(dS[i], vars[j])
            H[i, j] = hij
            H[j, i] = hij
        end
    end

    return H
end

# ============================================================
# Hessian
# ============================================================
function compute_Hessian_block_half(S::Basic, vars)
    n = length(vars)
    dS = [SymEngine.diff(S, v) for v in vars]
    H = Matrix{Basic}(undef, n, n)

    for i in 1:n
        H[i, i] = SymEngine.diff(dS[i], vars[i])
        for j in i+1:n
            hij = SymEngine.diff(dS[i], vars[j])
            H[i, j] = hij
            H[j, i] = Basic(0)
        end
    end

    return H
end

# ============================================================
# Evaluate EOMs numerically
# ============================================================
function check_EOMs(dS::Dict{Basic,Basic}, sd::SolveData{T}; γ=1) where {T<:Real}

    tol = get_tolerance()
    γsym = symbols("gamma")

    vals = build_value_dict(sd, γsym; γval=γ)

    all_zero = true

    for (v, expr) in dS
        val_sym = eval_symbolic(expr, vals)

        # convert to numeric
        val = to_complex_T(val_sym, T)

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
# Evaluate a precomputed symbolic Hessian block
# ============================================================
function evaluate_hessian_block(Hsym::Matrix{Basic},
                                sd::SolveData{T};
                                γ = one(T)) where {T<:Real}

    γsym = symbols("gamma")
    vals = build_value_dict(sd, γsym; γval=γ)

    n = size(Hsym, 1)
    H = Matrix{Complex{T}}(undef, n, n)

    for j in 1:n
        for i in 1:j
            val_sym = eval_symbolic(Hsym[i, j], vals)

            val = to_complex_T(val_sym, T)

            H[i, j] = val
            H[j, i] = val
        end
    end

    return H
end

# ============================================================
# Evaluate Hessian from precomputed first derivatives
# ============================================================
function evaluate_hessian_from_dS(
    dS_precomp::Vector{Basic},
    vars::Vector{Basic},
    sd::SolveData{T};
    γ = one(T)
) where {T<:Real}

    γsym = symbols("gamma")
    vals = build_value_dict(sd, γsym; γval=γ)

    n = length(vars)
    H = Matrix{Complex{T}}(undef, n, n)

    for j in 1:n
        for i in 1:j
            hij_sym = SymEngine.diff(dS_precomp[i], vars[j])

            val_sym = eval_symbolic(hij_sym, vals)

            val = to_complex_T(val_sym, T)

            H[i, j] = val
            H[j, i] = val
        end
    end

    return H
end

# ============================================================
# No symbolic Hessian storage:
# differentiate and evaluate on demand
# ============================================================
function evaluate_hessian_ondemand(S::Basic,
                                   vars,
                                   sd::SolveData{T};
                                   γ = one(T)) where {T<:Real}

    γsym = symbols("gamma")
    vals = build_value_dict(sd, γsym; γval=γ)

    n = length(vars)
    H = Matrix{Complex{T}}(undef, n, n)

    # first derivatives once
    dS = Vector{Basic}(undef, n)
    for i in 1:n
        dS[i] = SymEngine.diff(S, vars[i])
    end

    for j in 1:n
        for i in 1:j
            hij_sym = SymEngine.diff(dS[i], vars[j])
            val_sym = eval_symbolic(hij_sym, vals)

            val = to_complex_T(val_sym, T)

            H[i, j] = val
            H[j, i] = val
        end
    end

    return H
end

end # module