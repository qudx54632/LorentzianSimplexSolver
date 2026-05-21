module ActionEvaluation

using SymEngine
using DoubleFloats

export build_value_dict, eval_symbolic


# ------------------------------------------------------------
# Safe conversion to SymEngine.Basic
# ------------------------------------------------------------
@inline val_basic(x) = x isa Double64 ? Basic(string(x)) : Basic(x)

function build_value_dict(sd, γsym::Basic; γval=nothing)
    d = Dict{Basic,Basic}()

    for i in eachindex(sd.labels_vars)
        sym = sd.labels_vars[i]
        val = sd.flags_vars[i] ?
            (γval === nothing ?
                val_basic(sd.values_vars[i]) / γsym :
                val_basic(sd.values_vars[i]) / val_basic(γval)) :
            val_basic(sd.values_vars[i])

        d[sym] = val
    end

    for i in eachindex(sd.labels_bdry)
        sym = sd.labels_bdry[i]
        val = sd.flags_bdry[i] ?
            (γval === nothing ?
                val_basic(sd.values_bdry[i]) / γsym :
                val_basic(sd.values_bdry[i]) / val_basic(γval)) :
            val_basic(sd.values_bdry[i])

        d[sym] = val
    end

    for i in eachindex(sd.labels_η)
        sym = sd.labels_η[i]
        val = sd.flags_η[i] ?
            (γval === nothing ?
                val_basic(sd.values_η[i]) / γsym :
                val_basic(sd.values_η[i]) / val_basic(γval)) :
            val_basic(sd.values_η[i])

        d[sym] = val
    end

    d[γsym] = γval === nothing ? γsym : val_basic(γval)

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

end