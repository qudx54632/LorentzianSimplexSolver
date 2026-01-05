module KappaFromNormals

using LinearAlgebra
using ..PrecisionUtils: get_tolerance

export compute_kappa

# ------------------------------------------------------------
# Internal test matrix κ_test
# ------------------------------------------------------------
function kappatest(nabout, nabfrombivec, tetareasign)

    Ntet = length(tetareasign)

    # infer scalar type from data
    T = eltype(eltype(eltype(nabout)))
    tol = T(get_tolerance())

    return [
        [
            k == i ? 0 :
            begin
                # geometric normal times area sign
                v_geom = tetareasign[k][i] .* nabout[k][i]
                v_biv  = nabfrombivec[k][i]

                # find first component where both are nonzero
                idx = findfirst(
                    j -> abs(v_geom[j]) > tol && abs(v_biv[j]) > tol,
                    eachindex(v_geom)
                )

                if idx === nothing
                    0
                else
                    r = v_geom[idx] / v_biv[idx]   # r :: T
                    r > zero(T) ? 1 : -1
                end
            end
            for i in 1:Ntet
        ]
        for k in 1:Ntet
    ]
end

# ------------------------------------------------------------
# Public κ computation
# ------------------------------------------------------------
function compute_kappa(nabout, nabfrombivec, tetareasign)

    κtest = kappatest(nabout, nabfrombivec, tetareasign)

    Ntet = length(κtest)
    κ = Vector{Vector{Int}}(undef, Ntet)

    # First row defines reference orientation
    κ[1] = Int.(κtest[1])

    tol = get_tolerance()  # tolerance here is only for integer sanity checks

    # Remaining rows fixed relative to first row
    for i in 2:Ntet

        a = κtest[i][1]
        b = κtest[1][i]

        abs(a) > tol || error("compute_kappa: zero reference entry κ[$i,1]")
        abs(b) > tol || error("compute_kappa: zero reference entry κ[1,$i]")

        # relative orientation factor (must be ±1)
        denom = a / b
        abs(abs(denom) - 1) < tol ||
            error("compute_kappa: inconsistent orientation ratio = $denom")

        κ[i] = [
            j == i ? 0 :
            -Int(round(κtest[i][j] / denom))
            for j in 1:Ntet
        ]
    end

    return κ
end

end # module