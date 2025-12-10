module KappaFromNormals

using LinearAlgebra

export compute_kappa

function kappatest(nabout, nabfrombivec, tetareasign; tol=1e-12)
    Ntet = length(tetareasign)

    return [
        [
            k == i ? 0 :
            begin
                # geometric normal times area sign
                v_geom = tetareasign[k][i] .* nabout[k][i]
                v_biv  = nabfrombivec[k][i]

                # find first component where both are nonzero (after "chop")
                idx = findfirst(j -> abs(v_geom[j]) > tol && abs(v_biv[j]) > tol,
                                eachindex(v_geom))

                if idx === nothing
                    0
                else
                    r = v_geom[idx] / v_biv[idx]
                    r > 0 ? 1 : -1
                end
            end
            for i in 1:Ntet
        ]
        for k in 1:Ntet
    ]
end

function compute_kappa(nabout, nabfrombivec, tetareasign; tol=1e-12)

    kappatest_vals = kappatest(nabout, nabfrombivec, tetareasign)
    Ntet = length(kappatest_vals)
    kappa = Vector{Vector{Int}}(undef, Ntet)
    kappa[1] = [Int(x) for x in kappatest_vals[1]]

    # Remaining rows
    for i in 2:Ntet
        denom = kappatest_vals[i][1] / kappatest_vals[1][i]
        kappa[i] = [-Int(kappatest_vals[i][j] / denom) for j in 1:Ntet]
    end

    return kappa
end

end # module