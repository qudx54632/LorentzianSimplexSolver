module CriticalPoints

using LinearAlgebra

export compute_critical_data

# ------------------------------------------------------------
# 1. ξ critical parameters (xisoln)
# ------------------------------------------------------------
function compute_xisoln(bdyxi, sgndet, tetareasign, tetn0)
    ns   = length(bdyxi)
    ntet = length(bdyxi[1])

    # xisol[k][i][j] = Vector{Float64} of length 2
    xisol = Vector{Vector{Vector{Vector{Float64}}}}(undef, ns)

    for k in 1:ns
        xisol[k] = Vector{Vector{Vector{Float64}}}(undef, ntet)

        for i in 1:ntet
            nf = length(bdyxi[k][i])
            xisol[k][i] = Vector{Vector{Float64}}(undef, nf)

            for j in 1:nf
                ξ1 = bdyxi[k][i][j][1]  # spinor 1 (2-complex)
                ξ2 = bdyxi[k][i][j][2]  # spinor 2 (2-complex)

                if sgndet[k][i] == 1
                    # spacelike tetrahedron
                    θ  = asin(abs(ξ1[1]))
                    φ  = angle(ξ1[2]) - angle(ξ1[1])
                    xisol[k][i][j] = [θ, φ]

                elseif tetareasign[k][i][j] == 1
                    # timelike face, positive tetareasign
                    if tetn0[k][i][j] == 1
                        θ = acosh(abs(ξ1[1]))
                        φ = angle(ξ1[1]) - angle(ξ1[2])
                        xisol[k][i][j] = [θ, φ]
                    else
                        θ = acosh(abs(ξ2[1]))
                        φ = angle(ξ2[1]) - angle(ξ2[2])
                        xisol[k][i][j] = [θ, φ]
                    end

                else
                    # null / other case: {Arg(ξ12/ξ11), ξ11}
                    θ = angle(ξ1[2] / ξ1[1])
                    # store the real part of ξ11 as the second entry (MMA used the complex number directly)
                    xisol[k][i][j] = [θ, real(ξ1[1])]
                end
            end
        end
    end

    return xisol
end

# ------------------------------------------------------------
# 2. gdataof = inv(transpose(sl2c))
# ------------------------------------------------------------
function compute_gdataof(sl2c_all)
    ns   = length(sl2c_all)
    ntet = length(sl2c_all[1])

    return [
        [ Matrix(inv(transpose(sl2c_all[k][i]))) for i in 1:ntet ]
        for k in 1:ns
    ]
end

# ------------------------------------------------------------
# 3. getz helper
# ------------------------------------------------------------
function getz(ginvT::Matrix{ComplexF64}, ξ::Vector{ComplexF64})
    v = ginvT * ξ
    if abs(real(v[1])) < 1e-12 && abs(imag(v[1])) < 1e-12
        return v ./ v[2]
    else
        return v ./ v[1]
    end
end

# ------------------------------------------------------------
# 4. zdataf
# ------------------------------------------------------------
function compute_zdataf(kappa, tetareasign, gdataof, bdyxi)
    ns   = length(bdyxi)
    ntet = length(bdyxi[1])

    zdata = Vector{Vector{Vector{Vector{ComplexF64}}}}(undef, ns)

    for k in 1:ns
        zdata[k] = Vector{Vector{Vector{ComplexF64}}}(undef, ntet)

        for i in 1:ntet
            zdata[k][i] = Vector{Vector{ComplexF64}}(undef, ntet)

            for j in 1:ntet
                if kappa[k][i][j] == 1 && i != j
                    if tetareasign[k][i][j] > 0
                        ξ = bdyxi[k][i][j][1]
                    else
                        ξ = bdyxi[k][i][j][2]
                    end
                    ginvT = Matrix(inv(transpose(gdataof[k][i])))
                    zdata[k][i][j] = getz(ginvT, ξ)
                else
                    zdata[k][i][j] = ComplexF64[0, 0]
                end
            end
        end
    end

    return zdata
end

# ------------------------------------------------------------
# 5. spins j from areas and tetareasign
# ------------------------------------------------------------
function compute_spins(areas, tetareasign, gamma::Real)
    ns   = length(areas)
    ntet = length(areas[1])

    γ = float(gamma)

    jcrit = Vector{Vector{Vector{Float64}}}(undef, ns)

    for s in 1:ns
        jcrit[s] = Vector{Vector{Float64}}(undef, ntet)

        for i in 1:ntet
            jcrit[s][i] = Vector{Float64}(undef, ntet)

            for j in 1:ntet
                A = areas[s][i][j]
                if tetareasign[s][i][j] == 1
                    # spacelike: area = γ j  ⇒ j = area / γ
                    jcrit[s][i][j] = A / γ
                else
                    # timelike or others: area = j
                    jcrit[s][i][j] = A
                end
            end
        end
    end

    return jcrit
end

# ------------------------------------------------------------
# 6. main driver
# ------------------------------------------------------------
function compute_critical_data(geom; gamma::Real)
    ns = length(geom.simplex)

    xi_final    = [geom.simplex[i].bdyxi       for i in 1:ns]
    sgndet      = [geom.simplex[i].sgndet      for i in 1:ns]
    tetareasign = [geom.simplex[i].tetareasign for i in 1:ns]
    tetn0       = [geom.simplex[i].tetn0sign   for i in 1:ns]
    kappa       = [geom.simplex[i].kappa       for i in 1:ns]

    sl2c4 = [geom.simplex[s].solgsl2c for s in 1:ns]
    areas = [geom.simplex[s].areas    for s in 1:ns]

    xisoln  = compute_xisoln(xi_final, sgndet, tetareasign, tetn0)
    gdataof = compute_gdataof(sl2c4)
    zdataf  = compute_zdataf(kappa, tetareasign, gdataof, xi_final)
    jdataf  = compute_spins(areas, tetareasign, gamma)

    return (gdataof = gdataof,
            xisoln  = xisoln,
            zdataf  = zdataf,
            jdataf  = jdataf)
end

end # module










