module CriticalPoints

using LinearAlgebra
using ..PrecisionUtils: get_tolerance
using ..SpinAlgebra: imag_unit

export compute_bdy_critical_data

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
_realT(::Type{Complex{T}}) where {T<:Real} = T
_realT(::Type{T}) where {T<:Real} = T

# ------------------------------------------------------------
# 1. ξ critical parameters (xisoln)
# ------------------------------------------------------------
function compute_xisoln(bdyxi, sgndet, tetareasign, tetn0)
    ns   = length(bdyxi)
    ntet = length(bdyxi[1])

    CT = eltype(bdyxi[1][1][1][1])     # Complex{T}
    T  = _realT(CT)
    # xisol[k][i][j] = Vector{Float64} of length 2
    xisol = Vector{Vector{Vector{Vector{T}}}}(undef, ns)

    for k in 1:ns
        xisol[k] = Vector{Vector{Vector{T}}}(undef, ntet)

        for i in 1:ntet
            nf = length(bdyxi[k][i])
            xisol[k][i] = Vector{Vector{T}}(undef, nf)

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

    CT = eltype(sl2c_all[1][1]) # Complex{T}
    out = Vector{Vector{Matrix{CT}}}(undef, ns)

    for k in 1:ns
        out[k] = Vector{Matrix{CT}}(undef, ntet)
        for i in 1:ntet
            out[k][i] = inv(transpose(sl2c_all[k][i]))
        end
    end

    return out
end

# ------------------------------------------------------------
# 3. getz helper
# ------------------------------------------------------------
function getz(ginvT::AbstractMatrix{Complex{T}}, ξ::AbstractVector{Complex{T}}) where {T<:Real}
    v = ginvT * ξ
    tol = T(get_tolerance())

    if abs(real(v[1])) < tol && abs(imag(v[1])) < tol
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

    CT = eltype(bdyxi[1][1][1][1])
    T  = _realT(CT)
    zeroz = Complex{T}[zero(T), zero(T)]

    zdata = Vector{Vector{Vector{Vector{Complex{T}}}}}(undef, ns)

    for k in 1:ns
        zdata[k] = Vector{Vector{Vector{Complex{T}}}}(undef, ntet)

        for i in 1:ntet
            zdata[k][i] = Vector{Vector{Complex{T}}}(undef, ntet)

            for j in 1:ntet
                if kappa[k][i][j] == 1 && i != j
                    ξ = (tetareasign[k][i][j] > 0) ? bdyxi[k][i][j][1] : bdyxi[k][i][j][2]
                    ginvT = Matrix(inv(transpose(gdataof[k][i])))
                    zdata[k][i][j] = getz(ginvT, ξ)
                else
                    zdata[k][i][j] = copy(zeroz)
                end
            end
        end
    end

    return zdata
end

# ------------------------------------------------------------
# 6. main driver
# ------------------------------------------------------------
function compute_bdy_critical_data(geom)
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
    areadataf  = areas

    return (gdataof = gdataof,
            xisoln  = xisoln,
            zdataf  = zdataf,
            areadataf  = areadataf)
end

end # module



