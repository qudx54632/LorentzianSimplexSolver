module DefineAction

using LinearAlgebra
using ..SpinAlgebra: σ3

export halfedgeaction,
       halfedgeactiont,
       vertexaction

# ============================================================
# Half-edge action (spacelike face)
# ============================================================

function halfedgeaction(xi, z, g, κ, meta; signzz=1, γ=1)
    # scalar contractions
    A = :( $signzz * conj($xi)' * $meta * transpose($g) * $z )
    B = :( $signzz * conj(transpose($g) * $z)' * $meta * $xi )

    term1 = :( 2 * log( ($A)^(( $κ + det($meta) ) / 2)
                        * ($B)^(( -$κ + det($meta) ) / 2) ) )

    term2 = :( (im * $γ * $κ - det($meta)) *
               log( $signzz * conj(transpose($g) * $z)' * $meta * transpose($g) * $z ) )

    return :( $term1 + $term2 )
end


# ============================================================
# Half-edge action (timelike face)
# ============================================================

function halfedgeactiont(xi, z, g, κ, meta; signzz=1, γ=1)
    A = :( $signzz * conj($xi)' * $meta * transpose($g) * $z )
    B = :( $signzz * conj(transpose($g) * $z)' * $meta * $xi )

    term1 = :( 2 * $κ * log( sqrt( $A / $B ) ) )
    term2 = :( -im / $γ * $κ * log( $A * $B ) )

    return :( $term1 + $term2 )
end


# ============================================================
# Vertex action (single 4-simplex)
# ============================================================

"""
    vertexaction(jfdata, xidata, zdata, gdata,
                 κdata, sgndet, tetn0sign, facesigndata;
                 γ=1)

Direct translation of Mathematica `vertexaction`.
All inputs are 5×5 or length-5 arrays, exactly as in DefineVariables.
"""
function vertexaction(jfdata, xidata, zdata, gdata,
                      κdata, sgndet, tetn0sign, facesigndata;
                      γ=1)

    ntet = 5
    act_terms = Expr[]

    for i in 1:ntet, j in 1:ntet
        i == j && continue

        jf = jfdata[i][j]
        jf == 0 && continue

        κ = κdata[i][j]

        # choose z orientation
        z = κ == 1 ? zdata[i][j] : zdata[j][i]

        if facesigndata[i][j] > 0
            meta = sgndet[i] == 1 ? I(2) : σ3
            signzz = tetn0sign[i][j]

            he = halfedgeaction(
                xidata[i][j], z, gdata[i],
                κ, meta;
                signzz=signzz, γ=γ
            )
        else
            he = halfedgeactiont(
                xidata[i][j], z, gdata[i],
                κ, σ3;
                signzz=1, γ=γ
            )
        end

        push!(act_terms, :( $jf * $he ))
    end

    return foldl((a,b)->:( $a + $b ), act_terms; init=:(0))
end

end # module