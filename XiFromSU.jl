module XiFromSU

using LinearAlgebra

export get_xi_from_su

"""
    get_xi_from_su(hsu, sgndet, facesign, tetsign0)

Compute boundary spinors ξ from an SU(2) / SU(1,1) matrix `hsu`.

Inputs:
- `hsu`        :: 2×2 matrix (Complex{T})
- `sgndet`     :: Int, sign of tetra (±1)
- `facesign`   :: Int, sign of face
- `tetsign0`   :: Int, future / past sign of the timelike face-normal

Output:
- Vector `[ξ1, ξ2]`, each a length-2 complex vector.
"""
function get_xi_from_su(hsu::AbstractMatrix{Complex{T}},
                        sgndet::Int,
                        facesign::Int,
                        tetsign0::Int) where {T<:Real}

    z = zero(T)
    o = one(T)

    # Basis spinors
    e1 = Complex{T}[o, z]
    e2 = Complex{T}[z, o]
    invsqrt2 = inv(sqrt(T(2)))

    if sgndet > 0
        # SU(2) case: {hsu⋅(1,0), hsu⋅(0,1)}
        ξ1 = hsu * e1
        ξ2 = hsu * e2
        return [ξ1, ξ2]
    else
        if facesign > 0
            if tetsign0 > 0
                ξ1 = hsu * e1
                ξ2 = hsu * e2
                return [ξ1, ξ2]
            else
                ξ1 = hsu * e2
                ξ2 = hsu * e1
                return [ξ1, ξ2]
            end
        else
            # {hsu⋅(1,1)/√2, hsu⋅(1,-1)/√2}
            vplus  = invsqrt2 * Complex{T}[o,  o]
            vminus = invsqrt2 * Complex{T}[o, -o]
            ξ1 = hsu * vplus
            ξ2 = hsu * vminus
            return [ξ1, ξ2]
        end
    end
end

end # module