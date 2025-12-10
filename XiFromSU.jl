module XiFromSU

using LinearAlgebra

export get_xi_from_su

"""
    get_xi_from_su(hsu, sgndet, facesign, tetsign0)

Compute boundary spinors ξ from an SU(2) / SU(1,1) matrix `hsu`
Inputs:
- `hsu`        :: 2×2 matrix (SU(2) or SU(1,1) element)
- `sgndet`     :: Int, sign of tetra (±1)
- `facesign`   :: Int, sign of face
- `tetsign0`   :: Int, future / past sign of the timelike face-normal

Output:
- A vector of two spinors `[ξ1, ξ2]`, each a length-2 complex vector.
"""
function get_xi_from_su(hsu::AbstractMatrix{<:Number},
                        sgndet::Int,
                        facesign::Int,
                        tetsign0::Int)

    # Basis spinors
    e1 = ComplexF64[1.0, 0.0]
    e2 = ComplexF64[0.0, 1.0]
    invsqrt2 = 1 / sqrt(2.0)

    if sgndet > 0
        # sgndet > 0: always {hsu⋅(1,0), hsu⋅(0,1)}
        ξ1 = hsu * e1
        ξ2 = hsu * e2
        return [ξ1, ξ2]
    else
        if facesign > 0
            if tetsign0 > 0
                # {hsu⋅(1,0), hsu⋅(0,1)}
                ξ1 = hsu * e1
                ξ2 = hsu * e2
                return [ξ1, ξ2]
            else
                # {hsu⋅(0,1), hsu⋅(1,0)}
                ξ1 = hsu * e2
                ξ2 = hsu * e1
                return [ξ1, ξ2]
            end
        else
            # {hsu⋅(1,1)/√2, hsu⋅(1,-1)/√2}
            vplus  = invsqrt2 * ComplexF64[1.0,  1.0]
            vminus = invsqrt2 * ComplexF64[1.0, -1.0]
            ξ1 = hsu * vplus
            ξ2 = hsu * vminus
            return [ξ1, ξ2]
        end
    end
end

end # module