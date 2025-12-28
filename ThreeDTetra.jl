module ThreeDTetra

using LinearAlgebra
using ..PrecisionUtils: get_tolerance
using ..SpinAlgebra: Params
using ..LorentzGroup: bivec1tohalf

export ε, get3dtet, get3dvec, threetofour, getbivec, getbivec2d

η(::Type{T}) where {T<:Real} = Params{T}().eta
# ----------------------------------------------------------
# Levi-Civita tensor ε_{ijkl}
# ----------------------------------------------------------
const ε = let A = zeros(Int, 4, 4, 4, 4)
    for i in 1:4, j in 1:4, k in 1:4, l in 1:4
        idx = (i, j, k, l)
        if length(unique(idx)) < 4
            A[i, j, k, l] = 0
        else
            arr = collect(idx)
            inv = 0
            for a in 1:3, b in a+1:4
                if arr[a] > arr[b]
                    inv += 1
                end
            end
            A[i, j, k, l] = (inv % 2 == 0) ? 1 : -1
        end
    end
    A
end

# ----------------------------------------------------------
# get3dvec: just the transformed 4 → 4 vectors, no dropping
# ----------------------------------------------------------
function get3dvec(tetedgevec::AbstractVector{<:AbstractVector{T}},
                  tetsol13::AbstractMatrix{T}) where {T<:Real}
    invΛ = inv(tetsol13)
    return [invΛ * v for v in tetedgevec]
end


# ----------------------------------------------------------
# get3dtet:
# Convert 4D edge vectors into 3D vectors inside a tetrahedron
#
# Output:
#   (list_of_3d_edges, zero_position)
#
# If the zero index has length 1:
#   spacelike tet  → zero_pos = 1
#   timelike tet   → zero_pos = 4
# ----------------------------------------------------------
function get3dtet(tetedgevec::AbstractVector{<:AbstractVector{T}},
                  tetsol13::AbstractMatrix{T}) where {T<:Real}

    tet4d = get3dvec(tetedgevec, tetsol13)
    ncomp = length(tet4d[1])  # expect 4

    sum_components = [
        sum(abs(v[j]) for v in tet4d)
        for j in 1:ncomp
    ]

    tol = T(get_tolerance())
    zero_positions = findall(x -> isapprox(x, zero(T), atol=tol), sum_components)

    if length(zero_positions) != 1
        error("Could not identify unique 0-component. Check your data.")
    end

    pos = zero_positions[1]

    tet3d = [v[setdiff(1:ncomp, (pos,))] for v in tet4d]

    sign =
        pos == 4 ? -1 :
        pos == 1 ?  1 :
        error("Zero component position must be 1 (spacelike) or 4 (timelike). Got $pos.")

    return tet3d, pos, sign
end

# ----------------------------------------------------------
# threetofour:
# Embed 3-vector back into 4D by inserting leading or trailing 0
#
# If sgndet > 0 → prepend 0
# If sgndet < 0 → append 0
# ----------------------------------------------------------
function threetofour(n::AbstractVector{T}, sgndet::Int) where {T<:Real}
    z = zero(T)
    if sgndet > 0
        return vcat(T[z], n)
    else
        return vcat(n, T[z])
    end
end

# ----------------------------------------------------------
# getbivec(n1, n2)
# The 4D bivector:
#
# B_{ij} = ½ ε_{ijkl} η_{km} n1_m η_{ln} n2_n
#
# Output → B ⋅ η 
# ----------------------------------------------------------
function getbivec(n1::AbstractVector{T}, n2::AbstractVector{T}) where {T<:Real}
    B = zeros(T, 4, 4)
    ηv1 = η(T) * n1
    ηv2 = η(T) * n2

    half = one(T) / T(2)

    for i in 1:4, j in 1:4
        s = zero(T)
        for k in 1:4, l in 1:4
            s += T(ε[i, j, k, l]) * ηv1[k] * ηv2[l]
        end
        B[i, j] = half * s
    end

    return B * η(T)
end

# ----------------------------------------------------------
# getbivec2d(n1, n2)
#   1. compute 4D bivector
#   2. normalize by sqrt(|½ Tr(B⋅B)|)
#   3. convert to SL(2,C) via bivec1tohalf
# ----------------------------------------------------------
function getbivec2d(n1::AbstractVector{T}, n2::AbstractVector{T}) where {T<:Real}
    B = getbivec(n1, n2)

    val = real(LinearAlgebra.tr(B * B)) / T(2)
    Bnorm = B / sqrt(abs(val))

    return bivec1tohalf(Bnorm)  # returns Matrix{Complex{T}}
end

end # module