module ThreeDTetra

using LinearAlgebra
using ..TetraNormals: chop
using ..SpinAlgebra: eta
using ..LorentzGroup: bivec1tohalf

export ε, get3dtet, get3dvec, threetofour, getbivec, getbivec2d

# ----------------------------------------------------------
# Levi-Civita tensor ε_{ijkl}
# ----------------------------------------------------------
const ε = let A = zeros(Int, 4,4,4,4)
    for i in 1:4, j in 1:4, k in 1:4, l in 1:4
        idx = (i,j,k,l)
        # If any indices repeat, epsilon = 0
        if length(unique(idx)) < 4
            A[i,j,k,l] = 0
        else
            # Compute sign of permutation (1,2,3,4) -> (i,j,k,l)
            arr = collect(idx)
            inv = 0
            for a in 1:3, b in a+1:4
                if arr[a] > arr[b]
                    inv += 1
                end
            end
            A[i,j,k,l] = (inv % 2 == 0) ? 1 : -1
        end
    end
    A
end

# ----------------------------------------------------------
# get3dvec: just the transformed 4 → 4 vectors, no dropping
# ----------------------------------------------------------
function get3dvec(tetedgevec, tetsol13)
    invΛ = inv(tetsol13)
    return chop([invΛ * v for v in tetedgevec])
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
function get3dtet(tetedgevec, tetsol13)

    # ------------------------------------------------------------
    # Step 1: transform edges (Λ⁻¹ v)
    # ------------------------------------------------------------
    tet4d = get3dvec(tetedgevec, tetsol13)

    ncomp = length(tet4d[1])   # = 4 expected

    # ------------------------------------------------------------
    # Step 2: sum abs(v[j]) over all edges
    # find which coordinate component is identically zero
    # ------------------------------------------------------------
    sum_components = [
        sum(abs(v[j]) for v in tet4d) 
        for j in 1:ncomp
    ]

    # ------------------------------------------------------------
    # Step 3: find zero component
    # ------------------------------------------------------------
    zero_positions = findall(x -> isapprox(x, 0.0, atol=1e-12),
                             sum_components)

    if length(zero_positions) != 1
        error("Something is wrong — could not identify the unique 0-component. Check your data.")
    end

    pos = zero_positions[1]

    # ------------------------------------------------------------
    # Step 4: remove that component → 3D edge vectors
    # ------------------------------------------------------------
    tet3d = [ v[ setdiff(1:ncomp, (pos,)) ] for v in tet4d ]

    # ------------------------------------------------------------
    # Step 5: sign convention (your rule)
    #
    # pos == 4 → timelike → -1
    # pos == 1 → spacelike → +1
    # otherwise → no convention → error
    # ------------------------------------------------------------
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
function threetofour(n, sgndet::Int)
    if sgndet > 0
        return vcat([0], n)
    else
        return vcat(n, [0])
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
function getbivec(n1, n2)
    B = zeros(Float64, 4,4)
    ηv1 = eta * n1
    ηv2 = eta * n2

    for i in 1:4, j in 1:4
        s = 0
        for k in 1:4, l in 1:4
            s += ε[i,j,k,l] * ηv1[k] * ηv2[l]
        end
        B[i,j] = 1/2 * s
    end

    return B * eta
end

# ----------------------------------------------------------
# getbivec2d(n1, n2)
#   1. compute 4D bivector
#   2. normalize by sqrt(|½ Tr(B⋅B)|)
#   3. convert to SL(2,C) via bivec1tohalf
# ----------------------------------------------------------
function getbivec2d(n1, n2)

    # Step 1: 4D bivector
    B = getbivec(n1, n2)

    # Step 2: normalize
    val = 0.5 * real(tr(B * B))
    Bnorm = B / sqrt(abs(val))

    # Step 3: convert to spinor representation
    return bivec1tohalf(Bnorm)
end

end # module