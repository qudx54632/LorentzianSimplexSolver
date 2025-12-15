module LorentzGroup

using LinearAlgebra
using ..SpinAlgebra: eta, Jvec, jjvec
using ..Dihedral: theta_ab
using ..TetraNormals: chop

export getso13, getsl2c

# -------------------------------------------------------------
# Helper: trace
# -------------------------------------------------------------
tr(A) = LinearAlgebra.tr(A)

# -------------------------------------------------------------
# Helper: Minkowski norm squared
# -------------------------------------------------------------
minkowski_norm2(v::AbstractVector) = v' * eta * v

# -------------------------------------------------------------
# Helper: approximate vector comparison (for special cases)
# -------------------------------------------------------------
function vec_is(a::AbstractVector, b::NTuple{4,Real}; atol=1e-12)
    length(a) == 4 || return false
    for i in 1:4
        if abs(a[i] - float(b[i])) > atol
            return false
        end
    end
    return true
end

# -------------------------------------------------------------
# Bivector wedge product:
# B = (a ⊗ b - b ⊗ a) · η
#
# a,b are 4-vectors (Minkowski)
# Returns a 4×4 matrix representing a bivector in so(1,3).
# -------------------------------------------------------------
function wedge(a::AbstractVector, b::AbstractVector)
    @assert length(a) == 4 && length(b) == 4
    B = a * b' .- b * a'
    return B * eta
end

# -------------------------------------------------------------
# Compute SO(1,3) element Λ such that Λ * N_ref = N_a
# -------------------------------------------------------------
function getso13(Na::AbstractVector{<:Real})
    Na = collect(Float64, Na)
    Na2 = minkowski_norm2(Na)
    Nasign = round(Int, Na2)   # should be -1 (timelike) or +1 (spacelike)

    # Reference normal:
    #   if timelike (Nasign == -1): Nref = (±1,0,0,0) with sign = sign(Na[1])
    #   if spacelike (Nasign == +1): Nref = (0,0,0,1)
    ref = if Nasign == -1
        Na[1] > 0 ? [1.0, 0.0, 0.0, 0.0] : [-1.0, 0.0, 0.0, 0.0]
    else
        [0.0, 0.0, 0.0, 1.0]
    end

    # Special exact cases (identity / simple reflection)
    if vec_is(Na, (1, 0, 0, 0)) ||
       vec_is(Na, (0, 0, 0, 1)) ||
       vec_is(Na, (-1, 0, 0, 0))
        return Matrix{Float64}(I, 4, 4)
    elseif vec_is(Na, (0, 0, 0, -1))
        # DiagonalMatrix[{1,1,-1,-1}]
        return Matrix(Diagonal([1.0, 1.0, -1.0, -1.0]))
    end

    # General case
    # Determine dihedral angle θ_ref,a
    θ = theta_ab(ref, Na)

    dihedral = if Nasign == -1
        # timelike: |θ_ref,e|
        abs(θ)
    else
        # spacelike: -θ_ref,e
        -θ
    end

    # bivector B = Nref ∧ Na
    B = wedge(ref, Na)

    # normalized bivector B / |B|
    normB = sqrt(abs(0.5 * tr(B * B)))
    Bnorm = B / normB

    # exp(θ * Bnorm) ∈ SO(1,3)
    Λ = exp(dihedral * Bnorm)
    return chop(Λ)
end

# -------------------------------------------------------------
# Spin-1/2 rep of bivector: B -> 2x2 matrix in sl(2,C)
# -------------------------------------------------------------
function bivec1tohalf(bivec::AbstractMatrix)
    # coefficients α_i, β_i
    coeffs = ComplexF64[]

    # first 3: + Tr(B J_i)
    for i in 1:3
        push!(coeffs, tr(bivec * Jvec[i]))
    end

    # last 3: - Tr(B J_i) for i=4..6
    for i in 4:6
        push!(coeffs, -tr(bivec * Jvec[i]))
    end

    coeffs .*= 0.5

    # linear combination Σ coeff_i * jjvec[i]
    M = zeros(ComplexF64, 2, 2)
    for i in 1:6
        M .+= coeffs[i] .* jjvec[i]
    end
    return M
end


# -------------------------------------------------------------
# Pauli σ1 for special SL(2,C) case
# -------------------------------------------------------------
const σ1 = [0.0 + 0im  1.0 + 0im;
            1.0 + 0im  0.0 + 0im]

# -------------------------------------------------------------
# Compute SL(2,C) element g such that
# it corresponds to the same Lorentz transform as getso13(Na)
# -------------------------------------------------------------
function getsl2c(Na::AbstractVector{<:Real})
    Na = collect(Float64, Na)
    Na2 = minkowski_norm2(Na)
    Nasign = round(Int, Na2)

    # Reference normal as in getso13
    ref = if Nasign == -1
        Na[1] > 0 ? [1.0, 0.0, 0.0, 0.0] : [-1.0, 0.0, 0.0, 0.0]
    else
        [0.0, 0.0, 0.0, 1.0]
    end

    # Special cases
    if vec_is(Na, (1, 0, 0, 0)) ||
       vec_is(Na, (0, 0, 0, 1)) ||
       vec_is(Na, (-1, 0, 0, 0))
        return Matrix{ComplexF64}(I, 2, 2)
    elseif vec_is(Na, (0, 0, 0, -1))
        # I * PauliMatrix[1]
        return 1im * σ1
    end

    # General case
    θ = theta_ab(ref, Na)

    dihedral = if Nasign == -1
        abs(θ)
    else
        -θ
    end

    # bivector in vector rep
    B = wedge(ref, Na)

    # normalized bivector
    normB = sqrt(abs(0.5 * tr(B * B)))
    Bnorm = B / normB

    # spin-1/2 representation of Bnorm
    Bhalf = bivec1tohalf(Bnorm)

    # exp(θ * Bhalf) ∈ SL(2,C)
    g = exp(dihedral * Bhalf)
    return g
end

end # module