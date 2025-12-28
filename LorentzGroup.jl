module LorentzGroup

using LinearAlgebra
using GenericLinearAlgebra
using ..PrecisionUtils: get_tolerance
using ..SpinAlgebra: Params, Jvec, jjvec,  σ1, imag_unit
using ..Dihedral: theta_ab

export getso13, getsl2c

η(::Type{T}) where {T<:Real} = Params{T}().eta
J(::Type{T}) where {T<:Real} = Jvec(T)
jj(::Type{T}) where {T<:Real} = jjvec(T)

# -------------------------------------------------------------
# Helper: trace
# -------------------------------------------------------------
tr(A) = LinearAlgebra.tr(A)

# -------------------------------------------------------------
# Helper: Minkowski norm squared
# -------------------------------------------------------------
minkowski_norm2(v::AbstractVector{T}) where {T<:Real} = (v' * η(T) * v)[1]

# -------------------------------------------------------------
# Helper: approximate vector comparison (for special cases)
# -------------------------------------------------------------

function vec_is(a::AbstractVector{T}, b::NTuple{4,Real}; atol=get_tolerance()) where {T<:Real}
    length(a) == 4 || return false
    tol = T(atol)
    for i in 1:4
        if !isapprox(a[i], T(b[i]), atol=tol)
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
function wedge(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:Real}
    @assert length(a) == 4 && length(b) == 4
    B = a * b' .- b * a'
    return B * η(T)
end

function exp_so13(θ::T, B::AbstractMatrix{T}) where {T<:Real}
    I4 = Matrix{T}(I, 4, 4)
    B2 = B * B
    σ = tr(B2)

    if σ > zero(T)
        # boost
        return I4 + sinh(θ) * B + (cosh(θ) - one(T)) * B2
    else
        # rotation
        return I4 + sin(θ) * B + (one(T) - cos(θ)) * B2
    end
end

# -------------------------------------------------------------
# Compute SO(1,3) element Λ such that Λ * N_ref = N_a
# -------------------------------------------------------------
function getso13(Na::AbstractVector{T}) where {T<:Real}
    Na = collect(T, Na)
    Na2 = minkowski_norm2(Na)

    tol = T(get_tolerance())
    Nasign = abs(Na2 + one(T)) < tol ? -1 :
             abs(Na2 - one(T)) < tol ?  1 :
             error("Na not unit: Na2=$Na2")

    # Reference normal
    ref = if Nasign == -1
        Na[1] > zero(T) ? T[one(T), zero(T), zero(T), zero(T)] :
                          T[-one(T), zero(T), zero(T), zero(T)]
    else
        T[zero(T), zero(T), zero(T), one(T)]
    end

    # Special exact cases (identity / simple reflection)
    if vec_is(Na, (1,0,0,0); atol=tol) || vec_is(Na, (0,0,0,1); atol=tol) || vec_is(Na, (-1,0,0,0); atol=tol)
        return Matrix{T}(I, 4, 4)
    elseif vec_is(Na, (0,0,0,-1); atol=tol)
        return Matrix(Diagonal(T[one(T), one(T), -one(T), -one(T)]))
    end

    # General case
    θ = theta_ab(ref, Na)             # returns T
    dihedral = Nasign == -1 ? abs(θ) : -θ

    B = wedge(ref, Na)

    normB = sqrt(abs(tr(B * B) / (T(2))))
    Bnorm = B / normB

    Λ = exp_so13(dihedral, Bnorm)
    return Λ
end

# -------------------------------------------------------------
# Spin-1/2 rep of bivector: B -> 2x2 matrix in sl(2,C)
# -------------------------------------------------------------
function bivec1tohalf(bivec::AbstractMatrix{T}) where {T<:Real}
    # coefficients α_i, β_i
    coeffs = Vector{Complex{T}}(undef, 6)

    # first 3: + Tr(B J_i)
    for i in 1:3
        coeffs[i] = tr(bivec * J(T)[i])
    end

    # last 3: - Tr(B J_i) for i=4..6
    for i in 4:6
        coeffs[i] = -tr(bivec * J(T)[i])
    end

    coeffs .*= (one(T) / T(2))

    # linear combination Σ coeff_i * jjvec[i]
    M = zeros(Complex{T}, 2, 2)
    for i in 1:6
        M .+= coeffs[i] .* jj(T)[i]
    end
    return M
end


# exp(θ B) for 2×2 where B^2 is proportional to I
function exp2x2_from_square(θ::T, B::AbstractMatrix{Complex{T}}) where {T<:Real}
    I2 = Matrix{Complex{T}}(I, 2, 2)
    B2 = B * B
    α = real(tr(B2)) / T(2)

    tol = T(get_tolerance())

    if abs(α) < tol
        return I2 + θ * B
    elseif α > zero(T)
        s = sqrt(α)
        return cosh(θ*s) * I2 + (sinh(θ*s)/s) * B
    else
        s = sqrt(-α)
        return cos(θ*s) * I2 + (sin(θ*s)/s) * B
    end
end

# -------------------------------------------------------------
# Compute SL(2,C) element g such that
# it corresponds to the same Lorentz transform as getso13(Na)
# -------------------------------------------------------------
function getsl2c(Na::AbstractVector{T}) where {T<:Real}
    Na = collect(T, Na)
    Na2 = minkowski_norm2(Na)

    tol = T(get_tolerance())
    Nasign = abs(Na2 + one(T)) < tol ? -1 :
             abs(Na2 - one(T)) < tol ?  1 :
             error("Na not unit: Na2=$Na2")

    ref = if Nasign == -1
        Na[1] > zero(T) ? T[one(T),zero(T),zero(T),zero(T)] :
                          T[-one(T),zero(T),zero(T),zero(T)]
    else
        T[zero(T),zero(T),zero(T),one(T)]
    end

    # Special cases
    if vec_is(Na, (1,0,0,0); atol=tol) || vec_is(Na, (0,0,0,1); atol=tol) || vec_is(Na, (-1,0,0,0); atol=tol)
        return Matrix{Complex{T}}(I, 2, 2)
    elseif vec_is(Na, (0,0,0,-1); atol=tol)
        i = imag_unit(T)
        return i * Complex{T}.(σ1(T))
    end

    θ = theta_ab(ref, Na)
    dihedral = Nasign == -1 ? abs(θ) : -θ

    B = wedge(ref, Na)
    normB = sqrt(abs(tr(B * B) / (T(2))))
    Bnorm = B / normB

    Bhalf = bivec1tohalf(Bnorm)       # Complex{T} 2×2
    g = exp2x2_from_square(dihedral, Bhalf)
    return g
end

end # module