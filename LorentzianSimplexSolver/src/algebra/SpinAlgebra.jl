module SpinAlgebra

using LinearAlgebra

export Params, sigma4, sigmabar4, su11sigma, eta, Jvec, jjvec, σ1, σ2, σ3, imag_unit


"""
    imag_unit(T)
Return the imaginary unit i as Complex{T}.
"""
imag_unit(::Type{T}) where T = Complex{T}(zero(T), one(T))

# ============================================================
# Pauli matrices (precision-generic)
# ============================================================

σ1(::Type{T}) where T =
    Matrix{T}([zero(T)  one(T);
               one(T)   zero(T)])

function σ2(::Type{T}) where T
    i = imag_unit(T)
    Matrix{Complex{T}}([
        zero(T)   -i;
        i          zero(T)
    ])
end

σ3(::Type{T}) where T =
    Matrix{T}([ one(T)   zero(T);
                zero(T) -one(T)])



# ============================================================
# sigma matrices
# ============================================================
"""
    sigma4(T)

Return {I, σ1, σ2, σ3} as Vector{Matrix{Complex{T}}}.
"""
function sigma4(::Type{T}) where T
    [
        Matrix{Complex{T}}(I, 2, 2),
        Complex{T}.(σ1(T)),
        σ2(T),
        Complex{T}.(σ3(T))
    ]
end


"""
    sigmabar4(T)
Return {I, -σ1, -σ2, -σ3}.
"""
function sigmabar4(::Type{T}) where T
    s = sigma4(T)
    [s[1], -s[2], -s[3], -s[4]]
end

# ============================================================
# su(1,1) analog basis
# ============================================================

"""
    su11sigma(T)

Return su(1,1) generators.
"""
function su11sigma(::Type{T}) where T
    i = imag_unit(T)
    [
        Complex{T}.(σ3(T)),
        i * Complex{T}.(σ1(T)),
        i * σ2(T)
    ]
end


# ============================================================
# Minkowski metric
# ============================================================

"""
    eta(T)

Minkowski metric diag(-1,1,1,1).
"""
eta(::Type{T}) where T = Diagonal(T[-one(T), one(T), one(T), one(T)])

# ============================================================
# SO(1,3) generators (vector rep)
# ============================================================

"""
    Jvec(T)

Return SO(1,3) generators in vector representation.
"""
function Jvec(::Type{T}) where T
    z = zero(T)
    o = one(T)

    [
        T[ z  o  z  z;
           o  z  z  z;
           z  z  z  z;
           z  z  z  z ],

        T[ z  z  o  z;
           z  z  z  z;
           o  z  z  z;
           z  z  z  z ],

        T[ z  z  z  o;
           z  z  z  z;
           z  z  z  z;
           o  z  z  z ],

        T[ z  z  z  z;
           z  z  z  z;
           z  z  z  o;
           z  z -o  z ],

        T[ z  z  z  z;
           z  z  z -o;
           z  z  z  z;
           z  o  z  z ],

        T[ z  z  z  z;
           z  z  o  z;
           z -o  z  z;
           z  z  z  z ]
    ]
end

# ============================================================
# SL(2,C) generators
# ============================================================

"""
    jjvec(T)

Return SL(2,C) generators (σ_i/2, iσ_i/2).
"""
function jjvec(::Type{T}) where T
    half = one(T) / T(2)
    i = imag_unit(T)

    vcat(
        [
            half * Complex{T}.(σ1(T)),
            half * σ2(T),
            half * Complex{T}.(σ3(T))
        ],
        [
            half * i * Complex{T}.(σ1(T)),
            half * i * σ2(T),
            half * i * Complex{T}.(σ3(T))
        ]
    )
end

# ============================================================
# Parameter container
# ============================================================

Base.@kwdef mutable struct Params{T}
    sigma4    = sigma4(T)
    sigmabar4 = sigmabar4(T)
    su11sigma = su11sigma(T)
    eta       = eta(T)
end

end # module