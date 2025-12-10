module SpinAlgebra

using LinearAlgebra

export Params, sigma4, sigmabar4, su11sigma, eta, Jvec, jjvec

# --------------------------------------------
# Pauli matrices
# --------------------------------------------
const σ1 = [0 1;
            1 0]

const σ2 = [0 -im;
            im  0]

const σ3 = [1  0;
            0 -1]

# sigma4  = {I, σ1, σ2, σ3}
const sigma4 = [
    Matrix{ComplexF64}(I(2)),
    σ1,
    σ2,
    σ3
]

# sigmabar4 = {I, -σ1, -σ2, -σ3}
const sigmabar4 = [
    Matrix{ComplexF64}(I(2)),
    -σ1,
    -σ2,
    -σ3
]

# su(1,1) analog basis
const su11sigma = [
    σ3,
    1im * σ1,
    1im * σ2
]

# Minkowski metric diag(-1,1,1,1)
const eta = Diagonal([-1, 1, 1, 1])

# --------------------------------------------
# SO(1,3) generators in vector representation (4×4 matrices)
# --------------------------------------------
const Jvec = [
    # J1
    [ 0  1  0  0;
      1  0  0  0;
      0  0  0  0;
      0  0  0  0 ],

    # J2
    [ 0  0  1  0;
      0  0  0  0;
      1  0  0  0;
      0  0  0  0 ],

    # J3
    [ 0  0  0  1;
      0  0  0  0;
      0  0  0  0;
      1  0  0  0 ],

    # J4 = -(...)
    [  0  0  0   0;
       0  0  0   0;
       0  0  0   1;
       0  0 -1   0 ],

    # J5 = -(...)
    [  0  0  0   0;
       0  0  0   -1;
       0  0  0   0;
       0  1  0   0 ],

    # J6
    [  0  0  0   0;
       0  0  1   0;
       0 -1  0   0;
       0  0  0   0 ]
]

# --------------------------------------------
# SL(2,C) generators (Pauli and i*Pauli) / 2
# --------------------------------------------
const jjvec = vcat(
    [ 0.5 * σ1, 0.5 * σ2, 0.5 * σ3 ],
    [ 0.5im * σ1, 0.5im * σ2, 0.5im * σ3 ]
)

# --------------------------------------------
# Parameter struct container
# --------------------------------------------
Base.@kwdef mutable struct Params
    sigma4::Vector{Matrix{ComplexF64}} = sigma4
    sigmabar4::Vector{Matrix{ComplexF64}} = sigmabar4
    su11sigma::Vector{Matrix{ComplexF64}} = su11sigma
    eta::Diagonal = eta
end

end # module