# ============================================================
# runtests.jl
# Master test entry for LorentzianSimplexSolver
# ============================================================

using Test
using Symbolics

# Activate package if tests are run manually
try
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
catch
end

using LorentzianSimplexSolver

println("\n==============================")
println(" Running LorentzianSimplexSolver tests")
println("==============================\n")


println("\n==============================")
println(" All tests completed")
println("==============================\n")