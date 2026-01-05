# ============================================================
# test/interactive_driver.jl
# Manual interactive test driver
# ============================================================

using Pkg

# Activate package root
Pkg.activate(joinpath(@__DIR__, ".."))

# Run the interactive example
include(joinpath(@__DIR__, "..", "examples", "interactive_main.jl"))