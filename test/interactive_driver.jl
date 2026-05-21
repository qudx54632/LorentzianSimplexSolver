# ============================================================
# examples/interactive_main.jl
# Interactive driver for LorentzianSimplexSolver
# ============================================================

# Optional: make sure we are using the package environment when running this file directly
# (safe even if already activated)
try
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
catch
end

using LorentzianSimplexSolver

using LinearAlgebra
using Printf
using Dates
using SymEngine
using Symbolics
using DoubleFloats
# ------------------------------------------------------------
# Input helpers
# ------------------------------------------------------------

# Choose the numeric precision used throughout the computation.
function choose_scalartype()
    println("Choose scalar type:")
    println("  1) Float64")
    println("  2) BigFloat")
    print("> ")

    choice = strip(readline())

    if choice == "1"
        return Float64, 1e-8
    elseif choice == "2"
        return BigFloat, BigFloat(1e-20)
    else
        error("Invalid choice. Please enter 1, or 2.")
    end
end

# Read simplices from one line, e.g.
# [[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6]]
function read_simplices()
    println()
    println("Enter simplices:")
    println("e.x., [[1,2,3,4,6],...,[2,3,4,5,6]]")
    print("> ")

    line = strip(readline())
    isempty(line) && error("No simplices entered.")

    simplices = try
        Meta.parse(line) |> eval
    catch e
        error("Could not parse simplices input:\n$e")
    end

    if !(simplices isa Vector{<:Vector{<:Integer}})
        error("Input must be a vector of integer vectors.")
    end

    return [Int.(s) for s in simplices]
end

# Read one coordinate line per vertex.
# Example line:
# 0,-2.7,-0.9,-1.6
function read_coord_lines(nv::Int)
    println()
    println("Enter $nv coordinate lines (one per vertex):")
    println("Format: t,x,y,z")
    coord_lines = String[]

    for i in 1:nv
        print("[$i] ")
        line = strip(readline())
        isempty(line) && error("Empty coordinate line at entry $i.")
        push!(coord_lines, line)
    end

    return coord_lines
end

# ------------------------------------------------------------
# Start
# ------------------------------------------------------------

println("========================================")
println(" Spinfoam / Regge Interactive Driver")
println("========================================")
println()

ScalarT, tol = choose_scalartype()

if ScalarT === BigFloat
    setprecision(BigFloat, 100)
    LorentzianSimplexSolver.PrecisionUtils.set_big_precision!(100)
end
LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(tol)

println()
println("Using scalar type: $ScalarT")
println("Tolerance: $tol")

@variables γ

# ------------------------------------------------------------
# Read simplices
# ------------------------------------------------------------

simplices = read_simplices()

ns = length(simplices)
all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)
Nverts = length(all_vertices)

println()
println("Detected $ns simplices with $Nverts unique vertices.")

# ------------------------------------------------------------
# Read coordinates
# ------------------------------------------------------------

println()
println("Enter coordinates in the order of the vertices above.")

coord_lines = read_coord_lines(Nverts)

vertex_coords = Dict{Int, Vector{ScalarT}}()
for (i, v) in enumerate(all_vertices)
    nums = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(coord_lines[i], ScalarT)
    length(nums) == 4 || error("Vertex $v: expected 4 numbers, got $(length(nums))")
    vertex_coords[v] = nums
end

println()
println("Coordinates loaded.")

# ------------------------------------------------------------
# Read gamma
# ------------------------------------------------------------

println()
print("Enter γ: ")
gamma_val = parse(ScalarT, strip(readline()))

println("Using γ = $gamma_val")

# ------------------------------------------------------------
# Geometry and action setup
# ------------------------------------------------------------
println()
println("Building geometry...")
datasets = LorentzianSimplexSolver.GeometryTypes.GeometryDataset{ScalarT}[]

for (s, simplex) in enumerate(simplices)
    println("\n--- Processing simplex $s with vertices $simplex ---")
    bdypoints = [vertex_coords[v] for v in simplex]
    ds = LorentzianSimplexSolver.GeometryPipeline.run_geometry_pipeline(bdypoints)
    push!(datasets, ds)
end

geom = LorentzianSimplexSolver.GeometryTypes.GeometryCollection(datasets)
println("\n=== Geometry initialization complete ===\n")


# ------------------------------------------------------------
# Consistency checks for each simplex
# ------------------------------------------------------------
println("Would you like to check parallel transport conditions and closure conditions for each simplex? (y or n)")
if lowercase(strip(readline())) == "y"
    for (idx, simplex) in enumerate(geom.simplex)
        println("\n--- Checking simplex $idx ---")
        LorentzianSimplexSolver.GeometryConsistency.check_sl2c_parallel_transport(simplex.solgsl2c, simplex.bdybivec55)
        LorentzianSimplexSolver.GeometryConsistency.check_so13_parallel_transport(simplex.solgso13, simplex.bdybivec4d55)
        LorentzianSimplexSolver.GeometryConsistency.check_closure_bivectors(simplex.kappa, simplex.areas, simplex.bdybivec55)
    end
else
    println("\nSkipping consistency checks.")
end

if ns > 1
    println("Connect simplices and perform face matching? (y or n)")
    if lowercase(strip(readline())) == "y"
        println("\nFixing global κ-sign orientation ...")
        LorentzianSimplexSolver.KappaOrientation.fix_kappa_signs!(simplices, geom)

        println("\nBuilding global connectivity ...")
        conn = LorentzianSimplexSolver.FourSimplexConnectivity.build_global_connectivity(simplices, geom)
        push!(geom.connectivity, conn)

        LorentzianSimplexSolver.FaceXiMatching.run_face_xi_matching(geom; sector = :ref)
        println("Global connectivity is constructed.")

        println("\nWould you like to check parallel transport conditions and closure conditions after face matching? (y or n)")
        if lowercase(strip(readline())) == "y"
            LorentzianSimplexSolver.FaceMatchingChecks.check_all(geom)
        end

        println("\nPerform SU(2) and SU(1,1) gauge fixing ...")
        LorentzianSimplexSolver.GaugeFixingSU.run_su2_su11_gauge_fix(geom)
        println("\nGauge fixing finished.")
    else
        println("\nSkipping connectivity construction and face matching.")
    end
else
    println("\nOnly one simplex detected. Global connectivity is skipped.")
end

println("\nRegge data...")
γsym = LorentzianSimplexSolver.DefineAction.γsym()
_, dihedral_angles, _, _, iRegge = LorentzianSimplexSolver.ReggeAction.run_Regge_action(
    geom, simplices, vertex_coords
)
println()
println("iRegge:")
display(iRegge)

# ------------------------------------------------------------
# Symbols and action setup
# ------------------------------------------------------------
println("\nEvaluating spinfoam action at γ = $gamma_val ...")
LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom)
sd, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom);
S_symbols, phase_soln = LorentzianSimplexSolver.SFaction_no_phase.compute_action_no_bdry_phase(geom, sd, dihedral_angles; γ=γsym);
S_no_phase = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_symbols, phase_soln)
vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd, γsym; γval=gamma_val)
S_val = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_no_phase, vals)
SF_action = SymEngine.expand(S_val)

println()
println("Spinfoam action:")
display(SF_action)

# ------------------------------------------------------------
# Check equation motions
# ------------------------------------------------------------
println("\nWould you like to check the equations of motion? (y/n)")
if lowercase(strip(readline())) == "y"
    println("\nComputing equations of motion (symbolic)...")
    dS = LorentzianSimplexSolver.EOMsHessian.compute_EOMs(S_no_phase, sd)

    println("\nChecking equations of motion...")
    LorentzianSimplexSolver.EOMsHessian.check_EOMs(dS, sd; γ = gamma_val)
else
    println("\nSkipping equations-of-motion and Hessian computing.")
end

println("\nWould you like to compute Hessian? (y/n)")
if lowercase(strip(readline())) == "y"
    println("\nComputing Hessian matrix (symbolic)...")
    g_vars = geom.varias[:g_var]
    z_vars = geom.varias[:z_var]
    η_vars = geom.varias[:η_var]
    vars = vcat(g_vars, z_vars, η_vars)

    H_symbols = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block_half(S_no_phase, vars)

    println("\nEvaluating Hessian matrix...")
    H_eval = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_block(H_symbols, sd; γ = gamma_val)

    eigenvalues = eigvals(H_eval)
    Hess_eigenvals_sort = sort(eigenvalues, by=abs, rev=true)
    # println("The eigenvalues of Hessian matrix are: $Hess_eigenvals_sort")
else
    println("\nSkipping Hessian computing.")
end

println("\n=== Program finished ===\n")