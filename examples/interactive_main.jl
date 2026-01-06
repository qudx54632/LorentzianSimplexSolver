# ============================================================
# examples/interactive_main.jl
# Interactive driver for LorentzianSimplexSolver
# ============================================================

using LinearAlgebra
using Printf
using Dates
using Symbolics
using PythonCall

# Optional: make sure we are using the package environment when running this file directly
# (safe even if already activated)
try
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
catch
end

using LorentzianSimplexSolver

# If you want sympy available (only if your package needs it at runtime)
sympy = pyimport("sympy")

# ------------------------------------------------------------
# 1. Precision choice (user-controlled)
# ------------------------------------------------------------
println("\nSelect numerical precision:")
println("  1 → Float64 (fast, standard)")
println("  2 → BigFloat (high precision)")
print("Your choice [1 or 2]: ")

choice = strip(readline())

if choice != "1" && choice != "2"
    println("Invalid choice. Defaulting to Float64.")
    choice = "1"
end

ScalarT = if choice == "2"
    # println("\nUsing BigFloat arithmetic.")
    # print("Enter BigFloat precision in bits (default = 256): ")
    prec_input = strip(readline())
    prec = isempty(prec_input) ? 256 : parse(Int, prec_input)

    LorentzianSimplexSolver.PrecisionUtils.set_big_precision!(prec)
    LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(sqrt(eps(BigFloat)))
    BigFloat
else
    # println("\nUsing Float64 arithmetic.")
    LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(1e-10)
    Float64
end

println("Scalar type set to: $ScalarT")

# ------------------------------------------------------------
# 2. Read simplices
# ------------------------------------------------------------
println("\nEnter the list of simplices, for example:")
println("[[1,2,3,4,5],[1,3,4,5,6]]")
println("Press Enter on an empty line to finish the input.")

lines = String[]
while true
    line = readline()
    isempty(strip(line)) && break
    push!(lines, line)
end

simplices_text = join(lines, "\n")

simplices = try
    Meta.parse(simplices_text) |> eval
catch e
    error("Could not parse simplices input:\n$e")
end

ns = length(simplices)
println("\nNumber of simplices detected = $ns")
println("Simplices = ", simplices)

all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)
Nverts = length(all_vertices)

println("\nUnique vertex labels detected: ", all_vertices)
println("Total number of vertices = $Nverts\n")

# ------------------------------------------------------------
# 3. Read vertex coordinates
# ------------------------------------------------------------
println("Please enter coordinates for all vertices.")
println("Each line format: t, x, y, z  (commas or spaces both ok)")
println("Follow the order of the sorted vertex list shown above.\n")

vertex_coords = Dict{Int, Vector{ScalarT}}()

coord_lines = String[]
for i in 1:Nverts
    line = readline()
    isempty(strip(line)) && error("Empty line encountered at vertex $i. Need $Nverts coordinate lines.")
    push!(coord_lines, line)
end

for (i, v) in enumerate(all_vertices)
    nums = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(coord_lines[i], ScalarT)
    length(nums) == 4 || error("Vertex $v: expected 4 numbers, got $(length(nums))")
    vertex_coords[v] = nums
end

println("\n=== Parsed Vertex Coordinates ===")
for v in all_vertices
    println("Vertex $v  =>  ", vertex_coords[v])
end

# ------------------------------------------------------------
# 4. Build geometry
# ------------------------------------------------------------
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
# 5. Consistency checks for each simplex
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

# ------------------------------------------------------------
# 6. Connect simplices + face matching + gauge fixing
# ------------------------------------------------------------
if ns > 1
    println("\nConnect simplices and perform face matching? (y or n)")
    if lowercase(strip(readline())) == "y"
        println("\nFixing global κ-sign orientation ...")
        LorentzianSimplexSolver.KappaOrientation.fix_kappa_signs!(simplices, geom)

        println("\nBuilding global connectivity ...")
        conn = LorentzianSimplexSolver.FourSimplexConnectivity.build_global_connectivity(simplices, geom)
        push!(geom.connectivity, conn)
        println("Global connectivity constructed.")

        println("\nRunning face-ξ matching ...")
        LorentzianSimplexSolver.FaceXiMatching.run_face_xi_matching(geom)

        println("\nRunning final face-matching checks ...")
        LorentzianSimplexSolver.FaceMatchingChecks.check_all(geom)

        println("\nPerform SU(2) and SU(1,1) gauge fixing ...")
        LorentzianSimplexSolver.GaugeFixingSU.run_su2_su11_gauge_fix(geom)
        println("\nGauge fixing finished.")
    else
        println("\nSkipping connectivity construction and face matching.")
    end
else
    println("\nOnly one simplex detected. Global connectivity is skipped.")
end

# ------------------------------------------------------------
# 7. Symbols, action, EOMs, Hessian
# ------------------------------------------------------------
println("\nDefining symbols and separating boundary symbols from dynamical variables...")
LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom)

println("\nComputing boundary data and critical points for all symbols...")
sd, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom)
println("The action contains $(length(sd.labels_vars)) dynamical variables.")

println("\nConstructing the action...")
S = LorentzianSimplexSolver.DefineAction.compute_action(geom)

println("\nCompiling action into a Julia function...")
S_fn, labels = LorentzianSimplexSolver.SymbolicToJulia.build_action_function(S, sd)

println("\nEvaluating the action at the critical point...")
@variables γ
args = LorentzianSimplexSolver.SymbolicToJulia.build_argument_vector(sd, labels, γ)
S_sym = simplify(S_fn(args...))
println("The action at the critical point is $S_sym.")

println("\nWould you like to check the equations of motion? (y/n)")
if lowercase(strip(readline())) == "y"
    println("\nComputing equations of motion (symbolic)...")
    dS = LorentzianSimplexSolver.EOMsHessian.compute_EOMs(S, sd)

    println("\nChecking equations of motion...")
    grad_fns = LorentzianSimplexSolver.SymbolicToJulia.build_gradient_functions(dS, sd)
    LorentzianSimplexSolver.EOMsHessian.check_EOMs(grad_fns, sd; γ = one(ScalarT))
else
    println("\nSkipping equations-of-motion and Hessian computing.")
end

println("\nWould you like to compute Hessian? (y/n)")
if lowercase(strip(readline())) == "y"
    println("\nComputing Hessian matrix (symbolic)...")
    H = LorentzianSimplexSolver.EOMsHessian.compute_Hessian(S, sd)

    println("\nEvaluating Hessian matrix...")
    hess_fns = LorentzianSimplexSolver.SymbolicToJulia.build_hessian_functions(H, sd)
    H_eval, _ = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian(hess_fns, sd; γ = one(ScalarT))

    H_det = det(H_eval)
    println("The determinant of Hessian matrix is $H_det.")
else
    println("\nSkipping Hessian computing.")
end

println("\n=== Program finished ===\n")