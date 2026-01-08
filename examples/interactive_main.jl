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

geom_base = LorentzianSimplexSolver.GeometryTypes.GeometryCollection(datasets)
println("\n=== Geometry initialization complete ===\n")

# ------------------------------------------------------------
# 5. Consistency checks for each simplex
# ------------------------------------------------------------
println("Would you like to check parallel transport conditions and closure conditions for each simplex? (y or n)")
if lowercase(strip(readline())) == "y"
    for (idx, simplex) in enumerate(geom_base.simplex)
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
        LorentzianSimplexSolver.KappaOrientation.fix_kappa_signs!(simplices, geom_base)

        println("\nBuilding global connectivity ...")
        conn = LorentzianSimplexSolver.FourSimplexConnectivity.build_global_connectivity(simplices, geom_base)
        push!(geom_base.connectivity, conn)

        geom_ref    = deepcopy(geom_base)
        geom_parity = deepcopy(geom_base)
        LorentzianSimplexSolver.FaceXiMatching.run_face_xi_matching(geom_ref; sector=:ref)
        LorentzianSimplexSolver.FaceXiMatching.run_face_xi_matching(geom_parity; sector=:parity)
        println("Global connectivity constructed for both reference and parity orientations.")

        # println("\nRunning final face-matching checks ...")
        LorentzianSimplexSolver.FaceMatchingChecks.check_all(geom_ref)
        LorentzianSimplexSolver.FaceMatchingChecks.check_all(geom_parity)

        println("\nPerform SU(2) and SU(1,1) gauge fixing ...")
        LorentzianSimplexSolver.GaugeFixingSU.run_su2_su11_gauge_fix(geom_ref)
        LorentzianSimplexSolver.GaugeFixingSU.run_su2_su11_gauge_fix(geom_parity)
        println("\nGauge fixing finished.")
    else
        println("\nSkipping connectivity construction and face matching.")
    end
else
    geom_ref    = deepcopy(geom_base)
    geom_parity = deepcopy(geom_base)
    
    sl2c_ref = [geom_base.simplex[i].solgsl2c    for i in 1:ns]
    sgndet = [geom_base.simplex[i].sgndet    for i in 1:ns]
    geom_parity.simplex[1].solgsl2c = LorentzianSimplexSolver.FaceXiMatching.update_sl2ctest(sl2c_ref, sgndet)[1]
    println("\nOnly one simplex detected. Global connectivity is skipped.")
end

# ------------------------------------------------------------
# 7a. Symbols and action (reference orientation)
# ------------------------------------------------------------
println("\nConstructing the action...")
using Symbolics
@variables γ
LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom_ref)
sd_ref, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom_ref)
S_ref = LorentzianSimplexSolver.DefineAction.compute_action(geom_ref)
S_ref_fn, labels_ref = LorentzianSimplexSolver.SymbolicToJulia.build_action_function(S_ref, sd_ref)
args_ref = LorentzianSimplexSolver.SymbolicToJulia.build_argument_vector(sd_ref, labels_ref, γ)
args_ref_keep_j = LorentzianSimplexSolver.SymbolicToJulia.build_argument_vector_keep_j(sd_ref, labels_ref, γ)
S_ref_sym = expand(simplify(S_ref_fn(args_ref...)))
S_ref_sym_keep_j = expand(simplify(S_ref_fn(args_ref_keep_j...)))

# ------------------------------------------------------------
# 7b. Symbols and action (parity orientation)
# ------------------------------------------------------------
LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom_parity)
sd_parity, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom_parity)
S_parity = LorentzianSimplexSolver.DefineAction.compute_action(geom_parity)
S_parity_fn, labels_parity = LorentzianSimplexSolver.SymbolicToJulia.build_action_function(S_parity, sd_parity)
args_parity = LorentzianSimplexSolver.SymbolicToJulia.build_argument_vector(sd_parity, labels_parity, γ)
args_parity_keep_j = LorentzianSimplexSolver.SymbolicToJulia.build_argument_vector_keep_j(sd_parity, labels_parity, γ)
S_parity_sym = expand(simplify(S_parity_fn(args_parity...)))
S_parity_sym_keep_j = expand(simplify(S_ref_fn(args_parity_keep_j...)))

# ------------------------------------------------------------
# 7c. Regge action (parity orientation)
# ------------------------------------------------------------
phase = expand(simplify((S_ref_sym+S_parity_sym)//2))
S_regge_num,  S_regge_symbolics = LorentzianSimplexSolver.ReggeAction.run_Regge_action(geom_ref, γ);
println("The Regge action is $S_regge_num, and the common phase is $phase.")

orientation = LorentzianSimplexSolver.OrientationSelector.select_orientation(S_ref_sym_keep_j, S_parity_sym_keep_j, S_regge_symbolics, γ)

if orientation == :ref_neg || orientation == :parity_pos
    S_pos = expand(simplify(S_parity_sym - phase))
    S_neg = expand(simplify(S_ref_sym - phase))
else
    S_pos = expand(simplify(S_ref_sym - phase))
    S_neg = expand(simplify(S_parity_sym - phase))
end
println("Action at the positive-orientation critical point: $S_pos")
println("Action at the negative-orientation critical point: $S_neg")

# ------------------------------------------------------------
# 8. Check equation motions
# ------------------------------------------------------------
println("\nWould you like to check the equations of motion? (y/n)")
if lowercase(strip(readline())) == "y"
    println("\nComputing equations of motion (symbolic)...")
    dS_ref = LorentzianSimplexSolver.EOMsHessian.compute_EOMs(S_ref, sd_ref)

    println("\nChecking equations of motion...")
    grad_ref_fns = LorentzianSimplexSolver.SymbolicToJulia.build_gradient_functions(dS_ref, sd_ref)
    LorentzianSimplexSolver.EOMsHessian.check_EOMs(grad_ref_fns, sd_ref; γ = one(ScalarT))
else
    println("\nSkipping equations-of-motion and Hessian computing.")
end

println("\nWould you like to compute Hessian? (y/n)")
if lowercase(strip(readline())) == "y"
    println("\nComputing Hessian matrix (symbolic)...")
    H_ref = LorentzianSimplexSolver.EOMsHessian.compute_Hessian(S_ref, sd_ref)

    println("\nEvaluating Hessian matrix...")
    hess_ref_fns = LorentzianSimplexSolver.SymbolicToJulia.build_hessian_functions(H_ref, sd_ref)
    H_ref_eval, _ = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian(hess_ref_fns, sd_ref; γ = one(ScalarT))

    H_ref_det = det(H_ref_eval)
    println("The determinant of Hessian matrix is $H_ref_det.")
else
    println("\nSkipping Hessian computing.")
end

println("\n=== Program finished ===\n")