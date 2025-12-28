# ============================================================
# main.jl — load geometry or compute from scratch
# ============================================================
using LinearAlgebra, Combinatorics, Printf, Dates, PythonCall, Symbolics

sympy = pyimport("sympy")

include("PrecisionUtils.jl")
include("SpinAlgebra.jl")
include("SimplexGeometry.jl")
include("TetraNormals.jl")
include("DihedralAngles.jl")
include("LorentzGroup.jl")
include("ThreeDTetra.jl")
include("volume.jl")
include("Su2Su11FromBivector.jl")
include("XiFromSU.jl")
include("FaceNormals3D.jl")
include("KappaFromNormals.jl")
include("GeometryTypes.jl")          # defines GeometryDataset, GeometryCollection
include("GeometryPipeline.jl")
include("GeometryConsistency.jl")
include("KappaOrientation.jl")
include("FourSimplexConnectivity.jl")
include("FaceXiMatching.jl")
include("FaceMatchingChecks.jl")
include("GaugeFixing.jl")
include("CriticalPoints.jl")
include("DefineSymbols.jl")
include("DefineAction.jl")
include("SolveVars.jl")
include("SymbolicToJulia.jl")
include("EOMsHessian.jl")

using .PrecisionUtils: set_big_precision!, parse_numeric_line, set_tolerance!
using .GeometryTypes: GeometryDataset, GeometryCollection
using .GeometryPipeline: run_geometry_pipeline
using .GeometryConsistency: check_sl2c_parallel_transport, check_so13_parallel_transport, check_closure_bivectors
using .KappaOrientation: fix_kappa_signs!
using .FourSimplexConnectivity: build_global_connectivity
using .FaceXiMatching: run_face_xi_matching
using .FaceMatchingChecks: check_all
using .GaugeFixingSU: run_su2_su11_gauge_fix
using .CriticalPoints: compute_bdy_critical_data
using .DefineSymbols: run_define_variables
using .DefineAction: compute_action
using .SolveVars: run_solver
using .SymbolicToJulia: build_action_function, build_gradient_functions, build_hessian_functions, build_argument_vector
using .EOMsHessian: compute_EOMs, compute_Hessian, check_EOMs, evaluate_hessian

# ============================================================
# 1. Precision choice
# ============================================================

const ScalarT = Float64 # or BigFloat

if ScalarT === BigFloat
    set_big_precision!(256)
    set_tolerance!(1e-20)
else
    set_tolerance!(1e-10)
end

# ============================================================
# 2. Read simplices
# ============================================================

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
# ============================================================
# 3. Read vertex coordinates
# ============================================================
println("Please enter coordinates for all vertices.")
println("Each line format: t, x, y, z  (commas or spaces both ok)")
println("Follow the order of the sorted vertex list shown above.\n")

vertex_coords = Dict{Int, Vector{ScalarT}}()

lines = String[]
for i in 1:Nverts
    line = readline()
    isempty(strip(line)) && error("Empty line encountered at vertex $i. Need $Nverts coordinate lines.")
    push!(lines, line)
end

for (i, v) in enumerate(all_vertices)
    nums = parse_numeric_line(lines[i], ScalarT)  
    length(nums) == 4 || error("Vertex $v: expected 4 numbers, got $(length(nums))")
    vertex_coords[v] = nums
end

println("\n=== Parsed Vertex Coordinates ===")
for v in all_vertices
    println("Vertex $v  =>  ", vertex_coords[v])
end

# ============================================================
# 4. Build geometry
# ============================================================

datasets = GeometryDataset{ScalarT}[]

for (s, simplex) in enumerate(simplices)
    println("\n--- Processing simplex $s with vertices $simplex ---")

    # Build boundary points from user’s vertex coordinates
    bdypoints = [vertex_coords[v] for v in simplex]
    # Run geometry pipeline
    ds = run_geometry_pipeline(bdypoints)
    push!(datasets, ds)
end

geom = GeometryTypes.GeometryCollection(datasets);
println("\n=== Geometry initialization complete ===\n")

# ============================================================
# 5. Consistency checks for each simplex
# ============================================================

println("Would you like to check parallel transport conditions and closure conditions for each simplex? (y or n)")
if lowercase(readline()) == "y"
    for (idx, simplex) in enumerate(geom.simplex)
        println("\n--- Checking simplex $idx ---")
        check_sl2c_parallel_transport(simplex.solgsl2c, simplex.bdybivec55)
        check_so13_parallel_transport(simplex.solgso13, simplex.bdybivec4d55)
        check_closure_bivectors(simplex.kappa, simplex.areas, simplex.bdybivec55)
    end
else
    println("\nSkipping consistency checks.")
end

# =====================================================================================
# 6. Connect simplices, build global connectivity, match faces and perform gauge fixing
# =====================================================================================

if ns > 1
    println("\nConnect simplices and perform face matching? (y or n)")
    if lowercase(readline()) == "y"

        println("\nFixing global κ-sign orientation ...")
        fix_kappa_signs!(simplices, geom)

        println("\nBuilding global connectivity ...")
        conn = build_global_connectivity(simplices, geom)
        push!(geom.connectivity, conn)
        println("Global connectivity constructed.")

        println("\nRunning face-ξ matching ...")
        run_face_xi_matching(geom)

        println("\nRunning final face-matching checks ...")
        check_all(geom)

        println("\nPerform SU(2) and SU(1,1) gauge fixing ...")
        run_su2_su11_gauge_fix(geom);
        println("\nGauge fixing finished.")
    else
        println("\nSkipping connectivity construction and face matching.")
    end
else
    println("\nOnly one simplex detected. Global connectivity is skipped.")
end

println("\nDefining symbols and separating boundary symbols from dynamical variables...")
run_define_variables(geom)


println("\nComputing boundary data and critical points for all symbols...")
sd, _ = run_solver(geom)
println("The action contains $(length(sd.labels_vars)) dynamical variables.")

println("\nConstructing the action...")
S = compute_action(geom)

println("\nCompiling action into a Julia function...")
S_fn, labels = build_action_function(S, sd)

println("\nEvaluating the action at the critical point...")
@variables γ
args = build_argument_vector(sd, labels, γ);
S_sym = simplify(S_fn(args...))
println("The action at the critical point is $S_sym.")

println("\nWould you like to check the equations of motion? (y/n)")
if lowercase(strip(readline())) == "y"
    println("\nComputing equations of motion (symbolic)...")
    dS = compute_EOMs(S, sd)
    println("\nChecking equations of motion...")
    grad_fns = build_gradient_functions(dS, sd)
    check_EOMs(grad_fns, sd; γ = one(ScalarT))
else
    println("\nSkipping equations-of-motion and Hessian computing.")
end

println("\nWould you like to compute Hessian? (y/n)")
if lowercase(strip(readline())) == "y"
    println("\nComputing Hessian matrix (symbolic)...")
    H  = compute_Hessian(S, sd)
    println("\nEvaluating Hessian matrix...")
    hess_fns = build_hessian_functions(H, sd);
    H_eval, _ = evaluate_hessian(hess_fns, sd; γ = one(ScalarT))
    H_det = det(H_eval)
    println("The determinant of Hessian matrix is $H_det.")
else
    println("\nSkipping Hessian computing.")
end
println("\n=== Program finished ===\n")

