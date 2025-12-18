# ============================================================
# main.jl — load geometry or compute from scratch
# ============================================================
using LinearAlgebra, Combinatorics, Printf, Dates, PythonCall

# --- include all modules ---
include("GeometryTypes.jl")          # defines GeometryDataset, GeometryCollection
include("SimplexGeometry.jl")
include("SpinAlgebra.jl")
include("volume.jl")
include("TetraNormals.jl")
include("DihedralAngles.jl")
include("LorentzGroup.jl")
include("ThreeDTetra.jl")
include("Su2Su11FromBivector.jl")
include("XiFromSU.jl")
include("FaceNormals3D.jl")
include("KappaFromNormals.jl")
include("GeometryConsistency.jl")
include("GeometryPipeline.jl")
include("KappaOrientation.jl")
include("FourSimplexConnectivity.jl")
include("FaceXiMatching.jl")
include("FaceMatchingChecks.jl")
include("GaugeFixing.jl")
include("CriticalPoints.jl")
include("DefineSymbols.jl")
include("SolveVars.jl")
include("DefineAction.jl")

using .GeometryTypes: GeometryDataset, GeometryCollection
using .GeometryPipeline: run_geometry_pipeline
using .GeometryConsistency: check_sl2c_parallel_transport, check_so13_parallel_transport, check_closure_bivectors
using .KappaOrientation: fix_kappa_signs!
using .FourSimplexConnectivity: build_global_connectivity
using .FaceXiMatching: run_face_xi_matching
using .FaceMatchingChecks: check_all
using .GaugeFixingSU
using .CriticalPoints: compute_bdy_critical_data
using .DefineSymbols: run_define_variables
using .SolveVars: run_solver
using .DefineAction: vertexaction

sympy = pyimport("sympy")
# ============================================================
# 1. Ask user for connectivity list (list of vertex labels)
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

println("\nYou entered the following simplices:")
println(simplices)

# number of simplices
ns = length(simplices)
println("\nNumber of simplices detected = $ns")

# ============================================================
# 2. Read ALL vertex coordinates in one block of input
# ============================================================

all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)

Nverts = length(all_vertices)

println("\nUnique vertex labels detected:")
println(all_vertices)
println("Total number of vertices = $Nverts\n")

println("Please enter coordinates for all vertices.")
println("Each line should have the format: t, x, y, z")
println("Follow the order of the sorted vertex list shown above.")
println("Press Enter on an empty line to finish the input.\n")

vertex_coords = Dict{Int, Vector{Float64}}()

lines = String[]
for i in 1:Nverts
    line = readline()
    isempty(strip(line)) && break
    push!(lines, line)
end

if length(lines) != Nverts
    error("You entered $(length(lines)) lines, but $Nverts vertices are required.")
end

# Parse all coordinates at once
for (i, v) in enumerate(all_vertices)
    parts = split(lines[i], ',')
    length(parts) == 4 || error("Vertex $v: expected 4 numbers.")
    nums = parse.(Float64, strip.(parts))
    vertex_coords[v] = nums
end

println("\n=== Parsed Vertex Coordinates ===")
for v in all_vertices
    println("Vertex $v  =>  ", vertex_coords[v])
end

# ============================================================
# 3. Construct boundary points for each simplex
# ============================================================

println("\n=== Building geometry for each simplex ===\n")

datasets = GeometryDataset[]

for (s, simplex) in enumerate(simplices)
    println("\n--- Processing simplex $s with vertices $simplex ---")

    # Build boundary points from user’s vertex coordinates
    bdypoints = [vertex_coords[v] for v in simplex]

    # Run geometry pipeline
    ds = run_geometry_pipeline(bdypoints)
    push!(datasets, ds)
end

geom = GeometryCollection(datasets)
println("\n=== Geometry initialization complete ===\n")

# ============================================================
# 4. Consistency checks for each simplex
# ============================================================

println("Would you like to run consistency checks for each simplex? (y or n)")
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

# ============================================================
# 5. Connect simplices, build global connectivity, match faces
#    and perform gauge fixing
# ============================================================

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

println("\nDefining symbols and separating boundary symbols from dynamical variables ...")
run_define_variables(geom);

println("\nComputing boundary data and critical points for all symbols ...")
sol_vars, sol_bdry, γsym = run_solver(geom);
println("The action contains $(length(sol_vars)) dynamical variables.")



