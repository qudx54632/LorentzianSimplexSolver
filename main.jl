# ============================================================
# main.jl — load geometry or compute from scratch
# ============================================================
using LinearAlgebra, Combinatorics
using Printf
using Dates

include("GeometryTypes.jl")          # defines GeometryDataset, GeometryCollection
#include("load_geometry_data.jl")        # low-level parsers FIRST
#include("GeometryDataLoader.jl")        # defines GeometryDataset
include("write_geometry_data.jl")
# --- include all modules ---
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
include("write_connectivity.jl")
include("FaceXiMatching.jl")
include("FaceMatchingChecks.jl")


using .GeometryTypes: GeometryDataset, GeometryCollection
using .GeometryPipeline: run_geometry_pipeline
using .GeometryConsistency: check_sl2c_parallel_transport, check_so13_parallel_transport, check_closure_bivectors
using .KappaOrientation: fix_kappa_signs!
using .FourSimplexConnectivity: build_global_connectivity
using .write_connectivity_data: save_connectivity_data
using .FaceXiMatching: run_face_xi_matching
using .FaceMatchingChecks: check_all

# for mkpath
import Base.Filesystem: mkpath

# ============================================================
# 1. Ask user whether to load old dataset
# ============================================================

println("Load existing geometry dataset? (y/n)")
ans = readline()
use_existing = (lowercase(ans) == "y")

geom = nothing

if use_existing
    folder = "geom_output/4simplex-example/"
    println("Loading existing dataset from: $folder")

    # ---------------------------
    # Print boundary points
    # ---------------------------
    bdrypoints = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.5, 1.0, 1.0, 1.0]
    ]

    println("Boundary points used:")
    for p in bdrypoints
        println("  ", p)
    end

    geom = load_geometry_dataset(folder)
    
else
    # ============================================================
    # 2. Compute fresh geometry from user-provided boundary points
    # ============================================================
    base_folder = "geom_output/new-data/"
    mkpath(base_folder)

    println("\nHow many 4-simplices do you want to generate?")
    println("(Enter a positive integer, e.g. 1, 2, 3, ...)")

    ns = try
        parse(Int, strip(readline()))
    catch
        println("Invalid integer; defaulting to 1 simplex.")
        1
    end
    ns < 1 && (println("Number of simplices < 1; using 1."); ns = 1)

    datasets = GeometryDataset[]

    for s in 1:ns
        println("\n--- Simplex $s ---")
        subfolder = base_folder * "simplex$(s)/"
        mkpath(subfolder)

        println("Enter boundary points (t,x,y,z) comma-separated. Empty line to exit.")

        bdypoints = Vector{Vector{Float64}}()

        while true
            line = readline()
            isempty(strip(line)) && break

            parts = split(line, ',')

            if length(parts) != 4
                println("Error: expected exactly 4 comma-separated numbers.")
                continue
            end

            # trim whitespace around each field
            trimmed = strip.(parts)

            # parse as Float64
            nums = try
                parse.(Float64, trimmed)
            catch
                println("Error: couldn't parse numbers. Please try again.")
                continue
            end

            push!(bdypoints, nums)
        end

        println("Boundary points for simplex $s:", bdypoints)

        ds = run_geometry_pipeline(bdypoints, subfolder)  # GeometryDataset
        push!(datasets, ds)
    end
    
    geom = GeometryCollection(datasets)

    println("\n=== Geometry initialization complete ===\n")

    # ============================================================
    # 2. Consistency checks for each simplex
    # ============================================================
    println("\nWould you like to check consistency conditions for each simplex? (y/n)")

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
    # 4. If multiple simplices, ask for 4-simplex connectivity
    # ============================================================
    if ns > 1
        local four_simplices = Vector{Vector{Int}}()
        println("\nEnter four-simplices [[1,2,3,4,5], ...] — empty line ends input.")

        # Read multi-line user input
        input_lines = String[]
        while true
            line = readline()
            isempty(strip(line)) && break
            push!(input_lines, line)
        end

        usertext = join(input_lines, "\n")

        # Parse user input
        try
            four_simplices = Meta.parse(usertext) |> eval
        catch e
            error("Could not parse four-simplices input:\n$e")
        end

        println("\nFour-simplices parsed successfully:", four_simplices)

        # ============================================================
        # 4.1 Fix κ signs globally
        # ============================================================
        println("\nFixing global κ-sign orientation ...")
        fix_kappa_signs!(four_simplices, geom)

        # ============================================================
        # 4.2 Build connectivity + save to folder
        # ============================================================
        println("\nBuilding global connectivity ...")
        conn = build_global_connectivity(four_simplices, geom)
        push!(geom.connectivity, conn)
        println("Connectivity built!")

        # ============================================================
        # 4.3 Save connectivity to a new folder
        # ============================================================
        conn_folder = base_folder * "connectivity/"
        save_connectivity_data(conn_folder, conn)

        println("\nConnectivity saved to: $conn_folder\n")
        
        # ============================================================
        # 4.4 Face matching and geometric consistency
        # ============================================================
        println("Running face–xi matching...\n")
        run_face_xi_matching(geom)
        # ============================================================
        # 4.5 Final consistency checks after matching
        # ============================================================
        check_all(geom)
    end
end

