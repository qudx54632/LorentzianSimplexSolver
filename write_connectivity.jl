module write_connectivity_data

using Printf
using DelimitedFiles

export save_connectivity_data

"""
Ensure a folder exists.
"""
function ensure_dir(path::String)
    isdir(path) || mkpath(path)
end

# ============================================================
# Generic writer (THE ONLY ONE)
# ============================================================

"""
write_list(path, label, data)

Writes any Julia vector to a text file.
Each entry is labeled and printed using `string(item)`.

Works for:
- Vector{Any}
- Vector{Vector}
- Vector{Vector{Vector}}
- etc.

You never need multiple write functions again.
"""
function write_list(path::String, label::String, data)
    open(path, "w") do io
        println(io, "# $label")
        println(io)

        for (i, item) in enumerate(data)
            println(io, "[$i] = ", string(item))
        end
    end
    # println("Wrote $path")
end

# ============================================================
# MAIN SAVE FUNCTION
# ============================================================

"""
save_connectivity_data(folder, conn)

Writes the connectivity dictionary produced by `build_global_connectivity`
using unified `write_list`.
"""
function save_connectivity_data(folder::String, conn::Dict)

    ensure_dir(folder)

    write_list(joinpath(folder, "tets.out"),                 "Tetrahedra",              conn["Tets"])
    write_list(joinpath(folder, "tetfaces.out"),             "Tetrahedron Faces",       conn["TetFaces"])
    write_list(joinpath(folder, "triangles.out"),            "Triangles",               conn["Triangles"])
    write_list(joinpath(folder, "face_positions.out"),       "Face Positions",          conn["FacePosition"])

    write_list(joinpath(folder, "boundary_faces.out"),       "Boundary Faces",          conn["BDFaces"])
    write_list(joinpath(folder, "boundary_faces_pos.out"),   "Boundary Face Positions", conn["BDFacesPos"])

    write_list(joinpath(folder, "bulk_faces.out"),           "Bulk Faces",              conn["BulkFaces"])
    write_list(joinpath(folder, "bulk_faces_pos.out"),       "Bulk Face Positions",     conn["BulkFacesPos"])

    write_list(joinpath(folder, "shared_tets.out"),          "Shared Tets",             conn["sharedTets"])
    write_list(joinpath(folder, "shared_tets_pos.out"),      "Shared Tet Positions",    conn["sharedTetsPos"])

    write_list(joinpath(folder, "ordered_bulk_faces.out"),   "Ordered Bulk Faces",      conn["OrderBulkFaces"])
    write_list(joinpath(folder, "ordered_bdry_faces.out"),   "Ordered Boundary Faces",  conn["OrderBDryFaces"])

    println("\nConnectivity data saved to folder: $folder\n")
end

end # module