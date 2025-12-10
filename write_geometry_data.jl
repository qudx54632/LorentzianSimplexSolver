module write_geometry_data
# ============================================================
# write_geometry_data.jl
# Save spinfoam geometric data into separate .out files
# ============================================================

    using Printf
    using LinearAlgebra
    using DelimitedFiles

    export save_geometry_data
    """
    Ensure a folder exists.
    """
    function ensure_dir(path::String)
        isdir(path) || mkpath(path)
    end


    # ============================================================
    # Generic writers
    # ============================================================

    """
    Write a Vector{Matrix} to file, each matrix separated by blank line.
    """
    function write_matrix_list(filename::String, mats::Vector)
        open(filename, "w") do io
            for (k, M) in enumerate(mats)
                @printf(io, "# Tetrahedron %d\n", k)
                for row in eachrow(M)
                    @printf(io, "%s\n", join(row, " "))
                end
                @printf(io, "\n")
            end
        end
        # println("Wrote $filename")
    end


    """
    Write Vector{Vector{Matrix}} (e.g. bdysu) to file.
    """
    function write_block_matrix_list(filename::String, blocks::Vector{<:Vector})
        open(filename, "w") do io
            for (i, block) in enumerate(blocks)
                @printf(io, "# Tetrahedron %d\n", i)
                for (j, M) in enumerate(block)
                    @printf(io, "## Face %d\n", j)
                    for row in eachrow(M)
                        @printf(io, "%s\n", join(row, " "))
                    end
                    @printf(io, "\n")
                end
                @printf(io, "\n")
            end
        end
        # println("Wrote $filename")
    end

    """
    Write a list of vectors to file.
    """
    function write_vector_list(filename::String, vecs)
        open(filename, "w") do io
            for (k, v) in enumerate(vecs)
                @printf(io, "%d  %s\n", k, join(v, " "))
            end
        end
        # println("Wrote $filename")
    end

    """
    Write Vector{Vector{Vector}} (e.g. bdyxi) to file.
    """
    function write_nested_vectors(filename::String, data)
        open(filename, "w") do io
            for (i, block) in enumerate(data)
                @printf(io, "# Tetrahedron %d\n", i)
                for (j, subblock) in enumerate(block)
                    @printf(io, "## Faces %d\n", j)
                    for vec in subblock
                        @printf(io, "%s\n", join(vec, " "))
                    end
                    @printf(io, "\n")
                end
                @printf(io, "\n")
            end
        end
        # println("Wrote $filename")
    end

    """
    Write Vector{Float64} (e.g., sgndet) to file.
    """
    function write_scalar_list(filename::String, vals)
        open(filename, "w") do io
            for (k, v) in enumerate(vals)
                @printf(io, "# Tetrahedron %d\n", k)
                @printf(io, "%d  %.16g\n", k, v)
            end
        end
        # println("Wrote $filename")
    end

    """
    Write nested Vector{Vector{T}} such as dihedrals.
    """
    function write_nested_scalars(filename::String, nested::Vector{<:Vector})
        open(filename, "w") do io
            for (i, row) in enumerate(nested)
                @printf(io, "# Tetrahedron %d\n", i)
                @printf(io, "%s\n", join(row, " "))
            end
        end
        # println("Wrote $filename")
    end


    # ============================================================
    # Main SAVE FUNCTION
    # ============================================================

    """
    save_geometry_data(folder; kwargs...)

    Save all spinfoam geometric data to separate files.

    Keyword arguments correspond to your variables.
    """
    function save_geometry_data(folder::String;
        solgsl2c        = nothing,
        bdyxi           = nothing,
        bdysu           = nothing,
        areas           = nothing,
        kappa           = nothing,
        nabout          = nothing,
        solgso13        = nothing,
        threeto4dnormal = nothing,
        tetnormalvec    = nothing,
        dihedrals       = nothing,
        edgevec         = nothing,
        bdybivec4d55    = nothing,
        bdybivec55      = nothing,
        nabfrombivec    = nothing,
        sgndet          = nothing,
        tetareasign     = nothing,
        tetn0sign       = nothing,
        zdataf          = nothing
    )

    ensure_dir(folder)

    if solgsl2c       !== nothing; write_matrix_list(joinpath(folder, "sol_g_sl2c.out"), solgsl2c); end
    if bdyxi          !== nothing; write_nested_vectors(joinpath(folder, "bdy_xi.out"), bdyxi); end
    if bdysu          !== nothing; write_block_matrix_list(joinpath(folder, "bdy_su.out"), bdysu); end
    if areas          !== nothing; write_nested_scalars(joinpath(folder, "areas.out"), areas); end
    if kappa          !== nothing; write_nested_scalars(joinpath(folder, "kappa.out"), kappa); end
    if nabout         !== nothing; write_nested_vectors(joinpath(folder, "nabout.out"), nabout); end
    if solgso13       !== nothing; write_matrix_list(joinpath(folder, "sol_g_so13.out"), solgso13); end
    if threeto4dnormal !== nothing; write_vector_list(joinpath(folder, "threeto4dtetfacenormal.out"), threeto4dnormal); end
    if tetnormalvec   !== nothing; write_vector_list(joinpath(folder, "tetnormalvec.out"), tetnormalvec); end
    if dihedrals      !== nothing; write_nested_scalars(joinpath(folder, "dihedrals.out"), dihedrals); end
    if edgevec        !== nothing; write_block_matrix_list(joinpath(folder, "edgevec.out"), edgevec); end
    if bdybivec4d55   !== nothing; write_block_matrix_list(joinpath(folder, "bdy_bivec4d55.out"), bdybivec4d55); end
    if bdybivec55     !== nothing; write_block_matrix_list(joinpath(folder, "bdy_bivec55.out"), bdybivec55); end
    if nabfrombivec   !== nothing; write_nested_vectors(joinpath(folder, "nabfrombivec.out"), nabfrombivec); end
    if sgndet         !== nothing; write_scalar_list(joinpath(folder, "sgndet.out"), sgndet); end
    if tetareasign    !== nothing; write_nested_scalars(joinpath(folder, "tetareasign.out"), tetareasign); end
    if tetn0sign      !== nothing; write_nested_scalars(joinpath(folder, "tetn0sign.out"), tetn0sign); end
    if zdataf         !== nothing; write_nested_vectors(joinpath(folder, "zdataf.out"), zdataf); end

    println("All geometry data saved to '$folder'")
    end     
end
