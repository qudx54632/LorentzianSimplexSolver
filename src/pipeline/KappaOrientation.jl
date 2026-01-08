module KappaOrientation

using Combinatorics

export fix_kappa_signs!, build_tets, build_tetfaces

# ------------------------------------------------------------
# Build tetrahedra (4-subsets) of one 4-simplex
# ------------------------------------------------------------
function build_tets(four_simplex::Vector{Int})
    return collect(combinations(four_simplex, 4))
end

# ------------------------------------------------------------
# Build faces (triangles) for each tetrahedron
# Insert dummy [0,0,0] at position j
# ------------------------------------------------------------
function build_tetfaces(tets::Vector{Vector{Int}})
    out = Vector{Vector{Vector{Int}}}(undef, length(tets))
    for j in 1:length(tets)
        faces = collect(combinations(tets[j], 3))
        insert!(faces, j, [0, 0, 0])
        out[j] = faces
    end
    return out
end

# ------------------------------------------------------------
# Helper: remove diagonal j-th entry from a vector
# ------------------------------------------------------------
remove_diag(vec, j) = [vec[k] for k in 1:length(vec) if k != j]

# ------------------------------------------------------------
# Helper: flip all κ signs (works for Matrix or Vector-of-Vectors)
# ------------------------------------------------------------
function flip_kappa!(k::AbstractMatrix)
    @inbounds for i in axes(k,1), j in axes(k,2)
        k[i,j] = -k[i,j]
    end
end

function flip_kappa!(k::Vector{<:AbstractVector})
    @inbounds for row in k
        for i in eachindex(row)
            row[i] = -row[i]
        end
    end
end

# ------------------------------------------------------------
# Faithful translation of the Mathematica κ-orientation logic
# ------------------------------------------------------------
function fix_kappa_signs!(four_simplices, geom)

    ns = length(four_simplices)

    # Tets[i][j] = j-th tet of simplex i
    Tets = [build_tets(fs) for fs in four_simplices]

    # TetFaces[i][j] = 5 triangles for tet (dummy included)
    TetFaces = [build_tetfaces(Tets[i]) for i in 1:ns]

    # Copy κ from geometry so we can modify signs
    kappa0    = [geom.simplex[i].kappa for i in 1:ns]
    kappatest = [deepcopy(kappa0[i])   for i in 1:ns]

    # savedTet = list of visited simplex indices
    savedTet = Int[1]
    count = 0

    while length(savedTet) < ns
        count += 1
        i_curr = savedTet[count]

        # Loop over tetrahedra j = 1..5
        for j in 1:length(Tets[i_curr])

            tet = Tets[i_curr][j]

            # pos0 = all (simplex, tet_index) containing this tet
            pos0 = Tuple{Int,Int}[]
            for i2 in 1:ns
                for j2 in 1:length(Tets[i2])
                    if Tets[i2][j2] == tet
                        push!(pos0, (i2, j2))
                    end
                end
            end

            # Only relevant if tet is shared (appears > 1 time)
            if length(pos0) > 1
                # remove current (i_curr, j)
                pos1_candidates = [p for p in pos0 if p != (i_curr, j)]
                pos1 = pos1_candidates[1]
                i2, j2 = pos1

                # skip if already oriented
                if i2 in savedTet
                    continue
                end

                push!(savedTet, i2)

                # Compare TetFaces without the diagonal face
                facesA = remove_diag(TetFaces[i_curr][j], j)
                facesB = remove_diag(TetFaces[i2][j2],  j2)

                if facesA != facesB
                    @warn "faces do not match!" i_curr j i2 j2
                    continue
                end

                # κ-row of simplex i_curr, tet j
                rowA =
                    kappatest[i_curr] isa AbstractMatrix ?
                        collect(kappatest[i_curr][j, :]) :
                        collect(kappatest[i_curr][j])

                # κ-row of simplex i2, tet j2
                rowB =
                    kappatest[i2] isa AbstractMatrix ?
                        collect(kappatest[i2][j2, :]) :
                        collect(kappatest[i2][j2])

                # Remove diagonal entries j and j2
                kapA = remove_diag(rowA, j)
                kapB = remove_diag(rowB, j2)

                # If reduced rows are equal → flip whole κ of simplex i2
                if kapA == kapB
                    flip_kappa!(kappatest[i2])
                end
            end
        end

        # println(savedTet)
    end

    # Write the final κ back into geom
    for i in 1:ns
        geom.simplex[i].kappa .= kappatest[i]
    end

    return geom
end

end # module