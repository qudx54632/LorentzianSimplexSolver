module FourSimplexConnectivity

using Combinatorics

export build_global_connectivity

# ------------------------------------------------------------
# Build tetrahedra (4-subsets) for each 4-simplex
# four_simplices :: Vector{Vector{Int}} (each length 5)
# Returns: Tets[i][j] = j-th tet (Vector{Int}) in simplex i
# ------------------------------------------------------------
function build_tets_all(four_simplices)
    return [ [collect(a) for a in combinations(fs, 4)]
             for fs in four_simplices ]
end

# ------------------------------------------------------------
# Build face lists for each tetrahedron in each simplex,
# and insert a dummy face [0,0,0] at the diagonal position j.
#
# Tets[i][j]  : j-th tetrahedron in simplex i
# TetFaces[i][j][k] : k-th face (triangle) of that tetrahedron
# ------------------------------------------------------------
function build_tetfaces_all(Tets)

    TetFaces0 = [
        [ [collect(a) for a in combinations(Tets[i][j], 3)]
          for j in 1:length(Tets[i]) ]
        for i in 1:length(Tets)
    ]

    TetFaces = Vector{Vector{Vector{Vector{Int}}}}(undef, length(Tets))

    for i in 1:length(Tets)
        TetFaces[i] = Vector{Vector{Vector{Int}}}()

        for j in 1:length(Tets[i])
            faces = TetFaces0[i][j]
            insert!(faces, j, [0,0,0])   # dummy
            push!(TetFaces[i], faces)
        end
    end

    return TetFaces
end

# ------------------------------------------------------------
# Collect all distinct triangular faces from all tetrahedra
# ------------------------------------------------------------
function build_triangles_all(Tets)
    tris = Vector{Vector{Int}}()
    for i in 1:length(Tets)
        for j in 1:length(Tets[i])
            append!(tris, [collect(a) for a in combinations(Tets[i][j], 3)])
        end
    end
    return unique(tris)
end

# ------------------------------------------------------------
# For each triangle, record all positions (simplex, tet, face)
# where it appears in TetFaces.
#
# FacePos[n] = list of (i, j, k) with TetFaces[i][j][k] == Triangles[n]
# ------------------------------------------------------------
function build_face_positions_all(TetFaces, Triangles)

    FacePos = Vector{Vector{Vector{Int}}}(undef, length(Triangles))

    for (n, tri) in enumerate(Triangles)
        pos = Vector{Vector{Int}}()

        for i in 1:length(TetFaces)
            for j in 1:length(TetFaces[i])
                for k in 1:length(TetFaces[i][j])
                    if TetFaces[i][j][k] == tri
                        push!(pos, [i, j, k])  # NO TUPLES
                    end
                end
            end
        end

        FacePos[n] = pos
    end

    return FacePos
end

# ------------------------------------------------------------
# Classify triangles globally into boundary or bulk faces.
#
# BDFaces / BDFacesPos: boundary triangles and their positions.
# BulkFaces / BulkFacesPos: internal triangles and their positions.
# ------------------------------------------------------------
function classify_faces_global(Tets, Triangles, FacePos)

    BDFaces     = Vector{Vector{Int}}()
    BDFacesPos  = Vector{Vector{Vector{Int}}}()

    BulkFaces    = Vector{Vector{Int}}()
    BulkFacesPos = Vector{Vector{Vector{Int}}}()

    for i in 1:length(Triangles)
        pos = FacePos[i]    # each pos element is [i,j,k]

        parents = [Tets[p[1]][p[2]] for p in pos]
        unique_parents = unique(parents)

        if length(unique_parents) > length(pos) / 2
            push!(BDFaces, Triangles[i])
            push!(BDFacesPos, pos)

        elseif length(unique_parents) == length(pos) / 2
            push!(BulkFaces, Triangles[i])
            push!(BulkFacesPos, pos)
        end
    end

    return BDFaces, BDFacesPos, BulkFaces, BulkFacesPos
end

# ------------------------------------------------------------
# Find tetrahedra that are shared between different 4-simplices.
#
# shared      : list of shared tetrahedra (as vertex sets)
# shared_pos  : for each shared tet, its positions (simplex, tet_index)
# ------------------------------------------------------------
function find_shared_tets_global(Tets)

    shared     = Vector{Vector{Int}}()
    shared_pos = Vector{Vector{Vector{Int}}}()

    ns = length(Tets)

    for i in 1:ns
        for j in i+1:ns

            inter = intersect(Tets[i], Tets[j])  # still Vector{Vector{Int}}

            for tet in inter
                push!(shared, tet)

                pos_i = findfirst(x -> x == tet, Tets[i])
                pos_j = findfirst(x -> x == tet, Tets[j])

                push!(shared_pos, [[i, pos_i], [j, pos_j]])  # NO TUPLES
            end
        end
    end

    return shared, shared_pos
end

# ------------------------------------------------------------
# Order the bulk faces of a given triangle index k,
# using adjacency information encoded in shared_pos.
#
# FacePos[k] is a list of (simplex, tet, face) positions.
# The result is an ordered list of those positions.
# ------------------------------------------------------------
function find_positions_value(selectlinks, value)
    pos = Vector{Vector{Int}}()
    for i in 1:length(selectlinks)
        for j in 1:2           # each selectlinks[i] has 2 faces
            for k in 1:2       # each face has 2 vertices
                if selectlinks[i][j][k] == value
                    push!(pos, [i, j, k])
                end
            end
        end
    end
    return pos
end

function orderBulk(FacesPosition, sharedTetsPos, k)
    # FacesPosition[[k, All, 1;;2]] in Mathematica
    tempfaces = [fp[1:2] for fp in FacesPosition[k]]
    uniqfaces = unique(tempfaces)

    # Subsets[..., {2}]
    subsets2 = [[a, b] for (a, b) in combinations(uniqfaces, 2)]

    # If[MemberQ[sharedTetsPos, #], #, Sequence[]] & /@ ...
    selectlinks = [pair for pair in subsets2 if pair in sharedTetsPos]

    # this is the object you checked in Mma
    isempty(selectlinks) && return FacesPosition[k]

    # templist = selectlinks[[1]]
    templist = selectlinks[1]
    target_len = length(vcat(selectlinks...))


    while length(templist) < target_len
        # templist[[-1,1]]  → last face in templist, its first vertex
        value = templist[end][1]

        # Position[selectlinks, value]
        pos_all = find_positions_value(selectlinks, value)
        # If[#[[-1]] == 1, #, Sequence[]] & /@ ...
        pos = [p for p in pos_all if p[end] == 1]

        for p in pos
            i, j = p[1], p[2]

            pair = selectlinks[i][j]      # selectlinks[[i,j]]

            # MemberQ[templist, pair]
            if any(x -> x == pair, templist)
                continue
            end

            push!(templist, pair)         # AppendTo[templist, selectlinks[[i,j]]]

            other = j == 1 ? 2 : 1        # Complement[{1,2},{j}]
            push!(templist, selectlinks[i][other])
        end
    end

    tempfaces = [fp[1:2] for fp in FacesPosition[k]]
    # Now reproduce the last part of the Mathematica function:
    posnew = [findfirst(x -> x == f, tempfaces) for f in templist]

    out = [FacesPosition[k][i] for i in posnew]
    #FacesPosition[[k,#]] & /@ posnew
    return out
end

# ------------------------------------------------------------
# Order the bulk faces for each bulk triangle
# ------------------------------------------------------------
function order_bulk_faces_all(BulkFacesPos, sharedTetsPos)
    out = Vector{Vector{Vector{Int}}}(undef, length(BulkFacesPos))

    for j in 1:length(BulkFacesPos)
        out[j] = orderBulk(BulkFacesPos, sharedTetsPos, j)
    end
    return out
end

# ------------------------------------------------------------
# Order boundary faces with bulk chain inserted inside
# ------------------------------------------------------------
function order_bdry_faces(FacesPosition, sharedTetsPos, k)

    faces_k = FacesPosition[k]

    # Case 1: only two faces → return directly
    if length(faces_k) == 2
        return faces_k
    end

    # Case 2: bulk interior ordering
    innerorder = orderBulk(FacesPosition, sharedTetsPos, k)

    # Complement: remaining boundary faces
    comp = [f for f in faces_k if !(f in innerorder)]

    # The Mathematica condition: comp[[1,1]] == innerorder[[1,1]]
    if comp[1][1] == innerorder[1][1]
        # Join[{comp[[1]]}, innerorder, {comp[[2]]}]
        return vcat([comp[1]], innerorder, [comp[2]])
    else
        # Join[{comp[[2]]}, innerorder, {comp[[1]]}]
        return vcat([comp[2]], innerorder, [comp[1]])
    end
end

function order_bdry_faces_all(BDFacesPos, sharedTetsPos)
    return [order_bdry_faces(BDFacesPos, sharedTetsPos, k)
            for k in 1:length(BDFacesPos)]
end

function orient_boundary_faces_all(OrderBDryFacestest1, kappa)
    out = Vector{Vector{Vector{Int}}}(undef, length(OrderBDryFacestest1))

    for i in 1:length(out)
        seq = OrderBDryFacestest1[i]
        signs = [kappa[a][b][c] for (a, b, c) in seq]

        if signs[1] == 1
            out[i] = seq
        else
            out[i] = reverse(seq)
        end
    end
    return out
end

function orient_bulk_faces_all(OrderBulkFacestest1, kappa)
    # positions are stored as Vector{Int}, so keep that type
    out = Vector{Vector{Vector{Int}}}(undef, length(OrderBulkFacestest1))

    for i in 1:length(OrderBulkFacestest1)
        seq = OrderBulkFacestest1[i]   # seq :: Vector{Vector{Int}}

        # (a,b,_) pattern-matches each [a,b,c] vector
        signs = [kappa[a][b][c] for (a, b, c) in seq]

        if signs[1] == -1
            out[i] = seq
        else
            out[i] = reverse(seq)
        end
    end

    return out
end

#------------------------------------------
# Convert Dict{Symbol,Any} → Dict{String,Any}
#------------------------------------------
"""
Convert any Dict with Symbol keys → Dict{String,Any}
"""
function dict_string_keys(d::Dict{Symbol,T}) where T
    out = Dict{String,Any}()
    for (k, v) in d
        out[string(k)] = v
    end
    return out
end

# ------------------------------------------------------------
# Build global connectivity information for a list of 4-simplices.
#
# four_simplices :: Vector{Vector{Int}} (each length 5)
# kappa          :: not yet used here, reserved for later orientation
#
# Returns a Dict with all basic geometric connectivity data:
#   :Tets, :TetFaces, :Triangles, :FacePosition,
#   :BDFaces, :BDFacesPos, :BulkFaces, :BulkFacesPos,
#   :sharedTets, :sharedTetsPos
# ------------------------------------------------------------
function build_global_connectivity(four_simplices, geom)

    kappa = [geom.simplex[i].kappa for i in 1:length(four_simplices)]
    # Tetrahedra per simplex
    Tets = build_tets_all(four_simplices)

    # Faces of each tetrahedron
    TetFaces = build_tetfaces_all(Tets)

    # All distinct triangular faces
    Triangles = build_triangles_all(Tets)

    # Positions of each triangle in the full complex
    FacePos = build_face_positions_all(TetFaces, Triangles)

    # Boundary and bulk triangle classification
    BDFaces, BDFacesPos, BulkFaces, BulkFacesPos =
        classify_faces_global(Tets, Triangles, FacePos)

    # Shared tetrahedra between different simplices
    sharedTets, sharedTetsPos = find_shared_tets_global(Tets)

    # Step 7: order faces
    OrderBulkFacestest1 = order_bulk_faces_all(BulkFacesPos, sharedTetsPos)
    OrderBDryFacestest1 = order_bdry_faces_all(BDFacesPos, sharedTetsPos)

    # Step 8: κ–orientation
    OrderBulkFaces = orient_bulk_faces_all(OrderBulkFacestest1, kappa)
    OrderBDryFaces = orient_boundary_faces_all(OrderBDryFacestest1, kappa)

    raw = Dict(
        :Tets           => Tets,
        :TetFaces       => TetFaces,
        :Triangles      => Triangles,
        :FacePosition   => FacePos,
        :BDFaces        => BDFaces,
        :BDFacesPos     => BDFacesPos,
        :BulkFaces      => BulkFaces,
        :BulkFacesPos   => BulkFacesPos,
        :sharedTets     => sharedTets,
        :sharedTetsPos  => sharedTetsPos,
        :OrderBulkFaces => OrderBulkFaces,
        :OrderBDryFaces => OrderBDryFaces,
    )

    return dict_string_keys(raw)
end

end # module FourSimplexConnectivity