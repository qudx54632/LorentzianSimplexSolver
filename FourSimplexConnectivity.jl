module FourSimplexConnectivity

using Combinatorics

export build_global_connectivity

# ------------------------------------------------------------
# Build tetrahedra (4-subsets) for each 4-simplex
# ------------------------------------------------------------
function build_tets_all(four_simplices)
    return [ [collect(a) for a in combinations(fs, 4)]
             for fs in four_simplices ]
end

# ------------------------------------------------------------
# Build triangular faces for each tetrahedron
# Insert dummy face at index j
# ------------------------------------------------------------
function build_tetfaces_all(Tets)
    TetFaces0 = [
        [ [collect(a) for a in combinations(Tets[i][j], 3)]
          for j in 1:length(Tets[i]) ]
        for i in 1:length(Tets)
    ]

    TetFaces = Vector{Vector{Vector{Vector{Int}}}}(undef, length(Tets))

    for i in 1:length(Tets)
        faces_i = Vector{Vector{Vector{Int}}}()
        for j in 1:length(Tets[i])
            faces = TetFaces0[i][j]
            insert!(faces, j, [0,0,0])
            push!(faces_i, faces)
        end
        TetFaces[i] = faces_i
    end

    return TetFaces
end

# ------------------------------------------------------------
# Collect all distinct triangular faces
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
# For each triangle, find all (simplex, tet, face) locations
# ------------------------------------------------------------
function build_face_positions_all(TetFaces, Triangles)
    FacePos = Vector{Vector{Vector{Int}}}(undef, length(Triangles))

    for (n, tri) in enumerate(Triangles)
        pos = Vector{Vector{Int}}()
        for i in 1:length(TetFaces)
            for j in 1:length(TetFaces[i])
                for k in 1:length(TetFaces[i][j])
                    if TetFaces[i][j][k] == tri
                        push!(pos, [i,j,k])
                    end
                end
            end
        end
        FacePos[n] = pos
    end
    return FacePos
end

# ------------------------------------------------------------
# Classify triangles into boundary / bulk
# ------------------------------------------------------------
function classify_faces_global(Tets, Triangles, FacePos)

    BDFaces     = Vector{Vector{Int}}()
    BDFacesPos  = Vector{Vector{Vector{Int}}}()

    BulkFaces    = Vector{Vector{Int}}()
    BulkFacesPos = Vector{Vector{Vector{Int}}}()

    for i in 1:length(Triangles)
        pos = FacePos[i]
        parents = [Tets[p[1]][p[2]] for p in pos]
        unique_parents = unique(parents)

        if length(unique_parents) > length(pos) ÷ 2
            push!(BDFaces, Triangles[i])
            push!(BDFacesPos, pos)
        elseif length(unique_parents) == length(pos) ÷ 2
            push!(BulkFaces, Triangles[i])
            push!(BulkFacesPos, pos)
        end
    end

    return BDFaces, BDFacesPos, BulkFaces, BulkFacesPos
end

# ------------------------------------------------------------
# Find shared tetrahedra across simplices
# ------------------------------------------------------------
function find_shared_tets_global(Tets)
    shared     = Vector{Vector{Int}}()
    shared_pos = Vector{Vector{Vector{Int}}}()

    ns = length(Tets)

    for i in 1:ns
        for j in i+1:ns
            for tet_i in Tets[i]
                for tet_j in Tets[j]
                    if tet_i == tet_j
                        push!(shared, tet_i)

                        pos_i = findfirst(x -> x == tet_i, Tets[i])
                        pos_j = findfirst(x -> x == tet_j, Tets[j])

                        if pos_i !== nothing && pos_j !== nothing
                            push!(shared_pos, [[i,pos_i],[j,pos_j]])
                        end
                    end
                end
            end
        end
    end

    return shared, shared_pos
end

# ------------------------------------------------------------
# Helper: find positions of a value inside selectlinks
# ------------------------------------------------------------
function find_positions_value(selectlinks, value)
    pos = Vector{Vector{Int}}()
    for i in 1:length(selectlinks)
        for j in 1:2
            for k in 1:2
                if selectlinks[i][j][k] == value
                    push!(pos, [i,j,k])
                end
            end
        end
    end
    return pos
end

# ------------------------------------------------------------
# Order bulk faces (Mathematica translation)
# ------------------------------------------------------------
function orderBulk(FacesPosition, sharedTetsPos, k)

    tempfaces = [fp[1:2] for fp in FacesPosition[k]]
    uniqfaces = unique(tempfaces)

    subsets2 = [[a,b] for (a,b) in combinations(uniqfaces, 2)]

    selectlinks = [pair for pair in subsets2 if pair in sharedTetsPos]

    isempty(selectlinks) && return FacesPosition[k]

    templist = selectlinks[1]
    target_len = length(vcat(selectlinks...))

    while length(templist) < target_len
        value = templist[end][1]
        pos_all = find_positions_value(selectlinks, value)
        pos = [p for p in pos_all if p[end] == 1]

        for p in pos
            i, j = p[1], p[2]
            pair = selectlinks[i][j]

            if any(x -> x == pair, templist)
                continue
            end

            push!(templist, pair)

            other = j == 1 ? 2 : 1
            push!(templist, selectlinks[i][other])
        end
    end

    tempfaces = [fp[1:2] for fp in FacesPosition[k]]
    posnew = [findfirst(x -> x == f, tempfaces) for f in templist]

    return [FacesPosition[k][i] for i in posnew]
end

function order_bulk_faces_all(BulkFacesPos, sharedTetsPos)
    return [orderBulk(BulkFacesPos, sharedTetsPos, j)
            for j in 1:length(BulkFacesPos)]
end

# ------------------------------------------------------------
# Order boundary faces around bulk chain
# ------------------------------------------------------------
function order_bdry_faces(FacesPosition, sharedTetsPos, k)
    faces_k = FacesPosition[k]

    if length(faces_k) == 2
        return faces_k
    end

    inner = orderBulk(FacesPosition, sharedTetsPos, k)
    comp = [f for f in faces_k if !(f in inner)]

    if comp[1][1] == inner[1][1]
        return vcat([comp[1]], inner, [comp[2]])
    else
        return vcat([comp[2]], inner, [comp[1]])
    end
end

function order_bdry_faces_all(BDFacesPos, sharedTetsPos)
    return [order_bdry_faces(BDFacesPos, sharedTetsPos, k)
            for k in 1:length(BDFacesPos)]
end

# ------------------------------------------------------------
# Orientation by κ
# ------------------------------------------------------------
function orient_boundary_faces_all(OrderBDryFaces, kappa)
    out = Vector{Vector{Vector{Int}}}(undef, length(OrderBDryFaces))
    for i in 1:length(out)
        seq = OrderBDryFaces[i]
        signs = [kappa[p[1]][p[2]][p[3]] for p in seq]
        out[i] = signs[1] == 1 ? seq : reverse(seq)
    end
    return out
end

function orient_bulk_faces_all(OrderBulkFaces, kappa)
    out = Vector{Vector{Vector{Int}}}(undef, length(OrderBulkFaces))
    for i in 1:length(OrderBulkFaces)
        seq = OrderBulkFaces[i]
        signs = [kappa[p[1]][p[2]][p[3]] for p in seq]
        out[i] = signs[1] == -1 ? seq : reverse(seq)
    end
    return out
end

# ------------------------------------------------------------
# SU(2) gauge-fix selection
# ------------------------------------------------------------
function build_gauge_fix_sets(sharedTetsPos, sgndet)
    GaugeFixUpperTriangle = Vector{Vector{Int}}()
    oppositesl2c          = Vector{Vector{Int}}()

    for pair in sharedTetsPos
        s1, t1 = pair[1][1], pair[1][2]
        s2, t2 = pair[2][1], pair[2][2]

        if sgndet[s1][t1] == 1
            push!(GaugeFixUpperTriangle, [s1, t1])
            push!(oppositesl2c,          [s2, t2])
        end
    end

    return GaugeFixUpperTriangle, oppositesl2c
end

# ------------------------------------------------------------
# Build SU(1,1) gauge-fix triple sets
# ------------------------------------------------------------
# helper must exist BEFORE this function
@inline key2(v::Vector{Int}) = string(v[1], "_", v[2])

function build_timelike_data(sharedTetsPos, sgndet, tetareasign)
    timelike_pairs = Vector{Vector{Vector{Int}}}()
    gaugespacelike = Vector{Vector{Int}}()
    gaugetimelike  = Vector{Vector{Int}}()
    lookup         = Dict{String,Int}()

    for pair in sharedTetsPos
        s1, t1 = pair[1][1], pair[1][2]
        s2, t2 = pair[2][1], pair[2][2]

        if sgndet[s1][t1] == -1
            j_sp = findfirst(j -> tetareasign[s1][t1][j] == 1  && j != t1, 1:5)
            j_tm = findfirst(j -> tetareasign[s1][t1][j] == -1 && j != t1, 1:5)
            (j_sp === nothing || j_tm === nothing) && continue

            p1 = [s1, t1]
            p2 = [s2, t2]

            push!(timelike_pairs, [p1, p2])
            push!(gaugespacelike, [s1, t1, j_sp])
            push!(gaugetimelike,  [s1, t1, j_tm])

            idx = length(timelike_pairs)
            lookup[key2(p1)] = idx
            lookup[key2(p2)] = idx
        end
    end

    return timelike_pairs, gaugespacelike, gaugetimelike, lookup
end

# ------------------------------------------------------------
# Build SL(2,C) gauge-fix sets
# ------------------------------------------------------------
function compute_GaugeTet(sharedTetsPos, GaugeFixUpperTriangle, ns; ntet=5)
    # Flatten shared tet pairs (convert each [s,t] to Vector{Int})
    shared_flat = [Vector(p) for pair in sharedTetsPos for p in pair]

    # Step 1: Build BdryTet
    BdryTet = Vector{Vector{Vector{Int}}}(undef, ns)

    for i in 1:ns
        BdryTet[i] = Vector{Vector{Int}}()
        for j in 1:ntet
            pos = [i, j]
            if !(pos in shared_flat)
                push!(BdryTet[i], pos)
            end
        end
    end

    # Step 2: Build GaugeTet
    GaugeTet = Vector{Vector{Int}}(undef, ns)

    for i in 1:ns
        if isempty(BdryTet[i])
            # fallback to last non–GaugeFixUpperTriangle tet
            fallback = nothing
            for j in 1:ntet
                pos = [i, j]
                if !(pos in GaugeFixUpperTriangle)
                    fallback = pos
                end
            end
            GaugeTet[i] = fallback
        else
            # take first boundary tet (like MMA)
            GaugeTet[i] = BdryTet[i][1]
        end
    end

    return GaugeTet
end

# ------------------------------------------------------------
# Convert Dict{Symbol,Any} → Dict{String,Any}
# ------------------------------------------------------------
function dict_string_keys(d::Dict{Symbol,T}) where T
    out = Dict{String,Any}()
    for (k,v) in d
        out[string(k)] = v
    end
    return out
end

# ------------------------------------------------------------
# Build full connectivity for given simplices and geom structure
# ------------------------------------------------------------
function build_global_connectivity(four_simplices, geom)
    ns = length(four_simplices)
    kappa = [geom.simplex[i].kappa for i in 1:ns]
    sgnd  = [geom.simplex[i].sgndet     for i in 1:ns]
    area  = [geom.simplex[i].tetareasign for i in 1:ns]

    Tets = build_tets_all(four_simplices)
    TetFaces = build_tetfaces_all(Tets)
    Triangles = build_triangles_all(Tets)
    FacePos = build_face_positions_all(TetFaces, Triangles)

    BDFaces, BDFacesPos, BulkFaces, BulkFacesPos =
        classify_faces_global(Tets, Triangles, FacePos)

    sharedTets, sharedTetsPos =
        find_shared_tets_global(Tets)

    OrderBulkFacestest1 = order_bulk_faces_all(BulkFacesPos, sharedTetsPos)
    OrderBDryFacestest1 = order_bdry_faces_all(BDFacesPos, sharedTetsPos)

    OrderBulkFaces = orient_bulk_faces_all(OrderBulkFacestest1, kappa)
    OrderBDryFaces = orient_boundary_faces_all(OrderBDryFacestest1, kappa)

    GaugeFixUpperTriangle, oppositesl2c = build_gauge_fix_sets(sharedTetsPos, sgnd)
    timelike_pairs, gaugespacelike, gaugetimelike, lookup = build_timelike_data(sharedTetsPos, sgnd, area)

    GaugeTet = compute_GaugeTet(sharedTetsPos, GaugeFixUpperTriangle, ns)

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

        :GaugeFixUpperTriangle => GaugeFixUpperTriangle,
        :oppositesl2c   => oppositesl2c,
        :timelike_pairs => timelike_pairs,
        :gaugespacelike => gaugespacelike,
        :gaugetimelike  => gaugetimelike,
        :lookup         => lookup,
        :GaugeTet       => GaugeTet
    )

    return dict_string_keys(raw)
end

end # module