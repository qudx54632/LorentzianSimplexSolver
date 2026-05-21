module ReggeAction

using ForwardDiff
using Combinatorics
using ..Volume: distance_sq, V2sq, V3sq, V4sq
using ..FourSimplexConnectivity: build_tets_all, build_tetfaces_all

export run_iRegge_action


"""
θfunc(
    simplex_edges_syms::Vector{SymEngine.Basic},  # length 10, expressions for l_e^2
    teta_edges_syms::Vector{SymEngine.Basic},     # length 6, expressions for l_e^2
    tetb_edges_syms::Vector{SymEngine.Basic},     # length 6, expressions for l_e^2
    ah_edges_syms::Vector{SymEngine.Basic},       # length 3, expressions for l_e^2
    edge_to_diff_idx::Int                         # 1..10
)
Returns a SymEngine expression for the (real-convention) dihedral angle.
If the edge_to_diff_idx is invalid or the chosen entry contains no differentiable symbol,
this will return an expression with derivative = 0 or SymEngine may error depending on your case.
"""
function θfunc(simplex_edges, teta_edges, tetb_edges, ah_edges, edge_to_diff_idx::Int)
    # fixed floats
    Vtaua = V3sq(teta_edges[1], teta_edges[2], teta_edges[3],
                                                 teta_edges[4], teta_edges[5], teta_edges[6])
    Vtaub = V3sq(tetb_edges[1], tetb_edges[2], tetb_edges[3],
                                                 tetb_edges[4], tetb_edges[5], tetb_edges[6])
    Vt    = V2sq(ah_edges[1], ah_edges[2], ah_edges[3])

    x0 = simplex_edges[edge_to_diff_idx]

    f(x) = begin
        T = typeof(x)
        l2new = T.(simplex_edges)
        l2new[edge_to_diff_idx] = x
        V4sq(
            l2new[1],  l2new[2],  l2new[3],  l2new[4],  l2new[5],
            l2new[6],  l2new[7],  l2new[8],  l2new[9],  l2new[10]
        )
    end
    Vtauab = ForwardDiff.derivative(f, x0)
    term1 = (4^2) / Vt * Vtauab
    term2 = Vtaua / Vt
    term3 = Vtaub / Vt

    sqrt_term2 = term2 < 0 ? im*sqrt(-term2) : sqrt(term2)
    sqrt_term3 = term3 < 0 ? im*sqrt(-term3) : sqrt(term3)

    sqrt_product = term2*term3 - term1^2
    sqrt_product = sqrt_product < 0 ? im*sqrt(-sqrt_product) : sqrt(sqrt_product)

    c = (term1 - im*sqrt_product) / (sqrt_term2*sqrt_term3)
    creal = real(c)

    return creal < 0 ? -im*(log(-creal) - im*pi) : -im*log(creal)
end

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
_to_edge(a::Int,b::Int) = (min(a,b), max(a,b))

# all edges (2-subsets) of a 5-vertex simplex, returned in a fixed order i<j
function edges10_of5(v5::AbstractVector{Int})
    v = collect(v5)
    return [ _to_edge(v[i], v[j]) for i in 1:4 for j in (i+1):5 ]  # length 10
end

# tetrahedral edges (2-subsets) of a 4-vertex tetra, returned in fixed order i<j
function edges6_of4(v4::AbstractVector{Int})
    v = collect(v4)
    return [ _to_edge(v[i], v[j]) for i in 1:3 for j in (i+1):4 ]  # length 6
end

# triangle edges (3)
function tri_edges(tri::Vector{Int})
    a,b,c = collect(tri)
    return [ _to_edge(a,b), _to_edge(a,c), _to_edge(b,c) ]
end
_to_sorted3(tri::Vector{Int}) = sort(collect(tri))


"""
build_thetafunc_inputs_one_simplex(
    Vertices,
    simplex5::Vector{Int},          # 5 vertices of ONE 4-simplex
    tri3::Vector{Int},             # 3 vertices of the hinge triangle
)
Return:
(simplex_edges_syms, teta_edges_syms, tetb_edges_syms, ah_edges_syms, idx)
If tri3 is not a subset (order-independent) of simplex5, returns:
([], [], [], [], 0)
idx is the position (1..10) in simplex_edges_syms corresponding to bdry_pertu_edge
if that perturbed boundary edge lies in this simplex; otherwise idx=0 (and you can choose
to set idx differently if your θfunc expects a bulk perturbed edge).
"""
function build_thetafunc_inputs_one_simplex(
    Vertices,
    simplex5::Vector{Int},
    tri3::Vector{Int}
)
    simplex_set = Set(simplex5)
    tri = _to_sorted3(tri3)
    # triangle must be contained in this simplex
    if !(all(v -> v in simplex_set, tri))
        return ([], [], [], [], [])
    end

    simplex = collect(simplex5)

    edge_symbol_or_numeric(e::Tuple{Int,Int}) = begin
        i,j = e
        return distance_sq(Vertices[i], Vertices[j])
    end
    
    # simplex edges (10)
    simplex_edges = edges10_of5(simplex)
    simplex_edges_syms = [edge_symbol_or_numeric(e) for e in simplex_edges]

    # determine the two tetrahedra adjacent to this triangle inside the 4-simplex
    tri_set = Set(tri)
    rest = [v for v in simplex if !(v in tri_set)]
    @assert length(rest) == 2 "In a 4-simplex, a triangle must leave exactly 2 remaining vertices."
    vA, vB = rest[1], rest[2]
    tetraA = vcat(tri, [vA])  # 4 vertices
    tetraB = vcat(tri, [vB])  # 4 vertices
    teta_edges = edges6_of4(tetraA)
    tetb_edges = edges6_of4(tetraB)
    teta_edges_syms = [edge_symbol_or_numeric(e) for e in teta_edges]
    tetb_edges_syms = [edge_symbol_or_numeric(e) for e in tetb_edges]

    # triangle edges for ah
    ah_edges = tri_edges(tri)
    ah_edges_syms = [edge_symbol_or_numeric(e) for e in ah_edges]
    
    # opposite edge inside this 4-simplex: between the two vertices not in the hinge triangle
    opp_edge = _to_edge(vA, vB)                 # vA,vB already computed above
    idx = findfirst(==(opp_edge), simplex_edges)
    idx = idx === nothing ? 0 : idx
    return (simplex_edges_syms, teta_edges_syms, tetb_edges_syms, ah_edges_syms, idx)
end

function run_Regge_action_multiple_vertices(geom, simplices, vertex_coords)
    tetsfaces = geom.connectivity[1]["TetFaces"]
    bulktriangles = [tetsfaces[faces[1][1]][faces[1][2]][faces[1][3]] for faces in geom.connectivity[1]["OrderBulkFaces"]]
    bdrytriangles = [tetsfaces[faces[1][1]][faces[1][2]][faces[1][3]] for faces in geom.connectivity[1]["OrderBDryFaces"]]

    deficit_angles = []
    for tri in bulktriangles
        face_angles = []
        for simplex5 in simplices
            (simplex_edges_syms, teta_edges_syms, tetb_edges_syms, ah_edges_syms, idx) =
                build_thetafunc_inputs_one_simplex(vertex_coords, simplex5, tri)

            if !isempty(simplex_edges_syms)
                angle_val = θfunc(simplex_edges_syms, teta_edges_syms, tetb_edges_syms, ah_edges_syms, idx)
                push!(face_angles, angle_val)
            end
        end
        deficit_angle = 2*pi + sum(face_angles)
        push!(deficit_angles, deficit_angle/im)
    end
    areas_bulk = [geom.simplex[faces[1][1]].areas[faces[1][2]][faces[1][3]] for faces in geom.connectivity[1]["OrderBulkFaces"]]

    dihedral_angles = []
    for tri in bdrytriangles
        face_angles = []
        for simplex5 in simplices
            (simplex_edges_syms, teta_edges_syms, tetb_edges_syms, ah_edges_syms, idx) =
                build_thetafunc_inputs_one_simplex(vertex_coords, simplex5, tri)

            if !isempty(simplex_edges_syms)
                angle_val = θfunc(simplex_edges_syms, teta_edges_syms, tetb_edges_syms, ah_edges_syms, idx)
                push!(face_angles, angle_val)
            end
        end

        angle = sum(face_angles)
        phase = abs(real(angle)/pi) * pi
        dihedral_angle = angle + phase

        push!(dihedral_angles, dihedral_angle/im)
    end
    areas_bdry = [geom.simplex[faces[1][1]].areas[faces[1][2]][faces[1][3]] for faces in geom.connectivity[1]["OrderBDryFaces"]]

    Regge_action = sum(areas_bulk[i] * deficit_angles[i] for i in eachindex(deficit_angles)) +
                    sum(areas_bdry[i] * dihedral_angles[i] for i in eachindex(dihedral_angles))  

    return deficit_angles, dihedral_angles, areas_bulk, areas_bdry, im * Regge_action
end


function run_Regge_action(geom, simplices, vertex_coords)
    ns = length(simplices)
    if ns > 1
        return run_Regge_action_multiple_vertices(geom, simplices, vertex_coords)
    else
        simplex5 = simplices[1]
        tets = build_tets_all(simplices)
        tetsfaces = build_tetfaces_all(tets)
        kappaMat = [geom.simplex[i].kappa for i in 1:length(geom.simplex)]
        OrderBDryFaces = [kappaMat[1][i][j]== 1 ? [[1, i, j], [1, j, i]] : [[1, j, i], [1, i, j]] for i in 1:5 for j in 1:5 if i < j]
        bdrytriangles = [tetsfaces[faces[1][1]][faces[1][2]][faces[1][3]] for faces in OrderBDryFaces]
        Regge_action = 0
        dihedral_angles = []
        areas = []
        for tri3 in bdrytriangles
            (simplex_edges_syms, teta_edges_syms, tetb_edges_syms, ah_edges_syms, idx) =
            build_thetafunc_inputs_one_simplex(vertex_coords, simplex5, tri3)

            angle = θfunc(simplex_edges_syms, teta_edges_syms, tetb_edges_syms, ah_edges_syms, idx)
            dihedral_angle = imag(angle) 
            push!(dihedral_angles, dihedral_angle)
            area = sqrt(V2sq(ah_edges_syms[1], ah_edges_syms[2], ah_edges_syms[3]))
            push!(areas, area)
        end
        Regge_action = sum(areas[i] * dihedral_angles[i] for i in eachindex(dihedral_angles)) 

        return [0], dihedral_angles, [0], areas, im * Regge_action  
    end 
end

end
