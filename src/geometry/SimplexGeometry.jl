module SimplexGeometry

export Simplex, edge_or, face_or, eto_f, tet_order,
       edges_from_faces, edges_from_faces_num,
       build_tets, build_tetfaces

using Combinatorics

# -----------------------------------------------------------
# 4-simplex vertices
# -----------------------------------------------------------
const VERTICES = 1:5

"""Return the tetrahedra obtained by omitting one vertex from a 4-simplex."""
build_tets(four_simplex::AbstractVector{<:Integer}) =
    [collect(tet) for tet in combinations(four_simplex, 4)]

"""
Return the triangular faces of each tetrahedron, inserting a zero face at
the tetrahedron's diagonal position to preserve the package's 5x5 indexing.
"""
function build_tetfaces(tets::AbstractVector{<:AbstractVector{<:Integer}})
    return [
        insert!([collect(face) for face in combinations(tet, 3)], j, [0, 0, 0])
        for (j, tet) in enumerate(tets)
    ]
end

# -----------------------------------------------------------
# Oriented tetrahedra: all subsets of size 4
# -----------------------------------------------------------
const tet_order = collect(reverse(VERTICES))     # {5,4,3,2,1}
const TETRA = build_tets(collect(VERTICES))      # 5 tetrahedra

# -----------------------------------------------------------
# Edges and faces
# -----------------------------------------------------------
const edge_or = [ collect(combinations(tet, 2)) for tet in TETRA ]
const face_or = [ collect(combinations(tet, 3)) for tet in TETRA ]

# etof = {{1,2},{1,3},{2,3},{4,5}}
const eto_f = [[1,2], [1,3], [2,3], [4,5]]

# -----------------------------------------------------------
# Edges contained in each face
# -----------------------------------------------------------
const edges_from_faces = [
    [
        collect(combinations(face, 2)) for face in face_or[i]
    ]
    for i in 1:5
]

# -----------------------------------------------------------
# Helper: find index of an edge in the edge list
# -----------------------------------------------------------
function find_edge_index(edge_list, edge)
    e = sort(edge)
    for (k, e2) in enumerate(edge_list)
        if sort(e2) == e
            return k
        end
    end
    error("Edge $(edge) not found in edge list $(edge_list)")
end

# -----------------------------------------------------------
# Numerical index for edges inside faces
# -----------------------------------------------------------
const edges_from_faces_num = [
    [
        [ find_edge_index(edge_or[i], edge)
          for edge in edges_from_faces[i][j]
        ]
        for j in 1:4
    ]
    for i in 1:5
]

# -----------------------------------------------------------
# Simplex container with keyword constructor
# -----------------------------------------------------------
Base.@kwdef struct Simplex
    edge_or::Vector{Vector{Vector{Int}}} = edge_or
    face_or::Vector{Vector{Vector{Int}}} = face_or
    edges_from_faces::Vector = edges_from_faces
    edges_from_faces_num::Vector = edges_from_faces_num
end

end # module
