"""
    construct_curved_geometry(vertices_for_each_simplex)

Construct a Regge geometry when every 4-simplex has its own local vertex
coordinates.
"""
function construct_curved_geometry(vertices_for_each_simplex; verbose=false)
    T = eltype(vertices_for_each_simplex[1][1])
    datasets = GeometryDataset{T}[]

    for n in eachindex(vertices_for_each_simplex)
        verbose && println("Constructing 4-simplex $n")
        points = vertices_for_each_simplex[n]
        push!(datasets, run_geometry_pipeline(points))
    end

    return GeometryCollection(datasets)
end
