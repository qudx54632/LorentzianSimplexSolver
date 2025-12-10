module GeometryDataLoader

using ..GeometryTypes: GeometryDataset, GeometryCollection
using ..LoadGeometryData:
    load_matrix_list,
    load_block_matrix_list,
    load_nested_vectors,
    load_nested_scalars,
    load_nested_real_vectors,
    load_vector_list,
    load_scalar_list

export load_geometry_dataset, load_multiple_simplices

"""
Load ONE simplex from a folder.
"""
function load_geometry_dataset(folder::String)
    endswith(folder, "/") || (folder *= "/")

    solgsl2c       = load_matrix_list(folder * "sol_g_sl2c.out")
    solgso13       = load_matrix_list(folder * "sol_g_so13.out")

    bdyxi          = load_nested_vectors(folder * "bdy_xi.out")
    nabout         = load_nested_real_vectors(folder * "nabout.out")
    nabfrombivec   = load_nested_real_vectors(folder * "nabfrombivec.out")

    bdysu          = load_block_matrix_list(folder * "bdy_su.out")
    bdybivec4d55   = load_block_matrix_list(folder * "bdy_bivec4d55.out")
    bdybivec55     = load_block_matrix_list(folder * "bdy_bivec55.out")

    dihedrals      = load_nested_scalars(folder * "dihedrals.out")
    areas          = load_nested_scalars(folder * "areas.out")
    kappa          = load_nested_scalars(folder * "kappa.out")
    tetareasign    = load_nested_scalars(folder * "tetareasign.out")
    tetn0sign      = load_nested_scalars(folder * "tetn0sign.out")

    tetnormalvec   = load_vector_list(folder * "tetnormalvec.out")
    sgndet         = load_scalar_list(folder * "sgndet.out")

    return GeometryDataset(
        solgsl2c, solgso13,
        bdyxi, nabout, nabfrombivec,
        bdysu, bdybivec4d55, bdybivec55,
        dihedrals, areas, kappa, tetareasign, tetn0sign,
        tetnormalvec, sgndet
    )
end

"""
Load multiple simplex folders into a GeometryCollection.
USAGE:
    geom = load_multiple_simplices(["simp1/", "simp2/"])
"""
function load_multiple_simplices(folders::Vector{String})
    datasets = GeometryDataset[]
    for f in folders
        push!(datasets, load_geometry_dataset(f))
    end
    return GeometryCollection(datasets)
end

end # module