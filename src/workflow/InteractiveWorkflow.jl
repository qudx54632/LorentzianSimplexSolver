module InteractiveWorkflow

using LinearAlgebra
using SymEngine

import ..PrecisionUtils
using ..GeometryTypes
using ..GeometryPipeline
using ..GeometryConsistency
using ..KappaOrientation
using ..FourSimplexConnectivity
using ..FaceXiMatching
using ..FaceMatchingChecks
using ..GaugeFixingSU
using ..DefineAction
using ..DefineSymbols
using ..SolveVars
using ..SFaction_no_phase
using ..ActionEvaluation
using ..EOMs
using ..Hessian
using ..ReggeAction

export configure_precision!,
       construct_geometry,
       check_simplex_consistency,
       prepare_global_geometry!,
       compute_regge_action,
       compute_spinfoam_action,
       compute_eom,
       check_eom,
       compute_hessian,
       ReggeActionData,
       SpinfoamActionData,
       HessianData

"""Regge deficit angles, boundary angles, areas, and the value `i S_Regge`."""
Base.@kwdef struct ReggeActionData
    deficit_angles::Any
    dihedral_angles::Any
    bulk_areas::Any
    boundary_areas::Any
    iregge::Any
end

"""Symbolic spinfoam action together with its real geometric solution."""
Base.@kwdef struct SpinfoamActionData
    symbols::Any
    phase_solution::Any
    action_no_phase::Any
    value_dict::Any
    action::Any
    solve_data::Any
    gamma_symbol::Basic
end

"""Symbolic and numerical Hessian data for the selected variables."""
Base.@kwdef struct HessianData
    variables::Any
    symbols::Any
    matrix::Any
    eigenvalues::Any = nothing
end

"""Select `Float64` or `BigFloat` arithmetic for the complete workflow."""
configure_precision!(::Type{T}; precision::Integer=100) where {T<:AbstractFloat} =
    PrecisionUtils.configure_precision!(T; precision=precision)

function validate_geometry_precision(geom::GeometryTypes.GeometryCollection{T}) where {T<:AbstractFloat}
    PrecisionUtils.validate_active_precision(T)
    return T
end

"""Construct all 4-simplices from one global coordinate for each vertex."""
function construct_geometry(simplices, vertex_coords; verbose::Bool=false)
    T = eltype(first(values(vertex_coords)))
    PrecisionUtils.validate_precision(T, values(vertex_coords))
    datasets = GeometryTypes.GeometryDataset{T}[]

    for (idx, simplex) in enumerate(simplices)
        verbose && println("--- Processing simplex $idx with vertices $simplex ---")
        bdypoints = [vertex_coords[v] for v in simplex]
        push!(datasets, GeometryPipeline.run_geometry_pipeline(bdypoints))
    end

    return GeometryTypes.GeometryCollection(datasets)
end

"""Check the geometric critical-point equations separately in every 4-simplex."""
function check_simplex_consistency(geom)
    validate_geometry_precision(geom)
    failed_simplices = Int[]

    for (idx, simplex) in enumerate(geom.simplex)
        GeometryConsistency.critical_point_equations_satisfied(simplex) ||
            push!(failed_simplices, idx)
    end

    if isempty(failed_simplices)
        println("✓ Every 4-simplex satisfies the critical point equations.")
    else
        for idx in failed_simplices
            println("✗ 4-simplex $idx does not satisfy the critical point equations.")
        end
    end

    return nothing
end

"""
Connect the 4-simplices, fix their orientation, match shared faces, and gauge
fix the resulting boundary data.
"""
function prepare_global_geometry!(
    geom,
    simplices;
    sector::Symbol=:ref,
    match_faces::Bool=true,
    gauge_fix::Bool=true,
    check::Bool=false,
    verbose::Bool=false
)
    validate_geometry_precision(geom)

    if length(simplices) <= 1
        verbose && println("Only one simplex detected. Global connectivity is skipped.")
        return geom
    end

    verbose && println("Fixing global kappa-sign orientation.")
    KappaOrientation.fix_kappa_signs!(simplices, geom)

    verbose && println("Building global connectivity.")
    conn = FourSimplexConnectivity.build_global_connectivity(simplices, geom)
    if isempty(geom.connectivity)
        push!(geom.connectivity, conn)
    else
        geom.connectivity[1] = conn
    end

    if match_faces
        verbose && println("Matching face data.")
        FaceXiMatching.run_face_xi_matching(geom; sector=sector)
    end

    check && FaceMatchingChecks.check_all(geom)

    if gauge_fix
        verbose && println("Performing SU(2) and SU(1,1) gauge fixing.")
        GaugeFixingSU.run_su2_su11_gauge_fix(geom)
    end

    return geom
end

"""Compute the Lorentzian Regge action from global or simplex-local coordinates."""
function compute_regge_action(geom, simplices, vertex_coords)
    T = validate_geometry_precision(geom)
    PrecisionUtils.validate_precision(T, values(vertex_coords))

    deficit_angles, dihedral_angles, bulk_areas, boundary_areas, iregge =
        ReggeAction.run_Regge_action(geom, simplices, vertex_coords)

    return ReggeActionData(
        deficit_angles=deficit_angles,
        dihedral_angles=dihedral_angles,
        bulk_areas=bulk_areas,
        boundary_areas=boundary_areas,
        iregge=iregge,
    )
end

function curved_face_triangle(tetsfaces, face)
    return tetsfaces[face[1][1]][face[1][2]][face[1][3]]
end

function curved_local_angle(local_coords, simplices, simplex_number, triangle)
    data = ReggeAction.build_thetafunc_inputs_one_simplex(
        local_coords[simplex_number],
        simplices[simplex_number],
        triangle,
    )
    return ReggeAction.θfunc(data...)
end

function compute_regge_action(
    geom,
    simplices,
    vertices_for_each_simplex::AbstractVector,
)
    T = validate_geometry_precision(geom)
    for local_coordinates in vertices_for_each_simplex
        PrecisionUtils.validate_precision(T, local_coordinates)
    end

    local_coords = [
        Dict(
            simplices[k][i] => vertices_for_each_simplex[k][i]
            for i in eachindex(simplices[k])
        )
        for k in eachindex(simplices)
    ]

    if length(simplices) == 1
        return compute_regge_action(geom, simplices, local_coords[1])
    end

    tetsfaces = geom.connectivity[1]["TetFaces"]
    bulk_faces = geom.connectivity[1]["OrderBulkFaces"]
    boundary_faces = geom.connectivity[1]["OrderBDryFaces"]

    deficit_angles = [
        (
            2pi + sum(
                curved_local_angle(
                    local_coords,
                    simplices,
                    k,
                    curved_face_triangle(tetsfaces, face),
                )
                for k in eachindex(simplices)
                if all(
                    v -> v in simplices[k],
                    curved_face_triangle(tetsfaces, face),
                )
            )
        ) / im
        for face in bulk_faces
    ]

    dihedral_angles = [
        begin
            angle = sum(
                curved_local_angle(
                    local_coords,
                    simplices,
                    k,
                    curved_face_triangle(tetsfaces, face),
                )
                for k in eachindex(simplices)
                if all(
                    v -> v in simplices[k],
                    curved_face_triangle(tetsfaces, face),
                )
            )
            angle = angle + abs(real(angle) / pi) * pi
            angle / im
        end
        for face in boundary_faces
    ]

    bulk_areas = [
        geom.simplex[face[1][1]].areas[face[1][2]][face[1][3]]
        for face in bulk_faces
    ]
    boundary_areas = [
        geom.simplex[face[1][1]].areas[face[1][2]][face[1][3]]
        for face in boundary_faces
    ]
    iregge = im * (
        sum(bulk_areas .* deficit_angles) +
        sum(boundary_areas .* dihedral_angles)
    )

    return ReggeActionData(
        deficit_angles=deficit_angles,
        dihedral_angles=dihedral_angles,
        bulk_areas=bulk_areas,
        boundary_areas=boundary_areas,
        iregge=iregge,
    )
end

"""
Construct the phase-fixed symbolic spinfoam action and evaluate it at the real
critical or pseudo-critical configuration.
"""
function compute_spinfoam_action(
    geom,
    regge::ReggeActionData;
    γ=nothing,
    gamma=nothing,
    bulk_sum_form::Bool=true,
)
    T = validate_geometry_precision(geom)
    gamma_value = resolve_gamma(γ, gamma, nothing)
    PrecisionUtils.validate_number_precision(gamma_value, T, "gamma")
    gamma_symbol = DefineAction.γsym()

    DefineSymbols.run_define_variables(geom)
    solve_data, _ = SolveVars.run_solver(geom)

    action_symbols, phase_solution =
        SFaction_no_phase.compute_action_no_bdry_phase(
            geom,
            solve_data,
            regge.dihedral_angles;
            γ=gamma_symbol,
            bulk_sum_form=bulk_sum_form,
        )

    action_no_phase = ActionEvaluation.eval_symbolic(action_symbols, phase_solution)
    value_dict = ActionEvaluation.build_value_dict(solve_data, gamma_symbol; γval=gamma_value)
    action = SymEngine.expand(ActionEvaluation.eval_symbolic(action_no_phase, value_dict))

    return SpinfoamActionData(
        symbols=action_symbols,
        phase_solution=phase_solution,
        action_no_phase=action_no_phase,
        value_dict=value_dict,
        action=action,
        solve_data=solve_data,
        gamma_symbol=gamma_symbol,
    )
end

"""
Re-evaluate an existing symbolic spinfoam action at a new Immirzi parameter.
The geometry, critical data, boundary phases, and symbolic action are reused.
"""
function compute_spinfoam_action(prepared::SpinfoamActionData; γ=nothing, gamma=nothing)
    T = real_type(prepared.solve_data)
    PrecisionUtils.validate_active_precision(T)
    gamma_value = resolve_gamma(γ, gamma, nothing)
    PrecisionUtils.validate_number_precision(gamma_value, T, "gamma")
    value_dict = ActionEvaluation.build_value_dict(
        prepared.solve_data,
        prepared.gamma_symbol;
        γval=gamma_value,
    )
    action = SymEngine.expand(
        ActionEvaluation.eval_symbolic(prepared.action_no_phase, value_dict),
    )

    return SpinfoamActionData(
        symbols=prepared.symbols,
        phase_solution=prepared.phase_solution,
        action_no_phase=prepared.action_no_phase,
        value_dict=value_dict,
        action=action,
        solve_data=prepared.solve_data,
        gamma_symbol=prepared.gamma_symbol,
    )
end

"""Differentiate the symbolic spinfoam action with respect to internal variables."""
function compute_eom(action::SpinfoamActionData; variables=nothing)
    T = real_type(action.solve_data)
    PrecisionUtils.validate_active_precision(T)
    vars = variables === nothing ? action.solve_data.labels_vars : variables
    return EOMs.compute_EOMs(action.action_no_phase, vars)
end

"""Evaluate and print the equations of motion at the stored real configuration."""
function check_eom(action::SpinfoamActionData; γ=nothing, gamma=nothing, eom=nothing)
    T = real_type(action.solve_data)
    PrecisionUtils.validate_active_precision(T)
    gamma_value = resolve_gamma(γ, gamma, one(T))
    PrecisionUtils.validate_number_precision(gamma_value, T, "gamma")
    dS = eom === nothing ? compute_eom(action) : eom
    EOMs.check_EOMs(dS, action.solve_data; γ=gamma_value)
    return dS
end

"""Compute and evaluate the spinfoam Hessian for the selected variables."""
function compute_hessian(
    geom,
    action::SpinfoamActionData;
    γ=nothing,
    gamma=nothing,
    variables=nothing,
    eom=nothing,
    half::Bool=true,
    eigenvalues::Bool=false,
)
    T = validate_geometry_precision(geom)
    T === real_type(action.solve_data) || error(
        "Geometry and action use different scalar types.",
    )
    gamma_value = resolve_gamma(γ, gamma, one(T))
    PrecisionUtils.validate_number_precision(gamma_value, T, "gamma")
    vars = variables === nothing ? spinfoam_variables(geom) : variables

    H_symbols = half ?
        Hessian.compute_Hessian_block_half(action.action_no_phase, vars; eom=eom) :
        Hessian.compute_Hessian_block(action.action_no_phase, vars; eom=eom)

    H_matrix = Hessian.evaluate_hessian_block(H_symbols, action.solve_data; γ=gamma_value)
    H_eigenvalues = eigenvalues ? sort(eigvals(H_matrix), by=abs, rev=true) : nothing

    return HessianData(
        variables=vars,
        symbols=H_symbols,
        matrix=H_matrix,
        eigenvalues=H_eigenvalues,
    )
end

function spinfoam_variables(geom)
    return vcat(
        geom.varias[:g_var],
        geom.varias[:z_var],
        geom.varias[:η_var],
    )
end

real_type(::SolveVars.SolveData{T}) where {T<:Real} = T

function resolve_gamma(γ, gamma, default)
    if γ !== nothing && gamma !== nothing
        error("Pass either γ or gamma, not both.")
    end

    value = γ !== nothing ? γ : gamma
    return value === nothing ? default : value
end

end
