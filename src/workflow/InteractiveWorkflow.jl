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

Base.@kwdef struct ReggeActionData
    deficit_angles::Any
    dihedral_angles::Any
    bulk_areas::Any
    boundary_areas::Any
    iregge::Any
end

Base.@kwdef struct SpinfoamActionData
    symbols::Any
    phase_solution::Any
    action_no_phase::Any
    value_dict::Any
    action::Any
    solve_data::Any
    gamma_symbol::Basic
end

Base.@kwdef struct HessianData
    variables::Any
    symbols::Any
    matrix::Any
    eigenvalues::Any = nothing
end

configure_precision!(::Type{T}; precision::Integer=100) where {T<:AbstractFloat} =
    PrecisionUtils.configure_precision!(T; precision=precision)

function validate_geometry_precision(geom::GeometryTypes.GeometryCollection{T}) where {T<:AbstractFloat}
    PrecisionUtils.validate_active_precision(T)
    return T
end

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

function compute_spinfoam_action(geom, regge::ReggeActionData; γ=nothing, gamma=nothing)
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

function compute_eom(action::SpinfoamActionData; variables=nothing)
    T = real_type(action.solve_data)
    PrecisionUtils.validate_active_precision(T)
    vars = variables === nothing ? action.solve_data.labels_vars : variables
    return EOMs.compute_EOMs(action.action_no_phase, vars)
end

function check_eom(action::SpinfoamActionData; γ=nothing, gamma=nothing, eom=nothing)
    T = real_type(action.solve_data)
    PrecisionUtils.validate_active_precision(T)
    gamma_value = resolve_gamma(γ, gamma, one(T))
    PrecisionUtils.validate_number_precision(gamma_value, T, "gamma")
    dS = eom === nothing ? compute_eom(action) : eom
    EOMs.check_EOMs(dS, action.solve_data; γ=gamma_value)
    return dS
end

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
