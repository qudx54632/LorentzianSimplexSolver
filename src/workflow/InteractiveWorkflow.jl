module InteractiveWorkflow

using LinearAlgebra
using SymEngine

using ..PrecisionUtils
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
using ..EOMsHessian
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

function configure_precision!(::Type{T}; precision::Integer=100, tolerance=nothing) where {T<:Real}
    tol = tolerance === nothing ? default_tolerance(T) : tolerance

    if T === BigFloat
        setprecision(BigFloat, precision)
        PrecisionUtils.set_big_precision!(precision; tol=tol)
    else
        PrecisionUtils.set_tolerance!(tol)
    end

    return tol
end

default_tolerance(::Type{Float64}) = 1e-8
default_tolerance(::Type{BigFloat}) = BigFloat(1e-20)
default_tolerance(::Type{T}) where {T<:Real} = sqrt(eps(T))

function construct_geometry(simplices, vertex_coords; verbose::Bool=false)
    T = eltype(first(values(vertex_coords)))
    datasets = GeometryTypes.GeometryDataset{T}[]

    for (idx, simplex) in enumerate(simplices)
        verbose && println("--- Processing simplex $idx with vertices $simplex ---")
        bdypoints = [vertex_coords[v] for v in simplex]
        push!(datasets, GeometryPipeline.run_geometry_pipeline(bdypoints))
    end

    return GeometryTypes.GeometryCollection(datasets)
end

function check_simplex_consistency(geom)
    for (idx, simplex) in enumerate(geom.simplex)
        println("--- Checking simplex $idx ---")
        GeometryConsistency.check_sl2c_parallel_transport(simplex.solgsl2c, simplex.bdybivec55)
        GeometryConsistency.check_so13_parallel_transport(simplex.solgso13, simplex.bdybivec4d55)
        GeometryConsistency.check_closure_bivectors(simplex.kappa, simplex.areas, simplex.bdybivec55)
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
    gamma_value = resolve_gamma(γ, gamma, nothing)
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

compute_eom(action::SpinfoamActionData) =
    EOMsHessian.compute_EOMs(action.action_no_phase, action.solve_data)

function check_eom(action::SpinfoamActionData; γ=nothing, gamma=nothing)
    gamma_value = resolve_gamma(γ, gamma, one(real_type(action.solve_data)))
    dS = compute_eom(action)
    EOMsHessian.check_EOMs(dS, action.solve_data; γ=gamma_value)
    return dS
end

function compute_hessian(
    geom,
    action::SpinfoamActionData;
    γ=nothing,
    gamma=nothing,
    variables=nothing,
    half::Bool=true,
    eigenvalues::Bool=false,
)
    gamma_value = resolve_gamma(γ, gamma, one(real_type(action.solve_data)))
    vars = variables === nothing ? spinfoam_variables(geom) : variables

    H_symbols = half ?
        EOMsHessian.compute_Hessian_block_half(action.action_no_phase, vars) :
        EOMsHessian.compute_Hessian_block(action.action_no_phase, vars)

    H_matrix = EOMsHessian.evaluate_hessian_block(H_symbols, action.solve_data; γ=gamma_value)
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
