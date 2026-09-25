using Test
using LorentzianSimplexSolver

const LSS = LorentzianSimplexSolver

function delta3_curved_data()
    simplices = [[1,2,3,4,5], [1,2,3,5,6], [1,3,4,5,6]]

    P1 = [0.0, 0.0, 0.0, 0.0]
    P2 = [0.0, -2.774527633525212, -0.9809436521275707, -1.6990442448471228]
    P3 = [0.0, 0.0, 0.0, -3.3980884896942456]
    P4 = [-0.24028114141347542, -0.693631908381303, -0.9809436521275707, -1.6990442448471228]
    P5 = [0.0, 0.0, -2.942830956382712, -2.942830956382712]
    P6 = [0.8981363267013139, 2.743722483186861, -0.9809436519009252, -1.6990442448710555]
    P6_curved = [0.9036268647043613, 2.74552467008038, -0.9809436519009252, -1.6990442448710555]

    vertices = [
        [P1, P2, P3, P4, P5],
        [P1, P2, P3, P5, P6_curved],
        [P1, P3, P4, P5, P6],
    ]

    return simplices, vertices
end

function evaluated_value(expression, action)
    value = LSS.ActionEvaluation.eval_symbolic(expression, action.value_dict)
    return LSS.ActionEvaluation.to_complex_T(value, Float64)
end

@testset verbose=true "LorentzianSimplexSolver physical checks" begin
    configure_precision!(Float64)
    simplices, vertices = delta3_curved_data()

    flat_simplices = [simplices[1]]
    flat_coordinates = Dict(flat_simplices[1] .=> vertices[1])
    flat_geom = construct_geometry(flat_simplices, flat_coordinates)
    flat_regge = compute_regge_action(flat_geom, flat_simplices, flat_coordinates)

    @testset "1. Flat 4-simplex and Regge action" begin
        @test length(flat_geom.simplex) == 1
        @test isfinite(abs(flat_regge.iregge))
    end

    geom = construct_curved_geometry(vertices)
    prepare_global_geometry!(geom, simplices; check=false, gauge_fix=false)

    bivectors = [simplex.bdybivec55 for simplex in geom.simplex]
    sl2c = [simplex.solgsl2c for simplex in geom.simplex]
    kappa = [simplex.kappa for simplex in geom.simplex]
    areas = [simplex.areas for simplex in geom.simplex]
    shared_tetrahedra = geom.connectivity[1]["sharedTetsPos"]

    face_ok, face_residual = LSS.FaceMatchingChecks.check_face_matching_bivec(
        bivectors, shared_tetrahedra,
    )
    transport_ok, transport_residual = LSS.FaceMatchingChecks.check_parallel_transport(
        sl2c, bivectors,
    )
    closure_ok, closure_residual = LSS.FaceMatchingChecks.check_closure(
        bivectors, kappa, areas,
    )

    @testset "2. Curved Delta3: face matching and geometric constraints" begin
        @test length(geom.simplex) == 3
        @test length(geom.connectivity) == 1
        @test face_ok
        @test transport_ok
        @test closure_ok
    end

    LSS.GaugeFixingSU.run_su2_su11_gauge_fix(geom)

    regge = compute_regge_action(geom, simplices, vertices)
    spinfoam = compute_spinfoam_action(
        geom,
        regge;
        gamma=1.0,
        bulk_sum_form=false,
    )
    pseudo_action = evaluated_value(spinfoam.action, spinfoam)

    dS = compute_eom(spinfoam)
    non_eta_residuals = [
        abs(evaluated_value(expression, spinfoam))
        for (variable, expression) in dS
        if !startswith(string(variable), "η_")
    ]
    eta_residuals = [
        abs(evaluated_value(expression, spinfoam))
        for (variable, expression) in dS
        if startswith(string(variable), "η_")
    ]
    max_non_eta_residual = maximum(non_eta_residuals)
    max_eta_residual = maximum(eta_residuals)

    @testset "3. Real pseudo-critical point" begin
        @test isfinite(abs(regge.iregge))
        @test isfinite(abs(pseudo_action))
        @test max_non_eta_residual < 1e-10
        @test max_eta_residual > 1e-10
    end

    solution = solve_complex_critical_point(
        spinfoam.action_no_phase,
        spinfoam.solve_data,
        spinfoam.gamma_symbol,
        1.0;
        tol=1e-11,
        verbose=false,
    )
    max_complex_residual = maximum(abs.(solution.residual))

    @testset "4. Complex critical point" begin
        @test solution.converged
        @test max_complex_residual < 1e-11
        @test isfinite(abs(solution.action))
    end

    println()
    println("Physical results for the curved Delta3 example:")
    println("  Face-matching residual              = $face_residual")
    println("  SL(2,C) parallel-transport residual = $transport_residual")
    println("  Closure residual                    = $closure_residual")
    println("  Regge action i S_Regge              = $(regge.iregge)")
    println("  S at the real pseudo-critical point = $pseudo_action")
    println("  S at the complex critical point     = $(solution.action)")
    println("  Max group/spinor EOM residual       = $max_non_eta_residual")
    println("  Max bulk-area EOM |dS/dη|           = $max_eta_residual")
    println("  Complex-point EOM max |dS|          = $max_complex_residual")
    println()
    println("Interpretation:")
    println("  The real curved configuration solves the group and spinor equations.")
    println("  Its bulk-area equation is nonzero because the bulk deficit angle is nonzero.")
    println("  After complex Newton iteration, all equations of motion are satisfied.")
end

# Keep interactive output focused on the physical results above.
nothing
