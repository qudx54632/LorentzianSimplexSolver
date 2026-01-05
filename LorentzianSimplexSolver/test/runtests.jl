using Test
using SpinfoamGeometry

@testset "SpinfoamGeometry basic loading" begin
    @test isdefined(SpinfoamGeometry, :GeometryDataset)
    @test isdefined(SpinfoamGeometry, :GeometryCollection)
    @test isdefined(SpinfoamGeometry, :run_geometry_pipeline)
end

@testset "Precision utilities" begin
    # Float64 tolerance
    SpinfoamGeometry.PrecisionUtils.set_tolerance!(1e-10)
    tol = SpinfoamGeometry.PrecisionUtils.get_tolerance()
    @test tol ≈ 1e-10
end

@testset "Minimal single-simplex geometry pipeline" begin
    # A single 4-simplex in flat Minkowski space
    bdypoints = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    ds = run_geometry_pipeline(bdypoints)

    @test ds isa SpinfoamGeometry.GeometryTypes.GeometryDataset
    @test !isempty(ds.areas)
    @test length(ds.areas) == length(ds.kappa)
end

@testset "Action construction does not error" begin
    bdypoints = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    ds = run_geometry_pipeline(bdypoints)
    geom = SpinfoamGeometry.GeometryTypes.GeometryCollection([ds])

    # Should not throw
    @test begin
        S = compute_action(geom)
        true
    end
end