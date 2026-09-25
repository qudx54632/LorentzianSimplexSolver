module ComplexCriticalPoints

using LinearAlgebra
using SymEngine

using ..GeometryTypes: GeometryDataset, GeometryCollection
using ..GeometryPipeline: run_geometry_pipeline
using ..ActionEvaluation: eval_symbolic, to_complex_T
using ..EOMs: compute_EOMs
using ..Hessian: compute_Hessian_block

export construct_curved_geometry,
       prepare_complex_critical_point,
       solve_complex_critical_point,
       ComplexCriticalPointResult

include("CurvedReggeInput.jl")
include("PrepareComplexCriticalPoint.jl")
include("SolveComplexCriticalPoint.jl")

end
