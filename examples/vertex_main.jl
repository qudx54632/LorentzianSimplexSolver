# ============================================================
# examples/Delta3_main.jl
# ============================================================

using LinearAlgebra
using Printf
using Dates
using SymEngine

# Optional: make sure we are using the package environment when running this file directly
# (safe even if already activated)
try
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
catch
end

using LorentzianSimplexSolver

# ------------------------------------------------------------
# 1. Precision choice (user-controlled)
# ------------------------------------------------------------
const ScalarT = Float64
# const ScalarT = BigFloat

if ScalarT === BigFloat
    LorentzianSimplexSolver.PrecisionUtils.set_big_precision!(prec)
    LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(sqrt(eps(BigFloat)))
else
    LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(1e-10)
end

# ------------------------------------------------------------
# 2. Read simplices
# ------------------------------------------------------------
simplices = [[1,2,3,4,5]]

ns = length(simplices)

all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)

Nverts = length(all_vertices)

# ------------------------------------------------------------
# 3. Read vertex coordinates
# ------------------------------------------------------------
vertex_coords = Dict{Int, Vector{ScalarT}}()    

coords_lines = [
    "0, 0, 0, 0",
    "0, 0, 0, 1",
    "0, 0, 1, 1",
    "0, 1, 1, 1",
    "1//2, 1, 1, 1",
]

for (v, line) in zip(all_vertices, coords_lines)
    vertex_coords[v] = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(line, ScalarT)
end

# ------------------------------------------------------------
# 4. Build geometry
# ------------------------------------------------------------
datasets = LorentzianSimplexSolver.GeometryTypes.GeometryDataset{ScalarT}[]

for (s, simplex) in enumerate(simplices)
    println("\n--- Processing simplex $s with vertices $simplex ---")
    bdypoints = [vertex_coords[v] for v in simplex]
    ds = LorentzianSimplexSolver.GeometryPipeline.run_geometry_pipeline(bdypoints)
    push!(datasets, ds)
end

geom_base = LorentzianSimplexSolver.GeometryTypes.GeometryCollection(datasets);

# ------------------------------------------------------------
# 5. both orientation construction for single 4-simplex
# ------------------------------------------------------------
geom_ref    = deepcopy(geom_base)
geom_parity = deepcopy(geom_base)

sl2c_ref = [geom_base.simplex[i].solgsl2c    for i in 1:ns]
sgndet = [geom_base.simplex[i].sgndet    for i in 1:ns]
geom_parity.simplex[1].solgsl2c = LorentzianSimplexSolver.FaceXiMatching.update_sl2ctest(sl2c_ref, sgndet)[1]

# ------------------------------------------------------------
# 6a. Symbols and action (reference orientation)
# ------------------------------------------------------------
γ = LorentzianSimplexSolver.DefineAction.γsym()

LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom_ref)
sd_ref, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom_ref)
S_ref = LorentzianSimplexSolver.DefineAction.compute_action(geom_ref)

vals_ref = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd_ref, γ; γval=nothing)
S_ref_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_ref, vals_ref);
S_ref_vals = SymEngine.expand(S_ref_sym)

# ------------------------------------------------------------
# 6b. Symbols and action (parity orientation)
# ------------------------------------------------------------
LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom_parity)
sd_parity, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom_parity)
S_parity = LorentzianSimplexSolver.DefineAction.compute_action(geom_parity)

vals_parity = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd_parity, γ; γval=nothing)
S_parity_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_parity, vals_parity);
S_parity_vals = SymEngine.expand(S_parity_sym)

# ------------------------------------------------------------
# 6c. Regge action (parity orientation)
# ------------------------------------------------------------
phase = SymEngine.expand((S_ref_sym+S_parity_sym)//2)
S_regge_num,  S_regge_symbolics = LorentzianSimplexSolver.ReggeAction.run_Regge_action(geom_ref, γ);

# ------------------------------------------------------------
# 7. EOMs computation 
# ------------------------------------------------------------
dS_sym = LorentzianSimplexSolver.EOMsHessian.compute_EOMs(S_ref, sd_ref)
dS_vals = LorentzianSimplexSolver.EOMsHessian.check_EOMs(dS_sym, sd_ref; γ=1)

# ------------------------------------------------------------
# 8. Hessian matrix computation 
# ------------------------------------------------------------
g_vars = geom_ref.varias[:g_var]
z_vars = geom_ref.varias[:z_var]
j_vars = geom_ref.varias[:j_var]
xi_vars = geom_ref.varias[:xi_var]
vars = vcat(g_vars, z_vars, xi_vars, j_vars)
Hsym = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block_half(S_ref, vars)
H_evals = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_block(Hsym, sd_ref; γ=1);