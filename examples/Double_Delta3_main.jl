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
simplices = [[1,2,3,4,6], [1,2,3,5,6], [1,2,4,5,6],[1,2,3,4,7], [1,2,3,5,7], [1,2,4,5,7]]

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
    "-0.068000000000000005, -0.21988127663727278, -0.5316227766016838, -1.3316227766016839",
    "0, 0, 0, -3.398088489694245",
    "-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
    "0, 0, -2.942830956382712, -1.6990442448471226",
    "0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
    "-2.4696884592430974, -3.893218630529324, -1.3565336794679874, -1.9090667752920147",
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
# 5. Connect simplices + face matching + gauge fixing
# ------------------------------------------------------------
LorentzianSimplexSolver.KappaOrientation.fix_kappa_signs!(simplices, geom_base)

conn = LorentzianSimplexSolver.FourSimplexConnectivity.build_global_connectivity(simplices, geom_base)
push!(geom_base.connectivity, conn)

geom_ref    = deepcopy(geom_base)
LorentzianSimplexSolver.FaceXiMatching.run_face_xi_matching(geom_ref; sector=:ref)
LorentzianSimplexSolver.GaugeFixingSU.run_su2_su11_gauge_fix(geom_ref)

# ------------------------------------------------------------
# 6. Symbols and action (reference orientation)
# ------------------------------------------------------------
LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom_ref)
sd_ref, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom_ref)
S_ref = LorentzianSimplexSolver.DefineAction.compute_action(geom_ref)

γ = LorentzianSimplexSolver.DefineAction.γsym()
vals_ref = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd_ref, γ; γval=nothing)

S_ref_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_ref, vals_ref);
S_ref_vals = SymEngine.expand(S_ref_sym)

# ------------------------------------------------------------
# 7. EOMs computation 
# ------------------------------------------------------------
g_vars = geom_ref.varias[:g_var]
z_vars = geom_ref.varias[:z_var]
j_vars = geom_ref.varias[:j_var]
xi_vars = geom_ref.varias[:xi_var]
vars = vcat(g_vars, z_vars, xi_vars, j_vars)
dS_sym = LorentzianSimplexSolver.EOMsHessian.compute_EOMs(S_ref, sd_ref)
dS_vals = LorentzianSimplexSolver.EOMsHessian.check_EOMs(dS_sym, sd_ref; γ=1)

# ------------------------------------------------------------
# 8. Hessian matrix computation 
# ------------------------------------------------------------
Hsym = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block_half(S_ref, vars)
H_evals = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_block(Hsym, sd_ref; γ=1);