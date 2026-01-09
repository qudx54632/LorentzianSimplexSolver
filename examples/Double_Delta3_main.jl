# ============================================================
# examples/Delta3_main.jl
# ============================================================

using LinearAlgebra
using Printf
using Dates
using Symbolics
using PythonCall

# Optional: make sure we are using the package environment when running this file directly
# (safe even if already activated)
try
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
catch
end

using LorentzianSimplexSolver

# If you want sympy available (only if your package needs it at runtime)
sympy = pyimport("sympy")

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
simplices = [[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6],[1,2,3,4,7],[1,2,3,5,7],[1,3,4,5,7]]

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
geom_parity = deepcopy(geom_base)
LorentzianSimplexSolver.FaceXiMatching.run_face_xi_matching(geom_ref; sector=:ref)
LorentzianSimplexSolver.FaceXiMatching.run_face_xi_matching(geom_parity; sector=:parity)

LorentzianSimplexSolver.GaugeFixingSU.run_su2_su11_gauge_fix(geom_ref)
LorentzianSimplexSolver.GaugeFixingSU.run_su2_su11_gauge_fix(geom_parity)

# ------------------------------------------------------------
# 6a. Symbols and action (reference orientation)
# ------------------------------------------------------------
using Symbolics
@variables γ
LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom_ref)
sd_ref, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom_ref)
S_ref = LorentzianSimplexSolver.DefineAction.compute_action(geom_ref)
S_ref_fn, labels_ref = LorentzianSimplexSolver.SymbolicToJulia.build_action_function(S_ref, sd_ref)
args_ref = LorentzianSimplexSolver.SymbolicToJulia.build_argument_vector(sd_ref, labels_ref, γ)
args_ref_keep_j = LorentzianSimplexSolver.SymbolicToJulia.build_argument_vector_keep_j(sd_ref, labels_ref, γ)
S_ref_sym = expand(simplify(S_ref_fn(args_ref...)))
S_ref_sym_keep_j = expand(simplify(S_ref_fn(args_ref_keep_j...)))

# ------------------------------------------------------------
# 6b. Symbols and action (parity orientation)
# ------------------------------------------------------------
LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom_parity)
sd_parity, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom_parity)
S_parity = LorentzianSimplexSolver.DefineAction.compute_action(geom_parity)
S_parity_fn, labels_parity = LorentzianSimplexSolver.SymbolicToJulia.build_action_function(S_parity, sd_parity)
args_parity = LorentzianSimplexSolver.SymbolicToJulia.build_argument_vector(sd_parity, labels_parity, γ)
args_parity_keep_j = LorentzianSimplexSolver.SymbolicToJulia.build_argument_vector_keep_j(sd_parity, labels_parity, γ)
S_parity_sym = expand(simplify(S_parity_fn(args_parity...)))
S_parity_sym_keep_j = expand(simplify(S_ref_fn(args_parity_keep_j...)))

# ------------------------------------------------------------
# 6c. Regge action (parity orientation)
# ------------------------------------------------------------
phase = expand(simplify((S_ref_sym+S_parity_sym)//2))
S_regge_num,  S_regge_symbolics = LorentzianSimplexSolver.ReggeAction.run_Regge_action(geom_ref, γ);

orientation = LorentzianSimplexSolver.OrientationSelector.select_orientation(S_ref_sym_keep_j, S_parity_sym_keep_j, S_regge_symbolics, γ)

if orientation == :ref_neg || orientation == :parity_pos
    S_pos = expand(simplify(S_parity_sym - phase))
    S_neg = expand(simplify(S_ref_sym - phase))
else
    S_pos = expand(simplify(S_ref_sym - phase))
    S_neg = expand(simplify(S_parity_sym - phase))
end