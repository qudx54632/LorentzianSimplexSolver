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