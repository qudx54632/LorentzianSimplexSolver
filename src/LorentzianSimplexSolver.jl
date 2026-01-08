module LorentzianSimplexSolver

using Symbolics
using LinearAlgebra
using Combinatorics

# ============================================================
# LorentzianSimplexSolver
# ============================================================

# ---------------- utilities ----------------
include("utils/PrecisionUtils.jl")                 # precision control, tolerances

# ---------------- algebra ----------------
include("algebra/SpinAlgebra.jl")                  # su(2), sl(2,C), generators

# ---------------- basic geometry ----------------
include("geometry/SimplexGeometry.jl")             # simplex geometry from points
include("geometry/TetraNormals.jl")                # tetrahedron normals
include("geometry/DihedralAngles.jl")              # dihedral angles
include("algebra/LorentzGroup.jl")                 # SO(1,3), SL(2,C) actions
include("geometry/ThreeDTetra.jl")                 # intrinsic 3D tetra geometry
include("geometry/volume.jl")                      # volumes

# ---------------- bivectors and group data ----------------
include("algebra/Su2Su11FromBivector.jl")           # SU(2)/SU(1,1) from bivectors
include("algebra/XiFromSU.jl")                      # xi variables from SU data
include("geometry/FaceNormals3D.jl")               # 3D face normals
include("geometry/KappaFromNormals.jl")            # kappa signs

# ---------------- data containers ----------------
include("geometry/GeometryTypes.jl")               # GeometryDataset, GeometryCollection

# ---------------- geometry pipeline ----------------
include("pipeline/GeometryPipeline.jl")            # main geometry pipeline
include("pipeline/GeometryConsistency.jl")         # consistency checks
include("pipeline/KappaOrientation.jl")            # kappa sign fixing
include("pipeline/FourSimplexConnectivity.jl")     # simplex connectivity
include("pipeline/FaceXiMatching.jl")              # xi matching
include("pipeline/FaceMatchingChecks.jl")          # final face checks
include("pipeline/GaugeFixing.jl")                 # gauge fixing

# ---------------- action and critical points ----------------
include("action/CriticalPoints.jl")                # critical point data
include("action/DefineSymbols.jl")                  # Symbolics variables
include("action/DefineAction.jl")                   # spinfoam action
include("action/SolveVars.jl")                      # solve critical equations
include("action/SymbolicToJulia.jl")                # Symbolics → Julia
include("action/EOMsHessian.jl")                    # EOMs and Hessian
include("action/ReggeAction.jl")                    # Regge action
include("utils/OrientationSelector.jl")             # Select the orientation of action at critical points

# ---------------- public API ----------------
export
    GeometryDataset,
    GeometryCollection,
    run_geometry_pipeline,
    fix_kappa_signs!,
    run_face_xi_matching,
    run_define_variables,
    compute_action,
    run_solver,
    compute_EOMs,
    compute_Hessian,
    compute_bdy_critical_data

end