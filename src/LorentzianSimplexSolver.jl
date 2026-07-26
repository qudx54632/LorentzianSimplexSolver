module LorentzianSimplexSolver

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
include("geometry/Volume.jl")                      # volumes

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
include("action/SolveVars.jl")                     # solve critical equations
include("action/ActionEvaluation.jl")             # evaluate action at critical points      
include("action/EOMs.jl")                           # equations of motion
include("action/Hessian.jl")                       # Hessian matrix
include("action/ReggeAction.jl")                    # Regge action
include("action/DefineSFAction_no_phase.jl")        # boundary phase from dihedral angles
include("workflow/InteractiveWorkflow.jl")          # paper-friendly workflow API

# ---------------- public API ----------------
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

using .InteractiveWorkflow:
    configure_precision!,
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

end
