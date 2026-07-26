# LorentzianSimplexSolver

LorentzianSimplexSolver is a Julia package for constructing Lorentzian 4-simplex geometries and evaluating the Regge and EPRL spinfoam data used in the effective-action calculation.

The public workflow starts from a list of 4-simplices and vertex coordinates, then builds the geometry, matches shared faces, evaluates the Regge action, evaluates the spinfoam action at the critical point, and optionally computes equations of motion and Hessian blocks. Both `Float64` and `BigFloat` arithmetic are supported.

## Current physical scope

The action currently implemented in this package is the spacelike EPRL
spinfoam action. The geometry pipeline is more general: boundary data and
critical-point configurations can be constructed for complexes containing
timelike tetrahedra. Support for timelike tetrahedra here refers to the
geometry, boundary data, and critical-point construction; it should not be
interpreted as an implementation of the timelike EPRL action.

## Installation (development version)

Clone the repository and activate it as a Julia project:

    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()

Then load the package:

    using LorentzianSimplexSolver

Note:
This package is currently intended for research use and is not yet registered in the Julia General registry.

## Quick start (interactive workflow)

An interactive driver is provided under `examples/`.

    include("examples/interactive_main.jl")

From the package root:

    julia --project=. examples/interactive_main.jl

The interactive script will guide you through:

1. Choosing numerical precision (Float64 or BigFloat)
2. Entering simplices
3. Entering vertex coordinates
4. Building boundary geometry
5. Optional consistency checks
6. Face matching and gauge fixing
7. Action evaluation, equations of motion, and Hessian computation

## Quick start (function workflow)

The same workflow can be used directly from Julia code:

```julia
using LorentzianSimplexSolver

configure_precision!(Float64)

simplices = [[1,2,3,4,6], [1,2,3,5,6], [1,2,4,5,6]]
vertex_coords = Dict(
    1 => [0.0, 0.0, 0.0, 0.0],
    2 => [0.0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226],
    3 => [0.0, 0.0, 0.0, -3.398088489694245],
    4 => [-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226],
    5 => [0.0, 0.0, -2.942830956382712, -1.6990442448471226],
    6 => [0.8981365593438019, 2.7437225604241213, -0.9809436521275707, -1.6990442448471226],
)

geom = construct_geometry(simplices, vertex_coords)
prepare_global_geometry!(geom, simplices)

regge = compute_regge_action(geom, simplices, vertex_coords)
spinfoam = compute_spinfoam_action(geom, regge; gamma=0.1)

# Reuse the symbolic action when only gamma changes.
spinfoam_gamma2 = compute_spinfoam_action(spinfoam; gamma=0.2)

dS = compute_eom(spinfoam)
check_eom(spinfoam; gamma=0.1, eom=dS)
hessian = compute_hessian(geom, spinfoam; gamma=0.1, eom=dS)
```

The optional `eom=dS` argument reuses the first derivatives when checking the
equations of motion and constructing the Hessian.

Precision must be configured before numeric coordinates are created or parsed.
For each configured workflow, the whole package uses one tolerance defined in
`PrecisionUtils`: `1e-10` for Float64 and `1e-12` for BigFloat.

For a single 4-simplex, `prepare_global_geometry!` is optional.

## Package structure

```text
LorentzianSimplexSolver/
├── examples/
│   ├── interactive_main.jl
│   ├── example-complex.txt
├── src/
│   ├── LorentzianSimplexSolver.jl
│   ├── workflow/
│   │   └── InteractiveWorkflow.jl
│   ├── action/
│   │   ├── CriticalPoints.jl
│   │   ├── DefineAction.jl
│   │   ├── DefineSymbols.jl
│   │   ├── EOMs.jl
│   │   ├── Hessian.jl
│   │   ├── ReggeAction.jl
│   │   ├── SolveVars.jl
│   │   └── ActionEvaluation.jl
│   ├── algebra/
│   │   ├── LorentzGroup.jl
│   │   ├── SpinAlgebra.jl
│   │   ├── Su2Su11FromBivector.jl
│   │   └── XiFromSU.jl
│   ├── geometry/
│   │   ├── DihedralAngles.jl
│   │   ├── FaceNormals3D.jl
│   │   ├── GeometryTypes.jl
│   │   ├── KappaFromNormals.jl
│   │   ├── SimplexGeometry.jl
│   │   ├── TetraNormals.jl
│   │   ├── ThreeDTetra.jl
│   │   └── Volume.jl
│   ├── pipeline/
│   │   ├── FaceMatchingChecks.jl
│   │   ├── FaceXiMatching.jl
│   │   ├── FourSimplexConnectivity.jl
│   │   ├── GaugeFixing.jl
│   │   ├── GeometryConsistency.jl
│   │   ├── GeometryPipeline.jl
│   │   └── KappaOrientation.jl
│   └── utils/
│       └── PrecisionUtils.jl
├── test/
│   ├── interactive_driver.jl
│   └── Lorentzian_simplices_main.ipynb
├── Project.toml
└── README.md
```

## Main components

- workflow/
  Public functions for the paper-level numerical workflow

- algebra/
  Spin algebra, Lorentz group elements, bivector mappings

- geometry/
  Simplex geometry, normals, areas, volumes

- pipeline/
  Geometry construction, consistency checks, face matching, gauge fixing

- action/
  Symbolic action, critical points, equations of motion, Hessians

## Dependencies

Key dependencies include:

- LinearAlgebra
- GenericLinearAlgebra
- Combinatorics
- SymEngine
- GenericSchur

All dependencies are declared explicitly in Project.toml.

## Intended audience

This package is intended for:

- Researchers in Loop Quantum Gravity
- Spinfoam and Regge calculus studies
- Numerical investigations of spinfoam asymptotics
- Advanced graduate-level research projects

It is not designed as a general-purpose geometry library.
