# LorentzianSimplexSolver

LorentzianSimplexSolver is a Julia package for constructing, analyzing, and matching boundary geometries of 4-simplices in covariant Loop Quantum Gravity (LQG) and Lorentzian spinfoam models.

It provides a complete pipeline from discrete boundary data (vertex coordinates or simplices) to:
- construction of Lorentzian 4-simplex boundary geometry
- boundary bivectors, face normals, areas, and dihedral angles
- SL(2,ℂ) and SO(1,3) parallel-transport consistency checks
- closure and orientation consistency of bivectors
- global face matching across multiple simplices
- automatic κ-orientation fixing
- construction of parity-related geometries
- SU(2) and SU(1,1) gauge fixing
- symbolic spinfoam action construction
- separation of parity-even and parity-odd critical points
- extraction of the Regge action and common phase
- symbolic equations of motion
- symbolic and numerical Hessian matrices

The package is designed for research and numerical experimentation in spinfoam asymptotics, Regge geometry, and related quantum gravity models.

## Features

- Geometry construction for individual Lorentzian 4-simplices
- Boundary bivectors, face normals, areas, and dihedral angles
- SL(2,C) and SO(1,3) parallel-transport consistency checks
- Closure and orientation consistency of boundary bivectors
- Global face matching across multiple simplices
- Automatic κ-orientation fixing
- Construction of parity-related geometries
- SU(2) and SU(1,1) gauge fixing
- Symbolic spinfoam action construction using Symbolics.jl
- Separation of parity-even and parity-odd critical points
- Extraction of the Regge action and common phase
- Symbolic equations of motion
- Symbolic and numerical Hessian matrices
- Automatic generation of executable Julia functions for actions, gradients, and Hessians
- Supports both Float64 and arbitrary precision (BigFloat)

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

An interactive driver is provided under examples/.

    include("test/interactive_driver.jl")

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

## Package structure

```text
LorentzianSimplexSolver/
├── examples/
│   ├── Delta3_main.jl
│   ├── Double_Delta3_main.jl
│   ├── example-complex.txt
│   ├── interactive_main.jl
│   ├── Lorentzian_simplices_main.ipynb
│   └── vertex_main.jl
├── src/
│   ├── LorentzianSimplexSolver.jl
│   ├── action/
│   │   ├── CriticalPoints.jl
│   │   ├── DefineAction.jl
│   │   ├── DefineSymbols.jl
│   │   ├── EOMsHessian.jl
│   │   ├── ReggeAction.jl
│   │   ├── SolveVars.jl
│   │   └── SymbolicToJulia.jl
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
│   │   ├── ThreeDtetra.jl
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
├── test/
│   ├── interactive_driver.jl
│   └── runtests.jl
├── Project.toml
└── README.md
```

## Main components

- utils/
  Precision control, parsing, and symbolic variable definitions

- algebra/
  Spin algebra, Lorentz group elements, bivector mappings

- geometry/
  Simplex geometry, normals, areas, volumes

- pipeline/
  Geometry construction, consistency checks, face matching, gauge fixing

- action/
  Symbolic action, critical points, equations of motion, Hessians

## Precision control

The package supports user-controlled precision:

- Float64 for fast numerical experiments
- BigFloat for high-precision asymptotic analysis

Precision is typically selected at runtime in interactive workflows.

## Dependencies

Key dependencies include:

- LinearAlgebra
- Symbolics
- GenericLinearAlgebra
- Combinatorics
- PythonCall (optional, for symbolic backends)

All dependencies are declared explicitly in Project.toml.

## Intended audience

This package is intended for:

- Researchers in Loop Quantum Gravity
- Spinfoam and Regge calculus studies
- Numerical investigations of spinfoam asymptotics
- Advanced graduate-level research projects

It is not designed as a general-purpose geometry library.
