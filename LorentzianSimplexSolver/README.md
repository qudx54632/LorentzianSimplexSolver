SpinfoamGeometry

SpinfoamGeometry is a Julia package for constructing, analyzing, and matching boundary geometries of 4-simplices in covariant Loop Quantum Gravity (LQG) and spinfoam models.

It provides a complete pipeline from discrete boundary data (vertex coordinates or simplices) to:
	•	bivectors and normals
	•	parallel transport consistency checks
	•	face matching and gauge fixing
	•	symbolic action construction
	•	equations of motion and Hessians

The package is designed for research and numerical experimentation in spinfoam asymptotics, Regge geometry, and related quantum gravity models.

Features
	•	Geometry construction for individual 4-simplices
	•	Boundary bivectors, face normals, and dihedral angles
	•	SL(2,C) and SO(1,3) parallel transport consistency checks
	•	Global face matching across multiple simplices
	•	Automatic κ-orientation fixing
	•	SU(2) and SU(1,1) gauge fixing
	•	Symbolic action construction using Symbolics.jl
	•	Automatic generation of:
	•	equations of motion
	•	Hessian matrices
	•	Supports both Float64 and arbitrary precision (BigFloat)

Installation (development version)

Clone the repository and activate it as a Julia project:
using Pkg
Pkg.activate(".")
Pkg.instantiate()

Then load the package:
using SpinfoamGeometry

Note
This package is currently intended for research use and is not yet registered in the General registry.

Quick start (interactive workflow)
An interactive driver is provided under examples/.

From the package root:
julia --project=. examples/interactive_main.jl

The interactive script will guide you through:
	1.	Choosing numerical precision (Float64 or BigFloat)
	2.	Entering simplices
	3.	Entering vertex coordinates
	4.	Building boundary geometry
	5.	Optional consistency checks
	6.	Face matching and gauge fixing
	7.	Action evaluation, equations of motion, and Hessian computation

Package structure
SpinfoamGeometry/
├── src/
│   ├── SpinfoamGeometry.jl
│   ├── utils/
│   ├── algebra/
│   ├── geometry/
│   ├── pipeline/
│   └── action/
├── examples/
│   └── interactive_main.jl
├── test/
│   └── runtests.jl
├── Project.toml
└── README.md

Main components
	•	utils/
Precision control, parsing, and symbolic variable definitions
	•	algebra/
Spin algebra, Lorentz group elements, bivector mappings
	•	geometry/
Simplex geometry, normals, areas, volumes
	•	pipeline/
Geometry construction, consistency checks, face matching, gauge fixing
	•	action/
Symbolic action, critical points, equations of motion, Hessians

Precision control
The package supports user-controlled precision:
	•	Float64 for fast numerical experiments
	•	BigFloat for high-precision asymptotic analysis

Precision is typically selected at runtime in interactive workflows.

Dependencies

Key dependencies include:
	•	LinearAlgebra
	•	Symbolics
	•	GenericLinearAlgebra
	•	Combinatorics
	•	PythonCall (optional, for symbolic backends)

All dependencies are declared explicitly in Project.toml.

Intended audience
This package is intended for:
	•	Researchers in Loop Quantum Gravity
	•	Spinfoam and Regge calculus studies
	•	Numerical investigations of spinfoam asymptotics
	•	Advanced graduate-level research projects
It is not designed as a general-purpose geometry library.