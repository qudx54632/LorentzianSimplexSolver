# LorentzianSimplexSolver

LorentzianSimplexSolver is a Julia package for constructing Lorentzian 4-simplex geometries and evaluating the Regge and spacelike EPRL spinfoam action. It starts directly from vertex coordinates and supports both globally embedded flat data and curved Regge data with independent local coordinates in each 4-simplex.

The package constructs the boundary spinors and group elements, connects neighboring 4-simplices, matches shared faces, fixes the gauge, evaluates the Regge and spinfoam actions, and computes equations of motion and Hessian blocks. For curved Regge data it can continue a real pseudo-critical configuration to a complex critical point by Newton iteration.

## Physical scope

The action implemented here is the spacelike EPRL spinfoam action. The geometry code can also construct boundary data containing timelike tetrahedra, but this does not constitute an implementation of a timelike EPRL action.

Every 4-simplex is flat internally. "Curved Regge geometry" means that the whole triangulation need not admit one global Minkowski embedding: each 4-simplex has its own five local coordinates, while shared subsimplices carry matching intrinsic geometry.

## Three kinds of points

The following distinction is important in the curved calculation.

1. **Real critical point for flat geometry.** The globally embedded geometry gives a real solution of all spinfoam critical-point equations.
2. **Real pseudo-critical point for curved geometry.** Each 4-simplex is constructed from its local coordinates and the usual face matching and gauge fixing are performed. This gives a real geometric configuration from which the complex calculation starts. For the exact action used by default, a curved pseudo-critical point is generally not a stationary point: derivatives with respect to both the group-spinor variables and the bulk area variables `η` can be nonzero.
3. **Complex critical point.** Starting from the real pseudo-critical point, the internal group, spinor, and bulk area variables are complexified. Newton's method solves all equations `dS = 0`, and the action is then evaluated at the complex solution.

By default, each bulk face contributes the exact expression `η_h * (E_h - 1)`, selected by `bulk_sum_form=true`. The alternative logarithmic expression `η_h * log(E_h)` is available through `bulk_sum_form=false`. The complex solver does not impose either convention: it symbolically differentiates the action supplied to it. The workflow and examples below use the default exact form.

## Calculation pipeline

The public workflow follows the same order as the mathematical construction:

1. `configure_precision!` selects `Float64` or `BigFloat`.
2. `construct_geometry` or `construct_curved_geometry` constructs each local Lorentzian 4-simplex and its boundary data.
3. `prepare_global_geometry!` fixes the global orientation, builds connectivity, matches shared-face data, and gauge fixes.
4. `compute_regge_action` sums the local dihedral-angle contributions at each triangle.
5. `compute_spinfoam_action` defines the symbolic EPRL action, solves the real geometric data, and fixes the boundary spinor phases using the Regge boundary angles.
6. `compute_eom` constructs the symbolic equations of motion, while `check_eom` and `compute_hessian` evaluate the equations and Hessian at the stored real configuration.
7. `solve_complex_critical_point` uses the same symbolic action, fixes the boundary data, and starts Newton iteration from the real pseudo-critical configuration.

For a single 4-simplex, step 3 is optional.

## Installation

The package is not yet registered in the Julia General registry. Install the development version directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/qudx54632/LorentzianSimplexSolver.git")
using LorentzianSimplexSolver
```

To work from a local clone, run Julia in the package directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using LorentzianSimplexSolver
```

## Interactive workflow

From the package root, run:

```bash
julia --project=. examples/interactive_main.jl
```

The program asks whether the geometry is flat or curved, selects numerical precision, reads the simplices and coordinates, and then offers the geometry checks, action, equations of motion, complex critical point, and Hessian. Input coordinates use the order `(t,x,y,z)`.

For flat geometry, enter one coordinate for each global vertex. For curved geometry, enter five local coordinates for every 4-simplex in the vertex order printed by the program. Complete flat and curved input data are collected in `examples/example-complex.txt`.

## Flat geometry from functions

```julia
using LorentzianSimplexSolver

configure_precision!(Float64)

simplices = [[1,2,3,4,5]]
vertex_coords = Dict(
    1 => [0.0, 0.0, 0.0, 0.0],
    2 => [0.0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226],
    3 => [0.0, 0.0, 0.0, -3.398088489694245],
    4 => [-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226],
    5 => [0.0, 0.0, -2.942830956382712, -1.6990442448471226],
)

geom = construct_geometry(simplices, vertex_coords)
regge = compute_regge_action(geom, simplices, vertex_coords)
spinfoam = compute_spinfoam_action(geom, regge; gamma=0.1)

dS = compute_eom(spinfoam)
check_eom(spinfoam; gamma=0.1, eom=dS)
hessian = compute_hessian(geom, spinfoam; gamma=0.1, eom=dS)
```

For a complex with several 4-simplices, call `prepare_global_geometry!(geom, simplices)` before evaluating the actions.

## Curved geometry and complex critical point

The curved input contains one list of five local coordinates for each 4-simplex. The position of a coordinate in each list follows the corresponding entry of `simplices`:

```julia
using LorentzianSimplexSolver

configure_precision!(Float64)

simplices = [
    [1,2,3,4,5],
    [1,2,3,5,6],
    [1,3,4,5,6],
]

# The numerical P1, ..., P6_curved values are in examples/example-complex.txt.
vertices_for_each_simplex = [
    [P1, P2, P3, P4, P5],
    [P1, P2, P3, P5, P6_curved],
    [P1, P3, P4, P5, P6],
]

geom = construct_curved_geometry(vertices_for_each_simplex)
prepare_global_geometry!(geom, simplices)

regge = compute_regge_action(geom, simplices, vertices_for_each_simplex)
spinfoam = compute_spinfoam_action(
    geom,
    regge;
    gamma=1.0,
)

# For the exact action these residuals are generally nonzero.
dS_pseudo = compute_eom(spinfoam)
check_eom(spinfoam; gamma=1.0, eom=dS_pseudo)

# Full complex solution and S evaluated at that solution.
solution = solve_complex_critical_point(
    spinfoam.action_no_phase,
    spinfoam.solve_data,
    spinfoam.gamma_symbol,
    1.0;
    tol=1e-11,
)

solution.converged
maximum(abs.(solution.residual))
solution.action
```

`prepare_complex_critical_point` is also public when the symbolic equations, Hessian, and initial point are needed separately. Normally `solve_complex_critical_point` is the only function required.

## Precision

Precision must be configured before coordinates are created or parsed:

```julia
configure_precision!(Float64)
configure_precision!(BigFloat; precision=100)
```

The geometry and real-point checks use tolerances `1e-10` for `Float64` and `1e-30` for `BigFloat`. The complex Newton tolerance is set independently by the `tol` keyword.

## Repository layout

```text
examples/
├── interactive_main.jl      interactive user workflow
└── example-complex.txt      flat and curved coordinate data
notebooks/
└── Lorentzian_simplices_main.ipynb  research calculation notebook
test/
└── runtests.jl              automatic local regression tests
src/
├── algebra/                 spin algebra and Lorentz-group maps
├── geometry/                simplex geometry, normals, areas, and volumes
├── pipeline/                connectivity, orientation, face matching, gauge fixing
├── action/                  Regge/spinfoam actions, EOMs, and Hessians
├── complex_critical_point/  curved input and complex Newton solver
├── workflow/                public high-level workflow
└── LorentzianSimplexSolver.jl
```

The high-level functions are kept in `workflow/InteractiveWorkflow.jl`; the remaining folders implement the mathematical steps called by that workflow.

## Tests

Run the automated Delta3 regression test with:

```julia
using Pkg
Pkg.test("LorentzianSimplexSolver")
```

The test checks flat and curved geometry construction, shared-face matching, the expected nonzero exact-action residuals at the real curved starting point, and convergence of all equations of motion at the complex critical point.

## Research use

This package is intended for research in Lorentzian Regge calculus, spinfoam asymptotics, and effective-action calculations. It is not a general-purpose geometry library.
