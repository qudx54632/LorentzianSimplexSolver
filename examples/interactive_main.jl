# Interactive entry point for LorentzianSimplexSolver.

try
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
catch
end

using LorentzianSimplexSolver

const LSS = LorentzianSimplexSolver

function ask_yes_no(prompt; default::Bool=false)
    suffix = default ? "Y/n" : "y/N"
    print("$prompt ($suffix) ")
    answer = lowercase(strip(readline()))

    isempty(answer) && return default
    return answer in ("y", "yes")
end

function choose_scalartype()
    println("Choose scalar type:")
    println("  1) Float64")
    println("  2) BigFloat")
    print("> ")

    choice = strip(readline())
    choice == "1" && return Float64
    choice == "2" && return BigFloat

    error("Invalid choice. Please enter 1 or 2.")
end

function choose_geometry_type()
    println("Choose geometry type:")
    println("  1) Flat geometry: one global coordinate for each vertex")
    println("  2) Curved geometry: local coordinates for each 4-simplex")
    print("> ")

    choice = strip(readline())
    choice == "1" && return :flat
    choice == "2" && return :curved

    error("Invalid choice. Please enter 1 or 2.")
end

function parse_simplices(line::AbstractString)
    text = replace(strip(line), r"\s+" => "")
    startswith(text, "[[") && endswith(text, "]]") ||
        error("Input must look like [[1,2,3,4,5],[1,2,3,4,6]].")

    body = text[3:end-2]
    isempty(body) && error("No simplices entered.")

    return [parse_simplex(chunk) for chunk in split(body, "],[")]
end

function parse_simplex(text::AbstractString)
    vertices = parse.(Int, split(text, ","; keepempty=false))
    length(vertices) == 5 || error("Each 4-simplex must contain 5 vertices, got $(length(vertices)).")
    length(unique(vertices)) == 5 || error("Simplex vertices must be distinct: $vertices")
    return vertices
end

function read_simplices()
    println()
    println("Enter simplices, for example:")
    println("[[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6]]")
    print("> ")

    line = strip(readline())
    isempty(line) && error("No simplices entered.")
    return parse_simplices(line)
end

function read_vertex_coordinates(vertices, ::Type{T}) where {T<:Real}
    println()
    println("Enter coordinates in vertex order $(collect(vertices)).")
    println("Format: t,x,y,z")

    vertex_coords = Dict{Int, Vector{T}}()
    for v in vertices
        print("[$v] ")
        line = strip(readline())
        isempty(line) && error("Empty coordinate line for vertex $v.")

        coords = LSS.PrecisionUtils.parse_numeric_line(line, T)
        length(coords) == 4 || error("Vertex $v: expected 4 numbers, got $(length(coords)).")
        vertex_coords[v] = coords
    end

    return vertex_coords
end

function read_local_vertex_coordinates(simplices, ::Type{T}) where {T<:Real}
    println()
    println("Enter five local vertex coordinates for every 4-simplex.")
    println("Format: t,x,y,z")

    vertices_for_each_simplex = Vector{Vector{Vector{T}}}()
    for k in eachindex(simplices)
        println()
        println("4-simplex $k: $(simplices[k])")
        local_coordinates = Vector{Vector{T}}()

        for vertex in simplices[k]
            print("[$vertex] ")
            coordinates = LSS.PrecisionUtils.parse_numeric_line(
                strip(readline()), T,
            )
            length(coordinates) == 4 ||
                error("Vertex $vertex: expected 4 numbers, got $(length(coordinates)).")
            push!(local_coordinates, coordinates)
        end

        push!(vertices_for_each_simplex, local_coordinates)
    end

    return vertices_for_each_simplex
end

function read_gamma(::Type{T}) where {T<:Real}
    println()
    print("Enter gamma: ")
    values = LSS.PrecisionUtils.parse_numeric_line(strip(readline()), T)
    length(values) == 1 || error("Expected one numeric value for gamma.")
    return only(values)
end

println("========================================")
println(" Spinfoam / Regge Interactive Driver")
println("========================================")
println("Julia version: ", VERSION)
println("Package source: ", pathof(LorentzianSimplexSolver))

geometry_type = choose_geometry_type()
ScalarT = choose_scalartype()
tol = configure_precision!(ScalarT)

println()
println("Using scalar type: $ScalarT")
println("Tolerance: $tol")

simplices = read_simplices()
vertices = sort(unique(Iterators.flatten(simplices)))

println()
println("Detected $(length(simplices)) simplices with $(length(vertices)) unique vertices.")

if geometry_type == :flat
    geometry_input = read_vertex_coordinates(vertices, ScalarT)
else
    geometry_input = read_local_vertex_coordinates(simplices, ScalarT)
end
gamma_value = read_gamma(ScalarT)

println()
println("Building geometry.")
if geometry_type == :flat
    geom = construct_geometry(simplices, geometry_input; verbose=true)
else
    geom = construct_curved_geometry(geometry_input; verbose=true)
end

if ask_yes_no("Check simplex consistency?")
    check_simplex_consistency(geom)
end

if length(simplices) > 1 && ask_yes_no("Connect simplices, match faces, and gauge fix?", default=true)
    prepare_global_geometry!(
        geom,
        simplices;
        check=ask_yes_no("Run checks after face matching?"),
        verbose=true,
    )
end

println()
println("Computing Regge action.")
regge = compute_regge_action(geom, simplices, geometry_input)
display(regge.iregge)

println()
println("Computing spinfoam action.")
if geometry_type == :flat
    spinfoam = compute_spinfoam_action(geom, regge; gamma=gamma_value)
else
    spinfoam = compute_spinfoam_action(
        geom,
        regge;
        gamma=gamma_value,
        bulk_sum_form=false,
    )
end
display(spinfoam.action)

dS = nothing
if ask_yes_no("Check equations of motion?")
    dS = check_eom(spinfoam; gamma=gamma_value)
end

if geometry_type == :curved &&
   ask_yes_no("Solve the complex critical point?", default=true)
    complex_solution = solve_complex_critical_point(
        spinfoam.action_no_phase,
        spinfoam.solve_data,
        spinfoam.gamma_symbol,
        gamma_value,
    )

    println("Converged: ", complex_solution.converged)
    println("Newton iterations: ", complex_solution.iterations)
    println("max |dS|: ", maximum(abs.(complex_solution.residual)))
    println("S at the complex critical point: ", complex_solution.action)

    println("Bulk area variables:")
    for i in eachindex(complex_solution.variables)
        startswith(string(complex_solution.variables[i]), "η_") || continue
        println(
            complex_solution.variables[i],
            " = ",
            complex_solution.solution[i],
        )
    end
end

if ask_yes_no("Compute Hessian?")
    hessian = compute_hessian(
        geom,
        spinfoam;
        gamma=gamma_value,
        eom=dS,
        eigenvalues=true,
    )
    println("Hessian size: $(size(hessian.matrix))")
    println("Eigenvalues sorted by absolute value:")
    display(hessian.eigenvalues)
end

println()
println("=== Program finished ===")
