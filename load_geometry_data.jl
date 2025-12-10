module LoadGeometryData

using LinearAlgebra
import Base: im

# ============================================================
# TOKENIZER
# ============================================================
function extract_tokens(s::AbstractString)
    parts = split(strip(s))
    tokens = String[]
    i = 1
    while i <= length(parts)
        if i + 2 <= length(parts) &&
           (parts[i+1] == "+" || parts[i+1] == "-") &&
           endswith(parts[i+2], "im")

            push!(tokens, string(parts[i], parts[i+1], parts[i+2]))
            i += 3
        else
            push!(tokens, String(parts[i]))
            i += 1
        end
    end
    return tokens
end

# ============================================================
# NUMBER PARSER
# ============================================================
function parse_number(tok::AbstractString)
    t = replace(strip(String(tok)), " " => "")

    if t == "+"
        return 1.0
    elseif t == "-"
        return -1.0
    end

    # a+bi
    m = match(r"^([+-]?\d*\.?\d+)([+-]\d*\.?\d+)im$", t)
    if m !== nothing
        re  = parse(Float64, m.captures[1])
        imv = parse(Float64, m.captures[2])
        return complex(re, imv)
    end

    # pure imaginary
    if endswith(t, "im")
        coeff = t[1:end-2]
        return parse(Float64, coeff) * im
    end

    return parse(Float64, t)
end

# ============================================================
# MATRIX BUILDER
# ============================================================
function make_matrix(rows::Vector{Vector{ComplexF64}})
    nrows = length(rows)
    ncols = length(rows[1])
    M = Matrix{ComplexF64}(undef, nrows, ncols)
    for i in 1:nrows
        M[i, :] = ComplexF64.(rows[i])
    end
    return M
end

# ============================================================
# 1. LOAD LIST OF MATRICES
# ============================================================
function load_matrix_list(filename::String)
    matrices = Matrix{ComplexF64}[]
    rows = Vector{Vector{ComplexF64}}()

    open(filename, "r") do io
        for line in eachline(io)
            s = strip(line)

            if isempty(s) || startswith(s, "#")
                if !isempty(rows)
                    push!(matrices, make_matrix(rows))
                    empty!(rows)
                end
                continue
            end

            vals = parse_number.(extract_tokens(s))
            push!(rows, vals)
        end
    end

    if !isempty(rows)
        push!(matrices, make_matrix(rows))
    end

    return matrices
end

# ============================================================
# 2. LOAD BLOCK MATRIX LIST
# ============================================================
function load_block_matrix_list(filename::String)
    blocks = Vector{Vector{Matrix{ComplexF64}}}()
    current_block = Vector{Matrix{ComplexF64}}()
    current_rows = Vector{Vector{ComplexF64}}()

    open(filename, "r") do io
        for line in eachline(io)
            s = strip(line)

            if startswith(s, "# Tetrahedron")
                if !isempty(current_rows)
                    push!(current_block, make_matrix(current_rows))
                    empty!(current_rows)
                end
                if !isempty(current_block)
                    push!(blocks, copy(current_block))
                    empty!(current_block)
                end
                continue
            end

            if startswith(s, "## Face")
                if !isempty(current_rows)
                    push!(current_block, make_matrix(current_rows))
                    empty!(current_rows)
                end
                continue
            end

            if !isempty(s)
                vals = parse_number.(extract_tokens(s))
                push!(current_rows, vals)
            end
        end
    end

    if !isempty(current_rows)
        push!(current_block, make_matrix(current_rows))
    end
    if !isempty(current_block)
        push!(blocks, current_block)
    end

    return blocks
end

# ============================================================
# 3. LOAD VECTOR LIST (ALWAYS REAL)
# ============================================================
function load_vector_list(filename::String)
    vecs = Vector{Vector{Float64}}()

    open(filename, "r") do io
        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            startswith(s, "#") && continue

            parts = split(s)
            vals = parse.(Float64, parts[2:end])
            push!(vecs, vals)
        end
    end

    return vecs
end

# ============================================================
# 4. LOAD SCALAR LIST (Float64)
# ============================================================
function load_scalar_list(filename::String)
    vals = Float64[]

    open(filename, "r") do io
        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            startswith(s, "#") && continue

            parts = split(s)
            push!(vals, parse(Float64, parts[end]))
        end
    end

    return vals
end

# ============================================================
# 5. LOAD NESTED COMPLEX VECTORS (bdyxi)
# ============================================================
function load_nested_vectors(filename::String)
    tets = Vector{Vector{Vector{Vector{ComplexF64}}}}()
    current_tet = Vector{Vector{Vector{ComplexF64}}}()
    current_face = Vector{Vector{ComplexF64}}()

    open(filename, "r") do io
        for line in eachline(io)
            s = strip(line)

            if startswith(s, "# Tetrahedron")
                if !isempty(current_face)
                    push!(current_tet, current_face)
                    current_face = Vector{Vector{ComplexF64}}()
                end
                if !isempty(current_tet)
                    push!(tets, current_tet)
                    current_tet = Vector{Vector{Vector{ComplexF64}}}()
                end
                continue
            end

            if startswith(s, "## Faces")
                if !isempty(current_face)
                    push!(current_tet, current_face)
                    current_face = Vector{Vector{ComplexF64}}()
                end
                continue
            end

            if isempty(s)
                if !isempty(current_face)
                    push!(current_tet, current_face)
                    current_face = Vector{Vector{ComplexF64}}()
                end
                continue
            end

            vals = parse_number.(extract_tokens(s))
            push!(current_face, vals)
        end
    end

    if !isempty(current_face)
        push!(current_tet, current_face)
    end
    if !isempty(current_tet)
        push!(tets, current_tet)
    end

    return tets
end

# ============================================================
# 6. LOAD NESTED FLOAT64 VECTORS (nabout, nabfrombivec)
# ============================================================
function load_nested_real_vectors(filename::String)
    tets = Vector{Vector{Vector{Float64}}}()
    current_tet = Vector{Vector{Float64}}()
    current_face = Vector{Float64}()

    open(filename, "r") do io
        for line in eachline(io)
            s = strip(line)

            if startswith(s, "# Tetrahedron")
                if !isempty(current_face)
                    push!(current_tet, current_face)
                    current_face = Float64[]
                end
                if !isempty(current_tet)
                    push!(tets, current_tet)
                    current_tet = Vector{Float64}[]
                end
                continue
            end

            if startswith(s, "## Faces")
                if !isempty(current_face)
                    push!(current_tet, current_face)
                    current_face = Float64[]
                end
                continue
            end

            if isempty(s)
                if !isempty(current_face)
                    push!(current_tet, current_face)
                    current_face = Float64[]
                end
                continue
            end

            vals = parse.(Float64, split(s))
            current_face = vals
        end
    end

    if !isempty(current_face)
        push!(current_tet, current_face)
    end
    if !isempty(current_tet)
        push!(tets, current_tet)
    end

    return tets
end

# ============================================================
# 7. LOAD NESTED FLOAT SCALARS (dihedrals, areas)
# ============================================================
function load_nested_scalars(filename::String)
    rows = Vector{Vector{Float64}}()

    open(filename, "r") do io
        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            startswith(s, "#") && continue

            vals = parse.(Float64, split(s))
            push!(rows, vals)
        end
    end

    return rows
end

# ============================================================
# 8. LOAD NESTED INTS (kappa, tetareasign, tetn0sign)
# ============================================================
function load_nested_ints(filename::String)
    rows_f = load_nested_scalars(filename)
    return [Int.(row) for row in rows_f]
end

# ============================================================
# 9. LOAD INT SCALAR LIST (sgndet)
# ============================================================
function load_int_scalar_list(filename::String)
    vals_f = load_scalar_list(filename)
    return Int.(vals_f)
end

end # module LoadGeometryData