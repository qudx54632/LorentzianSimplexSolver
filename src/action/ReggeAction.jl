module ReggeAction

using Symbolics
using ..SymbolicToJulia: sympy_to_expr
using ..Dihedral: theta_ab

export run_Regge_action
# ------------------------------------------------------------
# One triangle (i,j)
# ------------------------------------------------------------
function regge_face(tetareasign_ij, area_ij, jlabel_ij, Na, Nb, γ)
    θ = theta_ab(Na, Nb)  # Float64
    area_number = tetareasign_ij == -1 ? -area_ij : area_ij
    area_symbol = tetareasign_ij == -1 ? -jlabel_ij : jlabel_ij * γ
    theta_out = tetareasign_ij == -1 ? pi - θ : θ
    return area_number * theta_out, area_symbol * theta_out
end

# ------------------------------------------------------------
# One 4-simplex
# ------------------------------------------------------------
function regge_simplex(area_mat, jlabel_mat, tetareasign_mat, N_mat, γ)
    S_num = 0
    S_symbol = 0
    for i in 1:4
        for j in (i+1):5
           regge_face_num, regge_face_symbol = regge_face(tetareasign_mat[i][j], area_mat[i][j], jlabel_mat[i][j], N_mat[i], N_mat[j], γ)
           S_num += regge_face_num
           S_symbol +=  regge_face_symbol
        end
    end
    return S_num, S_symbol
end

# ------------------------------------------------------------
# All simplices
# ------------------------------------------------------------
function run_Regge_action(geom, γ)
    ns = length(geom.simplex)

    area_all = [geom.simplex[k].areas for k in 1:ns]

    j_all = [
        [
            [Symbolics.variable(sympy_to_expr(j))
             for j in row]
            for row in geom.varias[:j_mat][k]
        ]
        for k in 1:ns
    ]

    tetareasign_all = [geom.simplex[k].tetareasign for k in 1:ns]
    N_all = [geom.simplex[k].tetnormalvec for k in 1:ns]

    S_num = 0
    S_symbol = 0
    for k in 1:ns
        regge_simplex_num, regge_simplex_symbol = regge_simplex(area_all[k], j_all[k], tetareasign_all[k], N_all[k], γ)
        S_num += regge_simplex_num
        S_symbol +=  regge_simplex_symbol
    end
    return S_num, S_symbol
end

end
