module ReggeAction

using ..Dihedral: theta_ab

export run_Regge_action
# ------------------------------------------------------------
# One triangle (i,j)
# ------------------------------------------------------------
function regge_face(tetareasign_ij, area_ij, Na, Nb)
    θ = theta_ab(Na, Nb)  # Float64
    area = tetareasign_ij == -1 ? -area_ij : area_ij
    theta_out = tetareasign_ij == -1 ? pi - θ : θ
    return area * theta_out
end

# ------------------------------------------------------------
# One 4-simplex
# ------------------------------------------------------------
function regge_simplex(area_mat, tetareasign_mat, N_mat)
    S = 0
    for i in 1:4
        for j in (i+1):5
           S += regge_face(tetareasign_mat[i][j], area_mat[i][j], N_mat[i], N_mat[j])
        end
    end
    return S
end

# ------------------------------------------------------------
# All simplices
# ------------------------------------------------------------
function run_Regge_action(geom)
    ns = length(geom.simplex)

    area_all = [geom.simplex[k].areas for k in 1:ns]

    tetareasign_all = [geom.simplex[k].tetareasign for k in 1:ns]
    N_all = [geom.simplex[k].tetnormalvec for k in 1:ns]

    S = 0
    for k in 1:ns
        S += regge_simplex(area_all[k], tetareasign_all[k], N_all[k])
    end
    return S
end

end