module SFaction_no_phase

using SymEngine
using DoubleFloats: Double64
using ..DefineAction: _I, symone, Eehss, EebssSource, EebssTarget, γsym, compute_action
using ..DefineSymbols: make_symbol
using ..ActionEvaluation: eval_symbolic, build_value_dict
using ..FourSimplexConnectivity: build_single_simplex_boundary_faces

export compute_action_no_bdry_phase

@inline val_basic(x) = x isa Double64 ? Basic(string(x)) : Basic(x)

function boundary_Eb_phase(gvariablesall, zvariablesall, zetabdryall, kappaMat, OrderBDryFaces; γ=γsym())

    Eb_phase_list = []
    Iconst = _I[]

    for faces in OrderBDryFaces
        xilist = [zetabdryall[k][i][j] for (k,i,j) in faces]
        zlist  = [zvariablesall[k][i][j] for (k,i,j) in faces]
        glist  = [gvariablesall[k][i] for (k,i,j) in faces]
        κlist  = [kappaMat[k][i][j] for (k,i,j) in faces]

        (k1,i1,j1) = faces[1]
        phase = make_symbol("Phi$(k1)_$(i1)_$(j1)")
        prodEh = symone(κlist[1])

        for i in 2:2:(length(faces)-1)
            prodEh *= Eehss((zlist[i-1], zlist[i+1]),
                            (glist[i], glist[i+1]),
                            (κlist[i], κlist[i+1]); γ=γ)
        end

        term = EebssSource(glist[1], zlist[1], SymEngine.exp(Iconst * phase) * xilist[1], κlist[1]; γ=γ) *
               prodEh *
               EebssTarget(glist[end], zlist[end-1], xilist[end], κlist[end]; γ=γ)

        Eb_phase = term

        push!(Eb_phase_list, Eb_phase) 
    end

    return Eb_phase_list
end

function solve_phase(gvariablesall, zvariablesall, zetabdryall, kappaMat, OrderBDryFaces, vals, dihedral_angles; γ=γsym())

    Eb_phase_list = boundary_Eb_phase(gvariablesall, zvariablesall, zetabdryall, kappaMat, OrderBDryFaces; γ)
    
    phase_solution = Dict{Basic,Basic}()
    I = _I[]

    for face_index in eachindex(Eb_phase_list)
        
        (k, i, j) = OrderBDryFaces[face_index][1]
        phase_sym = make_symbol("Phi$(k)_$(i)_$(j)")   
        Eb_phase_list1_val = eval_symbolic(Eb_phase_list[face_index], vals)
        Eb_phase_list1_simp = subs(SymEngine.expand(Eb_phase_list1_val), phase_sym => 0)
        rhs = val_basic(real(dihedral_angles[face_index])) * I * γ/ val_basic(2)
        phase_expr = (log(Eb_phase_list1_simp) - rhs) / I
        phase_solution[phase_sym] = phase_expr
    end

    return phase_solution
end

function compute_action_no_bdry_phase(
    geom,
    sd,
    dihedral_angles;
    γ=γsym(),
    bulk_sum_form::Bool=true,
)
    gvariablesall = geom.varias[:g_mat]
    zvariablesall = geom.varias[:z_mat]
    zetabdryall   = geom.varias[:xi_mat]
    kappaMat      = [geom.simplex[i].kappa for i in 1:length(geom.simplex)]
    
    ns = length(geom.simplex)
    if ns > 1
        OrderBDryFaces = geom.connectivity[1]["OrderBDryFaces"]
    else
        OrderBDryFaces = build_single_simplex_boundary_faces(kappaMat[1])
    end

    vals = build_value_dict(sd, γsym())
    phase_soln = solve_phase(gvariablesall, zvariablesall, zetabdryall, kappaMat, OrderBDryFaces, vals, dihedral_angles; γ=γ)

    for face_idx in eachindex(OrderBDryFaces)
        (k, i, j) = OrderBDryFaces[face_idx][1]
        phase_sym = make_symbol("Phi$(k)_$(i)_$(j)")
        geom.varias[:xi_mat][k][i][j] = zetabdryall[k][i][j] * exp(_I[] * phase_sym)
    end
    
    new_action_no_bdry_phase = compute_action(geom; γ=γ, bulk_sum_form=bulk_sum_form)
    
    return new_action_no_bdry_phase, phase_soln
end


end
