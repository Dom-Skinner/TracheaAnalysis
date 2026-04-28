# This file fits the liability threshold model to all genotypes and makes all relevant plots
# Sometimes the MAP optimization throws an error. Looking at the MCMC results
# when the MAP estimate converges it does fine, sometimes it decides to not converge. Adjust optim conditions and
# MCMC steps as necessary and it should work.

using Pkg
Pkg.activate(@__DIR__)

include("Utils.jl")
include("LiabilityThresholdCore.jl")
using MCMCChains


tag = ["WT", "F53S (GOF MEK)", "BNL MUTANT", "RESCUE"]
genotype_str = ["WT", "F53S", "Bnl", "Rescue"]
path="data/Raw Cell Counts - All Genotypes - Raw Counts - ".*tag.*".csv"

data_tot = [extract_data_one_genotype(p) for p in path]
 

####################################################################################
# First step is to fit the models (with MCMC + maximum a posteriori optimization)
####################################################################################

opt = Optim.Options(
    iterations = 20_000,      
    g_tol      = 1e-10,      
    f_tol      = 1e-12,
    x_tol      = 1e-12,
    store_trace = true,
    show_trace  = true,
)



chn_partially_withheld = []
map_partially_withheld = []

for i = 1:4
    exp_include = setdiff(1:4,i)
    dat_part = pool_genotypes(data_tot, exp_include)
    
    n_draws   = 1000
    n_chains  = 4
    model = phenom_two_thresh_no_py(dat_part.y.+1,dat_part.m_idx,dat_part.g_idx, dat_part.M)
    chn_pooled = sample(model, NUTS(),  MCMCThreads(), n_draws,n_chains)


    @assert all(rhat(chn_pooled).nt.rhat  .< 1.01) # ensure convergence

    map_est, _ = map_from_chain(model, chn_pooled; solver = LBFGS())
    println("MAP log-posterior: ", map_est.optim_result.minimum, " | Max MCMC log-posterior: ", maximum(chn_pooled[:lp]))    

    push!(map_partially_withheld, map_est)
    push!(chn_partially_withheld, chn_pooled)
end


dat = pool_genotypes(data_tot,1:4)

n_draws   = 1000
n_chains  = 4
    
model = phenom_two_thresh_no_py(dat.y.+1,dat.m_idx,dat.g_idx, dat.M)
chn_full = sample(model, NUTS(),  MCMCThreads(), n_draws,n_chains)
@assert all(rhat(chn_full).nt.rhat  .< 1.01) # ensure convergence


idx = argmax(vec(chn_full[:lp]))
map_full, _ = map_from_chain(model, chn_full; solver = LBFGS())


####################################################################################
# Plot the held out predictions for proportions of terminal cell counts
####################################################################################
nexp = [Int64(length(data_tot[i][1])/dat.M) for i in 1:4]
plt = plot( ylabel="Empirical freq.", xlabel="Predicted prob.")
for gtype in [1,3,2,4]

    mp_val = map_partially_withheld[gtype]
    p_vals = p_vals_compute(mp_val,gtype,dat.M)

    p_vals_err = p_val_emp_error_bars(chn_partially_withheld[gtype], dat.M,nexp)   
    yerr = (vec(p_vals) - vec(p_vals_err[gtype][:,:,1]), vec(p_vals_err[gtype][:,:,2]) - vec(p_vals))
    emp_freq = empirical_freq(data_tot[gtype][1], data_tot[gtype][2]; M=dat.M)
    scatter!(plt, vec(emp_freq),vec(p_vals), label=genotype_str[gtype],yerr=yerr)
end
plot!(plt,[0,1],[0,1],l=:dashed, color=:black,aspect_ratio=true,label=false,grid=false)
savefig("plots/full_predicted.pdf")


####################################################################################
# Plot the maximum a posteriori liability distributions for each genotype
####################################################################################
μ_vals = [map_full.values[Symbol("μ_metamere[$m]")] for m in 1:dat.M]
σ = map_full.values[:σ]
eff = genotype_effects(map_full.values[:Bnl_shift], map_full.values[:F53S_shift])

pal = cgrad(:YlGnBu, 10, categorical=true) 
    
xplt = range(-1.5, stop=2, length=200)
plt_tot = []
for i = 1:4
#    siml = simulate_single_embryo(draw, i)
    shift = genotype_shift(i, eff)
    
    plt = plot(grid=false)

    for met = 1:8
        plot!(plt, xplt, pdf(Normal(μ_vals[met]+shift, σ), xplt), label="$met", linecolor=pal[met+2])
    end
    #plot!(plt,xplt,pdf(Normal(siml.μ[met], siml.σ_raw[met]), xplt))
    plot!(plt,zeros(2),[0,10],ylims=ylims(plt),label=false,c=:black,ls=:dash,lw=2)
    plot!(plt,ones(2),[0,10],ylims=ylims(plt),label=false,c=:black,ls=:dash,lw=2,
        xlabel="liability", ylabel="Density")
    push!(plt_tot, plt)
end
plot(plt_tot..., layout=(2,2))
savefig("plots/liability_dists_full_fit.pdf")


####################################################################################
# Plot the maximum a posteriori liability distributions for each genotype 
# with model fit with held out data
####################################################################################
map_rescue = map_partially_withheld[4]
μ_vals = [map_rescue.values[Symbol("μ_metamere[$m]")] for m in 1:dat.M]
σ = map_rescue.values[:σ]
eff = genotype_effects(map_rescue.values[:Bnl_shift], map_rescue.values[:F53S_shift])

pal = cgrad(:YlGnBu, 10, categorical=true) 
    
xplt = range(-1.5, stop=2, length=200)
plt_tot = []
for i = 1:4
#    siml = simulate_single_embryo(draw, i)
    shift = genotype_shift(i, eff)
    
    plt = plot(grid=false)

    for met = 1:8
        plot!(plt, xplt, pdf(Normal(μ_vals[met]+shift, σ), xplt), label="$met", linecolor=pal[met+2])
    end
    #plot!(plt,xplt,pdf(Normal(siml.μ[met], siml.σ_raw[met]), xplt))
    plot!(plt,zeros(2),[0,10],ylims=ylims(plt),label=false,c=:black,ls=:dash,lw=2)
    plot!(plt,ones(2),[0,10],ylims=ylims(plt),label=false,c=:black,ls=:dash,lw=2,
        xlabel="liability", ylabel="Density")
    push!(plt_tot, plt)
end
plot(plt_tot..., layout=(2,2))
savefig("plots/liability_dists_prediction_rescue.pdf")




####################################################################################
# PPC: Draw samples and look at by eye - do they look like the data
####################################################################################
draws = extract_two_thresh_draws_for_ppc(chn_full, dat.M)
Sfull = draws.S
ntrials = 5
plot_tot = []
for i = 1:ntrials
    draw_id =  randperm(Sfull)[1]
    ysim = simulate_two_thresh_y(draws, draw_id, dat.m_idx, dat.g_idx)

    emp_freq_wt = empirical_freq(ysim[dat.g_idx.==1], dat.m_idx[dat.g_idx.==1], M=dat.M)
    emp_freq_bnl = empirical_freq(ysim[dat.g_idx.==3], dat.m_idx[dat.g_idx.==3], M=dat.M)
    emp_freq_rescue = empirical_freq(ysim[dat.g_idx.==4], dat.m_idx[dat.g_idx.==4], M=dat.M)
    emp_freq_F53S = empirical_freq(ysim[dat.g_idx.==2], dat.m_idx[dat.g_idx.==2], M=dat.M)

    p1 = groupedbar(1:size(emp_freq_wt, 1), emp_freq_wt[:,end:-1:1],
        bar_position = :stack,
        ylim   = (0, 1),
        ylabel = "Fraction of metameres",
        labels = false,
        palette = [ "#ff0000", "#d3d3d3","#1e90ff"],
        xticks = (1:8, ["Tr2","Tr3","Tr4","Tr5","Tr6","Tr7","Tr8","Tr9"])) 

    p2 = groupedbar(1:size(emp_freq_bnl, 1), emp_freq_bnl[:,end:-1:1],
        bar_position = :stack,
        ylim   = (0, 1),
        ylabel = "Fraction of metameres",
        labels = false,
        palette = [ "#ff0000", "#d3d3d3","#1e90ff"],
        xticks = (1:8, ["Tr2","Tr3","Tr4","Tr5","Tr6","Tr7","Tr8","Tr9"])) 

    p4 = groupedbar(1:size(emp_freq_rescue, 1), emp_freq_rescue[:,end:-1:1],
        bar_position = :stack,
        ylim   = (0, 1),
        ylabel = "Fraction of metameres",
        labels = false,
        palette = [ "#ff0000", "#d3d3d3","#1e90ff"],
        xticks = (1:8, ["Tr2","Tr3","Tr4","Tr5","Tr6","Tr7","Tr8","Tr9"])) 

    p3 = groupedbar(1:size(emp_freq_F53S, 1), emp_freq_F53S[:,end:-1:1],
        bar_position = :stack,
        ylim   = (0, 1),
        ylabel = "Fraction of metameres",
        labels = false,
        palette = [ "#ff0000", "#d3d3d3","#1e90ff"],
        xticks = (1:8, ["Tr2","Tr3","Tr4","Tr5","Tr6","Tr7","Tr8","Tr9"])) 
    push!(plot_tot, p1, p2, p3, p4)
end
plot(plot_tot..., layout=(ntrials, 4), size=(1500, 1200))
savefig("plots/stacked_bar_ppc.pdf")


####################################################################################
# Plot effect size for different contributions to variability
####################################################################################
σ_vals = vec(chn_full[:σ])
μ_metamere = [ vec(chn_full["μ_metamere[$m]"]) for m in 1:8]
q_range = x -> maximum(x) - minimum(x)
μ_range_vals = [std([μ_metamere[m][k] for m in 1:8]) for k in 1:length(σ_vals)]
ν_vals = vec(chn_full[:Bnl_shift]) 
η_vals = vec(chn_full[:F53S_shift])


boxplot(σ_vals,grid=false, label="Stochasticity", ylims=(0,0.8),outliers=false)
boxplot!(μ_range_vals, label="Metamere differences",outliers=false)
boxplot!(abs.(ν_vals), label="Bnl shift",ylabel="Effect size (σ units)",outliers=false)
boxplot!(abs.(η_vals), label="F53S shift",ylabel="Effect size (σ units)",outliers=false)
savefig("plots/effect_sizes_full.pdf")




####################################################################################
# PPC: max/min/avg defects per metamere 
####################################################################################

ppc_by_genotype = Dict{Int,Any}()
obs_by_genotype = Dict{Int,Any}()

total_plot = []
for gtype in [1,2,3,4]
    dat_g = pool_genotypes(data_tot, [gtype])

    obs_g = observed_metamere_defect_stats(
        dat_g;
        defect_score=DEFECT_SCORE,
    )

    ppc_g = posterior_metamere_defect_stats(
        chn_full,
        dat_g;
        ndraws=10_000,
        defect_score=DEFECT_SCORE,
    )

    obs_by_genotype[gtype] = obs_g
    ppc_by_genotype[gtype] = ppc_g

    p_max = ppc_hist_stat(
        ppc_g.samples_max,
        obs_g.max_defects;
        xlabel="Max defects in any metamere",
        title="$(genotype_str[gtype]): max",
    )

    p_min = ppc_hist_stat(
        ppc_g.samples_min,
        obs_g.min_defects;
        xlabel="Min defects in any metamere",
        title="$(genotype_str[gtype]): min",
    )

    p_avg = ppc_hist_stat(
        ppc_g.samples_avg,
        obs_g.avg_defects;
        xlabel="Avg defects per metamere",
        title="$(genotype_str[gtype]): avg",
    )
    push!(total_plot, p_max, p_min, p_avg)
end

plt =  plot(total_plot..., layout=(4, 3), size=(1100, 1200))
savefig(plt, "plots/posterior_checks_metamere_defects_total.pdf")


####################################################################################
# Look at variance and difference between metameres as function of genotype shift
####################################################################################
var_fun = p -> p[2] + 4*p[3] - (p[2] + 2*p[3])^2
μ_vals = [map_full.values[Symbol("μ_metamere[$m]")] for m in 1:dat.M]
σ = map_full.values[:σ]
eff = genotype_effects(map_full.values[:Bnl_shift], map_full.values[:F53S_shift])

shift_vals = -1.3:0.001:1.3
max_diff = zeros(length(shift_vals))
total_var = zeros(length(shift_vals))
for i in 1:length(shift_vals)
    
    p_vals = zeros(dat.M, 3)
    for m in 1:dat.M
        μ_obs = μ_vals[m] + shift_vals[i]
        
        p0, p1, p2 = ordinal3_probs( μ_obs, σ)
        p_vals[m,1] = p0; p_vals[m,2] = p1; p_vals[m,3] = p2
    end
    max_diff[i] = maximum(maximum(p_vals,dims=1) - minimum(p_vals,dims=1))
    total_var[i] = sum([2*var_fun(p_vals[m,:]) for m in 1:dat.M])
end
plt_Δ =  plot(shift_vals, max_diff,label=false,grid=false,xlabel="Genotype shift", ylabel="Max probability difference")
plot!(plt_Δ, [0,0],[0,1],l=:dash, color=:black,label="WT")
plot!(plt_Δ,genotype_shift(3, eff)*ones(2),[0,1],l=:dash, color=:red,label="Bnl shift",ylims=(0,0.55))
plot!(plt_Δ,genotype_shift(4, eff)*ones(2),[0,1],l=:dash, color=:blue,label="Rescue",ylims=(0,0.55))
plot!(plt_Δ,genotype_shift(2, eff)*ones(2),[0,1],l=:dash, color=:orange,label="MEK F53S shift",ylims=(0,0.55))


plt_var = plot(shift_vals, total_var,label=false,grid=false,xlabel="Genotype shift", ylabel="Variance of total terminal cells")
plot!(plt_var,[0,0],[0,4],l=:dash, color=:black,label="WT")
plot!(plt_var,genotype_shift(3, eff)*ones(2),[0,4],l=:dash, color=:red,label="Bnl shift")
plot!(plt_var,genotype_shift(4, eff)*ones(2),[0,4],l=:dash, color=:blue,label="Rescue")
plot!(plt_var,genotype_shift(2, eff)*ones(2),[0,4],l=:dash, color=:orange,label="MEK F53S shift")
plot(plt_Δ, plt_var, layout=(2,1))
savefig("plots/shift_effects_full.pdf")