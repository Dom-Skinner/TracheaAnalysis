# This file fits the liability threshold model to the WT and Het data and makes all relevant plots
using Pkg
Pkg.activate(@__DIR__)

include("Utils.jl")
include("LiabilityThresholdCore.jl")
using MCMCChains



tag = ["WT", "F53S (GOF MEK)", "BNL MUTANT", "RESCUE"]
genotype_str = ["WT", "F53S", "Bnl", "Rescue"]
path="data/Raw Cell Counts - All Genotypes - Raw Counts - ".*tag.*".csv"

data_tot = [extract_data_one_genotype(p) for p in path]


dat = pool_genotypes(data_tot,[1,3])

emp_freq_wt = empirical_freq(data_tot[1][1], data_tot[1][2]; M=dat.M)
println("max difference between metameres, WT:",maximum(maximum(emp_freq_wt,dims=1) - minimum(emp_freq_wt,dims=1)))
emp_freq_bnl = empirical_freq(data_tot[3][1], data_tot[3][2]; M=dat.M)
println("max difference between metameres, WT:",maximum(maximum(emp_freq_bnl,dims=1) - minimum(emp_freq_bnl,dims=1)))


n_draws   = 1000
n_chains  = 3
chn_full = sample(liability_threshold(dat.y.+1,dat.m_idx,dat.g_idx, dat.M), NUTS(),  MCMCThreads(), n_draws,n_chains)


@assert all(rhat(chn_full).nt.rhat  .< 1.01) # ensure convergence
plot(chn_full) # Inspect chain by eye as well


map_estimate = maximum_a_posteriori(liability_threshold(dat.y .+1,dat.m_idx,dat.g_idx, dat.M))
μ_vals = [map_estimate.values[Symbol("μ_metamere[$m]")] for m in 1:dat.M]
σ = map_estimate.values[:σ]
eff = genotype_effects(map_estimate.values[:Bnl_shift], map_estimate.values[:F53S_shift])


####################################################################################
# Plot the maximum a posteriori liability distributions for each genotype
####################################################################################
xplt = range(-1.5, stop=1.5, length=200)
pal = cgrad(:YlGnBu, 10, categorical=true)  # 8 ordered, discrete shades
plt_tot = []
for i = [1,3]

    shift = genotype_shift(i, eff)
    plt = plot()

    for met = 1:dat.M
        plot!(plt, xplt, pdf(Normal(μ_vals[met] + shift, σ), xplt), label="$(met+1)", linecolor=pal[met+2],lw=2)
    end
    #plot!(plt,xplt,pdf(Normal(siml.μ[met], siml.σ_raw[met]), xplt))
    plot!(plt,zeros(2),[0,10],ylims=ylims(plt),label=false,c=:black,ls=:dash,lw=2)
    plot!(plt,ones(2),[0,10],ylims=ylims(plt),label=false,c=:black,ls=:dash,lw=2,
        xlabel="liability", ylabel="Density",grid=false)
    push!(plt_tot, plt)
end
plot(plt_tot...)
savefig("plots/FitsForBnl.pdf")


####################################################################################
# Look at variance and difference between metameres as function of genotype shift
####################################################################################
var_fun = p -> p[2] + 4*p[3] - (p[2] + 2*p[3])^2
    
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


plt_var = plot(shift_vals, total_var,label=false,grid=false,xlabel="Genotype shift", ylabel="Variance of total terminal cells")
plot!(plt_var,[0,0],[0,2],l=:dash, color=:black,label="WT")
plot!(plt_var,genotype_shift(3, eff)*ones(2),[0,2],l=:dash, color=:red,label="Bnl shift")
plot(plt_Δ, plt_var, layout=(2,1))
savefig("plots/shift_effects.pdf")

####################################################################################
# Plot effect size for different contributions to variability
####################################################################################

σ_vals = vec(chn_full[:σ])
μ_metamere = [ vec(chn_full["μ_metamere[$m]"]) for m in 1:8]
q_range = x -> maximum(x) - minimum(x)
μ_range_vals = [std([μ_metamere[m][k] for m in 1:8]) for k in 1:length(σ_vals)]
ν_vals = vec(chn_full[:Bnl_shift]) 

boxplot(σ_vals,grid=false, label="Stochasticity", ylims=(0,0.9),outliers=false)
boxplot!(μ_range_vals, label="Metamere differences",outliers=false)
boxplot!(abs.(ν_vals), label="Bnl shift",ylabel="Effect size",outliers=false)
savefig("plots/effect_sizes_wt_bnl.pdf")


####################################################################################
# PPC: max/min/avg defects per metamere 
####################################################################################

ppc_by_genotype = Dict{Int,Any}()
obs_by_genotype = Dict{Int,Any}()

total_plot = []
for gtype in [1,3]
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

plt =  plot(total_plot..., layout=(2, 3), size=(1100, 600))
savefig(plt, "plots/posterior_checks_metamere_defects_WT_bnl.pdf")




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

    push!(plot_tot, p1, p2)
end
plot(plot_tot..., layout=(ntrials, 2), size=(750, 1200))
savefig("plots/stacked_bar_ppc_wt_bnl.pdf")



####################################################################################
# PPC: larva-to-larva dispersion in total terminal cell count
####################################################################################

larva_ppc_by_genotype = Dict{Int,Any}()
larva_obs_by_genotype = Dict{Int,Any}()
larva_summary_by_genotype = Dict{Int,Any}()

total_plot = []

for gtype in [1,3]
    dat_g = pool_genotypes_with_larvae(data_tot, [gtype])

    obs_g = observed_larva_dispersion_stats(
        dat_g;
        defect_score=DEFECT_SCORE,
    )

    ppc_g = posterior_larva_dispersion_stats(
        chn_full,
        dat_g;
        ndraws=20_000,
        defect_score=DEFECT_SCORE,
    )

    larva_obs_by_genotype[gtype] = obs_g
    larva_ppc_by_genotype[gtype] = ppc_g
    larva_summary_by_genotype[gtype] = ppc_larva_summary(obs_g, ppc_g)

    println("\nLarva-level PPC: $(genotype_str[gtype])")
    display(larva_summary_by_genotype[gtype])

    p_var = ppc_hist_stat(
        ppc_g.samples_var,
        obs_g.var_total;
        xlabel="Variance of total terminal cells across larvae",
        title="$(genotype_str[gtype]): larva variance",
    )

    p_range = ppc_hist_stat(
        ppc_g.samples_range,
        obs_g.range_total;
        xlabel="Range of total terminal cells across larvae",
        title="$(genotype_str[gtype]): larva range",
    )

    push!(total_plot, p_var, p_range)
end

plt = plot(total_plot..., layout=(2, 2), size=(950, 1200))
savefig(plt, "plots/posterior_checks_larva_dispersion_wt_bnl.pdf")
