using Plots
using Turing

@model function pooled_model(x, l_idx, L)
    # Priors (weakly informative on logit scale)
    mu     ~ Normal(0, 2.5)
    sigma_l  ~ truncated(Cauchy(0, 1), 0, Inf)  
    b_raw  ~ filldist(Normal(0, 1), L)
    b      = b_raw .* sigma_l

    N = size(x,1)
    for i in 1:N
        x[i,1] ~ Bernoulli( invlogit(mu + b[l_idx[i]]))
        x[i,2] ~ Bernoulli( invlogit(mu + b[l_idx[i]]))
    end
end

function pooled_model_sample(mu,sigma_l,b_raw,l_idx)
    N = length(l_idx)
    b      = b_raw .* sigma_l
    x = zeros(Int64,N,2)
    for i in 1:N
        x[i,:] = rand(Bernoulli( invlogit(mu + b[l_idx[i]])),2)
    end
    return x
end

# --- Model (ii): hierarchical per-metamere with partial pooling + larva intercept ---
@model function hier_model(x, l_idx, L, m_idx, M)
    mu     ~ Normal(0, 2.5)                             # baseline log-odds
    tau    ~ truncated(Cauchy(0, 1), 0, Inf)            # SD across metameres
    sigma_l  ~ truncated(Cauchy(0, 1), 0, Inf)            # SD across larvae
    alpha_raw ~ filldist(Normal(0, 1), M)               # non-centered
    b_raw     ~ filldist(Normal(0, 1), L)               # non-centered
    alpha = alpha_raw .* tau
    b     = b_raw .* sigma_l

    N = size(x,1)
    for i in 1:N
        x[i,1] ~ Bernoulli( invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]] ))
        x[i,2] ~ Bernoulli( invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]] ))
    end
end

function hier_model_sample(mu,tau,sigma,alpha_raw,b_raw,l_idx,m_idx)
    N = length(l_idx)
    alpha = alpha_raw .* tau
    b     = b_raw .* sigma
    x = zeros(Int64,N,2)
    
    for i in 1:N
        x[i,:] = rand(Bernoulli( invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]])),2)
    end
    return x
end

include("Utils.jl")


function observed_error_extremes(dat)
    counts = [sum(dat.z[dat.m_idx .== i]) for i in 1:dat.M]
    return (; counts, max_errors = maximum(counts), min_errors = minimum(counts))
end

function metamere_error_counts(zsim::AbstractMatrix{<:Integer}, dat)
    [sum(zsim[dat.m_idx .== i, :]) for i in 1:dat.M]
end

function extract_pooled_draws(chn, dat)
    mat = DataFrame(chn)
    return (
        mu = Vector(mat.mu),
        sigma_l = Vector(mat.sigma_l),
        b_raw = hcat([mat[!, Symbol("b_raw[$i]")] for i in 1:dat.L]...),
    )
end

function extract_hier_draws(chn, dat)
    mat = DataFrame(chn)
    return (
        mu = Vector(mat.mu),
        tau = Vector(mat.tau),
        sigma_l = Vector(mat.sigma_l),
        alpha_raw = hcat([mat[!, Symbol("alpha_raw[$i]")] for i in 1:dat.M]...),
        b_raw = hcat([mat[!, Symbol("b_raw[$i]")] for i in 1:dat.L]...),
    )
end

function posterior_extremes(draws, dat; model::Symbol)

    S = length(draws.mu)
    samples_max = Vector{Int}(undef, S)
    samples_min = Vector{Int}(undef, S)

    for s in 1:S
        zsim = if model == :pooled
            pooled_model_sample(
                draws.mu[s], draws.sigma_l[s], vec(draws.b_raw[s, :]), dat.l_idx
            )
        elseif model == :hier
            hier_model_sample(
                draws.mu[s], draws.tau[s], draws.sigma_l[s],
                vec(draws.alpha_raw[s, :]), vec(draws.b_raw[s, :]),
                dat.l_idx, dat.m_idx
            )
        else
            error("Unknown model: $model")
        end

        counts = metamere_error_counts(zsim, dat)
        samples_max[s] = maximum(counts)
        samples_min[s] = minimum(counts)
    end

    return (; samples_max, samples_min)
end


function log_meanexp(v)
    m = maximum(v)
    return m + log(mean(exp.(v .- m)))
end

function pooled_model_LL(x, l_idx, mu, sigma_l, b_raw)
    # Priors (weakly informative on logit scale)
    LL = 0.0
  
    b      = b_raw .* sigma_l

    N = size(x,1)

    for i in 1:N
        LL += logpdf(Bernoulli( invlogit(mu + b[l_idx[i]])), x[i,1])
        LL += logpdf(Bernoulli( invlogit(mu + b[l_idx[i]])), x[i,2])
    end
    return LL
end

function LOO_likelihood_pooled(chn_pooled, dat)
    draws_full = extract_pooled_draws(chn_pooled, dat)
    LL = zeros(length(draws_full.mu))
    for i = 1:size(LL,1)
        LL[i] = pooled_model_LL(hcat(dat.left,dat.right), dat.l_idx,
            draws_full.mu[i],  draws_full.sigma_l[i],
            draws_full.b_raw[i,:])
    end
    return log_meanexp(LL)
end

function compute_LOO_score_pooled(dat)
    n_draws   = 200
    n_chains  = 4
    LOO_score = []
    for j in dat.larvae
        I = findall(dat.l_idx .== j)
        larvae_data = (left=dat.left[I],right=dat.right[I], l_idx=dat.l_idx[I],L=dat.L)
        Ir = setdiff(1:size(dat.l_idx,1), I)
        chn_leaveoneout = sample(pooled_model(hcat(dat.left[Ir],dat.right[Ir]), dat.l_idx[Ir], dat.L), NUTS(),  MCMCThreads(), n_draws,n_chains)
        LL_val = LOO_likelihood_pooled(chn_leaveoneout, larvae_data)
        push!(LOO_score, LL_val)
        println("LOO progress: larva ", j, " done. Value: ", LL_val)
    end
    return LOO_score
end


function hier_model_LL(x, l_idx, m_idx, mu, tau,sigma_l, alpha_raw, b_raw)
    LL = 0.0
    b      = b_raw .* sigma_l
    alpha = alpha_raw .* tau
    N = size(x,1)
    
    for i in 1:N
        LL += logpdf(Bernoulli( invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]])), x[i,1])
        LL += logpdf(Bernoulli( invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]])), x[i,2])
    end
    return LL
end

function LOO_likelihood_hier(chn_pooled, dat)
    
    draws_full = extract_hier_draws(chn_pooled, dat)
    LL = zeros(length(draws_full.mu))
    for i = 1:size(LL,1)
        LL[i] = hier_model_LL(hcat(dat.left,dat.right), dat.l_idx, dat.m_idx,
            draws_full.mu[i], draws_full.tau[i], draws_full.sigma_l[i],
            draws_full.alpha_raw[i,:],
            draws_full.b_raw[i,:])
    end
    return log_meanexp(LL)
end

function compute_LOO_score_hier(dat)
    n_draws   = 200
    n_chains  = 4
    LOO_score = []
    for j in dat.larvae
        I = findall(dat.l_idx .== j)
        larvae_data = (left=dat.left[I],right=dat.right[I], l_idx=dat.l_idx[I],L=dat.L, m_idx = dat.m_idx[I], M = dat.M)
        Ir = setdiff(1:size(dat.l_idx,1), I)
        chn_leaveoneout = sample(hier_model(hcat(dat.left[Ir],dat.right[Ir]), dat.l_idx[Ir], dat.L, dat.m_idx[Ir], dat.M), NUTS(),  MCMCThreads(), n_draws,n_chains)
        LL_val = LOO_likelihood_hier(chn_leaveoneout, larvae_data)
        push!(LOO_score, LL_val)
        println("LOO progress: larva ", j, " done. Value: ", LL_val)
    end
    return LOO_score
end

function ppc_hist(samples::AbstractVector{<:Real}, observed::Real; xlabel::String, title::String)
    p = histogram(
        samples;
        xlabel=xlabel,
        ylabel="Density",
        title=title,
        lt=:stephist,
        grid=false,
        normalize=true,
        bins=minimum(samples)-0.5:max(maximum(samples), observed)+1,
        label="Model",
    )
    vline!(p, [observed]; label="Data")
    # add text to show tail probability
    tail_prob1 = mean(samples .>= observed)
    tail_prob2 = mean(samples .<= observed)
    tail_prob = min(tail_prob1, tail_prob2)
    annotate!(p, (observed, 0.05, text("Tail prob: $(round(tail_prob, digits=3))", :left, 10)))
    return p
end

LOO_score_arr = []
LOO_score_se_arr = []
LOO_score_arr_hier_tot = []
LOO_score_arr_pooled_tot = []
tau_plot_arr = []

for tag in ["RESCUE", "F53S (GOF MEK)", "BNL MUTANT", "WT"]
    
    path="data/Raw Cell Counts - All Genotypes - Binarized - "*tag*".csv"
    df = load_trachea_wide_csv(path)

    dat = build_model_data(df)
    obs = observed_error_extremes(dat)


    LOO_score_pooled = compute_LOO_score_pooled(dat)

    n_adapt   = 1000
    n_draws   = 2000
    n_chains  = 4

    chn_pooled = sample(pooled_model(hcat(dat.left,dat.right), dat.l_idx, dat.L), NUTS(),  MCMCThreads(), n_draws,n_chains)

    plot(chn_pooled)

    draws_pooled = extract_pooled_draws(chn_pooled, dat)
    ppc_pooled = posterior_extremes(draws_pooled, dat; model=:pooled)


    p_samples = invlogit.(draws_pooled.mu .+ draws_pooled.b_raw .* draws_pooled.sigma_l)
    p1 = histogram(vec(p_samples), xlabel="Error probability p", titlefontsize=10, 
        ylabel="Density", title="Metamere independent error rate (larva adjusted)",
        lt=:stephist, grid=false, normalize=true,label="Model")

    p2 = ppc_hist(ppc_pooled.samples_max, obs.max_errors;
        xlabel="Max errors in any metamere", title="Posterior check")
    p3 = ppc_hist(ppc_pooled.samples_min, obs.min_errors;
        xlabel="Min errors in any metamere", title="Posterior check")

    LOO_score_hier = compute_LOO_score_hier(dat)

    println("rhat convergence < 0.01?, ", maximum(abs.(1 .- summarize(chn_pooled).nt.rhat)))
    println("rhat convergence < 0.01?, ", maximum(abs.(1 .- summarize(chn_full).nt.rhat)))

    se = x -> sqrt(var(x)/length(x))
    push!(LOO_score_arr, mean(LOO_score_hier .- LOO_score_pooled))
    push!(LOO_score_se_arr, se(LOO_score_hier .- LOO_score_pooled))
    push!(LOO_score_arr_hier_tot,LOO_score_hier)
    push!(LOO_score_arr_pooled_tot,LOO_score_pooled)


    chn_full   = sample(hier_model(hcat(dat.left,dat.right), dat.l_idx, dat.L, dat.m_idx, dat.M), NUTS(),  MCMCThreads(), n_draws,n_chains)


    pvecs = posterior_probs_full(chn_full, dat.l_idx, dat.m_idx)

    pal = cgrad(:YlGnBu, 10, categorical=true)  # 8 ordered, discrete shades
    p4 = plot(label="metamere 2",title="Metamere-specific error rates (larva adjusted)",titlefontsize=10,
            grid=false,xlabel="Probability of error",ylabel="Density",xlims=(0,1))
    # subsequent series
    for j in 1:dat.M
        pvals = vcat([vec(pvecs[i]) for i in findall(dat.m_idx.==j)]...)
        histogram!(p4,pvals, label="metamere $(j+1)",
                lt=:stephist, normalize=true,
                linecolor=pal[j+2],lw=4.5)
    end

    draws_full = extract_hier_draws(chn_full, dat)
    ppc_full = posterior_extremes(draws_full, dat; model=:hier)

    p5 = ppc_hist(ppc_full.samples_max, obs.max_errors;
        xlabel="Max errors in any metamere", title="Posterior check")
    p6 = ppc_hist(ppc_full.samples_min, obs.min_errors;
        xlabel="Min errors in any metamere", title="Posterior check")


    p7 = histogram(
        draws_full.tau;
            xlabel="Metamere effect magnitude",
            ylabel="Density",
            lt=:stephist,
            grid=false,
            normalize=true,
            title = tag,
            label=false,
            xlims=(0,maximum(draws_full.tau)*1.1)
        )
        

    plot(p1,p2,p3,p4,p5,p6, layout=(2,3),size=(900,600))
    push!(tau_plot_arr, p7)

    savefig("plots/larva_adjusted_pooled_v_hierarchical_"*tag*".pdf")
end


plot([0.5,4.5], [0,0], ls=:dash, lc=:black, label=false)
scatter!([4,3,2,1],LOO_score_arr,yerr=LOO_score_se_arr, label=false, 
    xticks=(1:4, ["WT","BNL MUTANT", "F53S","RESCUE"]), xlims=(0.5,4.5), ylabel="ΔLOO",grid=false)
savefig("plots/LOO_scores_pooled_v_hierarchical.pdf")

plot(tau_plot_arr..., layout=(2,2), size=(800,600))
savefig("plots/tau_posteriors.pdf")
