import Pkg; Pkg.activate(@__DIR__)
cd(@__DIR__)
using Plots
using Turing

@model function pooled_lr_model(x, l_idx, L)
    mu       ~ Normal(0, 2.5)
    sigma_l  ~ truncated(Cauchy(0, 1), 0, Inf)
    sigma_lr ~ truncated(Cauchy(0, 1), 0, Inf)
    b_raw    ~ filldist(Normal(0, 1), L)
    c_raw    ~ filldist(Normal(0, 1), size(x, 1))
    b = b_raw .* sigma_l
    c = c_raw .* sigma_lr

    N = size(x, 1)
    for i in 1:N
        x[i,1] ~ Bernoulli(invlogit(mu + b[l_idx[i]] + c[i]))
        x[i,2] ~ Bernoulli(invlogit(mu + b[l_idx[i]] + c[i]))
    end
end

function pooled_lr_sample(mu, sigma_l, sigma_lr, b_raw, l_idx)
    N = length(l_idx)
    b = b_raw .* sigma_l
    x = zeros(Int64, N, 2)
    for i in 1:N
        c = sigma_lr * randn()
        x[i,:] = rand(Bernoulli(invlogit(mu + b[l_idx[i]] + c)), 2)
    end
    return x
end

@model function hier_lr_model(x, l_idx, L, m_idx, M)
    mu        ~ Normal(0, 2.5)
    tau       ~ truncated(Cauchy(0, 1), 0, Inf)
    sigma_l   ~ truncated(Cauchy(0, 1), 0, Inf)
    sigma_lr  ~ truncated(Cauchy(0, 1), 0, Inf)
    alpha_raw ~ filldist(Normal(0, 1), M)
    b_raw     ~ filldist(Normal(0, 1), L)
    c_raw     ~ filldist(Normal(0, 1), size(x, 1))
    alpha = alpha_raw .* tau
    b     = b_raw .* sigma_l
    c     = c_raw .* sigma_lr

    N = size(x, 1)
    for i in 1:N
        x[i,1] ~ Bernoulli(invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]] + c[i]))
        x[i,2] ~ Bernoulli(invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]] + c[i]))
    end
end

function hier_lr_sample(mu, tau, sigma_l, sigma_lr, alpha_raw, b_raw, l_idx, m_idx)
    N = length(l_idx)
    alpha = alpha_raw .* tau
    b     = b_raw .* sigma_l
    x = zeros(Int64, N, 2)
    for i in 1:N
        c = sigma_lr * randn()
        x[i,:] = rand(Bernoulli(invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]] + c)), 2)
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

function extract_pooled_lr_draws(chn, dat)
    mat = DataFrame(chn)
    return (
        mu       = Vector(mat.mu),
        sigma_l  = Vector(mat.sigma_l),
        sigma_lr = Vector(mat.sigma_lr),
        b_raw    = hcat([mat[!, Symbol("b_raw[$i]")] for i in 1:dat.L]...),
    )
end

function extract_hier_lr_draws(chn, dat)
    mat = DataFrame(chn)
    return (
        mu       = Vector(mat.mu),
        tau      = Vector(mat.tau),
        sigma_l  = Vector(mat.sigma_l),
        sigma_lr = Vector(mat.sigma_lr),
        alpha_raw = hcat([mat[!, Symbol("alpha_raw[$i]")] for i in 1:dat.M]...),
        b_raw     = hcat([mat[!, Symbol("b_raw[$i]")] for i in 1:dat.L]...),
    )
end

function posterior_extremes_lr(draws, dat; model::Symbol)
    S = length(draws.mu)
    samples_max = Vector{Int}(undef, S)
    samples_min = Vector{Int}(undef, S)

    for s in 1:S
        zsim = if model == :pooled
            pooled_lr_sample(
                draws.mu[s], draws.sigma_l[s], draws.sigma_lr[s],
                vec(draws.b_raw[s, :]), dat.l_idx
            )
        elseif model == :hier
            hier_lr_sample(
                draws.mu[s], draws.tau[s], draws.sigma_l[s], draws.sigma_lr[s],
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

function pooled_lr_model_LL(x, l_idx, mu, sigma_l, sigma_lr, b_raw; n_c=100)
    b  = b_raw .* sigma_l
    LL = 0.0
    for i in 1:size(x, 1)
        lls = [logpdf(Bernoulli(invlogit(mu + b[l_idx[i]] + c)), x[i,1]) +
               logpdf(Bernoulli(invlogit(mu + b[l_idx[i]] + c)), x[i,2])
               for c in sigma_lr .* randn(n_c)]
        LL += log_meanexp(lls)
    end
    return LL
end

function LOO_likelihood_pooled_lr(chn_pooled, dat)
    draws = extract_pooled_lr_draws(chn_pooled, dat)
    LL = zeros(length(draws.mu))
    for i in 1:length(LL)
        LL[i] = pooled_lr_model_LL(hcat(dat.left, dat.right), dat.l_idx,
            draws.mu[i], draws.sigma_l[i], draws.sigma_lr[i],
            draws.b_raw[i,:])
    end
    return log_meanexp(LL)
end

function compute_LOO_score_pooled_lr(dat)
    n_draws  = 200
    n_chains = 4
    LOO_score = []
    for j in dat.larvae
        I  = findall(dat.l_idx .== j)
        Ir = setdiff(1:size(dat.l_idx, 1), I)
        larvae_data = (left=dat.left[I], right=dat.right[I], l_idx=dat.l_idx[I], L=dat.L)
        chn_loo = sample(pooled_lr_model(hcat(dat.left[Ir], dat.right[Ir]), dat.l_idx[Ir], dat.L),
                         NUTS(), MCMCThreads(), n_draws, n_chains)
        LL_val = LOO_likelihood_pooled_lr(chn_loo, larvae_data)
        push!(LOO_score, LL_val)
        println("LOO progress: larva ", j, " done. Value: ", LL_val)
    end
    return LOO_score
end

function hier_lr_model_LL(x, l_idx, m_idx, mu, tau, sigma_l, sigma_lr, alpha_raw, b_raw; n_c=100)
    alpha = alpha_raw .* tau
    b     = b_raw .* sigma_l
    LL    = 0.0
    for i in 1:size(x, 1)
        lls = [logpdf(Bernoulli(invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]] + c)), x[i,1]) +
               logpdf(Bernoulli(invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]] + c)), x[i,2])
               for c in sigma_lr .* randn(n_c)]
        LL += log_meanexp(lls)
    end
    return LL
end

function LOO_likelihood_hier_lr(chn_hier, dat)
    draws = extract_hier_lr_draws(chn_hier, dat)
    LL = zeros(length(draws.mu))
    for i in 1:length(LL)
        LL[i] = hier_lr_model_LL(hcat(dat.left, dat.right), dat.l_idx, dat.m_idx,
            draws.mu[i], draws.tau[i], draws.sigma_l[i], draws.sigma_lr[i],
            draws.alpha_raw[i,:], draws.b_raw[i,:])
    end
    return log_meanexp(LL)
end

function compute_LOO_score_hier_lr(dat)
    n_draws  = 200
    n_chains = 4
    LOO_score = []
    for j in dat.larvae
        I  = findall(dat.l_idx .== j)
        Ir = setdiff(1:size(dat.l_idx, 1), I)
        larvae_data = (left=dat.left[I], right=dat.right[I], l_idx=dat.l_idx[I], L=dat.L,
                       m_idx=dat.m_idx[I], M=dat.M)
        chn_loo = sample(hier_lr_model(hcat(dat.left[Ir], dat.right[Ir]), dat.l_idx[Ir], dat.L,
                                       dat.m_idx[Ir], dat.M),
                         NUTS(), MCMCThreads(), n_draws, n_chains)
        LL_val = LOO_likelihood_hier_lr(chn_loo, larvae_data)
        push!(LOO_score, LL_val)
        println("LOO progress: larva ", j, " done. Value: ", LL_val)
    end
    return LOO_score
end


function concordant_fracs(x)
    n = size(x, 1)
    frac_00 = sum((x[:, 1] .== 0) .& (x[:, 2] .== 0)) / n
    frac_11 = sum((x[:, 1] .== 1) .& (x[:, 2] .== 1)) / n
    return frac_00, frac_11
end

function posterior_concordant_fracs(draws, dat; model::Symbol)
    S = length(draws.mu)
    fracs_00 = Vector{Float64}(undef, S)
    fracs_11 = Vector{Float64}(undef, S)

    for s in 1:S
        zsim = if model == :pooled
            pooled_lr_sample(
                draws.mu[s], draws.sigma_l[s], draws.sigma_lr[s],
                vec(draws.b_raw[s, :]), dat.l_idx
            )
        elseif model == :hier
            hier_lr_sample(
                draws.mu[s], draws.tau[s], draws.sigma_l[s], draws.sigma_lr[s],
                vec(draws.alpha_raw[s, :]), vec(draws.b_raw[s, :]),
                dat.l_idx, dat.m_idx
            )
        else
            error("Unknown model: $model")
        end
        fracs_00[s], fracs_11[s] = concordant_fracs(zsim)
    end
    return (; fracs_00, fracs_11)
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
    tail_prob1 = mean(samples .>= observed)
    tail_prob2 = mean(samples .<= observed)
    tail_prob = min(tail_prob1, tail_prob2)
    annotate!(p, (observed, 0.05, text("Tail prob: $(round(tail_prob, digits=3))", :left, 10)))
    return p
end

function ppc_hist_continuous(samples::AbstractVector{<:Real}, observed::Real; xlabel::String, title::String)
    p = histogram(
        samples;
        xlabel=xlabel,
        ylabel="Density",
        title=title,
        lt=:stephist,
        grid=false,
        normalize=true,
        label="Model",
    )
    vline!(p, [observed]; label="Data")
    tail_prob1 = mean(samples .>= observed)
    tail_prob2 = mean(samples .<= observed)
    tail_prob = min(tail_prob1, tail_prob2)
    annotate!(p, (observed, 0.05, text("Tail prob: $(round(tail_prob, digits=3))", :left, 10)))
    return p
end


LOO_score_arr         = []
LOO_score_se_arr      = []
LOO_score_arr_hier_tot   = []
LOO_score_arr_pooled_tot = []
tau_plot_arr          = []
sigma_lr_plot_arr     = []

for tag in ["RESCUE", "F53S (GOF MEK)", "BNL MUTANT", "WT"]

    path = "data/Raw Cell Counts - All Genotypes - Binarized - "*tag*".csv"
    df   = load_trachea_wide_csv(path)
    dat  = build_model_data(df)
    obs  = observed_error_extremes(dat)

    obs_frac_00, obs_frac_11 = concordant_fracs(hcat(dat.left, dat.right))

    LOO_score_pooled = compute_LOO_score_pooled_lr(dat)

    n_draws  = 2000
    n_chains = 4

    chn_pooled = sample(pooled_lr_model(hcat(dat.left, dat.right), dat.l_idx, dat.L),
                        NUTS(), MCMCThreads(), n_draws, n_chains)

    draws_pooled = extract_pooled_lr_draws(chn_pooled, dat)
    ppc_pooled   = posterior_extremes_lr(draws_pooled, dat; model=:pooled)
    conc_pooled  = posterior_concordant_fracs(draws_pooled, dat; model=:pooled)

    p_samples = invlogit.(draws_pooled.mu .+ draws_pooled.b_raw .* draws_pooled.sigma_l)
    p1 = histogram(vec(p_samples), xlabel="Error probability p", titlefontsize=10,
        ylabel="Density", title="Metamere independent error rate (larva adjusted)",
        lt=:stephist, grid=false, normalize=true, label="Model")

    p2 = ppc_hist(ppc_pooled.samples_max, obs.max_errors;
        xlabel="Max errors in any metamere", title="Posterior check")
    p3 = ppc_hist(ppc_pooled.samples_min, obs.min_errors;
        xlabel="Min errors in any metamere", title="Posterior check")

    LOO_score_hier = compute_LOO_score_hier_lr(dat)

    println("rhat pooled < 0.01?, ", maximum(abs.(1 .- summarize(chn_pooled).nt.rhat)))

    se = x -> sqrt(var(x)/length(x))
    push!(LOO_score_arr, mean(LOO_score_hier .- LOO_score_pooled))
    push!(LOO_score_se_arr, se(LOO_score_hier .- LOO_score_pooled))
    push!(LOO_score_arr_hier_tot, LOO_score_hier)
    push!(LOO_score_arr_pooled_tot, LOO_score_pooled)

    chn_full = sample(hier_lr_model(hcat(dat.left, dat.right), dat.l_idx, dat.L, dat.m_idx, dat.M),
                      NUTS(), MCMCThreads(), n_draws, n_chains)

    println("rhat hier < 0.01?, ", maximum(abs.(1 .- summarize(chn_full).nt.rhat)))

    draws_full  = extract_hier_lr_draws(chn_full, dat)
    ppc_full    = posterior_extremes_lr(draws_full, dat; model=:hier)
    conc_full   = posterior_concordant_fracs(draws_full, dat; model=:hier)

    pal = cgrad(:YlGnBu, 10, categorical=true)
    p4 = plot(label=false, title="Metamere-specific error rates (larva adjusted)", titlefontsize=10,
              grid=false, xlabel="Probability of error", ylabel="Density", xlims=(0,1))
    for j in 1:dat.M
        pvals = vcat([vec(invlogit.(draws_full.mu .+ draws_full.alpha_raw[:,j] .* draws_full.tau .+
                          draws_full.b_raw[:,dat.l_idx[k]] .* draws_full.sigma_l))
                      for k in findall(dat.m_idx .== j)]...)
        histogram!(p4, pvals, label="metamere $(j+1)",
                   lt=:stephist, normalize=true,
                   linecolor=pal[j+2], lw=4.5)
    end

    p5 = ppc_hist(ppc_full.samples_max, obs.max_errors;
        xlabel="Max errors in any metamere", title="Posterior check")
    p6 = ppc_hist(ppc_full.samples_min, obs.min_errors;
        xlabel="Min errors in any metamere", title="Posterior check")

    # (i) sigma_lr posterior
    p7 = histogram(
        draws_full.sigma_lr;
        xlabel="L-R effect magnitude (sigma_lr)",
        ylabel="Density",
        lt=:stephist,
        grid=false,
        normalize=true,
        title=tag,
        label=false,
        xlims=(0, maximum(draws_full.sigma_lr)*1.1)
    )
    push!(sigma_lr_plot_arr, p7)

    p8 = histogram(
        draws_full.tau;
        xlabel="Metamere effect magnitude (tau)",
        ylabel="Density",
        lt=:stephist,
        grid=false,
        normalize=true,
        title=tag,
        label=false,
        xlims=(0, maximum(draws_full.tau)*1.1)
    )
    push!(tau_plot_arr, p8)

    # (ii) concordant pair PPC
    p_c00 = ppc_hist_continuous(conc_full.fracs_00, obs_frac_00;
        xlabel="Fraction (0,0) pairs", title="PPC: concordant (0,0) — "*tag)
    p_c11 = ppc_hist_continuous(conc_full.fracs_11, obs_frac_11;
        xlabel="Fraction (1,1) pairs", title="PPC: concordant (1,1) — "*tag)

    plot(p1, p2, p3, p4, p5, p6, layout=(2,3), size=(900,600))
    savefig("plots/lr_posterior_checks_"*tag*".pdf")

    plot(p_c00, p_c11, layout=(1,2), size=(800,400))
    savefig("plots/lr_concordant_ppc_"*tag*".pdf")
end


plot([0.5, 4.5], [0, 0], ls=:dash, lc=:black, label=false)
scatter!([4, 3, 2, 1], LOO_score_arr, yerr=LOO_score_se_arr, label=false,
    xticks=(1:4, ["WT", "BNL MUTANT", "F53S", "RESCUE"]), xlims=(0.5, 4.5),
    ylabel="ΔLOO", grid=false)
savefig("plots/lr_LOO_scores_pooled_v_hierarchical.pdf")

plot(tau_plot_arr..., layout=(2,2), size=(800,600))
savefig("plots/lr_tau_posteriors.pdf")

plot(sigma_lr_plot_arr..., layout=(2,2), size=(800,600))
savefig("plots/lr_sigma_lr_posteriors.pdf")
