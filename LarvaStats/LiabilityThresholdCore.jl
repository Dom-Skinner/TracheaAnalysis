
using Turing, Distributions
using Plots, StatsPlots
using Optim

function pool_genotypes(data_tot, include::AbstractVector{<:Integer})
    y     = vcat((data_tot[i][1] for i in include)...)
    m_idx = vcat((data_tot[i][2] for i in include)...)
    g_idx = vcat((fill(i, length(data_tot[i][1])) for i in include)...)
    M = maximum(m_idx) 
    @assert length(unique(m_idx)) == M
    return (y=y, m_idx=m_idx, g_idx=g_idx, M=M)
end



@inline function genotype_effects(Bnl_shift, F53S_shift)
    bnl  = (0.0, 0.0, Bnl_shift, Bnl_shift)
    f53s = (0.0, F53S_shift, 0.0, F53S_shift)
    return (bnl=bnl, f53s=f53s)
end

@inline genotype_shift(g, eff) = eff.bnl[g] + eff.f53s[g]

@model function phenom_two_thresh_no_py(y::AbstractVector{<:Integer},
                                      m_idx::AbstractVector{<:Integer},
                                      g_idx::AbstractVector{<:Integer}, M)

    @assert length(y) == length(m_idx) == length(g_idx)

    
    σ ~ LogNormal(-1, 1.)
    μ_metamere ~ filldist(Normal(0.5, 1), M)
    Bnl_shift ~ Normal(-1,2)
    F53S_shift ~ Normal(1,2)

    N = length(y)
    
    eff = genotype_effects(Bnl_shift, F53S_shift)
    
    p_vals = Array{eltype(μ_metamere)}(undef, N, 3)
    for i in 1:N
        μ_obs = μ_metamere[m_idx[i]] + genotype_shift(g_idx[i], eff)
        p0, p1, p2 = ordinal3_probs(μ_obs, σ)
        p_vals[i,1] = p0; p_vals[i,2] = p1; p_vals[i,3] = p2
    end

    for i in 1:N
        y[i] ~ Categorical(p_vals[i,:])
    end
end



function _matrix_and_names_local(chn::Chains)
    
    A = Array(chn.value)                      # (iters, chains, params)
    A = permutedims(A,[1,3,2])
    niter, nch, npar = size(A)
    A2 = reshape(A, niter*nch, npar)    # collapse draws
    nobj = names(chn)
    pnames = hasproperty(nobj, :parameters) ? String.(nobj.parameters) : String.(nobj)
    A2, pnames
end


function pick_draw(chn::Chains; which=:random, rng=Random.GLOBAL_RNG)
    A2, pnames = _matrix_and_names_local(chn)
    idx = _name_to_col_local(pnames)
    nS, _ = size(A2)

    row = which === :random ? rand(rng, 1:nS) : which
          
    
    
    muraw_names = filter(n -> startswith(n, "μ_metamere["), pnames)
    M = length(muraw_names)
    
    
    μ_metamere = [ A2[row, _col(idx, "μ_metamere[$m]") ] for m in 1:M]
    σ = A2[row, _col(idx, "σ") ]
    Bnl_shift = A2[row, _col(idx, "Bnl_shift") ]
    F53S_shift = A2[row, _col(idx, "F53S_shift") ]

    return (
        M = M,
        μ_metamere = μ_metamere,
        σ = σ,
        Bnl_shift = Bnl_shift,
        F53S_shift = F53S_shift,
        row_index = row
    )
end

function predict_metameres(draw, genotype::Integer)
    eff = genotype_effects(draw.Bnl_shift, draw.F53S_shift)
    shift = genotype_shift(genotype, eff)

    M = draw.M
    μ_obs = similar(draw.μ_metamere)
    p = Matrix{eltype(draw.μ_metamere)}(undef, M, 3)

    for m in 1:M
        μ_obs[m] = draw.μ_metamere[m] + shift
        p0,p1,p2 = ordinal3_probs(μ_obs[m], draw.σ)
        p[m,1] = p0; p[m,2] = p1; p[m,3] = p2
    end
    return (μ=μ_obs, p=p, shift=eff.f53s[genotype])
end

function simulate_single_embryo(draw, genotype::Integer)
    pred = predict_metameres(draw, genotype)
    y = [rand(Categorical(pred.p[m,:])) - 1 for m in 1:draw.M] 
    return (y=y, p=pred.p, μ=pred.μ, shift=pred.shift, σ=draw.σ)
end

function posterior_estimates(chn_pooled,M)
    nsamples = length(vec(chn_pooled[:lp]))

    μ_WT = zeros(Float64, nsamples,M)
    σ = zeros(Float64, nsamples)
    μ_mut = zeros(Float64, nsamples,M)
    
    for idx = 1:nsamples
        draw = pick_draw(chn_pooled, which=idx)
        siml = predict_metameres(draw, 1)
        μ_WT[idx,:] = siml.μ
        σ[idx] = siml.σ
        siml = predict_metameres(draw, 3)
        μ_mut[idx,:] = siml.μ
    end
    quants = [0.025, 0.5, 0.975]
    μ_WT_range = [quantile(μ_WT[:,m], quants) for m in 1:M] |> hcat |> transpose
    σ_range = quantile(σ, quants)
    μ_mut_range = [quantile(μ_mut[:,m], quants) for m in 1:M] |> hcat |> transpose
    
    return (μ_WT=μ_WT_range, σ=σ_range, μ_mut=μ_mut_range)
end

function p_vals_compute(mp_val,gtype,M)

    p_vals = zeros(M, 3)
    μ_vals = [mp_val.values[Symbol("μ_metamere[$m]")] for m in 1:M]
    Bnl_shift = mp_val.values[:Bnl_shift]
    F53S_shift = mp_val.values[:F53S_shift]
    σ = mp_val.values[:σ]
    eff = genotype_effects(Bnl_shift, F53S_shift)
    for m in 1:M
        μ_obs = μ_vals[m] + genotype_shift(gtype, eff)
        
        p0, p1, p2 = ordinal3_probs( μ_obs, σ)
        p_vals[m,1] = p0; p_vals[m,2] = p1; p_vals[m,3] = p2
    end
    return p_vals
end


function best_sample_init(chn)
    lp = vec(Array(chn[:lp]))
    lin = argmax(lp)
    vals = Array(chn)          
    θvec = vec(vals[lin, :])   
    
    return θvec, lp[lin]
end


function map_from_chain(model, chn; solver=LBFGS(), solve_kwargs...)
    init, lp_best = best_sample_init(chn)
    map_est = maximum_a_posteriori(model, solver; initial_params=init, maxiters = 20_000,
    abstol = 1e-10,
    reltol = 1e-12, solve_kwargs...)
    return map_est, lp_best
end

function p_val_error_bars(chn,M)
    nsamples = length(vec(chn[:lp]))
    p_vals_tot = []
    for g = 1:4
        p_vals = zeros(nsamples, M, 3)
        for idx = 1:nsamples

            draw = pick_draw(chn, which=idx)
            siml  = predict_metameres(draw, g)
            p_vals[idx, :, :] .= siml.p
        end
        p_quantile = zeros(M, 3, 2)  # M x 3 x 2 (lower and upper)
        for l in 1:3, m in 1:M
            p_quantile[m, l, :] .= quantile(p_vals[:,m,l], [0.025, 0.975])
        end
        push!(p_vals_tot, p_quantile)
    end
    return p_vals_tot
end

function empirical_freq(y::Vector{Int}, m_idx::Vector{Int}; M=maximum(m_idx), K=3)
    counts = zeros(Int, M, K)
    for i in eachindex(y)
        counts[m_idx[i], y[i]+1] += 1  # y is 0/1/2
    end
    freq = counts ./ sum(counts, dims=2)
    return freq
end

####################################################################
# Posterior predictive checks: metamere-level defect totals
#
# Requires:
#   ordinal3_probs
#   genotype_effects
#   genotype_shift
#
# These functions expect dat.y to be on the original 0/1/2 scale,
# not dat.y .+ 1.

# Choose how a y value is converted into "defects".
# Default below treats y itself as the count being summed.
DEFECT_SCORE = y -> (y == 1 ? 0.0 : 1.0)   # count only the "1 terminal" category as a non-defect

function extract_two_thresh_draws_for_ppc(chn::Chains, M::Integer)
    σ = vec(Array(chn[:σ]))
    Bnl_shift = vec(Array(chn[:Bnl_shift]))
    F53S_shift = vec(Array(chn[:F53S_shift]))

    μ_metamere = hcat([
        vec(Array(chn["μ_metamere[$m]"]))
        for m in 1:M
    ]...)

    @assert length(σ) == length(Bnl_shift) == length(F53S_shift) == size(μ_metamere, 1)

    return (
        μ_metamere = μ_metamere,
        σ = σ,
        Bnl_shift = Bnl_shift,
        F53S_shift = F53S_shift,
        M = M,
        S = length(σ),
    )
end


function simulate_two_thresh_y(draws,
                               s::Integer,
                               m_idx::AbstractVector{<:Integer},
                               g_idx::AbstractVector{<:Integer};
                               rng=Random.GLOBAL_RNG)

    @assert length(m_idx) == length(g_idx)

    ysim = Vector{Int}(undef, length(m_idx))

    eff = genotype_effects(draws.Bnl_shift[s], draws.F53S_shift[s])
    σ = draws.σ[s]

    @inbounds for i in eachindex(m_idx)
        μ_obs = draws.μ_metamere[s, m_idx[i]] + genotype_shift(g_idx[i], eff)

        p0, p1, p2 = ordinal3_probs(μ_obs, σ)

        # Guard against tiny numerical negatives / sums like 0.999999999.
        p = max.(Float64[p0, p1, p2], 0.0)
        p ./= sum(p)

        # Model categories are 1/2/3, but dat.y is 0/1/2.
        ysim[i] = rand(rng, Categorical(p)) - 1
    end

    return ysim
end


function metamere_defect_counts(y::AbstractVector{<:Real},
                                m_idx::AbstractVector{<:Integer};
                                M::Integer=maximum(m_idx),
                                defect_score=identity)

    @assert length(y) == length(m_idx)

    counts = zeros(Float64, M)

    @inbounds for i in eachindex(y)
        counts[m_idx[i]] += Float64(defect_score(y[i]))
    end

    return counts
end


function observed_metamere_defect_stats(dat; defect_score=identity)
    counts = metamere_defect_counts(
        dat.y,
        dat.m_idx;
        M=dat.M,
        defect_score=defect_score,
    )

    return (
        counts = counts,
        max_defects = maximum(counts),
        min_defects = minimum(counts),
        avg_defects = mean(counts),
    )
end


function posterior_metamere_defect_stats(chn::Chains,
                                         dat;
                                         ndraws=nothing,
                                         defect_score=identity,
                                         rng=Random.GLOBAL_RNG)

    draws = extract_two_thresh_draws_for_ppc(chn, dat.M)
    Sfull = draws.S

    draw_ids = if ndraws === nothing || ndraws >= Sfull
        collect(1:Sfull)
    else
        randperm(rng, Sfull)[1:ndraws]
    end

    S = length(draw_ids)

    samples_counts = Matrix{Float64}(undef, S, dat.M)
    samples_max = Vector{Float64}(undef, S)
    samples_min = Vector{Float64}(undef, S)
    samples_avg = Vector{Float64}(undef, S)

    for (j, s) in pairs(draw_ids)
        ysim = simulate_two_thresh_y(draws, s, dat.m_idx, dat.g_idx; rng=rng)

        counts = metamere_defect_counts(
            ysim,
            dat.m_idx;
            M=dat.M,
            defect_score=defect_score,
        )

        samples_counts[j, :] .= counts
        samples_max[j] = maximum(counts)
        samples_min[j] = minimum(counts)
        samples_avg[j] = mean(counts)
    end

    return (
        samples_counts = samples_counts,
        samples_max = samples_max,
        samples_min = samples_min,
        samples_avg = samples_avg,
        draw_ids = draw_ids,
    )
end


function ppc_metamere_summary(obs, ppc)
    function one_row(samples, observed)
        p_ge = mean(samples .>= observed)
        p_le = mean(samples .<= observed)

        return (
            observed = observed,
            pred_mean = mean(samples),
            pred_q025 = quantile(samples, 0.025),
            pred_q50 = quantile(samples, 0.5),
            pred_q975 = quantile(samples, 0.975),
            p_ge = p_ge,
            p_le = p_le,
            tail_prob = min(p_ge, p_le),
        )
    end

    rows = [
        one_row(ppc.samples_max, obs.max_defects),
        one_row(ppc.samples_min, obs.min_defects),
        one_row(ppc.samples_avg, obs.avg_defects),
    ]

    return DataFrame(
        stat = [
            "max defects/metamere",
            "min defects/metamere",
            "avg defects/metamere",
        ],
        observed = [r.observed for r in rows],
        pred_mean = [r.pred_mean for r in rows],
        pred_q025 = [r.pred_q025 for r in rows],
        pred_q50 = [r.pred_q50 for r in rows],
        pred_q975 = [r.pred_q975 for r in rows],
        p_ge = [r.p_ge for r in rows],
        p_le = [r.p_le for r in rows],
        tail_prob = [r.tail_prob for r in rows],
    )
end


function ppc_hist_stat(samples::AbstractVector{<:Real},
                       observed::Real;
                       xlabel::String,
                       title::String)

    integer_like =
        all(x -> abs(x - round(x)) < 1e-10, samples) &&
        abs(observed - round(observed)) < 1e-10

    bins = if integer_like 
        (floor(Int, min(minimum(samples), observed)) - 0.5):(ceil(Int, max(maximum(samples), observed)) + 0.5)
    else
        30
    end

    p = histogram(
        samples;
        xlabel=xlabel,
        ylabel="Density",
        title=title,
        lt=:stephist,
        grid=false,
        normalize=true,
        bins=bins,
        label="Model",
    )

    vline!(p, [observed]; label="Data")

    tail_prob = min(mean(samples .>= observed), mean(samples .<= observed))

    annotate!(
        p,
        (
            observed,
            0.05,
            text("Tail prob: $(round(tail_prob, digits=3))", :left, 10),
        ),
    )

    return p
end


function p_val_emp_error_bars(chn,M,nexp)
    nsamples = length(vec(chn[:lp]))
    p_vals_tot = []
    for g = 1:4
        p_vals = zeros(nsamples, M, 3)
        for idx = 1:nsamples
            draw = pick_draw(chn, which=idx)
            siml  = predict_metameres(draw, g)
            for m in 1:M
                qsample = rand(Categorical(siml.p[m,:]), nexp[g])
                for i = 1:3
                    p_vals[idx, m, i] =  sum(qsample .== i) / nexp[g]
                end
            end

        end
        p_quantile = zeros(M, 3, 2)  # M x 3 x 2 (lower and upper)
        for l in 1:3, m in 1:M
            p_quantile[m, l, :] .= quantile(p_vals[:,m,l], [0.025, 0.975])
        end
        push!(p_vals_tot, p_quantile)
    end
    return p_vals_tot
end