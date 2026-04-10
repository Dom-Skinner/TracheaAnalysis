


### LOO computation

function log_meanexp(v)
    m = maximum(v)
    return m + log(mean(exp.(v .- m)))
end

function pooled_model_LL(x, l_idx, mu, sigma_l, c, b_raw)
    # Priors (weakly informative on logit scale)
    LL = 0.0
  
    b      = b_raw .* sigma_l
    b = b .- mean(b)                        # sum-to-zero for identifiability

    N = size(x,1)
    #erf.(1e-6:0.01:1)
    for i in 1:N
        LL += logpdf(Bernoulli( invlogit(mu + b[l_idx[i]] + c)), x[i,1])
        LL += logpdf(Bernoulli( invlogit(mu + b[l_idx[i]] - c)), x[i,2])
    end
    return LL
end

function LOO_likelihood_pooled(chn_pooled, dat)
    LL = zeros(size(chn_pooled,1),size(chn_pooled,3))
    for i = 1:size(LL,1)
        for k = 1:size(LL,2)
            LL[i,k] = pooled_model_LL(hcat(dat.left,dat.right), dat.l_idx,chn_pooled.value.data[i,1,k],
                chn_pooled.value.data[i,2,k],
                chn_pooled.value.data[i,3,k],
                hcat([chn_pooled.value.data[i,4+j,k] for j = 0:(dat.L-1)]...))
        end
    end
    return log_meanexp(LL)
end

function compute_LOO_score_pooled(dat)
    n_draws   = 500
    n_chains  = 4
    LOO_score = 0.0
    for j in dat.larvae
        I = findall(dat.l_idx .== j)
        larvae_data = (left=dat.left[I],right=dat.right[I], l_idx=dat.l_idx[I],L=dat.L)
        Ir = setdiff(1:size(dat.l_idx,1), I)
        chn_leaveoneout = sample(pooled_model(hcat(dat.left[Ir],dat.right[Ir]), dat.l_idx[Ir], dat.L), NUTS(),  MCMCThreads(), n_draws,n_chains)
        LL_val = LOO_likelihood_pooled(chn_leaveoneout, larvae_data)
        LOO_score += LL_val
        println("LOO progress: larva ", j, " done. Value: ", LL_val)
    end
    return LOO_score
end


function hier_model_LL(x, l_idx, m_idx, mu, tau,sigma_l, c,alpha_raw, b_raw)
    # Priors (weakly informative on logit scale)
    LL = 0.0
  
    b      = b_raw .* sigma_l
    b = b .- mean(b)                        # sum-to-zero for identifiability

    alpha = alpha_raw .* tau
    alpha = alpha .- mean(alpha)    
    N = size(x,1)
    #erf.(1e-6:0.01:1)
    for i in 1:N
        LL += logpdf(Bernoulli( invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]] + c)), x[i,1])
        LL += logpdf(Bernoulli( invlogit(mu + alpha[m_idx[i]] + b[l_idx[i]] - c)), x[i,2])
    end
    return LL
end

function LOO_likelihood_hier(chn_pooled, dat)
    LL = zeros(size(chn_pooled,1),size(chn_pooled,3))
    for i = 1:size(LL,1)
        for k = 1:size(LL,2)
            mu = chn_pooled.value.data[i,1,k]
            tau = chn_pooled.value.data[i,2,k]
            sigma_l = chn_pooled.value.data[i,3,k]
            alpha_raw = chn_pooled.value.data[i,4:(4+dat.M-1),k]
            b_raw = chn_pooled.value.data[i,(4+dat.M):(4+dat.M+dat.L-1),k]
            c = chn_pooled.value.data[i,4+dat.M+dat.L,k]
            LL[i,k] = hier_model_LL(hcat(dat.left,dat.right), dat.l_idx, dat.m_idx,
                mu, tau, sigma_l, c,
                alpha_raw,
                b_raw)
        end
    end
    return log_meanexp(LL)
end

function compute_LOO_score_hier(dat)
    n_draws   = 500
    n_chains  = 4
    LOO_score = 0.0
    for j in dat.larvae
        I = findall(dat.l_idx .== j)
        larvae_data = (left=dat.left[I],right=dat.right[I], l_idx=dat.l_idx[I],L=dat.L, m_idx = dat.m_idx[I], M = dat.M)
        Ir = setdiff(1:size(dat.l_idx,1), I)
        chn_leaveoneout = sample(hier_model(hcat(dat.left[Ir],dat.right[Ir]), dat.l_idx[Ir], dat.L, dat.m_idx[Ir], dat.M), NUTS(),  MCMCThreads(), n_draws,n_chains)
        LL_val = LOO_likelihood_hier(chn_leaveoneout, larvae_data)
        LOO_score += LL_val
        println("LOO progress: larva ", j, " done. Value: ", LL_val)
    end
    return LOO_score
end