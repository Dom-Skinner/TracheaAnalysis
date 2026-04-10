using CSV, DataFrames
using Distributions
using Random
using MCMCChains
using Statistics
using StatsBase
using StatsPlots

# --- Utilities ---
invlogit(x) = 1 / (1 + exp(-x))  # logistic link
drop_missing(x) = x[:,1:findlast(c -> !all(ismissing, c), eachcol(x))]


function extract_pyTag_data(file)
    df = CSV.read(file, DataFrame)
    y_tot = vcat(df.Tr2,df.Tr3,df.Tr4,df.Tr5,df.Tr6,df.Tr7,df.Tr8,df.Tr9)
    m_idx_tot = vcat(fill(1, size(df,1)), fill(2, size(df,1)), fill(3, size(df,1)), fill(4, size(df,1)), 
                    fill(5, size(df,1)), fill(6, size(df,1)), fill(7, size(df,1)), fill(8, size(df,1)))
    idx_not_missing = findall(x-> !ismissing(x) & !iszero(x), y_tot)
    y_tot = Float64.(y_tot[idx_not_missing])
    m_idx_tot = m_idx_tot[idx_not_missing]
    return y_tot, m_idx_tot
end

function extract_data_one_genotype(path)
    df = load_trachea_wide_csv(path)
    dat = build_model_data(df)

    total_counts_l = min.(dat.left,2)
    total_counts_r = min.(dat.right,2)

    y = vcat(total_counts_l, total_counts_r)
    m_idx = vcat(dat.m_idx, dat.m_idx)

    return y, m_idx
end

function load_trachea_wide_csv(path)
    df = CSV.read(path, DataFrame; header = false)
    df_left  = df[3:10,2:4:end]
    df_right = df[3:10,3:4:end]
    df_left = drop_missing(df_left)
    df_right = drop_missing(df_right)

    recs = DataFrame( larva=Int[], metamere=Int[], left=Int[], right=Int[])
    nlarvae = ncol(df_left)

    for l in 1:nlarvae
        for r in 1:nrow(df_left)
            push!(recs, ( l, 10-r, parse(Int,df_left[r,l]), parse(Int,df_right[r,l])))
        end
    end

    return recs
end

function build_model_data(df_long::DataFrame)
    df = deepcopy(df_long)
    df.z = map((l,r)->  l + r, df.left, df.right)

    # indices for larva and metamere
    larvae = sort(unique(df.larva))
    L = length(larvae)
    larva_to_idx = Dict(larvae[i] => i for i in eachindex(larvae))
    df.l_idx = [larva_to_idx[x] for x in df.larva]

    metameres = sort(unique(df.metamere))  # should be [2,3,...,9]
    M = length(metameres)
    metamere_to_idx = Dict(metameres[i] => i for i in eachindex(metameres))
    df.m_idx = [metamere_to_idx[x] for x in df.metamere]

    # pack vectors for modeling
    z     = Vector{Int}(df.z)
    left = Vector{Int}(df.left)
    right = Vector{Int}(df.right)
    l_idx = Vector{Int}(df.l_idx)
    m_idx = Vector{Int}(df.m_idx)

    return (z=z, left=left,right=right, l_idx=l_idx, m_idx=m_idx, L=L, M=M, larvae=larvae, metameres=metameres, df=df)
end




# -- helper: pull a dense draws-by-parameter matrix and ordered names
function _matrix_and_names(chn::Chains)
    A = Array(chn)                      # size = (iters, chains, params)
    nobj = names(chn)
    # try to get parameter names robustly across MCMCChains versions
    pnames = hasproperty(nobj, :parameters) ? String.(nobj.parameters) : String.(nobj)
    return A, pnames
end

# -- helper: map name -> column index
function _name_to_col(pnames::Vector{String})
    Dict(name => j for (j, name) in pairs(pnames))
end


_name_to_col_local(pnames::Vector{String}) = Dict(name => j for (j, name) in pairs(pnames))

function _col(idx::Dict{String,Int}, key::String)
    if haskey(idx, key)
        return idx[key]
    end
    error("Parameter column not found for '$key' (tried alternates $(alts))")
end


function posterior_mean_probs(chn::Chains, l_idx::Vector{Int}; left_side::Bool)
    A, pnames = _matrix_and_names(chn)
    idx = _name_to_col(pnames)

    # required scalars/vectors per draw
    mu       = A[:, idx["mu"]]
    sigma_l  = A[:, idx["sigma_l"]]

    L = maximum(l_idx)
    Braw = hcat([A[:, idx["b_raw[$ℓ]"]] for ℓ in 1:L]...)  # (S, L)
    B    = Braw .* sigma_l                                  # broadcast over columns

    # posterior-predictive mean prob for each observation
    N = length(l_idx)
    pbar = Vector{Float64}(undef, N)
    for i in 1:N
        # prob for this obs across draws
        p_draws = @views 1 ./(1 .+ exp.(-(mu .+ B[:, l_idx[i]])))
        pbar[i] = mean(p_draws)
    end
    return pbar
end

function posterior_mean_probs(chn::Chains, l_idx::Vector{Int},m_idx::Vector{Int}; left_side::Bool)
    A, pnames = _matrix_and_names(chn)
    idx = _name_to_col(pnames)

    # required scalars/vectors per draw
    mu       = A[:, idx["mu"]]
    tau   = A[:, idx["tau"]]
    sigma_l  = A[:, idx["sigma_l"]]

    L = maximum(l_idx)
    Braw = hcat([A[:, idx["b_raw[$ℓ]"]] for ℓ in 1:L]...)  # (S, L)
    B    = Braw .* sigma_l                                  # broadcast over columns

    M = maximum(m_idx)
    alpha_raw = hcat([A[:, idx["alpha_raw[$m]"]] for m in 1:M]...) 
    alpha = alpha_raw .* tau 

    # posterior-predictive mean prob for each observation
    N = length(l_idx)
    pbar = Vector{Float64}(undef, N)
    for i in 1:N
        # prob for this obs across draws
        p_draws = @views 1 ./(1 .+ exp.(-(mu .+ B[:, l_idx[i]] .+ alpha[:, m_idx[i]])))
        pbar[i] = mean(p_draws)
    end
    return pbar
end





function posterior_probs_full(chn::Chains, l_idx::Vector{Int},m_idx::Vector{Int})
    A, pnames = _matrix_and_names(chn)
    idx = _name_to_col(pnames)

    # required scalars/vectors per draw
    mu       = A[:, idx["mu"]]
    tau   = A[:, idx["tau"]]
    sigma_l  = A[:, idx["sigma_l"]]

    L = maximum(l_idx)
    Braw = hcat([A[:, idx["b_raw[$ℓ]"]] for ℓ in 1:L]...)  # (S, L)
    B    = Braw .* sigma_l                                  # broadcast over columns

    M = maximum(m_idx)
    alpha_raw = hcat([A[:, idx["alpha_raw[$m]"]] for m in 1:M]...) 
    alpha = alpha_raw .* tau

    # posterior-predictive mean prob for each observation
    N = length(l_idx)
    pbar = []
    for i in 1:N
        # prob for this obs across draws
        pl = 1 ./(1 .+ exp.(-(mu .+ B[:, l_idx[i]] .+ alpha[:, m_idx[i]])))
        pr = 1 ./(1 .+ exp.(-(mu .+ B[:, l_idx[i]] .+ alpha[:, m_idx[i]])))
        push!(pbar,hcat(pl,pr))   # each row: draws for left and right
    end
    return pbar
end



# -- randomized PIT for Bernoulli predictive prob p and observed y∈{0,1}
function randomized_pit(p::AbstractVector{<:Real}, y::AbstractVector{<:Integer})
    @assert length(p) == length(y)
    u = similar(p, Float64)
    for i in eachindex(p)
        pi = p[i]
        if y[i] == 0
            u[i] = rand() * (1 - pi)                # Uniform(0, 1-p)
        else
            u[i] = (1 - pi) + rand() * pi           # Uniform(1-p, 1)
        end
    end
    return u
end

# -- scoring rules (per-point and means)
struct Scores
    brier_per_point::Vector{Float64}
    log_per_point::Vector{Float64}
    mean_brier::Float64
    mean_logscore::Float64
end

function scores_for_binary(p::AbstractVector{<:Real}, y::AbstractVector{<:Integer})
    @assert length(p) == length(y)
    eps = 1e-12
    pp  = clamp.(p, eps, 1 - eps)
    brier = (pp .- y) .^ 2                                     # == CRPS for Bernoulli
    logpt = .-( y .* log.(pp) .+ (1 .- y) .* log.(1 .- pp) )   # negative log predictive
    Scores(brier, logpt, mean(brier), mean(logpt))
end


function reliability_curve(p::AbstractVector{<:Real},
    y::AbstractVector{<:Integer};
    nbins::Int=10)

    @assert length(p) == length(y)
    N = length(p)

    xs = Float64[]; ys = Float64[]; ws = Int[]

    # sort by predicted prob and slice into ~equal-size chunks
    ord = sortperm(p)
    # integer cut points: 0, floor(N/nbins), ..., N
    cuts = floor.(Int, (0:nbins) .* (N/nbins))
    cuts[end] = N
    for k in 1:nbins
        lo = cuts[k] + 1
        hi = cuts[k+1]
        if hi >= lo
            idx = ord[lo:hi]
            push!(xs, mean(p[idx]))
            push!(ys, mean(y[idx]))
            push!(ws, length(idx))
        end
    end

    return xs, ys, ws
end
