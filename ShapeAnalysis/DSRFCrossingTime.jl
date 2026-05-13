import Pkg; Pkg.activate(@__DIR__)
cd(@__DIR__)
using Statistics, StatsBase, Plots, CSV, DataFrames

df = CSV.read("data/processed/data_with_alignment.csv", DataFrame)

df_Terminal = filter(row -> !occursin("FusionCell", row.Cell_Type), df)

mutant_filter = .!occursin.("_bnl_", df_Terminal.exp_id)
failure_filter = [id ∈ ["251001_DSRFsfGFP_bnl_Data_Tr8","251005_DSRFsfGFP_bnl_Data_Tr7",
                         "251006_DSRFsfGFP_bnl_Data_Tr8","251008_DSRFsfGFP_bnl_Data_Tr7",
                         "251008_DSRFsfGFP_bnl_Data_Tr8"] for id in df_Terminal.unique_id]

df_cross = copy(df_Terminal)
df_cross.is_WT               = mutant_filter
df_cross.is_successful_mutant = occursin.("_bnl_", df_cross.exp_id) .& .!failure_filter


function first_crossing_time(t, y, T)
    good = [
        !ismissing(ti) && !ismissing(yi) && isfinite(Float64(ti)) && isfinite(Float64(yi))
        for (ti, yi) in zip(t, y)
    ]
    t = Float64.(t[good])
    y = Float64.(y[good])
    length(t) == 0 && return missing
    ord = sortperm(t)
    t = t[ord]
    y = y[ord]
    y[1] >= T && return t[1]
    for i in 2:length(y)
        if y[i-1] < T && y[i] >= T
            return t[i-1] + (T - y[i-1]) * (t[i] - t[i-1]) / (y[i] - y[i-1])
        end
    end
    return missing
end

function crossing_table_for_T(df_cross, T; time_col = :t_shifted_PC)
    rows = NamedTuple[]
    for g in groupby(df_cross, :unique_id)
        is_WT               = g.is_WT[1]
        is_successful_mutant = g.is_successful_mutant[1]
        if is_WT || is_successful_mutant
            tcross = first_crossing_time(getproperty(g, time_col),
                                         g.Normalized_DSRF_Intensity_scaled, T)
            if !ismissing(tcross)
                push!(rows, (unique_id = g.unique_id[1],
                              genotype  = is_WT ? "WT" : "successful mutant",
                              T         = T,
                              crossing_time = tcross))
            end
        end
    end
    return DataFrame(rows)
end

function crossing_summary(df_cross, Tvals; time_col = :t_shifted_PC)
    df_crossings = vcat([crossing_table_for_T(df_cross, Tval; time_col) for Tval in Tvals]...)
    return combine(groupby(df_crossings, [:genotype, :T]),
        :crossing_time => mean => :mean_crossing_time,
        :crossing_time => sem  => :std_crossing_time,
        nrow => :n)
end

function crossing_plot(df_summary; kwargs...)
    p = plot(; xlabel = "Threshold value", ylabel = "First crossing time",
               legend = :topleft, grid = false, kwargs...)
    for genotype in ["WT", "successful mutant"]
        s = sort(df_summary[df_summary.genotype .== genotype, :], :T)
        plot!(p, s.T, s.mean_crossing_time, ribbon = s.std_crossing_time,
              lw = 3, fillalpha = 0.2, label = genotype * " mean ± se")
    end
    return p
end


included_maxima = Float64[]
for g in groupby(df_cross, :unique_id)
    if g.is_WT[1] || g.is_successful_mutant[1]
        vals = [Float64(v) for v in g.Normalized_DSRF_Intensity_scaled
                if !ismissing(v) && isfinite(Float64(v))]
        !isempty(vals) && push!(included_maxima, maximum(vals))
    end
end

max_T_all_cross = minimum(included_maxima)
Tvals = range(7, 0.95 * max_T_all_cross, length = 50)

p_aligned = crossing_plot(crossing_summary(df_cross, Tvals);               title = "Time aligned")
p_raw     = crossing_plot(crossing_summary(df_cross, Tvals; time_col = :t); title = "Raw time")
savefig(plot(p_aligned, p_raw, layout = (1, 2), size = (900, 400)),
        "plots/DSRF_threshold_crossing_time_vs_T.pdf")