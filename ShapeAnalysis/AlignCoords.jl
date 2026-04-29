import Pkg; Pkg.activate(@__DIR__)
cd(@__DIR__)
using Optimization, OptimizationOptimisers, SciMLSensitivity,Zygote, ForwardDiff
using StatsBase
using Plots
using CSV, DataFrames
include("src/Utils.jl")

function core_obj_fun(time_shift,value_scaled,f)
    return sum(filter(!isnan,(f.(time_shift) .- value_scaled).^2))
end

function obj_function_scale_only(time,value,scale,up,f)
    value_scaled = value .* scale[up]
    return core_obj_fun(time,value_scaled,f)
end


function obj_function_shift_only(time,value,shift,up,f)
    return core_obj_fun(time .- shift[up],value,f)
end


function scale_fit_only(time_shifted,up,value,p)

    scale = ones(length(unique(up)))        
    f = (x-> p[1]*tanh.(p[2]*(x-p[3])) .- p[4])


    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> obj_function_scale_only(time_shifted,
    value, x,up,f), adtype)

    iter = Ref(0)  # mutable reference to track iteration count

    function print_callback(state, f)
        iter[] += 1
        if iter[] % 500 == 0
            println("Iteration $(iter[]): f = $f")
        end
        return false  # continue optimization
    end
   
    optprob = Optimization.OptimizationProblem(optf,copy(scale))
    result = Optimization.solve(optprob, Adam(),maxiters = 100_000,callback=print_callback)

    scale .= result.u
    
    
    return scale
end


function joint_tanh_shift_fit(time,up,value,p0=[3,0.02,100,0.0])
    
    t_shift = zeros(length(unique(up)))
    
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> obj_function_shift_only(time,value,x[1:end-4],
        up,(y-> x[end-3]*tanh.(x[end-2]*(y-x[end-1])) .- x[end]) ),
            adtype)
    optprob = Optimization.OptimizationProblem(optf,vcat(copy(t_shift),p0))

    iter = Ref(0)  # mutable reference to track iteration count

    function print_callback(state, f)
        iter[] += 1
        if iter[] % 500 == 0
            println("Iteration $(iter[]): f = $f")
        end
        return false  # continue optimization
    end
   
    result = Optimization.solve(optprob, Adam(),maxiters = 200_000,callback=print_callback)
    t_shift .= result.u[1:end-4]
    p = result.u[end-3:end]
    
    return t_shift,p

end


df = CSV.read("data/processed/data_with_PC.csv",DataFrame)

df_Terminal = filter(row -> .!occursin("FusionCell", row.Cell_Type), df)
df_Fusion = filter(row -> occursin("FusionCell", row.Cell_Type), df)
unique_pair = [findfirst(unique(df_Terminal.unique_pair) .== i) for i in df_Terminal.unique_pair] # unique pair scaled to 1:n

tlims = (-50,290)
# ==================================================================================================
# First show the data unaligned - as is.
# ==================================================================================================

p1 = plot(df_Terminal.t,df_Terminal.elongation,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="elongation")
p2 = plot(df_Terminal.t,df_Terminal.sphericity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="sphericity")
p3 = plot(df_Terminal.t,df_Terminal.volumeRatio,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="volume ratio")
p4 = plot(df_Terminal.t,df_Terminal.chords_nonzero,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="chords_nonzero")
p5 = plot(df_Terminal.t,df_Terminal.area,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="area")
p6 = plot(df_Terminal.t,df_Terminal.eccentricity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="eccentricity")
p7 = plot(df_Terminal.t,df_Terminal.perimeter,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="perimeter")
p8 = plot(df_Terminal.t,df_Terminal.circularity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="circularity")
p9 = plot(df_Terminal.t,df_Terminal.relative_volume,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="relative_volume")
p10 = plot(df_Terminal.t,df_Terminal.nuclei_distance,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="nuclei_distance")

p11 = plot(df_Terminal.t,df_Terminal.PC1,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="PC1")
p12 = plot(df_Terminal.t,df_Terminal.Normalized_DSRF_Intensity_scaled,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="DSRF Intensity")

plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,layout=(3,4),size=(1200,900))
savefig("plots/Unaligned_data.pdf")

# ==================================================================================================
# First, let us align to the PC1 coord
# ==================================================================================================
early_time = df_Terminal.t .< 175
shift_PC,p = joint_tanh_shift_fit(df_Terminal.t[early_time],unique_pair[early_time],df_Terminal.PC1[early_time])
t_shifted_PC = df_Terminal.t .- shift_PC[unique_pair]

f = (x-> p[1]*tanh.(p[2]*(x-p[3]))-p[4])
plot(t_shifted_PC,df_Terminal.PC1,group=df_Terminal.unique_cell,xlabel="Time",ylabel="PC1")
plot!(-70:290, f.(-70:290), label="Fitted PC1", color=:red, lw=2)


p1 = plot(t_shifted_PC,df_Terminal.elongation,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="elongation")
p2 = plot(t_shifted_PC,df_Terminal.sphericity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="sphericity")
p3 = plot(t_shifted_PC,df_Terminal.volumeRatio,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="volume ratio")
p4 = plot(t_shifted_PC,df_Terminal.chords_nonzero,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="chords_nonzero")
p5 = plot(t_shifted_PC,df_Terminal.area,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="area")
p6 = plot(t_shifted_PC,df_Terminal.eccentricity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="eccentricity")
p7 = plot(t_shifted_PC,df_Terminal.perimeter,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="perimeter")
p8 = plot(t_shifted_PC,df_Terminal.circularity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="circularity")
p9 = plot(t_shifted_PC,df_Terminal.relative_volume,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="relative_volume")
p10 = plot(t_shifted_PC,df_Terminal.nuclei_distance,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="nuclei_distance")
p11 = plot(t_shifted_PC,df_Terminal.PC1,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="PC1")
p12 = plot(t_shifted_PC,log.(df_Terminal.Normalized_DSRF_Intensity_scaled),group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="DSRF Intensity")
plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,layout=(3,4),size=(1200,900))
savefig("plots/Aligned_to_PC_data.pdf")



mutant_filter = .!occursin.("_bnl_",df_Terminal.exp_id)
failure_filter = [id ∈["251001_DSRFsfGFP_bnl_Data_Tr8","251005_DSRFsfGFP_bnl_Data_Tr7","251006_DSRFsfGFP_bnl_Data_Tr8","251008_DSRFsfGFP_bnl_Data_Tr7","251008_DSRFsfGFP_bnl_Data_Tr8"] for id in df_Terminal.unique_id]
remainder_filter = .!mutant_filter .& .!failure_filter
catagory_arr = mutant_filter .* 1 .+ failure_filter .* 2 .+ remainder_filter .* 3    
p1 =  plot(t_shifted_PC,df_Terminal.elongation,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="elongation",c=catagory_arr,msw=0.1,ms=5)
p2 = plot(t_shifted_PC,df_Terminal.sphericity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="sphericity",c=catagory_arr,msw=0.1,ms=5)
p3 = plot(t_shifted_PC,df_Terminal.volumeRatio,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="volume ratio",c=catagory_arr,msw=0.1,ms=5)
p4 = plot(t_shifted_PC,df_Terminal.chords_nonzero,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="chords_nonzero",c=catagory_arr,msw=0.1,ms=5)
p5 = plot(t_shifted_PC,df_Terminal.area,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="area",c=catagory_arr,msw=0.1,ms=5)
p6 = plot(t_shifted_PC,df_Terminal.eccentricity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="eccentricity",c=catagory_arr,msw=0.1,ms=5)
p7 = plot(t_shifted_PC,df_Terminal.perimeter,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="perimeter",c=catagory_arr,msw=0.1,ms=5)
p8 = plot(t_shifted_PC,df_Terminal.circularity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="circularity",c=catagory_arr,msw=0.1,ms=5)
p9 = plot(t_shifted_PC,df_Terminal.relative_volume,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="relative_volume",c=catagory_arr,msw=0.1,ms=5)
p10 = plot(t_shifted_PC,df_Terminal.nuclei_distance,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="nuclei_distance",c=catagory_arr,msw=0.1,ms=5)
p11 = plot(t_shifted_PC,df_Terminal.PC1,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="PC1",c=catagory_arr,msw=0.1,ms=5)
p12 = plot(t_shifted_PC,df_Terminal.Normalized_DSRF_Intensity_scaled,yaxis=:log10,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="DSRF Intensity",c=catagory_arr,msw=0.1,ms=5)
plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,layout=(3,4),size=(1200,900))
savefig("plots/Aligned_to_PC_data.pdf")





clim_vals = extrema(t_shifted_PC)   # (min, max)
scatter(df_Terminal.PC1[mutant_filter], df_Terminal.PC2[mutant_filter], 
        xlabel="PC1", ylabel="PC2", 
        aspect_ratio=:equal, 
        size=(800,800),clims=clim_vals,
        ms=5, c =:Greys, msc=:black,
        label=false,msw=0.5,zcolor=t_shifted_PC[mutant_filter])
scatter!(df_Terminal.PC1[failure_filter], df_Terminal.PC2[failure_filter],
        ms=5,msc=:black,clims=clim_vals,
        label=false,msw=0.5,zcolor=t_shifted_PC[failure_filter],c=:Reds)
scatter!(df_Terminal.PC1[remainder_filter], df_Terminal.PC2[remainder_filter],
    zcolor=t_shifted_PC[remainder_filter],
    ms=5,msc=:black,clims=clim_vals,
        label=false,msw=0.5,c=:Blues,grid=false)
savefig("plots/PC_full_tmp2.pdf")



p1 = plot(df_Terminal.t,df_Terminal.Normalized_DSRF_Intensity_scaled,yaxis=:log10,group=df_Terminal.unique_id,legend=false,
    xlabel="Unscaled time",ylabel="DSRF Intensity (30%)",c=catagory_arr,msw=0.1,ms=5)
p2 = plot(t_shifted_PC,df_Terminal.Normalized_DSRF_Intensity_scaled,yaxis=:log10,group=df_Terminal.unique_id,legend=false,
    xlabel="Scaled time",ylabel="DSRF Intensity (30%)",c=catagory_arr,msw=0.1,ms=5)
p3 = plot(df_Terminal.t,df_Terminal.Normalized_DSRF_Intensity,yaxis=:log10,group=df_Terminal.unique_id,legend=false,
xlabel="Unscaled time",ylabel="DSRF Intensity (max)",c=catagory_arr,msw=0.1,ms=5)
p4 = plot(t_shifted_PC,df_Terminal.Normalized_DSRF_Intensity,yaxis=:log10,group=df_Terminal.unique_id,legend=false,
xlabel="Scaled time",ylabel="DSRF Intensity (max)",c=catagory_arr,msw=0.1,ms=5)
plot(p1,p2,p3,p4,layout=(2,2),size=(800,600))
savefig("plots/DSRF_Intensity_in_time.pdf")

p1 = plot(t_shifted_PC,df_Terminal.Normalized_DSRF_Intensity_scaled,yaxis=:log10,group=df_Terminal.unique_id,legend=false,
    xlabel="Scaled time",ylabel="DSRF::sfGFP Intensity",c=catagory_arr,msw=0.1,ms=5,grid=false)
p2 = plot(t_shifted_PC,df_Terminal.Normalized_DSRF_Intensity_scaled,group=df_Terminal.unique_id,legend=false,
    xlabel="Scaled time",ylabel="DSRF::sfGFP Intensity",c=catagory_arr,msw=0.1,ms=5,grid=false)
plot(p1,p2,size=(800,400),layout=(1,2))
savefig("plots/DSRF_Scaled_Intensity_in_time.pdf")



p1 = plot(df_Terminal.t,df_Terminal.Normalized_DSRF_Intensity_scaled,yaxis=:log10,group=df_Terminal.unique_id,legend=false,
    xlabel="Unscaled Time (min)",ylabel="DSRF::sfGFP Intensity",c=catagory_arr,msw=0.1,ms=5,grid=false)
p2 = plot(t_shifted_PC,df_Terminal.Normalized_DSRF_Intensity_scaled,yaxis=:log10,group=df_Terminal.unique_id,legend=false,
    xlabel="Scaled Time (min)",ylabel="DSRF::sfGFP Intensity",c=catagory_arr,msw=0.1,ms=5,grid=false)
p3 = plot(df_Terminal.t,df_Terminal.PC1,group=df_Terminal.unique_id,legend=false,xlabel="Unscaled Time (min)",
    ylabel="Morphology PC1",c=catagory_arr,msw=0.1,ms=5,grid=false)
p4 = plot(t_shifted_PC,df_Terminal.PC1,group=df_Terminal.unique_id,legend=false,xlabel="Scaled Time (min)",
    ylabel="Morphology PC1",c=catagory_arr,msw=0.1,ms=5,grid=false)
plot(p1,p2,p3,p4,layout=(2,2),size=(800,600))
savefig("plots/DSRF_versus_Morphology.pdf")

# ==================================================================================================
# Now align to DSRF with no scaling
# ==================================================================================================
shift_DSRF,p = joint_tanh_shift_fit(df_Terminal.t,unique_pair,df_Terminal.Normalized_DSRF_Intensity)
t_shifted_DSRF = df_Terminal.t .- shift_DSRF[unique_pair]
f = (x-> p[1]*tanh.(p[2]*(x-p[3]))-p[4])

# now we can also rescale the DSRF (optional)
scale = scale_fit_only(t_shifted_DSRF,unique_pair,df_Terminal.Normalized_DSRF_Intensity,p)
intensity_shifted = df_Terminal.Normalized_DSRF_Intensity .* scale[unique_pair]

plot(t_shifted_DSRF,intensity_shifted,group=df_Terminal.unique_cell,xlabel="Time",ylabel="PC1")
#plot(t_shifted_DSRF,df_Terminal.Normalized_DSRF_Intensity,group=df_Terminal.unique_cell,xlabel="Time",ylabel="PC1")
plot!(-50:290, f.(-50:290), label="Fitted PC1", color=:red, lw=2)


p1 = plot(t_shifted_DSRF,df_Terminal.elongation,group=df_Terminal.unique_cell,legend=false,xlabel="Time",ylabel="elongation")
p2 = plot(t_shifted_DSRF,df_Terminal.sphericity,group=df_Terminal.unique_cell,legend=false,xlabel="Time",ylabel="sphericity")
p3 = plot(t_shifted_DSRF,df_Terminal.volumeRatio,group=df_Terminal.unique_cell,legend=false,xlabel="Time",ylabel="volume ratio")
p4 = plot(t_shifted_DSRF,df_Terminal.chords_nonzero,group=df_Terminal.unique_cell,legend=false,xlabel="Time",ylabel="chords_nonzero")
p5 = plot(t_shifted_DSRF,df_Terminal.PC1,group=df_Terminal.unique_cell,legend=false,xlabel="Time",ylabel="PC1")
p6 = plot(t_shifted_DSRF,intensity_shifted,group=df_Terminal.unique_pair,legend=false,xlabel="Time",ylabel="DSRF Intensity")
plot(p1,p2,p3,p4,p5,p6,layout=(2,3),size=(800,600))
savefig("plots/Aligned_by_DSRF_data.pdf")

# ==================================================================================================
# Compare alignments
# ==================================================================================================
scatter(shift_DSRF,shift_PC,aspect_ratio=true,
    xlabel="Time shift to align DSRF intensity", ylabel="Time shift to align morphology",
    title="Comparison of time shifts",
    label=false)
savefig("plots/AlignCoords.pdf")
# ==================================================================================================

# Save info
add_info!(df,t_shifted_DSRF,"t_shifted_DSRF")
add_info!(df,t_shifted_PC,"t_shifted_PC")
CSV.write("data/processed/data_with_alignment.csv", df)


using Statistics

# ================================================================================================
# First DSRF threshold crossing time, using t_shifted_PC
# ================================================================================================

df_cross = copy(df_Terminal)
df_cross.t_shifted_PC = t_shifted_PC

df_cross.is_WT = .!occursin.("_bnl_", df_cross.exp_id)
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
        if y[i - 1] < T && y[i] >= T
            # Linear interpolation between the two sampled time points
            return t[i - 1] + (T - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1])
        end
    end

    return missing
end

function crossing_table_for_T(df_cross, T)
    rows = NamedTuple[]

    for g in groupby(df_cross, :unique_id)
        is_WT = g.is_WT[1]
        is_successful_mutant = g.is_successful_mutant[1]

        if is_WT || is_successful_mutant
            tcross = first_crossing_time(
                g.t_shifted_PC,
                g.Normalized_DSRF_Intensity_scaled,
                T,
            )

            if !ismissing(tcross)
                push!(
                    rows,
                    (
                        unique_id = g.unique_id[1],
                        genotype = is_WT ? "WT" : "successful mutant",
                        T = T,
                        crossing_time = tcross,
                    ),
                )
            end
        end
    end

    return DataFrame(rows)
end


# ================================================================================================
# Sweep thresholds and plot mean ± sem ribbon
# ================================================================================================

# Choose thresholds guaranteed to be below the final max of every successful trajectory
included_maxima = Float64[]

for g in groupby(df_cross, :unique_id)
    if g.is_WT[1] || g.is_successful_mutant[1]
        vals = [
            Float64(v)
            for v in g.Normalized_DSRF_Intensity_scaled
            if !ismissing(v) && isfinite(Float64(v))
        ]
        !isempty(vals) && push!(included_maxima, maximum(vals))
    end
end

max_T_all_cross = minimum(included_maxima)

# Edit this range if desired
Tvals = range(7, 0.95 * max_T_all_cross, length=50)

df_crossings = vcat([crossing_table_for_T(df_cross, Tval) for Tval in Tvals]...)

df_summary = combine(
    groupby(df_crossings, [:genotype, :T]),
    :crossing_time => mean => :mean_crossing_time,
    :crossing_time => sem => :std_crossing_time,
    nrow => :n,
)

p = plot(
    xlabel = "Threshold value",
    ylabel = "First crossing time",
    legend = :topleft,
    grid = false,
)

for genotype in ["WT", "successful mutant"]
    s = df_summary[df_summary.genotype .== genotype, :]
    sort!(s, :T)

    plot!(
        p,
        s.T,
        s.mean_crossing_time,
        ribbon = s.std_crossing_time,
        lw = 3,
        fillalpha = 0.2,
        label = genotype * " mean ± se",
    )
end
display(p)

savefig(p, "plots/DSRF_threshold_crossing_time_vs_T.pdf")

