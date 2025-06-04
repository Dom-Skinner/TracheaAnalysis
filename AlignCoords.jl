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

df_Terminal = filter(row -> occursin("TerminalCell", row.Cell_Type), df)
df_Fusion = filter(row -> occursin("FusionCell", row.Cell_Type), df)
unique_pair = [findfirst(unique(df_Terminal.unique_pair) .== i) for i in df_Terminal.unique_pair] # unique pair scaled to 1:n

tlims = (-50,290)
# ==================================================================================================
# First show the data unaligned - as is.
# ==================================================================================================


p1 = plot(df_Terminal.t,df_Terminal.elongation,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="elongation")
p2 = plot(df_Terminal.t,df_Terminal.sphericity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="sphericity")
p3 = plot(df_Terminal.t,df_Terminal.chords_nonzero,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="chords_nonzero")
p4 = plot(df_Terminal.t,df_Terminal.PC1,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="PC1")
p5 = plot(df_Terminal.t,df_Terminal.Normalized_DSRF_Intensity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="DSRF Intensity")
p6 = plot(df_Terminal.t,df_Terminal.Normalized_DSRF_Intensity,group=df_Terminal.unique_id,xlabel="Time",ylabel="DSRF Intensity")
plot(p1,p2,p3,p4,p5,p6,layout=(2,3),size=(800,600))
savefig("plots/Unaligned_data.pdf")

# ==================================================================================================
# First, let us align to the PC1 coord
# ==================================================================================================
shift_PC,p = joint_tanh_shift_fit(df_Terminal.t,unique_pair,df_Terminal.PC1)
t_shifted_PC = df_Terminal.t .- shift_PC[unique_pair]

f = (x-> p[1]*tanh.(p[2]*(x-p[3]))-p[4])
plot(t_shifted_PC,df_Terminal.PC1,group=df_Terminal.unique_cell,xlabel="Time",ylabel="PC1")
plot!(-70:290, f.(-70:290), label="Fitted PC1", color=:red, lw=2)


p1 = plot(t_shifted_PC,df_Terminal.elongation,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="elongation")
p2 = plot(t_shifted_PC,df_Terminal.sphericity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="sphericity")
p3 = plot(t_shifted_PC,df_Terminal.chords_nonzero,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="chords_nonzero")
p4 = plot(t_shifted_PC,df_Terminal.PC1,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="PC1")
p5 = plot(t_shifted_PC,df_Terminal.Normalized_DSRF_Intensity,group=df_Terminal.unique_id,legend=false,xlabel="Time",ylabel="DSRF Intensity")
p6 = plot(t_shifted_PC,df_Terminal.Normalized_DSRF_Intensity,group=df_Terminal.unique_id,xlabel="Time",ylabel="DSRF Intensity")
plot(p1,p2,p3,p4,p5,p6,layout=(2,3),size=(800,600))
savefig("plots/Aligned_to_PC_data.pdf")


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
p3 = plot(t_shifted_DSRF,df_Terminal.chords_nonzero,group=df_Terminal.unique_cell,legend=false,xlabel="Time",ylabel="chords_nonzero")
p4 = plot(t_shifted_DSRF,df_Terminal.PC1,group=df_Terminal.unique_cell,legend=false,xlabel="Time",ylabel="PC1")
p5 = plot(t_shifted_DSRF,intensity_shifted,group=df_Terminal.unique_pair,legend=false,xlabel="Time",ylabel="DSRF Intensity")
p6 = plot(t_shifted_DSRF,intensity_shifted,group=df_Terminal.unique_id,xlabel="Time",ylabel="DSRF Intensity")
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