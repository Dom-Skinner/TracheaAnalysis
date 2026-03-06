using DataFrames, CSV, MultivariateStats, LinearAlgebra, StatsBase

function prepare_df(file_name)
    df = CSV.read(file_name, DataFrame)
    if "Normalized_DSRF_Intensity" ∉ names(df)
        df.Normalized_DSRF_Intensity = NaN*zeros(size(df,1))
    end
    if "Normalized_Pnt_Intensity" ∉ names(df)
        df.Normalized_Pnt_Intensity = NaN*zeros(size(df,1))
    end
    exp_id = string(split(file_name,"/")[3][1:end-4])
    df.exp_id = fill(exp_id,size(df,1))

    parse_fun = x-> x=="N/A" ? NaN : (x isa Number ? x : parse(Float64, x))
    df.x = parse_fun.(df.x)
    df.y = parse_fun.(df.y)
    df.z = parse_fun.(df.z)
    df.Normalized_DSRF_Intensity = parse_fun.(df.Normalized_DSRF_Intensity)
    df.Normalized_Pnt_Intensity = parse_fun.(df.Normalized_Pnt_Intensity)
    return df
end    


function create_adjusted_xy!(df)
    df[!,:x_adj] = copy(df.x)
    df[!,:y_adj] = copy(df.y)

    for exp_id in unique(df.exp_id)

        
        vecs_arr = []
        mean_arr = []

        for s  in ["7", "8"]
            cond = occursin.(s,df.Cell_Type) .& (df.exp_id .== exp_id) .& (.!isnan.(df.x)) .& (.!isnan.(df.y))    

            X = hcat(df.x[cond],df.y[cond])
            t_local = df.t[cond]

            pca_fit = MultivariateStats.fit(PCA, X'; maxoutdim=2, pratio=1.0)
            
            Y = MultivariateStats.transform(pca_fit, X')
            vecs = copy(pca_fit.proj)
            if det(vecs) < 0 # ensure PC is rotation 
                Y[1,:] .*= -1
                vecs[1,:] .*= -1
            end

            sign_correct = (cor(Y[1,:],t_local) > 0) ? 1 : -1 
            Y[:,] .*= sign_correct
            vecs[:,] .*= sign_correct

            df.x_adj[cond] = Y[1,:]
            df.y_adj[cond] = Y[2,:]

            push!(vecs_arr,vecs)
            push!(mean_arr,pca_fit.mean)
        end


        y_sign_correct = 1
        if vecs_arr[1][2,:]' *(mean_arr[1] .- mean_arr[2]) > 0
            y_sign_correct = -1
        end
        
        for s  in ["7", "8"]
            cond =  occursin.(s,df.Cell_Type) .& (df.exp_id .== exp_id) .& (.!isnan.(df.x)) .& (.!isnan.(df.y))    
            df.y_adj[cond] .*= y_sign_correct
        end
    end
end


function add_unique_cell!(df)
    u = unique(df.Cell_Type .* df.exp_id)
    unique_cell = [findfirst( df.Cell_Type[i]*df.exp_id[i] .== u) for i in 1:size(df,1)]
    df.unique_cell = unique_cell
end  

function add_unique_pair!(df)
    segment = [split.(f,"_")[1] for f in df.Cell_Type]
    u = unique(segment .* df.exp_id)
    unique_pair = [findfirst( segment[i]*df.exp_id[i] .== u) for i in 1:size(df,1)]
    df.unique_pair = unique_pair
end


function add_Δ!(df)
    t_pts = unique(df.t)
    pairs = unique(df.unique_pair)
    Δ = zeros(length(df.y_adj))

    for i = 1:length(pairs)
        idx = df.unique_pair .== pairs[i]
        for j = 1:length(t_pts)
            idx_t = df.t .== t_pts[j]
            idx_t = idx_t .& idx
            f = findall(idx_t)
            if length(f) == 2 # might not have this time point for all pairs
                Δ[f[1]] = df.y_adj[f[1]] .- df.y_adj[f[2]]
                Δ[f[2]] = df.y_adj[f[2]] .- df.y_adj[f[1]]
            end
        end
    end
    df.Delta_y = Δ
end

function add_flip_state!(df)
    Δs_total = zeros(Int64,size(df,1))

    for p = 1:maximum(df.unique_pair)
        p1,p2 = unique(df.unique_cell[findall(df.unique_pair.==p)])

        Δ =  df.Delta_y[df.unique_cell.==p1]
        Δs = findfirst(sign.(Δ[2:end]) .!= sign.(Δ[1:end-1]))
        if !isnothing(Δs)
            Δs_total[findall(df.unique_cell.==p1)[Δs]] = 1
            Δs_total[findall(df.unique_cell.==p2)[Δs]] = 1
        end
    end
   
    df.flip_state = Δs_total
end


function add_dist_state!(df)
    Δ_total = zeros(Float64,size(df,1))

    for p = 1:length(Δ_total)
        pts = findall((df.unique_pair.==df.unique_pair[p]) .& (df.t.==df.t[p]))
        @assert length(pts) == 2
        Δ_total[p] = sqrt((df.x_adj[pts[1]] .- df.x_adj[pts[2]]).^2 .+ (df.y_adj[pts[1]] .- df.y_adj[pts[2]]).^2)
    end
    
    df.nuclei_distance = Δ_total
end


function add_velocity!(df)
    # Compute average cell velocity
    df[!,:avg_vel] = 0*copy(df.x)
    
    for i  in unique(df.unique_cell)
        df_tmp = df[df.unique_cell .== i,:]
        @assert issorted(df_tmp.t)

        Δ = sqrt.((df_tmp.x_adj[2:end] .- df_tmp.x_adj[1:end-1]).^2 .+ (df_tmp.y_adj[2:end] .- df_tmp.y_adj[1:end-1]).^2)
        avg_vel = zeros(size(df_tmp.x_adj))
        dt = 8
        avg_vel[1] = Δ[1]/dt
        for i = 2:length(avg_vel)-1
            avg_vel[i] = (Δ[i-1] + Δ[i])/2/dt
        end
        avg_vel[end] = Δ[end]/dt
        df[df.unique_cell .== i,:avg_vel] .= avg_vel
    end

end    

function temporary_remove!(df)
    # remove missing frames and correct time
    deleteat!(df,findall((df.exp_id .== "250404_DSRFsfGFP_Data") .& (df.t .< 16)))
    df.t[df.exp_id .== "250404_DSRFsfGFP_Data"] .-= 16
end 

function remove_some_stalk!(df)
    # Delete stalk if terminal exists
    for exp_id in unique(df.exp_id)
        for s  in ["7", "8"]
            if any(occursin.(s*"_Terminal",df.Cell_Type) .& (df.exp_id .== exp_id))
                cond = occursin.(s*"_StalkCell",df.Cell_Type) .& (df.exp_id .== exp_id) 
                deleteat!(df,findall(cond)) 
            end
        end
    end    

end

function rescale_intensity!(df)
    exp_strs = unique(df.exp_id)
    df.Normalized_DSRF_Intensity_scaled = copy(df.Normalized_DSRF_Intensity)
    for e in exp_strs
        cond = occursin.(e,df.exp_id)
        q_scale = quantile(df.Normalized_DSRF_Intensity[cond],0.3)
        df.Normalized_DSRF_Intensity_scaled[cond] ./= q_scale
    end
end

function load_data()
    files=filter(x -> occursin(".csv",x),readdir("data/raw"))
    all_files = vcat(["data/raw/"*f for f in files]...)

    df = vcat([prepare_df(f) for f in all_files]...)
    remove_some_stalk!(df) 
    temporary_remove!(df) # a few frames are missing in the segmentation?
    create_adjusted_xy!(df)
    add_unique_cell!(df)
    add_unique_pair!(df)
    add_Δ!(df)
    add_flip_state!(df)
    add_velocity!(df)
    add_dist_state!(df)
    rescale_intensity!(df)
    return df
end


df = load_data()
mkpath("data/processed")
CSV.write("data/processed/combined_data.csv",df)


using Plots
exp_strs = unique(df.exp_id)
plt_scale = plot(xlabel="Intensity (scaled to 30th percentile)", ylabel="Cumulative Fraction")
plt_uscale = plot(xlabel="Intensity (scaled to max value)", ylabel="Cumulative Fraction")
for e in exp_strs
    df_one_exp =df[occursin.(e,df.exp_id),:]
    N = length(df_one_exp.Normalized_DSRF_Intensity)
    q_scale = quantile(df_one_exp.Normalized_DSRF_Intensity,0.3)
    plot!(plt_scale,sort(df_one_exp.Normalized_DSRF_Intensity)/q_scale,(1:N)/N, label=e)
    plot!(plt_uscale,sort(df_one_exp.Normalized_DSRF_Intensity),(1:N)/N, label=e)
    #histogram!(plt,df_one_exp.Normalized_DSRF_Intensity,bins=20,lt=:stephist,label=e)
end

plt_scale_zoom = deepcopy(plt_scale)
plt_uscale_zoom = deepcopy(plt_uscale)

plot!(plt_scale_zoom, xlims=(0,5), ylims=(0,1),legend=false)
plot!(plt_uscale_zoom, xlims=(0,0.15), ylims=(0,1),legend=false)
plot(plt_scale,plt_uscale, plt_scale_zoom, plt_uscale_zoom, layout=(2,2),size=(800,800))
savefig("plots/intensity_normalizing.pdf")