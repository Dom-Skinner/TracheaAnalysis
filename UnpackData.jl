using DataFrames, CSV, MultivariateStats, LinearAlgebra

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

    for exp in unique(df.exp_id)
        for s  in ["7", "8"]

            cond_fusion = (df.Cell_Type .== "Tr"*s*"_FusionCell") .& 
                (df.exp_id .== exp) .& (.!isnan.(df.x)) .& (.!isnan.(df.y))    
            cond_terminal = (df.Cell_Type .== "Tr"*s*"_TerminalCell") .& 
                (df.exp_id .== exp) .& (.!isnan.(df.x)) .& (.!isnan.(df.y))

            cond = cond_fusion .| cond_terminal

            X = hcat(df.x[cond],df.y[cond])
            t_local = df.t[cond]
            fusion_local = findall(df.Cell_Type[cond] .== "Tr"*s*"_FusionCell")
            terminal_local = findall(df.Cell_Type[cond] .== "Tr"*s*"_TerminalCell")

            pca_fit = MultivariateStats.fit(PCA, X'; maxoutdim=2, pratio=1.0)
            
            Y = MultivariateStats.transform(pca_fit, X')
            if det(pca_fit.proj) < 0 # ensure PC is rotation 
                Y[1,:] .*= -1
            end

            sign_correct = (cor(Y[1,:],t_local) > 0) ? 1 : -1 
            Y[:,] .*= sign_correct

            Δ = (Y[2,fusion_local][argmax(t_local[fusion_local])] - Y[2,terminal_local][argmax(t_local[terminal_local])])
            y_sign_correct = (Δ > 0) ? 1 : -1 
            Y[2,:] .*= y_sign_correct

            df.x_adj[cond] = Y[1,:]
            df.y_adj[cond] = Y[2,:]
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

function load_data()
    files=filter(x -> occursin(".csv",x),readdir("data/raw"))
    all_files = vcat(["data/raw/"*f for f in files]...)

    df = vcat([prepare_df(f) for f in all_files]...)
    deleteat!(df,findall(occursin.("Stalk",df.Cell_Type))) # don't care about stalk cells for now
    temporary_remove!(df) # a few frames are missing in the segmentation?
    create_adjusted_xy!(df)
    add_unique_cell!(df)
    add_unique_pair!(df)
    add_Δ!(df)
    add_flip_state!(df)
    add_velocity!(df)
    return df
end


df = load_data()
mkpath("data/processed")
CSV.write("data/processed/combined_data.csv",df)

