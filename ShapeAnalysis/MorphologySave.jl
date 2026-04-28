using CSV, DataFrames
using MultivariateStats
using StatsBase
using ReadVTK
using HDF5


using LinearAlgebra

"""
Compute (area, perimeter4, circularity, eccentricity, centroid, eigvals)
for a binary mask (0/1). Perimeter is 4-neighbor pixel-edge perimeter.
"""
function shape_metrics(img_converted,id)

    Z = maximum(img_converted,dims=3)[:,:,1]
    mask = Z .== id

    H, W = size(mask)

    coords = findall(!iszero, mask)          # Vector{CartesianIndex{2}}
    A = length(coords)

    ys = Float64[c[1] for c in coords]       # row index = y
    xs = Float64[c[2] for c in coords]       # col index = x

    cy, cx = mean(ys), mean(xs)

    dx = xs .- cx
    dy = ys .- cy
    C = [mean(dx .* dx)  mean(dx .* dy);
            mean(dx .* dy)  mean(dy .* dy)]

    vals = eigen(Symmetric(C)).values
    λ1, λ2 = maximum(vals), minimum(vals)
    ecc = λ1 == 0 ? 0.0 : log( λ1 / λ2)

    # 4-neighbor pixel-edge perimeter
    P = sum(mask[y, x] != 0 && (y == 1 || mask[y-1, x] == 0) for y in 1:H, x in 1:W) +
        sum(mask[y, x] != 0 && (y == H || mask[y+1, x] == 0) for y in 1:H, x in 1:W) +
        sum(mask[y, x] != 0 && (x == 1 || mask[y, x-1] == 0) for y in 1:H, x in 1:W) +
        sum(mask[y, x] != 0 && (x == W || mask[y, x+1] == 0) for y in 1:H, x in 1:W)

    circ = P == 0 ? 0.0 : (4π*A) / (P^2)

    return (area=A, perimeter=P, circularity=circ, eccentricity=ecc)
end
    


function get_quantile(img_converted, id)
    coords = findall(img_converted .== id)

    x_coords = [c[1] for c in coords]
    y_coords = [c[2] for c in coords]
    z_coords = 9.39* [c[3] for c in coords]
    X = hcat(x_coords, y_coords, z_coords) |> permutedims

    pca = fit(PCA, X; maxoutdim=1);
    Yte = MultivariateStats.transform(pca, X)

    q = quantile(Yte[1,:], [0.05, 0.95])
    return abs(q[2]-q[1])
end



function get_relative_volumes(img_converted, id)
    coords = findall(img_converted .== id)

    x_coords = [c[1] for c in coords]
    y_coords = [c[2] for c in coords]
    z_coords = 9.39* [c[3] for c in coords]
    X = hcat(x_coords, y_coords, z_coords) |> permutedims

    pca = fit(PCA, X; maxoutdim=1);
    Yte = MultivariateStats.transform(pca, X)

    q = quantile(Yte[1,:], [0.05, 0.95])
    midpoint = (q[1]/3 + 2*q[2]/3) 
    L1 = sum( Yte[1,:] .>= midpoint ) / length( Yte[1,:])
    midpoint = (2*q[1]/3 + q[2]/3)
    L2 = sum( Yte[1,:] .<= midpoint ) / length( Yte[1,:])
    return max(L1,L2)
end

df = DataFrame("exp_id" => String[], "unique_id"=> String[], "t" => Float64[], 
    "elongation" => Float64[],  "branch" => String[], "sphericity" => Float64[],
     "volumeRatio" => Float64[],"chords_nonzero" => Float64[], 
     "relative_volume"=>Float64[],"chords_mean" => Float64[],
        "area" => Float64[], "perimeter" => Float64[], "circularity" => Float64[],
        "eccentricity" => Float64[])

dirs = filter(x->isdir("data/vtk/"*x),readdir("data/vtk/"))

for dir in dirs
    in_dir = "data/vtk/"*dir*"/"

    vtk_files = readdir(in_dir)

    exp_id = dir*"_Data"#format_folder_string(in_dir)
    
    println("Processing ", exp_id)


    ext_arr = zeros(length(vtk_files),2)
    relative_volumes = zeros(length(vtk_files),2)
    area = zeros(length(vtk_files),2)
    perimeter = zeros(length(vtk_files),2)
    circularity = zeros(length(vtk_files),2)
    eccentricity = zeros(length(vtk_files),2)
    back_arr = zeros(length(vtk_files),2)
    back_arr_normed = zeros(length(vtk_files),2)
    
    for i = 1:length(vtk_files)
        vtk = VTKFile(in_dir*vtk_files[i])

        cell_data = get_point_data(vtk)
        img_converted = get_data_reshaped(cell_data["raw"]) 

        for id = 1:2
            ext_arr[i,id] = get_quantile(img_converted, id)
            relative_volumes[i,id] = get_relative_volumes(img_converted, id)
            metrics = shape_metrics(img_converted, id)
            area[i,id] = metrics.area
            perimeter[i,id] = metrics.perimeter
            circularity[i,id] = metrics.circularity
            eccentricity[i,id] = metrics.eccentricity
        end
    end

    t = 1:length(vtk_files)
    sph = h5read("data/processed/sphericity/"*dir*".h5", "/sphericity")
    volumeRatio = h5read("data/processed/sphericity/"*dir*".h5", "/volumeRatio")

    chords_mean = h5read("data/processed/chords/"*dir*".h5", "/total_mean")
    chords_nonzero = h5read("data/processed/chords/"*dir*".h5", "/nonzero_fraction")

    
    new_rows = DataFrame("exp_id" => repeat([exp_id], 2*length(t)), 
                        "unique_id" => vcat(repeat([exp_id*"_Tr7"],length(t)),repeat( [exp_id*"_Tr8"], length(t))),
                        "branch" => vcat(repeat(["Tr7"],length(t)),repeat( ["Tr8"], length(t))),
                     "t" => repeat(t,2), 
                     "elongation" => vcat(ext_arr[:,1], ext_arr[:,2]), 
                     "relative_volume" => vcat(relative_volumes[:,1], relative_volumes[:,2]),
                     "volumeRatio" => vcat(volumeRatio[:,1], volumeRatio[:,2]),
                     "sphericity" => vcat(sph[:,1], sph[:,2]),
                    "chords_nonzero" => vcat(chords_nonzero[:,1], chords_nonzero[:,2]),
                    "chords_mean" => vcat(chords_mean[:,1], chords_mean[:,2]),
                    "area" => vcat(area[:,1], area[:,2]),
                    "perimeter" => vcat(perimeter[:,1], perimeter[:,2]),
                    "circularity" => vcat(circularity[:,1], circularity[:,2]),
                    "eccentricity" => vcat(eccentricity[:,1], eccentricity[:,2]) )

    # Append new rows to the existing DataFrame
    append!(df, new_rows)
    
end

#= 
# can test some plots to inspect features
using Plots
plot(df.t, df.volumeRatio, group=df.unique_id)
=#

CSV.write("data/processed/morphology.csv",df)
df = CSV.read("data/processed/morphology.csv", DataFrame)
# Now we need to merge this with the fluorescence data

df_full = CSV.read("data/processed/combined_data.csv", DataFrame)
df_full = filter(row -> row.exp_id in unique(df.exp_id), df_full)

df_full[!,:branch] = [tp[1:3] for tp in df_full.Cell_Type]
df.t .= (df.t .- 1) .* 8 # convert to match times 

combined = leftjoin(df_full, df, on = [:exp_id, :t, :branch])

@assert size(combined,1) == size(df_full,1)
@assert size(combined,1) == 2*size(df,1)

CSV.write("data/processed/data_with_morphology.csv", combined)
