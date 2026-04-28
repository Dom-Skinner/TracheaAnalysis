using WriteVTK
using ReadVTK
using TiffImages

using LinearAlgebra
using MultivariateStats
using StatsBase

function find_PC(mask,v0)
    coords = findall(mask)
    x_coords = [c[1] for c in coords]
    y_coords = [c[2] for c in coords]
    z_coords = 9.39*[c[3] for c in coords]

    center_of_mass = [mean(x_coords),mean(y_coords),mean(z_coords)]
    dist_to_com = sqrt.((x_coords .- center_of_mass[1]).^2 .+ (y_coords .- center_of_mass[2]).^2 .+ (z_coords .- center_of_mass[3]).^2)
    idx_com = argmin(dist_to_com)

    X = hcat(x_coords, y_coords, z_coords) |> permutedims
    pca = fit(PCA, X; maxoutdim=1);
    if dot(pca.proj , v0) < 0 
        pca.proj .*=-1
    end
    Y = MultivariateStats.transform(pca, X)[1, :]
    idx0 = argmin(Y)


    return idx0,idx_com,pca
end


function lazy_convert(img)
    img_converted = zeros(Float64, size(img))
    for i in 1:size(img,1)
        for j in 1:size(img,2)
            for k in 1:size(img,3)
                if typeof(img[i,j,k].val) <: Number
                    img_converted[i,j,k] = Float64(img[i,j,k].val)
                else
                    img_converted[i,j,k] = Float64(img[i,j,k].val.i)
                end
            end
        end
    end
    img_converted = img_converted/minimum(filter(x->x>0,img_converted)) # normalize since values get read in weird sometimes
    return img_converted
end

function create_pair_dict(df_orig::DataFrame)
    df = copy(df_orig)
    df[!,:rows] = 1:size(df,1)
    dict = Dict{Int, Int}()
    groups = groupby(df, [:t, :unique_pair], sort=false)
    for group in groups
            rows = group.rows
            @assert length(rows) == 2
            dict[rows[1]] = rows[2]
            dict[rows[2]] = rows[1]
    end
    return dict
end



function add_info!(df,Y,str)
    ids = findall(occursin.("FusionCell", df.Cell_Type))
    pair_dict = create_pair_dict(df)
    df[!,str] = zeros(size(df,1))
    df[ids,str] = Y
    df[[pair_dict[i] for i in ids],str] = Y # add same value for both tip cells
end