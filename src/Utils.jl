using WriteVTK
using ReadVTK
using TiffImages
using Graphs, SimpleWeightedGraphs

using LinearAlgebra
using MultivariateStats
using SparseArrays
using Arpack
using StatsBase


function prune_connections!(g)

    for e in collect(edges(g))
        if (e.weight != 1) && (e.weight != 9.39)
            nb1 = neighbors(g,e.src)
            if length(nb1) == 26
                continue
            end
            nb2 = neighbors(g,e.dst)
            if length(nb2) == 26
                continue
            end

            close_nb1 = filter(x-> g.weights[x,e.src] ∈ [1,9.39], nb1)
            close_nb2 = filter(x-> g.weights[x,e.dst] ∈ [1,9.39], nb2)
            nb = intersect(close_nb1,close_nb2)
            if length(nb) == 0
                rem_edge!(g,e)
            end
        end

    end
end


function make_connected!(g,t)
    if !is_connected(g)
        println("Graph is not connected")
        v = findall(t)
        x_coords = [c[1] for c in v]
        y_coords = [c[2] for c in v]
        z_coords = 9.39* [c[3] for c in v]
        X = hcat(x_coords, y_coords, z_coords) |> permutedims
        
        comps = connected_components(g)
        comp_sizes = [length(c) for c in comps]
        println("Size of components: ", comp_sizes)
        main_comp = argmax(comp_sizes)
        # The adding of edges is fairly uncontrolled/arbitrary. 
        # Effectively assumes 99.9% is part of the main component 
        # and what is left is small floating islands
        X_main  = X[:,comps[main_comp]]
        for i in setdiff(1:length(comps),main_comp)
            for j in comps[i]
                nn = comps[main_comp][argmin(sum((X_main .- X[:,j]).^2,dims=1)[1,:])]
                add_edge!(g,nn,j)
            end
        end
    end
end

function graph_construct(skeleton)
    a,b,c = 1.0,1.0,9.39
    skel = copy(skeleton)
    vertex_dict = Dict(findall(skel) .=> 1:length(findall(skel)))
    edges = []
    weights = []

    len = 0
    for i in 1:size(skel,1), j in 1:size(skel,2), k in 1:size(skel,3)
        if !skel[i,j,k]
            continue
        end

         for ii in maximum([1,i-1]):minimum([size(skel,1),i+1])
            for jj in maximum([1,j-1]):minimum([size(skel,2),j+1])
                for kk in maximum([1,k-1]):minimum([size(skel,3),k+1])
                    if any([ii,jj,kk] .!= [i,j,k]) && skel[ii,jj,kk]
                        push!(edges, [vertex_dict[CartesianIndex(i,j,k)],
                             vertex_dict[CartesianIndex(ii,jj,kk)]])
                        push!(weights, sqrt(a^2*(ii-i)^2 + b^2*(jj-j)^2 + c^2*(kk-k)^2))
                    end
                end
            end
        end
        skel[i,j,k] = false
                
    end
    g = SimpleWeightedGraph([ee[1] for ee in edges],[ee[2] for ee in edges] , Float64.(weights))
    prune_connections!(g)
    make_connected!(g,skeleton)
    return g,vertex_dict
end

function initial_tip_vector(img_converted,id)
    Vu = (img_converted .== id)
    coords = findall(Vu)
    x_coords = [c[1] for c in coords]
    y_coords = [c[2] for c in coords]
    z_coords = 9.39* [c[3] for c in coords]

    X = hcat(x_coords, y_coords, z_coords) |> permutedims
    M = MultivariateStats.fit(PCA, X; maxoutdim=1);
    v = copy(M.proj)
    @warn "You are assuming the z coords point down"
    return -v*sign(v[end])
end


function grid_euclidean_distance(mask,idx0)
    
    coords = findall(mask)
    

    x_coords = [c[1] for c in coords]
    y_coords = [c[2] for c in coords]
    z_coords = 9.39*[c[3] for c in coords]


    length_array = zeros(length(coords))
    
    for i = 1:length(coords)
        Δx = [x_coords[i]-x_coords[idx0],y_coords[i]-y_coords[idx0],z_coords[i]-z_coords[idx0]]
        length_array[i] = grid_distance(Δx)
    end
    return length_array
end

function grid_distance(Δx)
    scale = [1,1,9.39]
    n = [abs(Δx[1]),abs(Δx[2]),abs(Δx[3]/9.39)]

    i0 = argmin(n)
    i2 = argmax(n)
    i1 = setdiff(1:3,[i0,i2])[1]
    
    Δ = n[i0]*sqrt(sum(scale.^2))
    Δ += (n[i1]-n[i0])*sqrt(sum(scale[[i1,i2]].^2))

    return Δ + (n[i2] - n[i1])*scale[i2]
    
end



function backtrack_search(mask,idx0,pca)
    
    coords = findall(mask)
    g,_ = graph_construct(mask)

    paths = dijkstra_shortest_paths(SimpleGraph(g), idx0,g.weights)

    backtrack_array = zeros(nv(g))
    pathlength_array = zeros(nv(g))
    
    for i = 1:length(backtrack_array)
        path = enumerate_paths(paths, i)
        len, len_backtrack = calculate_length(path,coords,pca)
        backtrack_array[i] = len_backtrack
        pathlength_array[i] = len
    end

    return backtrack_array,pathlength_array
end


function calculate_length(path, coords, pca)
    # If backtrack is true, we only count the length of the path not in the direction of PC 1
    len_full = 0.0
    len_backtrack = 0.0
    for i = 2:length(path)
        dx = coords[path[i]][1] - coords[path[i-1]][1]
        dy = coords[path[i]][2] - coords[path[i-1]][2]
        dz = coords[path[i]][3] - coords[path[i-1]][3]
        dl = sqrt(dx^2 + dy^2 + (9.39 * dz)^2)
        
        len_full += dl
        
        projection = pca.proj[1] * dx + pca.proj[2] * dy + 9.39 * pca.proj[3] * dz
        if projection < 0
            len_backtrack += dl
        end

    end
    return len_full, len_backtrack
end


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
    @assert maximum(img_converted) == 2.0 # sanity check
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
    ids = findall(occursin.("TerminalCell", df.Cell_Type))
    pair_dict = create_pair_dict(df)
    df[!,str] = zeros(size(df,1))
    df[ids,str] = Y
    df[[pair_dict[i] for i in ids],str] = Y # add same value for both tip cells
end