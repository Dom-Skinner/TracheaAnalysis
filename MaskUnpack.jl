### ========= Load tiff files and save as vtk files ========== ###

include("src/Utils.jl")
mkpath("data/vtk/")



in_data = filter(x->isdir("data/raw/"*x),readdir("data/raw/"))
for str_in in in_data[2:end]

    mkpath("data/vtk/"*str_in)

    in_files = filter(x->x!=".DS_Store",readdir("data/raw/"*str_in))
    in_files_id = [parse(Int64,split(f,"_")[2]) for f in in_files]
    in_files = in_files[sortperm(in_files_id)]


    # Take the initial PC vector from first image for consistent orientation throughout.
    img = TiffImages.load("data/raw/"*str_in*"/"*in_files[1]) |> lazy_convert
    v1 = initial_tip_vector(img,1)
    v2 = initial_tip_vector(img,2)
    v = [v1,v2]

    for i in 1:length(in_files)

        img = TiffImages.load("data/raw/"*str_in*"/"*in_files[i])
        img_converted = lazy_convert(img)
        
        x = 1:size(img,1)
        y = 1:size(img,2)
        z = 9.39*(1:size(img,3))

        path = []
        for id in [1,2]
    
            idx0,idx_com,pca = find_PC(img_converted .== id,v[id])
            _, pathlength_array = backtrack_search(img_converted .== id,idx0,pca)
    
            path_dist = zeros(size(img_converted))
            path_dist[img_converted.==id] .= pathlength_array
            push!(path,copy(path_dist))
        end

        vtk_grid("data/vtk/"*str_in*"/masked_"*lpad(i,3,"0"), x, y, z) do vtk
            vtk["raw"] = img_converted
            vtk["path1"] = path[1]
            vtk["path2"] = path[2]
        end
    end
    println("Converted data/raw/"*str_in*" to vtk format")
end