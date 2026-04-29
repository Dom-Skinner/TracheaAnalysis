import Pkg; Pkg.activate(@__DIR__)
cd(@__DIR__)
using CSV, DataFrames
### ========= Load tiff files and save as vtk files ========== ###

include("src/Utils.jl")
mkpath("data/vtk/")

in_data = filter(x->isdir("data/raw/"*x),readdir("data/raw/"))
for str_in in in_data

    mkpath("data/vtk/"*str_in)

    in_files = filter(x->x!=".DS_Store",readdir("data/raw/"*str_in))
    in_files_id = [parse(Int64,split(f,"_")[2]) for f in in_files]
    in_files = in_files[sortperm(in_files_id)]

    for i in 1:length(in_files)

        img = TiffImages.load("data/raw/"*str_in*"/"*in_files[i])
        img_converted = lazy_convert(img)
        
        x = 1:size(img,1)
        y = 1:size(img,2)
        z = 9.39*(1:size(img,3))

        vtk_grid("data/vtk/"*str_in*"/masked_"*lpad(i,3,"0"), x, y, z) do vtk
            vtk["raw"] = img_converted
        end
    end
    println("Converted data/raw/"*str_in*" to vtk format")
end