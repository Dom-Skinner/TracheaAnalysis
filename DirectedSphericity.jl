# This script calculates the sphericity of all images in a vtk folder. Both directed and undirected
# It calls a matlab script SurfaceArea.m, which is only needed to compute the surface area and volume
# from an alpha shape. Starting up matlab a bunch of times is inefficient, but the 
# overall code runs in a reasonable time.

using MultivariateStats
using ReadVTK
using Plots
using HDF5

function surfaceArea_Volume(in_file)
    
    tmp_out = "tmp_surface.h5"
    rm(tmp_out,force=true)

    run(`/Applications/MATLAB_R2024a.app/bin/matlab -nodisplay -nosplash -nodesktop -r "in_file='../$in_file',out_file='../$tmp_out'; run('src/SurfaceArea.m'); exit"`)

    
    surfaceArea = h5read(tmp_out, "/totalSurfaceArea")
    Volume = h5read(tmp_out, "/totalVolume")
    

    rm(tmp_out,force=true)
    return surfaceArea,Volume
end

function sphericity_wrapper(in_file,n_vol)
    
    vtk = VTKFile(in_file)
    cell_data = get_point_data(vtk)
    img_converted = get_data_reshaped(cell_data["raw"]) 

    # This block selects only the top n_vol pixels in the image, for a volume conserved sphericity
    for id = 1:2    
        path_data = get_data_reshaped(cell_data["path"*string(id)])
        coords = findall(img_converted .== id)
        Y = path_data[coords]
        nn = n_vol < length(Y) ? n_vol : length(Y)
        Yc = sort(Y,rev=true)[nn]
        img_converted[(img_converted .== id) .& (path_data .< Yc)] .= 0
    end
    h5write("tmp.h5","/segmented_points",img_converted)
    s,v = surfaceArea_Volume("tmp.h5")
    rm("tmp.h5",force=true)
    return s,v

end

mkpath("data/processed/sphericity")
dirs = filter(x->isdir("data/vtk/"*x),readdir("data/vtk/"))
for current_iter_dir in dirs

    in_dir = "data/vtk/"*current_iter_dir*"/"
    out_file = "data/processed/sphericity/"*current_iter_dir*".h5"
    vtk_files = filter(x->occursin("masked",x),readdir(in_dir))

    n_vol = 100_000

    sph_arr = zeros(length(vtk_files),2)
    sph_arr_full = zeros(length(vtk_files),2)
    for i = 1:length(vtk_files)
        println(i)
        in_file = in_dir*vtk_files[i]
        s,v = sphericity_wrapper(in_file,n_vol)
        sph_arr[i,:] = (π^(1/3) * (6 * v).^(2/3))./s

        s,v = sphericity_wrapper(in_file,Inf)
        sph_arr_full[i,:] = (π^(1/3) * (6 * v).^(2/3))./s 
    end

    rm(out_file,force=true)
    h5write(out_file,"/sphericity",sph_arr_full)
    h5write(out_file,"/sphericity_directed",sph_arr)
    println("Computed sphericity for $(current_iter_dir) and saved to $(out_file)")
end


#=
plot(sph_arr[:,1],label="Tr 7")
plot!(sph_arr[:,2],label="Tr 8",xlabel="Frame",ylabel="Sphericity")

plot!(sph_arr_full[:,1],label="Tr 7")
plot!(sph_arr_full[:,2],label="Tr 8",xlabel="Frame",ylabel="Sphericity")
=#