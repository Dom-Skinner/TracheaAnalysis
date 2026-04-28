# This script calculates the sphericity and convex volume ratio of all images in a vtk folder. 
# It calls a matlab script SurfaceArea.m, which is only needed to compute the surface area and volume
# from an alpha shape and convex hull. Starting up matlab a bunch of times is inefficient, but the 
# overall code runs in a reasonable time.

using MultivariateStats
using ReadVTK
using Plots
using HDF5

function sphericity_wrapper(in_file)
    
    vtk = VTKFile(in_file)
    cell_data = get_point_data(vtk)
    img_converted = get_data_reshaped(cell_data["raw"]) 

    save_h5 = "tmp.h5"
    rm(save_h5,force=true)
    tmp_out = "tmp_surface.h5"
    rm(tmp_out,force=true)

    h5write(save_h5,"/segmented_points",img_converted)

    run(`/Applications/MATLAB_R2024a.app/bin/matlab -nodisplay -nosplash -nodesktop -r "in_file='../$save_h5',out_file='../$tmp_out'; run('src/SurfaceArea.m'); exit"`)

    
    surfaceArea = h5read(tmp_out, "/totalSurfaceArea")
    Volume = h5read(tmp_out, "/totalVolume")
    volumeRatio = h5read(tmp_out, "/volumeRatio")
    

    rm(tmp_out,force=true)
    rm(save_h5,force=true)
    return surfaceArea, Volume, volumeRatio

end

mkpath("data/processed/sphericity")
dirs = filter(x->isdir("data/vtk/"*x),readdir("data/vtk/"))
for current_iter_dir in dirs

    in_dir = "data/vtk/"*current_iter_dir*"/"
    out_file = "data/processed/sphericity/"*current_iter_dir*".h5"
    vtk_files = filter(x->occursin("masked",x),readdir(in_dir))

    sph_arr_full = zeros(length(vtk_files),2)
    vr_arr_full = zeros(length(vtk_files),2)
    for i = 1:length(vtk_files)
        println(i)
        in_file = in_dir*vtk_files[i]

        s,v,vr = sphericity_wrapper(in_file)
        sph_arr_full[i,:] = (π^(1/3) * (6 * v).^(2/3))./s 
        vr_arr_full[i,:] .= vr
    end

    rm(out_file,force=true)
    h5write(out_file,"/sphericity",sph_arr_full)
    h5write(out_file,"/volumeRatio", vr_arr_full)
    println("Computed sphericity for $(current_iter_dir) and saved to $(out_file)")
end


#=
plot(sph_arr[:,1],label="Tr 7")
plot!(sph_arr[:,2],label="Tr 8",xlabel="Frame",ylabel="Sphericity")

plot!(sph_arr_full[:,1],label="Tr 7")
plot!(sph_arr_full[:,2],label="Tr 8",xlabel="Frame",ylabel="Sphericity")
=#