# This script calculates chord statistics of all images in a vtk folder. Random chords are drawn between
# two points and how much of that chord is outside of the alpha shape is recorded. The mean of this value
# number of non-zero values and the variance of the non-zero values are recorded.
# It calls a matlab script ChordStats.m, which performs the core computations.
# Starting up matlab a bunch of times is inefficient, but the overall code runs in a reasonable time.
import Pkg; Pkg.activate(@__DIR__)
cd(@__DIR__)
using MultivariateStats
using ReadVTK
using Plots
using HDF5

function ChordStats(in_file)
    
    tmp_out = "tmp_surface.h5"
    rm(tmp_out,force=true)

    run(`/Applications/MATLAB_R2024a.app/bin/matlab -nodisplay -nosplash -nodesktop -r "in_file='../$in_file',out_file='../$tmp_out'; run('src/ChordStats.m'); exit"`)

    
    total_mean = h5read(tmp_out, "/total_mean")
    nonzero_fraction = h5read(tmp_out, "/nonzero_fraction")
    conditional_variance = h5read(tmp_out, "/conditional_variance")
    

    rm(tmp_out,force=true)
    return total_mean, nonzero_fraction, conditional_variance
end

function ChordStats_wrapper(in_file)
    
    vtk = VTKFile(in_file)
    cell_data = get_point_data(vtk)
    img_converted = get_data_reshaped(cell_data["raw"]) 

    h5write("tmp.h5","/segmented_points",img_converted)
    total_mean, nonzero_fraction, conditional_variance = ChordStats("tmp.h5")
    rm("tmp.h5",force=true)
    return total_mean, nonzero_fraction, conditional_variance

end

mkpath("data/processed/chords")
dirs = filter(x->isdir("data/vtk/"*x),readdir("data/vtk/"))
for current_iter_dir in dirs

    in_dir = "data/vtk/"*current_iter_dir*"/"
    out_file = "data/processed/chords/"*current_iter_dir*".h5"
    vtk_files = filter(x->occursin("masked",x),readdir(in_dir))


    mean_arr = zeros(length(vtk_files),2)
    nonzero_arr = zeros(length(vtk_files),2)
    var_arr = zeros(length(vtk_files),2)
    for i = 1:length(vtk_files)
        println(i)
        in_file = in_dir*vtk_files[i]
        total_mean, nonzero_fraction, conditional_variance = ChordStats_wrapper(in_file)
        mean_arr[i,:] = total_mean
        nonzero_arr[i,:] = nonzero_fraction
        var_arr[i,:] = conditional_variance
    end

    rm(out_file,force=true)
    h5write(out_file,"/total_mean",mean_arr)
    h5write(out_file,"/nonzero_fraction",nonzero_arr)
    h5write(out_file,"/conditional_variance",var_arr)



#=
plot(mean_arr[:,1],label="Tr 7")
plot!(mean_arr[:,2],label="Tr 8",xlabel="Frame",ylabel="mean")

plot!(nonzero_arr[:,1],label="Tr 7")
plot!(nonzero_arr[:,2],label="Tr 8",xlabel="Frame",ylabel="Non zero")

plot(var_arr[:,1],label="Tr 7")
plot!(var_arr[:,2],label="Tr 8",xlabel="Frame",ylabel="Conditional Variance")
=#
end