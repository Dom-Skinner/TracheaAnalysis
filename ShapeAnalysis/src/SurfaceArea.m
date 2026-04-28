% This script calculates the total surface area and volume of the points in in_file
% Assumed inputs are in_file and out_file

disp(in_file)
disp(out_file)
totalsurfarea = zeros(2,1);
totalVolume = zeros(2,1);
volumeRatio = zeros(2,1);

I = h5read(in_file,"/segmented_points");

[rows, cols, slices] = size(I);

for j = 1:2
    indices = find(I == j);
    [x, y, z] = ind2sub([rows, cols, slices], indices);
    
    
    nonzero_coords = [x y 9.39*z];
    
    shp = alphaShape(nonzero_coords,12.5);
    %plot(shp)
    
    totalsurfarea(j) = surfaceArea(shp);
    totalVolume(j) = volume(shp);

    conv_hull = alphaShape(nonzero_coords,1e8);
    volumeRatio(j) = volume(shp)/volume(conv_hull);
end


h5create(out_file, '/totalSurfaceArea', size(totalsurfarea))
h5write(out_file, '/totalSurfaceArea', totalsurfarea)

h5create(out_file, '/totalVolume', size(totalVolume))
h5write(out_file, '/totalVolume', totalVolume)


h5create(out_file, '/volumeRatio', size(volumeRatio))
h5write(out_file, '/volumeRatio', volumeRatio)
disp('done')