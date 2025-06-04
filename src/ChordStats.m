% This script computes statistics of chords drawn between random points and
% whether they leave the alpha shape based on the points in in_file
% Assumed inputs are in_file and out_file

disp(in_file)
disp(out_file)

I = h5read(in_file,"/segmented_points");

[rows, cols, slices] = size(I);

total_mean = zeros(2,1);
nonzero_fraction = zeros(2,1);
conditional_variance = zeros(2,1);


for j = 1:2
    indices = find(I == j);
    [x, y, z] = ind2sub([rows, cols, slices], indices);
    
    
    nonzero_coords = [x y 9.39*z];
    
    shp = alphaShape(nonzero_coords,12.5);
    
    %plot(shp)
    q_vals = zeros(100000,1);
    for i = 1:length(q_vals)
        p = randperm(length(x),2);
        p1 = nonzero_coords(p(1),:);
        p2 = nonzero_coords(p(2),:);
        q_vals(i) = lineInAlphaShape(p1, p2, shp);
    end
    
    total_mean(j)= mean(q_vals);
    nonzero_fraction(j) = sum(q_vals>0)/length(q_vals);
    conditional_variance(j) = var(q_vals(q_vals>0));
end


h5create(out_file, '/total_mean', size(total_mean))
h5write(out_file, '/total_mean', total_mean)

h5create(out_file, '/nonzero_fraction', size(nonzero_fraction))
h5write(out_file, '/nonzero_fraction', nonzero_fraction)

h5create(out_file, '/conditional_variance', size(conditional_variance))
h5write(out_file, '/conditional_variance', conditional_variance)

disp('done')

function outsideFraction = lineInAlphaShape(p1, p2, shp)
    % Get triangulation and points from alpha shape
    
    % Sample points along the line segment
    numSamples = 100;  % Adjust based on needed precision
    t = linspace(0, 1, numSamples);
    samplePoints = p1 + (p2 - p1) .* t';
    
    % Check each sample point
    insideFlags = inShape(shp, samplePoints);
    
    % Calculate fraction outside
    outsideFraction = 1 - sum(insideFlags) / numSamples;
end