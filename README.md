

csv of fluorescence data -> formatted single data set (UnpackData.jl)

Raw image data -> vtk files (MaskUnpack.jl)
vtk files -> sphericity (DirectedSphericity.jl)
vtk files -> ChordStats (ChordStats.jl)
vtk files + sphericity + ChordStats -> morphology (MorphologySave.jl)

morphology + fluorescence -> single csv (MorphologySave.jl)

Add PCA coordinate to csv, make PCA plots (ShapePCA.jl)


Make plots, align with gene expression and morphology (AlignCoords.jl)
also makes data/processed/data_with_alignment.csv, the final csv file