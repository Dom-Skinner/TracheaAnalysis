import Pkg; Pkg.activate(@__DIR__)
cd(@__DIR__)
using MultivariateStats
using DataFrames, CSV
using StatsBase
using Plots
include("src/Utils.jl")


df = CSV.read("data/processed/data_with_morphology.csv",DataFrame)

df_Terminal = filter(row -> !occursin("FusionCell", row.Cell_Type), df)
df_Fusion = filter(row -> occursin("FusionCell", row.Cell_Type), df)

columns_to_keep = [:elongation, :sphericity,:chords_nonzero,:volumeRatio,:area,:eccentricity,:perimeter,:circularity,:relative_volume,:nuclei_distance]
df_selected = select(df_Terminal, columns_to_keep)

# apply PCA 
X = Matrix(df_selected)
mutant_filter = .!occursin.("_bnl_",df_Terminal.exp_id)
# Z-score each column
#X_scaled = (X .- mean(X[mutant_filter,:], dims=1)) ./ std(X[mutant_filter,:], dims=1)
X_scaled = (X .- mean(X, dims=1)) ./ std(X, dims=1)


M = fit(PCA, X_scaled[mutant_filter,:]', maxoutdim=20, pratio=0.999)
explained_variance = principalvars(M)
total_variance = sum(explained_variance)
explained_variance_ratio = explained_variance ./ total_variance

p1 = plot(1:length(explained_variance_ratio),explained_variance_ratio,seriestype=:bar,
    xlabel="Principal Component",ylabel="Proportion of Variance Explained",label=false,alpha=0.7,ylims=(0,1))

failure_filter = [id ∈["251001_DSRFsfGFP_bnl_Data_Tr8","251005_DSRFsfGFP_bnl_Data_Tr7","251006_DSRFsfGFP_bnl_Data_Tr8","251008_DSRFsfGFP_bnl_Data_Tr7","251008_DSRFsfGFP_bnl_Data_Tr8"] for id in df_Terminal.unique_id]
remainder_filter = .!mutant_filter .& .!failure_filter
    

# Above plot tells you how many PCs to keep. Adjust nPC accordingly
nPC = 2
M = fit(PCA, X_scaled', maxoutdim=nPC)
Y = MultivariateStats.transform(M, X_scaled')

for i = 1:nPC
        add_info!(df,Y[i,:],"PC$i")
end
CSV.write("data/processed/data_with_PC.csv", df)




scatter(Y[1,mutant_filter], Y[2,mutant_filter], 
        xlabel="PC1", ylabel="PC2", 
        aspect_ratio=:equal, 
        size=(800,800),
        ms=5,
        label=false,msw=0.1,mc=colorant"#66c2a5")
scatter!(Y[1,failure_filter], Y[2,failure_filter], 
        ms=5,
        label=false,msw=0.1,mc=colorant"#8da0cb")
scatter!(Y[1,remainder_filter], Y[2,remainder_filter],        
    ms=5,
        label=false,msw=0.1,mc=colorant"#fc8d62")  


marker_dict = Dict("250404_DSRFsfGFP_Data_Tr7"=>:square,
                    "250404_DSRFsfGFP_Data_Tr8"=>:circle,
                    "250405_DSRFsfGFP_Data_Tr7"=>:diamond,
                    "250405_DSRFsfGFP_Data_Tr8"=>:hexagon,
                    "250406_DSRFsfGFP_Data_Tr7"=>:utriangle,
                    "250406_DSRFsfGFP_Data_Tr8"=>:star4,
                    "250408_DSRFsfGFP_Data_Tr7"=>:star5,
                "250408_DSRFsfGFP_Data_Tr8"=>:dtriangle,
                "250412_DSRFsfGFP_Data_Tr7"=>:pentagon,
                "250412_DSRFsfGFP_Data_Tr8"=>:cross,
                "251001_DSRFsfGFP_bnl_Data_Tr7"=>:square,
                "251001_DSRFsfGFP_bnl_Data_Tr8"=>:circle,
                "251005_DSRFsfGFP_bnl_Data_Tr7"=>:diamond,
                "251005_DSRFsfGFP_bnl_Data_Tr8"=>:hexagon,
                "251006_DSRFsfGFP_bnl_Data_Tr7"=>:utriangle,
                "251006_DSRFsfGFP_bnl_Data_Tr8"=>:star4,
                "251008_DSRFsfGFP_bnl_Data_Tr7"=>:star5,
                 "251008_DSRFsfGFP_bnl_Data_Tr8"=>:dtriangle)



                 
p2 = plot() 
for id in keys(marker_dict)
        idx = findall(df_Terminal.unique_id .== id)

        scatter!(p2, Y[1,idx], Y[2,idx], 
        marker=marker_dict[id], 
        label=id,
        zcolor=df_Terminal.t[idx],
        msw=0.1)
end
plot!(p2, 
        xlabel="PC1", ylabel="PC2", 
        aspect_ratio=:equal, 
        size=(800,800),
        colorbar_title="Time")


l = @layout [a{0.25w} b{0.75w}]
plot(p1,p2,size=(800,400),layout=l)
savefig("plots/PCA_plots.pdf")





