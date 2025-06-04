using MultivariateStats
using DataFrames, CSV
using StatsBase
using Plots
include("src/Utils.jl")


df = CSV.read("data/processed/data_with_morphology.csv",DataFrame)

df_Terminal = filter(row -> occursin("TerminalCell", row.Cell_Type), df)
df_Fusion = filter(row -> occursin("FusionCell", row.Cell_Type), df)

columns_to_keep = [:elongation, :sphericity,:chords_nonzero]
df_selected = select(df_Terminal, columns_to_keep)

# apply PCA 
X = Matrix(df_selected)
# Z-score each column
X_scaled = (X .- mean(X, dims=1)) ./ std(X, dims=1)


M = fit(PCA, X_scaled', maxoutdim=20, pratio=0.999)
explained_variance = principalvars(M)
total_variance = sum(explained_variance)
explained_variance_ratio = explained_variance ./ total_variance

p1 = plot(1:length(explained_variance_ratio),explained_variance_ratio,seriestype=:bar,
    xlabel="Principal Component",ylabel="Proportion of Variance Explained",label=false,alpha=0.7,ylims=(0,1))


# Above plot tells you how many PCs to keep. Adjust nPC accordingly
nPC = 2
M = fit(PCA, X_scaled', maxoutdim=nPC)
Y = MultivariateStats.transform(M, X_scaled')
for i = 1:nPC
        add_info!(df,Y[i,:],"PC$i")
end
CSV.write("data/processed/data_with_PC.csv", df)




marker_dict = Dict("250404_DSRFsfGFP_Data_Tr7"=>:square,
                    "250404_DSRFsfGFP_Data_Tr8"=>:circle,
                    "250405_DSRFsfGFP_Data_Tr7"=>:diamond,
                    "250405_DSRFsfGFP_Data_Tr8"=>:hexagon,
                    "250406_DSRFsfGFP_Data_Tr7"=>:utriangle,
                    "250406_DSRFsfGFP_Data_Tr8"=>:star4,
                    "250408_DSRFsfGFP_Data_Tr7"=>:star5,
                "250408_DSRFsfGFP_Data_Tr8"=>:dtriangle,
                "250412_DSRFsfGFP_Data_Tr7"=>:pentagon,
                "250412_DSRFsfGFP_Data_Tr8"=>:cross)

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





