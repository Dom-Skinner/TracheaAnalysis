# This script is simply to make the demonstration plots that are found in Fig 4 B-C.
using Plots
using Distributions
include("Utils.jl")

function μ_σ_to_simplex(μ, σ)
    simplex_vals = [ordinal3_probs(μ[i], σ[i]) for i in 1:length(μ)]
    V0 = [0.0, 0.0]
    V1 = [0.5, sqrt(3.0) / 2.0]
    V2 = [1.0, 0.0]
    xy_coords = [ simplex_vals[i][1] * V0 + simplex_vals[i][2] * V1 + simplex_vals[i][3] * V2 for i in 1:length(simplex_vals) ]
    return [x[1] for x in xy_coords], [x[2] for x in xy_coords]
end


xplt = range(-1, stop=2, length=400)
plt1 = plot(xplt, pdf(Normal(0.5, 1/sqrt(8*pi)), xplt), label=false, lw=2)
plot!(plt1, xplt, (xplt .> 0) .+ (xplt .> 1), label=false, lw=2,grid=false)

savefig("plots/threshold_illustration.pdf")



t_vals = range(0, stop=1, length=6)

V0 = [0.0, 0.0]
V1 = [0.5, sqrt(3.0) / 2.0]
V2 = [1.0, 0.0]

plt1= plot(V0[1].*t_vals .+ V1[1] .*(1 .- t_vals),V0[2].*t_vals .+ V1[2] .*(1 .- t_vals),label=false, color=:black,xaxis=false, yaxis=false)
plot!(plt1, V2[1].*t_vals .+ V1[1] .*(1 .- t_vals),V2[2].*t_vals .+ V1[2] .*(1 .- t_vals),label=false, color=:black)
plot!(plt1, V2[1].*t_vals .+ V0[1] .*(1 .- t_vals),V2[2].*t_vals .+ V0[2] .*(1 .- t_vals),label=false, color=:black)
σ_vals_pts = 0.2 .+ 0.5 *  sin.(π * t_vals)
μ_vals_pts = t_vals 
xcoords, ycoords = μ_σ_to_simplex(μ_vals_pts, σ_vals_pts)
scatter!(plt1, xcoords, ycoords, label=false, color=:red,aspect_ratio=true, grid=false)
plt2= scatter(μ_vals_pts, σ_vals_pts, label=false, color=:red)


t_vals = range(0, stop=1, length=1001)

for σ_line in 0.2:0.2:80
    μ_vals = 200*((t_vals .- 0.5).^3)
    σ_vals = fill(σ_line, length(t_vals))
    skip = σ_line > 10 ? 5 : 1
    μ_vals = μ_vals[1:skip:end]
    σ_vals = σ_vals[1:skip:end]
    μ_vals[1] = -1000
    μ_vals[end] = 1000
   xcoord,ycoord = μ_σ_to_simplex(μ_vals, σ_vals)
    plot!(plt1, xcoord, ycoord, label=false, color=:grey, lw=0.5)
    plot!(plt2,μ_vals, σ_vals,grid=false,label=false,color=:grey,lw=0.5)
end


for μ_line in -10:0.2:10
    σ_vals = vcat(range(0, stop=10, length=501)[2:end],11:80)
    μ_vals = fill(μ_line, length(σ_vals))
    xcoord,ycoord = μ_σ_to_simplex(μ_vals, σ_vals)
    plot!(plt1, xcoord, ycoord, label=false, color=:grey, lw=0.5)
    plot!(plt2,μ_vals, σ_vals,grid=false,label=false,color=:grey,lw=0.5)
end

plot!(plt2,ylims=(0,1.03),xlims=(-0.5,1.5))
plot(plt1, plt2, layout=(1,2))

savefig("plots/simplex_traj.pdf")

xplt = range(-1.5, stop=2.5, length=400)

plt_tot = []
pie_tot = []

for i in 1:length(μ_vals_pts)
    println(i)
    plt = plot(xplt, pdf(Normal(μ_vals_pts[i], σ_vals_pts[i]), xplt), label=false, lw=2,ylims=(0,2),grid=false,
        yticks=(0:2))
    pie_plt = pie([ordinal3_probs(μ_vals_pts[i], σ_vals_pts[i])...],c=["#1e90ff", "#d3d3d3", "#ff0000"], label=false)
    push!(plt_tot, plt)
    push!(pie_tot, pie_plt)
end
plot(plt_tot...,pie_tot...)
savefig("plots/probabilities_traj.pdf")