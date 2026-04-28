using CSV
using DataFrames
using Statistics
using HypothesisTests
using Plots

control_file = "data/CONTROL_pytag_puncta_branch_metrics_refactored_25IQR.csv"
mutant_file  = "data/MUTANT_pytag_puncta_branch_metrics_refactored_25IQR.csv"


function mean_no_missing(x)
    return mean(Float64.(collect(skipmissing(x))))
end

function embryo_means(file)
    df = CSV.read(file, DataFrame)

    return combine(groupby(df, :Embryo_num),
        :puncta_mean_intensity => mean_no_missing => :intensity,
        :n_puncta              => mean_no_missing => :puncta
    )
end

control = embryo_means(control_file)
mutant  = embryo_means(mutant_file)

control_intensity = log.(control.intensity)
mutant_intensity  = log.(mutant.intensity)

control_puncta = log.(control.puncta)
mutant_puncta  = log.(mutant.puncta)

scale_label = "log embryo mean"

println("\nShapiro-Wilk normality tests")
println("================================")

normality_tests = [
    ("Control intensity", control_intensity),
    ("Mutant intensity",  mutant_intensity),
    ("Control puncta",    control_puncta),
    ("Mutant puncta",     mutant_puncta),
]

for (name, x) in normality_tests
    test = ShapiroWilkTest(x)
    println("\n$name")
    #println(test)
    println("p-value = ", pvalue(test))
end

p1 = histogram(control_intensity,
    title = "Control: intensity",
    xlabel = scale_label,
    ylabel = "count",
    legend = false,
    bins=0:0.1:5,
    xlims=(3.0,4),
    grid=false)

p2 = histogram(mutant_intensity,
    title = "Mutant: intensity",
    xlabel = scale_label,
    ylabel = "count",
    legend = false,
    bins=0:0.1:5,
    xlims=(3.0,4),
    grid=false)

p3 = histogram(control_puncta,
    title = "Control: puncta",
    xlabel = scale_label,
    ylabel = "count",
    legend = false,
    bins=0:0.2:5,
    xlims=(1.0,4.2),
    grid=false)

p4 = histogram(mutant_puncta,
    title = "Mutant: puncta",
    xlabel = scale_label,
    ylabel = "count",
    legend = false,
    bins=0:0.2:5,
    xlims=(1.0,4.2),
    grid=false)

fig = plot(p1, p3, p2, p4,
    layout = (2, 2),
    size = (900, 650))

savefig(fig, "control_mutant_histograms.png")

println("\nEqual variance tests: Control vs Mutant")
println("=======================================")

intensity_var_test = VarianceFTest(control_intensity, mutant_intensity)
puncta_var_test    = VarianceFTest(control_puncta, mutant_puncta)

println("\nIntensity variance test")
println(intensity_var_test)
println("p-value = ", pvalue(intensity_var_test))

println("\nPuncta variance test")
println(puncta_var_test)
println("p-value = ", pvalue(puncta_var_test))





function embryo_means(file)
    df = CSV.read(file, DataFrame)

    return combine(groupby(df, :Embryo_num),
        :puncta_mean_intensity => mean_no_missing => :intensity,
        :n_puncta              => mean_no_missing => :puncta
    )
end

control = CSV.read(control_file, DataFrame)
mutant  = CSV.read(mutant_file, DataFrame)
thresh_val = 0:0.1:70
thresh_fun_control = [mean_no_missing(control.n_puncta .> t) for t in thresh_val]
thresh_fun_mutant  = [mean_no_missing(mutant.n_puncta .> t) for t in thresh_val]
plot(thresh_val, thresh_fun_control, label="Control", xlabel="puncta mean intensity threshold", ylabel="fraction of embryos above threshold")
plot!(thresh_val, thresh_fun_mutant, label="Mutant")