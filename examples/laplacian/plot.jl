import DelimitedFiles
import PyPlot

numprocs = [1, 2, 4, 8]
ts = Any[]
for np in numprocs
	global ns
	global ts
	data = DelimitedFiles.readdlm("times_$np.csv", ',')
	ns = data[:, 1]
	push!(ts, data[:, 2])
end

#plot strong scaling behavior
fig, ax = PyPlot.subplots()
ts_strong = map(x->x[end], ts)
ts_strong_ideal = ts[1][end] ./ numprocs
ax.plot(numprocs, ts_strong, label="actual")
ax.plot(numprocs, ts_strong_ideal, label="ideal")
ax.set(xlabel="Number of processors", ylabel="Time (seconds)", title="Strong scaling")
ax.legend()
display(fig)
println()
PyPlot.close(fig)

#plot weak scaling behavior
fig, ax = PyPlot.subplots()
ratio_weak = Float64[]
for i = 2:length(ts[end]) - 1
	push!(ratio_weak, ts[4][i + 1] / ts[1][i] )
end
ratio_weak_ideal = fill(1, length(ts[1]) - 2)
ax.semilogx(ns[2:end - 1], ratio_weak, label="actual")
ax.semilogx(ns[2:end - 1], ratio_weak_ideal, label="ideal")
ax.set(xlabel="Number of unknowns in 1 core problem", ylabel="(8 core time) / (1 core time)", title="Weak scaling")
ax.legend()
display(fig)
println()
PyPlot.close(fig)
