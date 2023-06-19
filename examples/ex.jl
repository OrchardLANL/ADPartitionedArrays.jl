using Test
import ADPartitionedArrays
import DPFEHM
import GaussianRandomFields
import HYPRE
import MPI
import PartitionedArrays
import Random
MPI.Init()

Random.seed!(0)

function Array(v::PartitionedArrays.PVector)
	v_on_main = PartitionedArrays.to_trivial_partition(v, PartitionedArrays.trivial_partition(PartitionedArrays.partition(axes(v, 1))))
	v_standard_array = zeros(eltype(v), length(v_on_main))
	PartitionedArrays.map_main(PartitionedArrays.partition(v_on_main)) do myv
		v_standard_array .= myv
	end
	return v_standard_array
end

function neighbors2newneighbors(neighbors, numnodes)
	newneighbors = map(x->Int[], 1:numnodes)
	neighborindices = map(x->Int[], 1:numnodes)
	for (i, (node1, node2)) in enumerate(neighbors)
		push!(newneighbors[node1], node2)
		push!(newneighbors[node2], node1)
		push!(neighborindices[node1], i)
		push!(neighborindices[node2], i)
	end
	return newneighbors, neighborindices
end

@ADPartitionedArrays.equations exclude=(neighbors, neighborindices, areasoverlengths, volumes, isfreenode, nodei2freenodei, freenodei2nodei, rowindices) function gw(h, Ks, neighbors, neighborindices, areasoverlengths, dirichleths, Qs, specificstorage, volumes, isfreenode, nodei2freenodei, freenodei2nodei, rowindices)
	for node1_free = PartitionedArrays.local_to_global(rowindices)
		node1 = freenodei2nodei[node1_free]
		ADPartitionedArrays.addterm(node1_free, -Qs[node1] / (specificstorage[node1] * volumes[node1]))
		for (node2, neighborindex) in zip(neighbors[node1], neighborindices[node1])
			if isfreenode[node2]
				node2_free = nodei2freenodei[node2]
				ADPartitionedArrays.addterm(node1_free, Ks[neighborindex] * (h[node1_free] - h[node2_free]) * areasoverlengths[neighborindex] / (specificstorage[node1] * volumes[node1]))
			else
				ADPartitionedArrays.addterm(node1_free, Ks[neighborindex] * (h[node1_free] - dirichleths[node2]) * areasoverlengths[neighborindex] / (specificstorage[node1] * volumes[node1]))
			end
		end
	end
end

function bamg_solver(A, b; kwargs...)
	solver = HYPRE.BoomerAMG(; PrintLevel=-1, Tol=1e-8, MaxIter=10 ^ 3, kwargs...)
	hfree = HYPRE.solve(solver, A, b)
	return hfree
end

function gw_steadystate(Ks, oldneighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; linear_solver::Function=bamg_solver, solve_in_parallel::Bool=true, kwargs...)
	neighbors, neighborindices = neighbors2newneighbors(oldneighbors, length(Qs))
	isfreenode, nodei2freenodei, freenodei2nodei = DPFEHM.getfreenodes(length(Qs), dirichletnodes)
	if solve_in_parallel
		np = MPI.Comm_size(MPI.COMM_WORLD)
		ranks = PartitionedArrays.distribute_with_mpi(LinearIndices((np,)))
	else
		np = 1
		ranks = LinearIndices((np,))
	end
	row_partition = PartitionedArrays.uniform_partition(ranks, sum(isfreenode))
	args_no_rowindices = (zeros(sum(isfreenode)), Ks, neighbors, neighborindices, areasoverlengths, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)), isfreenode, nodei2freenodei, freenodei2nodei)
	#assemble the rhs, b
	f = rowindices->gw_residuals(args_no_rowindices..., rowindices)
	IV = map(f, row_partition)
	I, V = PartitionedArrays.tuple_of_arrays(IV)
	b = PartitionedArrays.pvector!(I, V, row_partition) |> fetch
	#assemble the matrix, A
	g = rowindices->gw_h(args_no_rowindices..., rowindices)
	IJV = map(g, row_partition)
	I, J, V = PartitionedArrays.tuple_of_arrays(IJV)
	col_partition = row_partition
	A = PartitionedArrays.psparse!(I, J, V, row_partition, col_partition) |> fetch
	#solve A*hfree=b
	hfree = linear_solver(A, -b; kwargs...)
	#convert to a normal Julia array on the main mode
	hfree_standard_array = Array(hfree)
	#add the dirichlet boundary conditions to the solution on the free nodes
	h = map(i->isfreenode[i] ? hfree_standard_array[nodei2freenodei[i]] : dirichleths[i], 1:length(Qs))
	return h
end

#set up the grid
mins = [0, 0, 0]; maxs = [50, 50, 5]#size of the domain, in meters
ns = round.(Int, 2 ^ (9 / 3) * [50, 50, 5])#number of nodes on the grid
#ns = [50, 50, 5]#number of nodes on the grid
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid3d(mins, maxs, ns)#build the grid

#set up the boundary conditions
Qs = randn(size(coords, 2))
dirichletnodes = Int[size(coords, 2)]#fix the pressure in the upper right corner
dirichleths = zeros(size(coords, 2))
dirichleths[size(coords, 2)] = 0.0

#set up the conductivity field
#println("geostats")
#@time begin
#	lambda = 50.0#meters -- correlation length of log-conductivity
#	sigma = 1.0#standard deviation of log-conductivity
#	mu = -9.0#mean of log conductivity -- ~1e-4 m/s, like clean sand here https://en.wikipedia.org/wiki/Hydraulic_conductivity#/media/File:Groundwater_Freeze_and_Cherry_1979_Table_2-2.png
#	cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
#	x_pts = range(mins[1], maxs[1]; length=ns[1])
#	y_pts = range(mins[2], maxs[2]; length=ns[2])
#	num_eigenvectors = 200
#	grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
#	logKs = zeros(reverse(ns)...)
#	logKs2d = mu .+ GaussianRandomFields.sample(grf)'#generate a random realization of the log-conductivity field
#	for i = 1:ns[3]#copy the 2d field to each of the 3d layers
#		v = view(logKs, i, :, :)
#		v .= logKs2d
#	end
#end
ts = Float64[]
for _ = 1:2
	t = @elapsed begin
		logKs = zeros(reverse(ns)...)
		logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))#convert from permeabilities at the nodes to permeabilities connecting the nodes
		Ks_neighbors = logKs2Ks_neighbors(logKs)
		h = gw_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
	end
	push!(ts, t)
end


if MPI.Comm_rank(MPI.COMM_WORLD) == 0
	@show log10(prod(ns)), minimum(ts)
end

nothing
