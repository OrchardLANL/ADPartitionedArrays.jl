using Test
import ADPartitionedArrays
import DPFEHM
import Graphs
import HYPRE
import Metis
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

function pcg_bamg_solver(A, b; kwargs...)
	precond = HYPRE.BoomerAMG()
	solver = HYPRE.PCG(; Precond=precond, Tol=1e-8, MaxIter=10 ^ 3)
	hfree = HYPRE.solve(solver, A, b)
	return hfree
end

function bamg_solver(A, b; kwargs...)
	solver = HYPRE.BoomerAMG(; PrintLevel=-1, Tol=1e-8, MaxIter=10 ^ 3, kwargs...)
	hfree = HYPRE.solve(solver, A, b)
	return hfree
end

function gw_steadystate(Ks, oldneighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; linear_solver::Function=bamg_solver, solve_in_parallel::Bool=true, use_metis=false, kwargs...)
	t_extra_metis = @elapsed if use_metis
		isfreenode, nodei2freenodei, freenodei2nodei = DPFEHM.getfreenodes(length(Qs), dirichletnodes)
		g = Graphs.SimpleGraph(sum(isfreenode))
		for (node1, node2) in oldneighbors
			if isfreenode[node1] && isfreenode[node2]
				Graphs.add_edge!(g, nodei2freenodei[node1], nodei2freenodei[node2])
			end
		end
		metis_partition = Metis.partition(g, MPI.Comm_size(MPI.COMM_WORLD))
		metis_reordering_subarrays = map(x->Int[], 1:MPI.Comm_size(MPI.COMM_WORLD))
		for i = 1:length(metis_partition)
			push!(metis_reordering_subarrays[metis_partition[i]], i)
		end
		metis_reordering = [map(i->freenodei2nodei[i], vcat(metis_reordering_subarrays...)); dirichletnodes]#put the dirichletnodes at the end in the reordering, since Metis hasn't considered those
		original_to_reorder = Dict{Int, Int}(zip(metis_reordering, 1:length(metis_reordering)))
		reordered_neighbors = Pair{Int, Int}[]
		for (node1, node2) in oldneighbors
			push!(reordered_neighbors, original_to_reorder[node1]=>original_to_reorder[node2])
		end
		Qs = Qs[metis_reordering]
		dirichleths = dirichleths[metis_reordering]
		dirichletnodes = map(i->original_to_reorder[i], dirichletnodes)
	else
		reordered_neighbors = oldneighbors
	end
	if MPI.Comm_rank(MPI.COMM_WORLD) == 0
		@show t_extra_metis
	end
	isfreenode, nodei2freenodei, freenodei2nodei = DPFEHM.getfreenodes(length(Qs), dirichletnodes)
	neighbors, neighborindices = neighbors2newneighbors(reordered_neighbors, length(Qs))
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
	if use_metis
		h = map(i->h[original_to_reorder[i]], 1:length(h))
	end
	return h
end

#set up the grid
mins = [0, 0, 0]; maxs = [50, 50, 5]#size of the domain, in meters
ns = round.(Int, 2 ^ (9 / 3) * [3, 3, 2])#number of nodes on the grid
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid3d(mins, maxs, ns)#build the grid

#set up the boundary conditions
Qs = randn(size(coords, 2))
dirichletnodes = Int[2]#fix the pressure at node #2
dirichleths = zeros(size(coords, 2))
dirichleths[2] = 0.0

ts = Float64[]
for run_num = 1:5
	t = @elapsed begin
		logKs = zeros(reverse(ns)...)
		logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))#convert from permeabilities at the nodes to permeabilities connecting the nodes
		Ks_neighbors = logKs2Ks_neighbors(logKs)
		t_metis = @elapsed h = gw_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; solve_in_parallel=true, linear_solver=pcg_bamg_solver, use_metis=true)
		t_no_metis = @elapsed h2 = gw_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; solve_in_parallel=true, linear_solver=pcg_bamg_solver, use_metis=false)
		#h3 = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
		if MPI.Comm_rank(MPI.COMM_WORLD) == 0 && run_num >1
			#=
			@show h
			@show h2
			@show h - h2
			@show sum((h - h3) .^ 2)
			=#
			@show sum((h - h2) .^ 2)
			@show t_metis, t_no_metis, t_no_metis / t_metis
		end
	end
	push!(ts, t)
end

nothing
