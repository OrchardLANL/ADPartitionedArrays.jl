using Test
import ADPartitionedArrays
import DPFEHM
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
ns = round.(Int, 2 ^ (0 / 3) * [50, 50, 5])#number of nodes on the grid
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid3d(mins, maxs, ns)#build the grid

#set up the boundary conditions
Qs = randn(size(coords, 2))
dirichletnodes = Int[size(coords, 2)]#fix the pressure in the upper right corner
dirichleths = zeros(size(coords, 2))
dirichleths[size(coords, 2)] = 0.0

ts = Float64[]
for _ = 1:2
	t = @elapsed begin
		logKs = zeros(reverse(ns)...)
		logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))#convert from permeabilities at the nodes to permeabilities connecting the nodes
		Ks_neighbors = logKs2Ks_neighbors(logKs)
			t_new = @elapsed h = gw_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; solve_in_parallel=true, linear_solver=pcg_bamg_solver)
			#=
			t_old = @elapsed A = DPFEHM.groundwater_h(h, Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
			t_old2 = @elapsed A \ randn(size(A, 1))
			@show t_old + t_old2, t_new
			=#
			@show t_new
	end
	push!(ts, t)
end

nothing
