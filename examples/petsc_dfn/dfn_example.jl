import DelimitedFiles
import MPI
import ADPartitionedArrays
import SparseArrays
import DPFEHM
import FEHM
import HDF5

using Gridap
using GridapPETSc
using GridapPETSc.PETSC: PetscScalar, PetscInt, PETSC
import PartitionedArrays
using LinearAlgebra
using Test
import JLD

MPI.Init()

struct VirtualZero{T}
end
function Base.getindex(vz::VirtualZero{T}, args...) where T
	return zero(T)
end
function Base.eltype(vz::VirtualZero{T}) where T
	return T
end

function loaddir(dirname)
	coords, volumes, neighbors, areas, lengths = DPFEHM.load_uge("$dirname/full_mesh_vol_area.uge")
	areasoverlengths = areas ./ lengths
	leftnodes = FEHM.readzone("$dirname/pboundary_left_w.zone")[2][1]
	rightnodes = FEHM.readzone("$dirname/pboundary_right_e.zone")[2][1]
	dirichletnodes = [leftnodes; rightnodes]
	dirichleths = [2e6 * ones(length(leftnodes)); 1e6 * ones(length(rightnodes))]
	Qs = zeros(size(coords, 2))
	Ks = HDF5.h5read("$dirname/dfn_properties.h5", "Permeability")
	Ks2Ks_neighbors(Ks) = sqrt.((Ks[map(p->p[1], neighbors)] .* Ks[map(p->p[2], neighbors)]))#convert from permeabilities at the nodes to permeabilities connecting the nodes
	Ks_neighbors = Ks2Ks_neighbors(Ks)
	dirichleths2 = zeros(length(Qs))
	for (i, j) in enumerate(dirichletnodes)
		dirichleths2[j] = dirichleths[i]
	end
	return Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths2, Qs
end

function variable_partition_from_partition(partition_in)
    n_global = PartitionedArrays.getany(map(PartitionedArrays.global_length,partition_in))
    n_own = map(PartitionedArrays.own_length,partition_in)
    PartitionedArrays.variable_partition(n_own,n_global)
end

function relabel_global_and_local_ids(
    partition_in,
    new_gids=PartitionedArrays.variable_partition_from_partition(partition_in))

    v = PartitionedArrays.PVector{Vector{Int}}(undef,partition_in)
    map(PartitionedArrays.own_values(v),new_gids) do own_v, new_gids
        own_v .= PartitionedArrays.own_to_global(new_gids)
    end
    PartitionedArrays.consistent!(v) |> wait
    new_ids_with_ghost = map(PartitionedArrays.ghost_values(v),new_gids,partition_in) do ghost_v,new_gids,gids
        n_global = PartitionedArrays.global_length(gids)
        ghost_to_new_gid = ghost_v
        rank = PartitionedArrays.part_id(new_gids)
        own = PartitionedArrays.OwnIndices(n_global,rank,PartitionedArrays.collect(PartitionedArrays.own_to_global(new_gids)))
        ghost = PartitionedArrays.GhostIndices(n_global,ghost_to_new_gid,PartitionedArrays.ghost_to_owner(gids))
        PartitionedArrays.OwnAndGhostIndices(own,ghost)
    end
    new_to_old = map(partition_in) do gids
        vcat(PartitionedArrays.own_to_local(gids),PartitionedArrays.ghost_to_local(gids))
    end
    old_to_new = map(PartitionedArrays.local_permutation,partition_in)
    new_ids_with_ghost, new_to_old, old_to_new
end

function relabel_system(A,b,x0)
    n = length(b)
    new_partition = variable_partition_from_partition(PartitionedArrays.partition(axes(A,1)))
    rank = PartitionedArrays.linear_indices(new_partition)
    new_row_partition = map(new_partition,rank) do gids,rank
        own = PartitionedArrays.OwnIndices(n,rank,collect(PartitionedArrays.own_to_global(gids)))
        ghost = PartitionedArrays.GhostIndices(n,Int[],Int32[])
        PartitionedArrays.OwnAndGhostIndices(own,ghost)
    end
    new_col_partition, new_to_old, old_to_new = relabel_global_and_local_ids(PartitionedArrays.partition(axes(A,2)),new_partition)
    b_new = PartitionedArrays.PVector(PartitionedArrays.partition(b),new_row_partition)
    x0_new_partition = map(PartitionedArrays.partition(x0),new_to_old) do x0,new_to_old
        x0[new_to_old]
    end
    x0_new = PartitionedArrays.PVector(x0_new_partition,new_col_partition)
    A_new_partition = map(PartitionedArrays.partition(A),old_to_new) do A,old_to_new
        m,n = size(A)
        I,J,V = PartitionedArrays.findnz(A)
        f = (old) -> old_to_new[old]
        J .= f.(J)
        SparseArrays.sparse(I,J,V,m,n)
    end
    A_new = PartitionedArrays.PSparseMatrix(A_new_partition,new_row_partition,new_col_partition)
    A_new, b_new, x0_new
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
	for node1_free in PartitionedArrays.local_to_global(rowindices)
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

function gw_steadystate(Ks, oldneighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; solve_in_parallel::Bool=true)#, kwargs...)
	neighbors, neighborindices = neighbors2newneighbors(oldneighbors, length(Qs))
	isfreenode, nodei2freenodei, freenodei2nodei = DPFEHM.getfreenodes(length(Qs), dirichletnodes)
   
    if solve_in_parallel
		np = MPI.Comm_size(MPI.COMM_WORLD)
		ranks = PartitionedArrays.distribute_with_mpi(LinearIndices((np,)))
	else
		np = 8
		ranks = LinearIndices((np,))
		#ranks = PartitionedArrays.DebugArray(LinearIndices((np,)))
	end
    
    t = PartitionedArrays.PTimer(ranks,verbose=true)

    lognp = Int(log2(np))
	p1 = 2 ^ (div(lognp, 3) + (mod(lognp, 3) > 0 ? 1 : 0))
	p2 = 2 ^ (div(lognp, 3) + (mod(lognp, 3) > 1 ? 1 : 0))
	p3 = 2 ^ div(lognp, 3)

    #row_partition = PartitionedArrays.uniform_partition(ranks, (p1, p2, p3), (grid_size, grid_size, grid_size))
    row_partition = PartitionedArrays.uniform_partition(ranks, sum(isfreenode))
	args_no_rowindices = (zeros(sum(isfreenode)), Ks, neighbors, neighborindices, areasoverlengths, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)), isfreenode, nodei2freenodei, freenodei2nodei)
    
	#assemble the rhs, b
    PartitionedArrays.tic!(t)
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
    PartitionedArrays.toc!(t,"assembly")
    cols = axes(A,2)
    x0 = PartitionedArrays.pzeros(PartitionedArrays.partition(cols))
    
    # Solve using petsc
    x = copy(x0)
    A_new,b_new,x_new = relabel_system(A,b,x)
    #jacobi_gmres_no_monitor
    #options = "-ksp_type gmres -ksp_converged_reason -ksp_error_if_not_converged true -pc_type jacobi -ksp_rtol 1.0e-12"
    #gamg_cg_monitor
    #options = "-pc_type gamg -ksp_type cg -ksp_monitor -ksp_converged_reason -ksp_rtol 1.0e-12"
    #gamg_cg_no_monitor
    options = "-pc_type gamg -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-5"
    x_new = GridapPETSc.with(args=split(options)) do
        b_petsc = convert(GridapPETSc.PETScVector,b_new)
        x_petsc = convert(GridapPETSc.PETScVector,x_new)
        A_petsc = convert(GridapPETSc.PETScMatrix,A_new)
        solver = GridapPETSc.PETScLinearSolver()
        PartitionedArrays.tic!(t)
        ss = GridapPETSc.symbolic_setup(solver,A_petsc)
        ns = GridapPETSc.numerical_setup(ss,A_petsc)
        PartitionedArrays.toc!(t,"setup_petsc")
        PartitionedArrays.tic!(t)
        solve!(x_petsc,ns,b_petsc)
        PartitionedArrays.toc!(t,"solve_petsc")
        PartitionedArrays.PVector(x_petsc,axes(x_new,1))
    end
    
    x .= x_new
    #x = map(i->isfreenode[i] ? x_new[nodei2freenodei[i]] : dirichleths[i], 1:length(Qs))

    # Calculate the norm
    #res = b - A * x
    #calc_norm = norm(res)/norm(b)
    #if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    #    @show calc_norm
    
    timing = PartitionedArrays.statistics(t)
    
    display(t)
    
    return x,timing
end


network_dirs = ["thomas_tpl_p2_x01", "thomas_tpl_p3_x01", "thomas_tpl_p5_x01", "thomas_tpl_p10_x01", "thomas_tpl_p20_x01"]
network_dirs = [network_dirs; ["var_b_0.0", "var_b_0.5", "var_b_1.0"]]
network_dirs = [network_dirs; ["TSA250_50"]]
need_to_download_data = false
for dir in network_dirs
	if !isdir(dir)
		need_to_download_data = true
	end
end
if need_to_download_data
	download("https://zenodo.org/record/5213727/files/results.tar.gz?download=1", "./results.tar.gz")
	run(`tar xzf results.tar.gz`)
	download("https://zenodo.org/record/5213727/files/meshes.tar.gz?download=1", "./meshes.tar.gz")
	run(`tar xzf meshes.tar.gz`)
end

assembly = Float64[]
setup = Float64[]
solve = Float64[]
ts = Float64[]
ns = []

times = Dict()
chs = Dict()
for (i, network_dir) in enumerate(network_dirs)
    resultsfilename = "results/" * replace(network_dir, "/"=>"_") * "_results.jld"
    if isfile(resultsfilename)
        thesetimes, thesechs = JLD.load(resultsfilename, "times", "chs")
        times[network_dir] = thesetimes
        chs[network_dir] = thesechs
    else
        times[network_dir] = Dict()
        chs[network_dir] = Dict()
    end

    Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs = loaddir(network_dir)

    time = @elapsed x,t = gw_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; solve_in_parallel=true)
    push!(ts, time)
    push!(ns, network_dir)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
		@show network_dir, time
        push!(assembly,t["assembly"][:avg])
        push!(setup,t["setup_petsc"][:avg])
        push!(solve,t["solve_petsc"][:avg])
	end
end

np = MPI.Comm_size(MPI.COMM_WORLD)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
	DelimitedFiles.writedlm("times_dfn_$(np).csv", hcat(ns, ts, assembly, setup, solve), ',')

end
