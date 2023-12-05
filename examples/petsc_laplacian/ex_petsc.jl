import DelimitedFiles
import MPI
import ADPartitionedArrays
import SparseArrays
#using IterativeSolvers

using Gridap
using GridapPETSc
using GridapPETSc.PETSC: PetscScalar, PetscInt, PETSC
import PartitionedArrays
using LinearAlgebra
using Test

MPI.Init()

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

#construct A in Ax=b
@ADPartitionedArrays.equations exclude=(grid_size, h, row_indices) function laplacian(u, grid_size, row_indices)
	h = 1 / grid_size
	for global_row in PartitionedArrays.local_to_global(row_indices)
		k = mod(global_row - 1, grid_size) + 1
		j = mod(div(global_row - k, grid_size), grid_size) + 1
		i = div(global_row - (j - 1) * grid_size + k, grid_size * grid_size) + 1
		ADPartitionedArrays.addterm(global_row, -6 * u[global_row] / h ^ 2 - 1 / h ^ 3)
		if i > 1
			ADPartitionedArrays.addterm(global_row, u[global_row - grid_size * grid_size] / h ^ 2)
		end
		if i < grid_size
			ADPartitionedArrays.addterm(global_row, u[global_row + grid_size * grid_size] / h ^ 2)
		end
		if j > 1
			ADPartitionedArrays.addterm(global_row, u[global_row - grid_size] / h ^ 2)
		end
		if j < grid_size
			ADPartitionedArrays.addterm(global_row, u[global_row + grid_size] / h ^ 2)
		end
		if k > 1
			ADPartitionedArrays.addterm(global_row, u[global_row - 1] / h ^ 2)
		end
		if k < grid_size
			ADPartitionedArrays.addterm(global_row, u[global_row + 1] / h ^ 2)
		end
	end
end

struct VirtualZero{T}
end
function Base.getindex(vz::VirtualZero{T}, args...) where T
	return zero(T)
end
function Base.eltype(vz::VirtualZero{T}) where T
	return T
end

function solve_laplacian(grid_size; solve_in_parallel::Bool=true)
	
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

    #row_partition = PartitionedArrays.uniform_partition(ranks, grid_size ^ 3)
	row_partition = PartitionedArrays.uniform_partition(ranks, (p1, p2, p3), (grid_size, grid_size, grid_size))

	#assemble the rhs, b
	virtual_zero = VirtualZero{Float64}()
    PartitionedArrays.tic!(t)
	f = row_indices->laplacian_residuals(virtual_zero, grid_size, row_indices)
	IV = map(f, row_partition)
	I, V = PartitionedArrays.tuple_of_arrays(IV)
	b = PartitionedArrays.pvector!(I, V, row_partition) |> fetch
	
    #assemble the matrix, A
	g = rowindices->laplacian_u(virtual_zero, grid_size, rowindices)
	#PartitionedArrays.tic!(t)
    IJV = map(g, row_partition)
    #PartitionedArrays.toc!(t,"IJV")
	I, J, V = PartitionedArrays.tuple_of_arrays(IJV)
    
    #PartitionedArrays.tic!(t)
	col_partition = row_partition
	A = PartitionedArrays.psparse!(I, J, V, row_partition, col_partition) |> fetch
    PartitionedArrays.toc!(t,"assembly")
    cols = axes(A,2)
    x0 = PartitionedArrays.pzeros(PartitionedArrays.partition(cols))
    
    # When this call returns, x has the correct answer only in the owned values.
    # The values at ghost ids can be recovered with consistent!(x) |> wait
    #x = copy(x0)
    #IterativeSolvers.cg!(x,A,b,verbose=PartitionedArrays.i_am_main(rank))

    ## Relabel global ids and solve again
    #x = copy(x0)
    #A_new,b_new,x_new = relabel_system(A,b,x)
    #IterativeSolvers.cg!(x_new,A_new,b_new,verbose=PartitionedArrays.i_am_main(rank))
    
    # Now with petsc
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

    # Calculate the norm
    #res = b - A * x
    #calc_norm = norm(res)/norm(b)
    #if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    #    @show calc_norm
    
    timing = PartitionedArrays.statistics(t)
    
    display(t)
    
	return x,timing
end

ts = Float64[]
ns = Int[]

assembly = Float64[]
setup = Float64[]
solve = Float64[]

for i = 2:8
	time = @elapsed x,t = solve_laplacian(2 ^ i; solve_in_parallel=true)
    @show 
	push!(ts, time)
	push!(ns, 2 ^ (3 * i))
	if MPI.Comm_rank(MPI.COMM_WORLD) == 0
		@show i, time
        push!(assembly,t["assembly"][:avg])
        push!(setup,t["setup_petsc"][:avg])
        push!(solve,t["solve_petsc"][:avg])
	end
end
np = MPI.Comm_size(MPI.COMM_WORLD)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
	DelimitedFiles.writedlm("times_3dpartition_$(np).csv", hcat(ns, ts, assembly, setup, solve), ',')

end
