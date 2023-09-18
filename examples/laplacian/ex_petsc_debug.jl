import DelimitedFiles
import MPI
import ADPartitionedArrays

using Gridap
using GridapPETSc
using GridapPETSc.PETSC: PetscScalar, PetscInt, PETSC
import PartitionedArrays
using LinearAlgebra
MPI.Init()

function variable_partition_from_partition(partition_in)
    n_global = PartitionedArrays.getany(map(PartitionedArrays.global_length,partition_in))
    n_own = map(PartitionedArrays.own_length,partition_in)
    PartitionedArrays.variable_partition(n_own,n_global)
end

function relabel_global_ids(
    partition_in,
    new_gids=PartitionedArrays.variable_partition_from_partition(partition_in))

    v = PartitionedArrays.PVector{Vector{Int}}(undef,partition_in)
    map(PartitionedArrays.own_values(v),new_gids) do own_v, new_gids
        own_v .= PartitionedArrays.own_to_global(new_gids)
    end
    PartitionedArrays.consistent!(v) |> wait
    map(PartitionedArrays.ghost_values(v),new_gids,partition_in) do ghost_v,new_gids,gids
        n_global = PartitionedArrays.global_length(gids)
        ghost_to_new_gid = ghost_v
        ghost = PartitionedArrays.GhostIndices(n_global,ghost_to_new_gid,PartitionedArrays.ghost_to_owner(gids))
        ids = PartitionedArrays.replace_ghost(new_gids,ghost)
        perm = PartitionedArrays.local_permutation(gids)
        PartitionedArrays.permute_indices(ids,perm)
    end
end

function relabel_system(A,b,x0)
    new_partition = variable_partition_from_partition(PartitionedArrays.partition(axes(A,1)))
    # NB. The next line is provably not needed in your case and you can simply take new_row_partition = new_partition
    #new_row_partition = relabel_global_ids(PartitionedArrays.partition(axes(A,1)),new_partition)
    new_row_partition = new_partition
	new_col_partition = relabel_global_ids(PartitionedArrays.partition(axes(A,2)),new_partition)
    b_new = PartitionedArrays.PVector(PartitionedArrays.partition(b),new_row_partition)
    x0_new = PartitionedArrays.PVector(PartitionedArrays.partition(x0),new_col_partition)
    A_new = PartitionedArrays.PSparseMatrix(PartitionedArrays.partition(A),new_row_partition,new_col_partition)
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
	end

	row_partition = PartitionedArrays.uniform_partition(ranks, grid_size^3)

	#assemble the rhs, b
	virtual_zero = VirtualZero{Float64}()
	f = row_indices->laplacian_residuals(virtual_zero, grid_size, row_indices)
	IV = map(f, row_partition)
	I, V = PartitionedArrays.tuple_of_arrays(IV)
	b = PartitionedArrays.pvector!(I, V, row_partition) |> fetch

	#assemble the matrix, A
	g = rowindices->laplacian_u(virtual_zero, grid_size, rowindices)
	IJV = map(g, row_partition)
	I, J, V = PartitionedArrays.tuple_of_arrays(IJV)
	col_partition = row_partition

    A = PartitionedArrays.psparse!(I, J, V, row_partition, col_partition) |> fetch
	display(PartitionedArrays.partition(A))
    B = convert(GridapPETSc.PETScMatrix,A)
    PETSC.@check_error_code PETSC.MatView(B.mat[],PETSC.@PETSC_VIEWER_STDOUT_WORLD)  

    x0 = copy(b)
	A_new, b_new, x0_new = relabel_system(A, b, x0)

	println("Before solver")
	solver = GridapPETSc.PETScLinearSolver()
	println("After solver")

    ss = Gridap.symbolic_setup(solver,A_new)
	println("After symbolic setup")

    ns = Gridap.numerical_setup(ss,A_new)
	println("After numerical setup")

	x_new = solve!(x,ns,b_new)
	x .= x_new
	return x
end


GridapPETSc.Init(args=split("-ksp_type gmres -ksp_converged_reason -ksp_error_if_not_converged true -pc_type jacobi -ksp_rtol 1.0e-12"))

ts = Float64[]
ns = Int[]
for i = 2:8
	t = @elapsed x = solve_laplacian(2 ^ i; solve_in_parallel=true)
	push!(ts, t)
	push!(ns, 2 ^ (3 * i))
	if MPI.Comm_rank(MPI.COMM_WORLD) == 0
		@show i, t
	end
end
np = MPI.Comm_size(MPI.COMM_WORLD)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
	DelimitedFiles.writedlm("times_3dpartition_$(np).csv", hcat(ns, ts), ',')
end
