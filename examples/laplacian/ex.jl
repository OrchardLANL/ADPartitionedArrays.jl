import ADPartitionedArrays
import DelimitedFiles
import MPI
import HYPRE
import PartitionedArrays
MPI.Init()

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
		np = 1
		ranks = LinearIndices((np,))
	end
	row_partition = PartitionedArrays.uniform_partition(ranks, grid_size ^ 3)
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
	#solve A*x=b
	solver = HYPRE.BoomerAMG(; PrintLevel=-1, Tol=1e-8, MaxIter=10 ^ 3)
	x = HYPRE.solve(solver, A, b)
	return x
end

ts = Float64[]
ns = Int[]
for i = 2:9
	t = @elapsed x = solve_laplacian(2 ^ i)
	push!(ts, t)
	push!(ns, 2 ^ (3 * i))
	@show i, t
end
np = MPI.Comm_size(MPI.COMM_WORLD)
DelimitedFiles.writedlm("times_$(np).csv", hcat(ns, ts), ',')
