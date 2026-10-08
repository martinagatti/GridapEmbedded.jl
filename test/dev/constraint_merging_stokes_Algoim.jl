module DistributedAgFEM4P4estMeshes

using Gridap
using Gridap.Helpers
using GridapEmbedded
using GridapEmbedded.Interfaces: CUT
using GridapEmbedded.LevelSetCutters: Leaf
using GridapDistributed
using PartitionedArrays
using MPI

using Gridap.Geometry: get_faces
using Gridap.FESpaces
using Gridap.Arrays
using Gridap.ReferenceFEs: Quadrature

using P4est_wrapper
using GridapP4est
using GridapP4est: generate_local_fe_spaces_and_constraints

using GridapDistributed: DistributedCellField, DistributedCellDof

using DrWatson
using Plots

# REFERENCES
#
# 1. p4est: SCALABLE ALGORITHMS FOR PARALLEL ADAPTIVE MESH REFINEMENT ON FORESTS OF OCTREES
# 2. A GENERIC FINITE ELEMENT FRAMEWORK ON PARALLEL TREE-BASED ADAPTIVE MESHES https://arxiv.org/pdf/1907.03709
# 3. THE AGGREGATED UNFITTED FINITE ELEMENT METHOD ON PARALLEL TREE-BASED ADAPTIVE MESHES https://arxiv.org/pdf/2006.05373

# BRANCHES
# 1. GridapP4est.jl: https://github.com/gridap/GridapP4est.jl/tree/fe_space_on_triangulation
# 2. Gridap.jl: https://github.com/gridap/Gridap.jl/tree/constraints
# 3. GridapDistributed.jl: https://github.com/gridap/GridapDistributed.jl/tree/agfem
# 4. GridapEmbedded.jl: https://github.com/gridap/GridapEmbedded.jl/tree/distributed_aggregate_p4est_meshes

# MyFESpace, Gridap.Adaptivity.get_model(::OctreeDistributedDiscreteModel), and
# NonConformingGridTopology now live in GridapEmbedded.Distributed (promoted from
# this script) — see GridapEmbedded/src/Distributed/DistributedAgFEMNonConforming.jl
using GridapEmbedded.Distributed: NonConformingGridTopology
using GridapEmbedded.Distributed: generate_amr_constraints
using GridapEmbedded.Distributed: generate_agfem_constraints
using GridapEmbedded.Distributed: generate_constrained_fe_space
using GridapEmbedded.Distributed: MyFESpace

function generate_unfitted_model(distribute,ls;
                                 num_parts=1,initial_uniform_refs=4,degree=4)
  ranks = distribute(LinearIndices((num_parts,)))
  coarse_model = CartesianDiscreteModel((-1,1,0,1),(2,1))
  num_ghost_layers = 1
  dmodel = OctreeDistributedDiscreteModel(
    ranks,coarse_model,initial_uniform_refs;num_ghost_layers=num_ghost_layers
  )
  # From here on we illustrate how to refine and coarsen the model based on the cut geometry
  # Beware that is_c and fmodel_refine_coarsen_flags are distributed MPI arrays
  # https://partitionedarrays.github.io/PartitionedArrays.jl/stable/usage/#Basic-usage
  # See also tests from PartitionedArrays.jl and GridapDistributed.jl for more examples
  #
  # is_c[cell] is true iff the algoim surface (phase=CUT, the default) quadrature
  # on that cell is non-empty, i.e. iff the level set crosses the cell. This plays
  # the same role as `compute_bgcell_to_inoutcut(cutgeo,geo) .!= CUT` did with the
  # CSG-based cutter, without ever constructing a cutgeo.
  squad = Quadrature(algoim,ls,degree)
  _,is_c = CellQuadratureAndActiveMask(dmodel,squad)
  fmodel_refine_coarsen_flags =
    map(ranks,
        partition(get_cell_gids(dmodel.dmodel)),
        is_c) do rank,indices,is_c
      flags = zeros(Int,length(indices))
      flags .= nothing_flag
      tocoarsen = findall(!,is_c)
      flags[tocoarsen] .= coarsen_flag
      flags
  end
  fmodel,_ = Gridap.Adaptivity.adapt(dmodel,fmodel_refine_coarsen_flags);
  # Exercice: Plot the model at this stage to see the effect of the refinement and coarsening flags
  for i in 1:initial_uniform_refs-1
    squad = Quadrature(algoim,ls,degree)
    _,is_c = CellQuadratureAndActiveMask(fmodel,squad)
    fmodel_refine_coarsen_flags =
      map(partition(get_cell_gids(fmodel)),
          is_c) do indices,is_c
        flags = zeros(Int,length(indices))
        flags .= nothing_flag
        tocoarsen = findall(!,is_c)
        flags[tocoarsen] .= coarsen_flag
        flags
    end
    fmodel,_ = Gridap.Adaptivity.adapt(fmodel,fmodel_refine_coarsen_flags);
  end
  if num_parts > 1
    fmodel, _ = GridapDistributed.redistribute(fmodel)
  end
  fmodel = Gridap.Adaptivity.get_model(fmodel)
  return fmodel, ls
end

function generate_aggregates(fmodel, ls, degree)
  cell_gids = get_cell_gids(fmodel)
  ncgt = NonConformingGridTopology(fmodel)
  Ω = Triangulation(fmodel)
  # is_a: cell has non-empty OUT-phase quadrature (the domain we solve on)
  # is_c: cell has non-empty CUT-phase (boundary) quadrature
  vquad = Quadrature(algoim,ls,degree,phase=OUT)
  _,is_a = CellQuadratureAndActiveMask(Ω,vquad)
  squad = Quadrature(algoim,ls,degree)
  _,is_c = CellQuadratureAndActiveMask(Ω,squad)
  cell_to_root = map(
    local_views(Ω),partition(cell_gids),is_a,is_c,ncgt
  ) do trian,cell_indices,is_a,is_c,ncgt
    lid_to_gid = local_to_global(cell_indices)
    cell_to_lcellin = aggregate(trian,is_a,is_c,OUT;grid_topology=ncgt,lid_to_gid=lid_to_gid)
    cell_to_gcellin = zeros(Int,length(cell_to_lcellin))
    for i in eachindex(cell_to_gcellin)
      if !iszero(cell_to_lcellin[i])
        cell_to_gcellin[i] = lid_to_gid[cell_to_lcellin[i]]
      end
    end
    cell_to_gcellin
  end
  return cell_to_root
end

function in_fe_space_if_u_order_two()
  u(x) = VectorValue( x[1]*x[1], x[2] )
  p(x) = x[1]
  f(x) = -Δ(u)(x) + ∇(p)(x)
  g(x) = (∇⋅u)(x)
  ∇u(x) = ∇(u)(x)
  return (u, p, f, g, ∇u)
end

function out_fe_space()
  @notimplemented "Out FE space example is not implemented yet"
end

function solve_on_model(model,ls;
                        order=2::Integer,
                        problem::Tuple=in_fe_space_if_u_order_two(),
                        write_solution::Bool=false)

  D = 2
  degree = 2*order*D

  Ω = Triangulation(model)

  # v_cell_quad/is_a: OUT-phase quadrature and active-cell mask -> drives Ωa/dΩ
  # s_cell_quad/is_c: CUT-phase (boundary) quadrature and mask  -> drives Γ/dΓ
  # There is no algoim analogue of a trimmed PHYSICAL_OUT triangulation for
  # visualization: dΩ already integrates over the correctly clipped OUT region
  # on cut cells (it uses the OUT-phase quadrature points), but Ωa itself is the
  # (untrimmed) active-cell mesh, used only to build the FE space support.
  vquad = Quadrature(algoim,ls,degree,phase=OUT)
  v_cell_quad,is_a = CellQuadratureAndActiveMask(Ω,vquad)
  squad = Quadrature(algoim,ls,degree)
  s_cell_quad,is_c = CellQuadratureAndActiveMask(Ω,squad)

  Ωa,dΩ,_ = TriangulationAndMeasure(Ω,v_cell_quad,is_a)
  _,dΓ,_  = TriangulationAndMeasure(Ω,s_cell_quad,is_c)
  # [WARNING] Sign of normal is flipped because OUT domain: normal(ls,Ω) points
  # from IN (φ<0) to OUT (φ>0), i.e. outward from the aggregated IN region, which
  # is the *inward* direction w.r.t. the OUT domain we are solving on.
  nΓ = -normal(ls,Ω)

  if write_solution
    writevtk(Ω,datadir("model"));
    writevtk(Ωa,datadir("active"));
  end

  cell_to_root = generate_aggregates(model,ls,degree)

  if write_solution
    writevtk(Triangulation(model),datadir("aggregates"),celldata=["aggregate"=>cell_to_root]);
  end

  reffeᵘ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
  spacesᵘ = map(local_views(Ωa)) do trian
    TestFESpace(trian,reffeᵘ;conformity=:H1,dirichlet_tags="boundary")
      # dirichlet_tags=[1,2,5, 7,8, 3,4,6],
      # dirichlet_masks=[(false,true),(false,true),(false,true),
      #                  (false,true),(false,true),
      #                  (false,true),(false,true),(false,true)])
  end
  V = generate_constrained_fe_space(model, Ωa, spacesᵘ, reffeᵘ, cell_to_root)

  reffeᵖ = ReferenceFE(lagrangian,Float64,order-1,space=:P)
  spacesᵖ = map(local_views(Ωa)) do trian
    TestFESpace(trian,reffeᵖ)
  end
  Q = generate_constrained_fe_space(model, Ωa, spacesᵖ, reffeᵖ, cell_to_root)

  K = ConstantFESpace(model)

  u, p, f, g, ∇u = problem

  U = TrialFESpace(V,u)
  P = TrialFESpace(Q)
  L = TrialFESpace(K)

  Y = MultiFieldFESpace([V,Q,K])
  X = MultiFieldFESpace([U,P,L])

  # First sanity check: interpolate a FE function that can be exactly represented by the FE space that combines hanging and aggregate constraints
  uᵢ, pᵢ, _ = interpolate_everywhere([u,p,0.0],X)

  cell_meas = map(get_cell_measure∘Triangulation,local_views(Ω))
  meas = map(minimum,cell_meas) |> PartitionedArrays.getany
  h = meas^(1/D)
  γ = 10.0

  α(u,v) = ∫( ∇(u)⊙∇(v) )dΩ
  β(v,q) = ∫( q*(∇⋅v) )dΩ

  γˡ(u,v,p,q) = ∫( (γ/h)*(v⋅u) +
    v⋅(nΓ⋅∇(u)) + (nΓ⋅∇(v))⋅u - (p*nΓ)⋅v - (q*nΓ)⋅u )dΓ
  γʳ(v,q) = ∫( (γ/h)*(v⋅u) + (nΓ⋅∇(v))⋅u - (q*nΓ)⋅u )dΓ

  r(p,ℓ) = ∫( p*ℓ )dΩ

  a((u,p,l),(v,q,ℓ)) = 
    α(u,v) - β(v,p) - β(u,q) + 
    γˡ(u,v,p,q) + r(p,ℓ) + r(q,l)
  b((v,q,l)) = γʳ(v,q) + ∫( v⋅f - q*g )dΩ

  op = AffineFEOperator(a,b,X,Y)
  uₕ, pₕ, _ = solve(op)

  # Second sanity check: If the first sanity check passes, but the solution to the problem is incorrect, the most typical cause is a problem in the definition of the weak form, e.g. a missing term or a wrong sign.

  if write_solution
    # No PHYSICAL_OUT-equivalent trimmed triangulation is available with algoim
    # quadratures (only quadrature-point clouds), so only the active mesh is
    # written here; use AlgoimUtils.visualization_data(dΩ,...) for a point-cloud
    # export of the physical domain if needed.
    writevtk(Ωa,datadir("check_active"),
            cellfields=["ui"=>uᵢ,"eui"=>u-uᵢ,"p"=>pᵢ,"epi"=>p-pᵢ,
                        "uh"=>uₕ,"euh"=>u-uₕ,"ph"=>pₕ,"eph"=>p-pₕ]);
  end

end

# MAIN PROGRAMME STARTS HERE

ls = AlgoimCallLevelSetFunction(
  x -> ( x[1]*x[1] + x[2]*x[2] ) - 1.0,
  x -> VectorValue( 2*x[1], 2*x[2] ) )
initial_uniform_refs = 4
order = 2
problem = in_fe_space_if_u_order_two()
write_solution = true

# The model is generated by coarsening a maximumly refined model, so that the
# cut geometry is well resolved. If you start with an analytical LS of disk,
# you can start with a coarse model (initial_uniform_refs=1) and refine on cut
# cells until achieving the desired mesh size on the cuts
model, _ = with_mpi() do distribute
  generate_unfitted_model(distribute,ls,initial_uniform_refs=initial_uniform_refs)
end

# Although it does not run on a distributed environment,
# the input arrays are distributed, so all the derived
# types from them are distributed
solve_on_model(
  model, ls;
  order,
  problem,
  write_solution
)

end # module