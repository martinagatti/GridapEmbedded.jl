# Support for combining AMR hanging-node constraints (non-conforming p4est meshes)
# with AgFEM aggregation constraints into a single DistributedFESpace. Promoted from
# GridapEmbedded/test/dev/constraint_merging_stokes.jl.

using GridapP4est
using GridapP4est: OctreeDistributedDiscreteModel
using GridapP4est: NonConformingGlue

using Gridap.Arrays: Table, length_to_ptrs!
using Gridap.Arrays: append_tables_locally
using Gridap.Geometry: GridTopology
using Gridap.Geometry: UnstructuredGridTopology
using Gridap.Geometry: get_grid_topology
import Gridap.Geometry: get_faces
import Gridap.Geometry: OrientationStyle, RegularityStyle
import Gridap.Geometry: get_cell_type, get_polytopes
import Gridap.ReferenceFEs: get_vertex_coordinates

# ---------------------------------------------------------------------------
# NonConformingGridTopology
#
# get_faces(model) on an OctreeDistributedDiscreteModel's local views returns the
# conforming (fine-mesh) topology, which does not know about hanging faces. Cell
# aggregation (AgFEM) needs face-to-cell/cell-to-face connectivity that accounts for
# hanging faces so that aggregation does not cross non-conforming interfaces
# incorrectly. This wraps the conforming topology with the hanging-face glue info.
# ---------------------------------------------------------------------------

struct NonConformingGridTopology{Dc,Dp,T,O} <: GridTopology{Dc,Dp}
  conforming_grid_topology::UnstructuredGridTopology{Dc,Dp,T,O}
  coarse_cell_to_hanging_faces::Vector{Table{Int32,Vector{Int32},Vector{Int32}}}
  hanging_faces_to_coarse_cell::Vector{Table{Int32,Vector{Int32},Vector{Int32}}}
end

function NonConformingGridTopology(model::OctreeDistributedDiscreteModel)
  map(local_views(model),model.non_conforming_glue) do model,ncg
    conforming_grid_topology = get_grid_topology(model)
    coarse_cell_to_hanging_faces, hanging_faces_to_coarse_cell =
      generate_hanging_faces_to_coarse_cell_glue(ncg,
                                                 num_cells(model),
                                                 num_dims(model))
    NonConformingGridTopology(conforming_grid_topology,
                              coarse_cell_to_hanging_faces,
                              hanging_faces_to_coarse_cell)
  end
end

function generate_hanging_faces_to_coarse_cell_glue(
  ncg::NonConformingGlue,num_cells::Int,D::Int)

  hanging_faces_to_cell_data =
    map(ncg.hanging_faces_glue) do glue
      isempty(glue) ? Int32[] : map(x -> Int32(first(x)), glue)
    end

  hanging_faces_to_cell_ptrs =
    map(Broadcasting(zeros),tfill(Int32,Val{D}()),
        ncg.num_regular_faces.+ncg.num_hanging_faces.+1)
  map(_ones_on_hanging_faces!,hanging_faces_to_cell_ptrs,
                              ncg.num_hanging_faces)
  map(length_to_ptrs!,hanging_faces_to_cell_ptrs)

  cell_to_hanging_faces_data =
    map(sortperm,hanging_faces_to_cell_data)
  cell_to_hanging_faces_data =
    map(Broadcasting(+),cell_to_hanging_faces_data,ncg.num_regular_faces)

  cell_to_hanging_faces_ptrs =
    map(_count_owned_hanging_faces,
        hanging_faces_to_cell_data,
        tfill(num_cells,Val{D}()))
  map(length_to_ptrs!,cell_to_hanging_faces_ptrs)

  # Convert data from Int64 to Int32
  cell_to_hanging_faces_data =
    map(Broadcasting(Int32),cell_to_hanging_faces_data)
  hanging_faces_to_cell_data =
    map(Broadcasting(Int32),hanging_faces_to_cell_data)

  coarse_cell_to_hanging_faces = map(cell_to_hanging_faces_data,
                                     cell_to_hanging_faces_ptrs) do d,p
    Table(d,p)
  end
  hanging_faces_to_coarse_cell = map(hanging_faces_to_cell_data,
                                     hanging_faces_to_cell_ptrs) do d,p
    Table(d,p)
  end

  coarse_cell_to_hanging_faces, hanging_faces_to_coarse_cell
end

function _ones_on_hanging_faces!(
  ptrs::Vector{Int32},num_hanging_faces::Int)
  ptrs[end-num_hanging_faces+1:end] .= 1
  return nothing
end

function _count_owned_hanging_faces(x::AbstractArray{<:Integer},num_cells::Int)
  r = zeros(Int32,num_cells+1)
  length(x) == 0 && return r
  for xi in x
    r[xi+1] += 1
  end
  return r
end

OrientationStyle(
  ::Type{<:NonConformingGridTopology{Dc,Dp,T,O}}) where
  {Dc,Dp,T,O} = O()

RegularityStyle(
  ::Type{<:NonConformingGridTopology{Dc,Dp,T,O}}) where
  {Dc,Dp,T,O} = Irregular()

get_vertex_coordinates(ncgt::NonConformingGridTopology) =
  ncgt.conforming_grid_topology.vertex_coordinates

get_cell_type(ncgt::NonConformingGridTopology) =
  ncgt.conforming_grid_topology.cell_type

get_polytopes(ncgt::NonConformingGridTopology) =
  collect1d(ncgt.conforming_grid_topology.polytopes)

function get_faces(ncgt::NonConformingGridTopology{Dc,Dp,T,O},
                   dimfrom::Integer,dimto::Integer) where {Dc,Dp,T,O}
  get_faces(ncgt,Val{dimfrom}(),Val{dimto}())
end

function get_faces(ncgt::NonConformingGridTopology{Dc,Dp,T,O},
                   ::Val{Df},::Val{Dt}) where {Dc,Dp,T,O,Df,Dt}
  get_faces(ncgt.conforming_grid_topology,Df,Dt)
end

function get_faces(ncgt::NonConformingGridTopology{Dc,Dp,T,O},
                   ::Val{Dc},::Val{Dt}) where {Dc,Dp,T,O,Dt}
  r = get_faces(ncgt.conforming_grid_topology,Dc,Dt)
  h = ncgt.coarse_cell_to_hanging_faces[Dt+1]
  append_tables_locally(r,h)
end

function get_faces(ncgt::NonConformingGridTopology{Dc,Dp,T,O},
                   ::Val{Df},::Val{Dc}) where {Dc,Dp,T,O,Df}
  r = get_faces(ncgt.conforming_grid_topology,Df,Dc)
  h = ncgt.hanging_faces_to_coarse_cell[Df+1]
  append_tables_locally(r,h)
end

# TODO: Overload case dimfrom == dimto == Dc to avoid ambiguity
function get_faces(ncgt::NonConformingGridTopology{Dc,Dp,T,O},
                   ::Val{Dc},::Val{Dc}) where {Dc,Dp,T,O}
  get_faces(ncgt.conforming_grid_topology,Dc,Dc)
end

# ---------------------------------------------------------------------------
# MyFESpace
#
# Builds a DistributedFESpace with linear constraints from per-rank slave-DOF
# tables, respecting the input triangulation (without adding back all the ghosts).
# ---------------------------------------------------------------------------

function MyFESpace(
  trian::DistributedTriangulation,reffe,spaces,
  sDOF_to_dof,sDOF_to_dofs,sDOF_to_coeffs;split_own_and_ghost=false,constraint=nothing,kwargs...
)
  myfespaces = map(sDOF_to_dof,sDOF_to_dofs,sDOF_to_coeffs,spaces) do sDOF_to_dof,sDOF_to_dofs,sDOF_to_coeffs,spaces
    FESpaceWithLinearConstraints(sDOF_to_dof,
                                 sDOF_to_dofs,
                                 sDOF_to_coeffs,
                                 spaces)
  end
  gids = generate_gids(trian,myfespaces)
  vector_type = _find_vector_type(myfespaces,gids;split_own_and_ghost=split_own_and_ghost)
  space = DistributedSingleFieldFESpace(myfespaces,gids,trian,vector_type)
  return GridapDistributed._add_distributed_constraint(space,reffe,constraint)
end

# ---------------------------------------------------------------------------
# AMR + AgFEM constraint merging
#
# Combines hanging-node constraints coming from AMR (non-conforming p4est
# refinement) and aggregation constraints coming from AgFEM into a single
# DistributedFESpace with linear constraints, preferring the AMR constraints
# in case of conflict (i.e. for ill-posed AgFEM-aggregated DOFs that are also
# hanging).
# ---------------------------------------------------------------------------

function generate_amr_constraints(model, trian, spaces, reffe)
  cell_gids = get_cell_gids(model)
  models, non_conforming_glue = GridapP4est._generate_active_models_and_non_conforming_glue(
    model.pXest_type, model.pXest_refinement_rule_type, trian, cell_gids, model.non_conforming_glue
  )
  sDOF_to_dof, sDOF_to_dofs, sDOF_to_coeffs = GridapP4est.generate_constraints(
    model.pXest_refinement_rule_type, reffe, models, non_conforming_glue, spaces
  )
  # [TODO] amr dofs are Int64, agg dofs are Int32, we must unify this
  sDOF_to_dofs = map(sDOF_to_dofs) do sDOF_to_dofs
    Table(Int32.(sDOF_to_dofs.data), sDOF_to_dofs.ptrs)
  end
  return sDOF_to_dof, sDOF_to_dofs, sDOF_to_coeffs
end

function generate_agfem_constraints(trian, spaces, bgcell_to_bgroot)
  shfns = DistributedCellField(map(get_fe_basis, spaces), trian)
  dofs = DistributedCellDof(map(get_fe_dof_basis, spaces), trian)
  bgcell_to_gcell = map(local_views(trian)) do trian
    glue = get_glue(trian,Val(num_cell_dims(trian)))
    glue.mface_to_tface
  end
  sDOF_gids, mfdof_gids, mddof_gids, mDOF_to_dof, sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs =
  generate_aggregated_space_constraints(
    trian, spaces, bgcell_to_bgroot, shfns, dofs, bgcell_to_gcell
  )
  return sDOF_gids, mfdof_gids, mddof_gids, mDOF_to_dof, sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs
end

function generate_constrained_fe_space(model, Ωa, spaces, reffe, cell_to_root)
  amr_sDOF_to_dof, amr_sDOF_to_dofs, amr_sDOF_to_coeffs = generate_amr_constraints(model,Ωa,spaces,reffe);
  _, _, _, agg_mDOF_to_dof, agg_sDOF_to_dof, agg_sDOF_to_mdofs, agg_sDOF_to_coeffs =
    generate_agfem_constraints(Ωa,spaces,cell_to_root);

  agg_sDOF_to_dofs = map(agg_mDOF_to_dof, agg_sDOF_to_mdofs) do mDOF_to_dof, sDOF_to_mdofs
    T = eltype(sDOF_to_mdofs.data)
    data = Vector{T}(undef, length(sDOF_to_mdofs.data))
    @inbounds for i in eachindex(sDOF_to_mdofs.data)
      mdof = sDOF_to_mdofs.data[i]
      data[i] = ifelse(mdof > zero(T), mDOF_to_dof[mdof], mdof)
    end
    return Table(data, sDOF_to_mdofs.ptrs)
  end

  # Remove ill-posed constraints that are also hanging
  agg_sDOF_keep = map(
    spaces, amr_sDOF_to_dof, amr_sDOF_to_dofs, agg_sDOF_to_dof, agg_sDOF_to_dofs,
  ) do space, amr_sDOF_to_dof, amr_sDOF_to_dofs, agg_sDOF_to_dof, agg_sDOF_to_dofs
    ndofs = num_free_dofs(space)
    is_agg_dof = falses(ndofs)
    is_agg_dof[agg_sDOF_to_dof] .= true
    amr_only_sDOF = findall(dof->!is_agg_dof[dof],amr_sDOF_to_dof)
    is_mfdof_of_well_posed_hanging = falses(ndofs)
    for sDOF in amr_only_sDOF
      masters = amr_sDOF_to_dofs[sDOF]
      fmasters = masters[masters .> 0]
      is_mfdof_of_well_posed_hanging[fmasters] .= true
    end
    keep_agg_constraint = falses(length(agg_sDOF_to_dof))
    for (sDOF,dof) in enumerate(agg_sDOF_to_dof)
      keep_agg_constraint[sDOF] = !is_mfdof_of_well_posed_hanging[dof]
    end
    return findall(keep_agg_constraint)
  end;
  agg_sDOF_to_dof, agg_sDOF_to_dofs, agg_sDOF_to_coeffs = map(
    agg_sDOF_keep, agg_sDOF_to_dof, agg_sDOF_to_dofs, agg_sDOF_to_coeffs
  ) do agg_sDOF_keep, agg_sDOF_to_dof, agg_sDOF_to_dofs, agg_sDOF_to_coeffs
    return agg_sDOF_to_dof[agg_sDOF_keep], agg_sDOF_to_dofs[agg_sDOF_keep], agg_sDOF_to_coeffs[agg_sDOF_keep]
  end |> tuple_of_arrays;

  # Merge and close the constraint tables, preferring the AMR constraints in case of conflict
  # for ill-posed dofs that are also hanging
  sDOF_to_dof, sDOF_to_dofs, sDOF_to_coeffs = map(
    spaces, agg_sDOF_to_dof, agg_sDOF_to_dofs, agg_sDOF_to_coeffs, amr_sDOF_to_dof, amr_sDOF_to_dofs, amr_sDOF_to_coeffs
  ) do space, agg_sDOF_to_dof, agg_sDOF_to_dofs, agg_sDOF_to_coeffs, amr_sDOF_to_dof, amr_sDOF_to_dofs, amr_sDOF_to_coeffs
    on_conflict(dof, dofs1, coeffs1, offset1, dofs2, coeffs2, offset2) = (dofs2, coeffs2, offset2)
    sDOF_to_dof, sDOF_to_dofs, sDOF_to_coeffs, sDOF_to_offsets = FESpaces.merge_slave_constraint_tables(
      space,
      agg_sDOF_to_dof, agg_sDOF_to_dofs, agg_sDOF_to_coeffs,
      amr_sDOF_to_dof, amr_sDOF_to_dofs, amr_sDOF_to_coeffs;
      on_conflict
    )
    sDOF_to_dof, sDOF_to_dofs, sDOF_to_coeffs, sDOF_to_offsets = FESpaces.close_slave_constraint_tables(
      space, sDOF_to_dof, sDOF_to_dofs, sDOF_to_coeffs, sDOF_to_offsets
    )
  end |> tuple_of_arrays;

  V = MyFESpace(Ωa,reffe,spaces,
                sDOF_to_dof,sDOF_to_dofs,sDOF_to_coeffs)
  return V
end
