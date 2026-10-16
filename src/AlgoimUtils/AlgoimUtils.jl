module AlgoimUtils

using CxxWrap
using Algoim
using Gridap

import Base.prod
@inline prod(x::Point{N,T}) where {N,T} = prod(x.data)

import Algoim: to_array
@inline to_array(x::Point{N,T}) where {N,T} = collect(x.data)

using Gridap.Helpers
using Gridap.ReferenceFEs
using Gridap.Arrays: collect1d, CompressedArray, Table
using Gridap.Adaptivity
using Gridap.Fields: testitem
using Gridap.CellData
using Gridap.CellData: GenericField, GenericCellField, get_data
using Gridap.CellData: _point_to_cell_cache, _point_to_cell!
using Gridap.Geometry

using PartitionedArrays
using GridapDistributed
using GridapDistributed: DistributedDiscreteModel
using GridapDistributed: DistributedTriangulation
using GridapDistributed: DistributedMeasure
using GridapDistributed: DistributedCellField
using GridapDistributed: add_ghost_cells
using GridapDistributed: DistributedFESpace
import GridapDistributed: local_views

using GridapP4est: OctreeDistributedDiscreteModel

using GridapEmbedded.Interfaces
using GridapEmbedded.Interfaces: Simplex

using MiniQhull
using FillArrays

import Algoim: CachedLevelSetValue
import Algoim: CachedLevelSetGradient
import Algoim: AlgoimCallLevelSetFunction
import Algoim: normal
import Gridap.ReferenceFEs: Quadrature

export CellQuadratureAndActiveMask
export TriangulationAndMeasure
export algoim
export Quadrature
export is_cell_active
export restrict_measure
export fill_cpp_data
export compute_closest_point_projections
export compute_normal_displacement
export compute_normal_displacement!
export compute_distance_fe_function
export delaunaytrian
export convexhull

struct Algoim <: QuadratureName end
const algoim = Algoim()

(φ::CachedLevelSetValue{<:CellField})(p,i::Float32) = begin
  _p = Point(p)
  (carr,cgetindex,ceval) = φ.c
  evaluate!(ceval,getindex!(cgetindex,carr,Int(i)),_p)
end

(φ::CachedLevelSetGradient{<:CellField})(p,i::Float32) = begin
  _p = Point(p)
  (carr,cgetindex,ceval) = φ.c
  _val = evaluate!(ceval,getindex!(cgetindex,carr,Int(i)),_p)
  ConstCxxRef(to_uvector(to_array(_val)))
end

function AlgoimCallLevelSetFunction(φ::CellField,∇φ::CellField)
  φtrian = get_triangulation(φ)
  ∇φtrian = get_triangulation(∇φ)
  xφ = testitem(testitem(get_cell_coordinates(φtrian)))
  x∇φ = testitem(testitem(get_cell_coordinates(∇φtrian)))
  cellφ = get_array(φ)
  cell∇φ = get_array(∇φ)
  cache_φ = (cellφ,array_cache(cellφ),return_cache(testitem(cellφ),xφ))
  cache_∇φ = (cell∇φ,array_cache(cell∇φ),return_cache(testitem(cell∇φ),x∇φ))
  AlgoimCallLevelSetFunction{typeof(φ),typeof(∇φ),typeof(cache_φ),typeof(cache_∇φ)}(φ,∇φ,cache_φ,cache_∇φ)
end

struct DistributedAlgoimCallLevelSetFunction{A<:AbstractArray{<:AlgoimCallLevelSetFunction}} <: GridapType
  values   ::DistributedCellField
  gradients::DistributedCellField
  levelsets::A
end

local_views(a::DistributedAlgoimCallLevelSetFunction) = a.levelsets

function AlgoimCallLevelSetFunction(φ::DistributedCellField,∇φ::DistributedCellField)
  levelsets = map((v,g)->AlgoimCallLevelSetFunction(v,g),local_views(φ),local_views(∇φ))
  DistributedAlgoimCallLevelSetFunction(φ,∇φ,levelsets)
end

function normal(phi::AlgoimCallLevelSetFunction,x::AbstractVector{<:Point},cell_id::Int=1)
  map(xi->normal(phi,xi,cell_id),x)
end

function normal(phi::AlgoimCallLevelSetFunction,trian::Triangulation)
  f = [ x -> normal(phi,x,i) for i in 1:num_cells(trian) ]
  f = map(GenericField,f)
  GenericCellField(f,trian,PhysicalDomain())
end

function normal(phi::AlgoimCallLevelSetFunction,trian::DistributedTriangulation)
  normals = map(t->normal(phi,t),local_views(trian))
  DistributedCellField(normals,trian)
end

function normal(phi::DistributedAlgoimCallLevelSetFunction,_trian::DistributedTriangulation)
  trian = add_ghost_cells(_trian)
  normals = map((φ,t)->normal(φ,t),local_views(phi),local_views(trian))
  DistributedCellField(normals,trian)
end

function normal(ls::AlgoimCallLevelSetFunction{<:CellField,<:CellField},x::Point,cell_id::Int=1)
  (carr,cgetindex,ceval) = ls.cache_∇φ
  gx = evaluate!(ceval,getindex!(cgetindex,carr,cell_id),x)
  gx/norm(gx)
end

function Quadrature(trian::Grid,::Algoim,phi::LevelSetFunction,degree::Int;kwargs...)
  ctype_polytope = map(get_polytope,get_reffes(trian))
  @notimplementedif !all(map(is_n_cube,ctype_polytope))
  cell_to_coords = get_cell_coordinates(trian)
  cell_to_bboxes = collect1d(lazy_map(a->(a[1],a[end]),cell_to_coords))
  jls = JuliaFunctionLevelSet(phi,Val{num_dims(trian)}())
  cell_to_quad = map(enumerate(cell_to_bboxes)) do (cell_id,bbox)
    bbmin, bbmax = bbox
    Quadrature(cell_id,bbmin,bbmax,jls,phi,degree;kwargs...)
  end
  CompressedArray(cell_to_quad,1:length(cell_to_quad))
end

function Quadrature(trian::Grid,::Algoim,phi::LevelSetFunction,
                    own_to_local::AbstractVector,degree::Int;kwargs...)
  ctype_polytope = map(get_polytope,get_reffes(trian))
  @notimplementedif !all(map(is_n_cube,ctype_polytope))
  cell_to_coords = get_cell_coordinates(trian)
  cell_to_bboxes = collect1d(lazy_map(a->(a[1],a[end]),cell_to_coords))
  jls = JuliaFunctionLevelSet(phi,Val{num_dims(trian)}())
  cell_to_quad = map(enumerate(cell_to_bboxes)) do (own_cell_id,bbox)
    bbmin, bbmax = bbox
    cell_id = own_to_local[own_cell_id]
    Quadrature(cell_id,bbmin,bbmax,jls,phi,degree;kwargs...)
  end
  CompressedArray(cell_to_quad,1:length(cell_to_quad))
end

function Quadrature(trian::Grid,::Algoim,
                    phi1::LevelSetFunction,phi2::LevelSetFunction,
                    degree::Int;kwargs...)
  ctype_polytope = map(get_polytope,get_reffes(trian))
  @notimplementedif !all(map(is_n_cube,ctype_polytope))
  cell_to_coords = get_cell_coordinates(trian)
  cell_to_bboxes = collect1d(lazy_map(a->(a[1],a[end]),cell_to_coords))
  jls1 = JuliaFunctionLevelSet(phi1,Val{num_dims(trian)}())
  jls2 = JuliaFunctionLevelSet(phi2,Val{num_dims(trian)}())
  cell_to_quad = map(enumerate(cell_to_bboxes)) do (cell_id,bbox)
    bbmin, bbmax = bbox
    Quadrature(cell_id,bbmin,bbmax,jls1,jls2,phi1,phi2,degree;kwargs...)
  end
  CompressedArray(cell_to_quad,1:length(cell_to_quad))
end

function Quadrature(cell_id::Int,
                    xmin::Point{N,T},
                    xmax::Point{N,T},
                    jls::LevelSetFunction,
                    phi::LevelSetFunction,
                    degree::Int;
                    phase::Int=CUT) where {N,T}
  coords, weights = fill_quad_data(jls,phi,xmin,xmax,phase,degree,cell_id)
  GenericQuadrature(coords,weights,"Algoim quadrature of degree $degree")
end

function Quadrature(cell_id::Int,
                    xmin::Point{N,T},
                    xmax::Point{N,T},
                    jls1::LevelSetFunction,
                    jls2::LevelSetFunction,
                    phi1::LevelSetFunction,
                    phi2::LevelSetFunction,
                    degree::Int;
                    phase1::Int=IN,
                    phase2::Int=IN) where {N,T}
  coords, weights = fill_quad_data(jls1,jls2,phi1,phi2,xmin,xmax,
                                   phase1,phase2,degree,cell_id)
  GenericQuadrature(coords,weights,"Algoim quadrature of degree $degree")
end

function is_cell_active(cell_quad::AbstractArray{<:Quadrature})
  has_non_empty_quad(x) = num_points(x) > 0
  lazy_map(has_non_empty_quad,cell_quad)
end

function is_cell_active(meas::Measure)
  is_cell_active(get_data(meas.quad))
end

function is_cell_active(meas::DistributedMeasure)
  map(local_views(meas)) do meas
    is_cell_active(get_data(meas.quad))
  end
end

# Splits the (possibly expensive) construction of the per-cell Algoim
# quadrature from the active-cell mask, so that both can be reused later
# (e.g. to build an aggregated model and only then the triangulation/measure)
# instead of recomputing the quadrature from scratch.
function _cell_quadrature_and_active_mask(trian::Grid,
                                          ::Algoim,phi,args;kwargs)
  cell_quad = Quadrature(trian,algoim,phi,args...;kwargs...)
  cell_to_is_active = is_cell_active(cell_quad)
  cell_quad, cell_to_is_active
end

function _cell_quadrature_and_active_mask(trian::DistributedTriangulation,
    ::Algoim,
    phi::AlgoimCallLevelSetFunction,
    args;kwargs)
  ltrians = local_views(trian)
  cell_quad = map(t->Quadrature(t,algoim,phi,args...;kwargs...),ltrians)
  cell_to_is_active = map(cq->is_cell_active(cq),cell_quad)
  cell_quad, cell_to_is_active
end

function _cell_quadrature_and_active_mask(trian::DistributedTriangulation,
    ::Algoim,
    phi::DistributedAlgoimCallLevelSetFunction,
    args;kwargs)
  ltrians = local_views(trian); lphis = local_views(phi)
  phitrian = get_triangulation(phi.values)
  gids = get_cell_gids(get_background_model(phitrian))
  own_to_local = map(local_views(phitrian),local_views(gids)) do t,g
    findall(!iszero,local_to_own(g)[t.tface_to_mface])
  end
  cell_quad = map(
    (t,p,otl)->Quadrature(t,algoim,p,otl,args...;kwargs...),ltrians,lphis,own_to_local)
  cell_to_is_active = map(cq->is_cell_active(cq),cell_quad)
  cell_quad, cell_to_is_active
end

function _cell_quadrature_and_active_mask(trian,
    name,_args::Tuple{<:AlgoimCallLevelSetFunction,Any};kwargs)
  phi, args = _args
  _cell_quadrature_and_active_mask(trian,name,phi,args;kwargs)
end

function _cell_quadrature_and_active_mask(trian,
    name,_args::Tuple{<:DistributedAlgoimCallLevelSetFunction,Any};kwargs)
  phi, args = _args
  _cell_quadrature_and_active_mask(trian,name,phi,args;kwargs)
end

function CellQuadratureAndActiveMask(trian,
    quad::Tuple{<:QuadratureName,Any,Any})
  name, args, kwargs = quad
  _cell_quadrature_and_active_mask(trian,name,args;kwargs)
end

function CellQuadratureAndActiveMask(
    model::Union{DiscreteModel,DistributedDiscreteModel},
    quad::Tuple{<:QuadratureName,Any,Any})
  CellQuadratureAndActiveMask(Triangulation(model),quad)
end

function restrict_measure(meas::Measure,trian::Triangulation)
  ocell_quad = meas.quad
  cell_quad   = lazy_map(Reindex(ocell_quad.cell_quad),trian.tface_to_mface)
  cell_point  = lazy_map(Reindex(ocell_quad.cell_point),trian.tface_to_mface)
  cell_weight = lazy_map(Reindex(ocell_quad.cell_weight),trian.tface_to_mface)
  dds = ocell_quad.data_domain_style
  ids = ocell_quad.integration_domain_style
  Measure(CellQuadrature(cell_quad,cell_point,cell_weight,trian,dds,ids))
end

function restrict_measure(meas::Measure,trian::Triangulation,gids::AbstractLocalIndices)
  oquad = meas.quad
  tface_to_own_mface = local_to_own(gids)[trian.tface_to_mface]
  cell_quad   = lazy_map(Reindex(oquad.cell_quad),tface_to_own_mface)
  cell_point  = lazy_map(Reindex(oquad.cell_point),tface_to_own_mface)
  cell_weight = lazy_map(Reindex(oquad.cell_weight),tface_to_own_mface)
  dds = oquad.data_domain_style
  ids = oquad.integration_domain_style
  Measure(CellQuadrature(cell_quad,cell_point,cell_weight,trian,dds,ids))
end

function restrict_measure(meas::DistributedMeasure,trian::DistributedTriangulation)
  gids = local_views(get_cell_gids(get_background_model(trian)))
  DistributedMeasure(map(restrict_measure,local_views(meas),local_views(trian),gids),trian)
end

@inline _measure(Ωbg,quad,phi::AlgoimCallLevelSetFunction) =
  map(lt->Measure(lt,quad,data_domain_style=PhysicalDomain()),local_views(Ωbg))

@inline function _measure(Ωbg,quad,phi::DistributedAlgoimCallLevelSetFunction)
  name, _args, kwargs = quad
  _, args = _args
  ltrians = local_views(Ωbg); lphis = local_views(phi)
  map((lt,lp)->Measure(lt,(name,(lp,args),kwargs),data_domain_style=PhysicalDomain()),ltrians,lphis)
end

function _triangulation_and_measure(Ωbg::Triangulation,quad::Tuple)
  dΩbg = Measure(Ωbg,quad,data_domain_style=PhysicalDomain())
  # RMK: This is a hack, but algoim interface does not let you
  # know if a (given) cell intersects the interior of the level
  # set. The hack consists in inferring this from the size of
  # each quadrature. I do not expect this hack implies a lot of
  # extra operations with regards to the proper way to do it.
  cell_to_is_active = is_cell_active(dΩbg)
  Ωᵃ = Triangulation(Ωbg,cell_to_is_active)
  dΩᵃ = restrict_measure(dΩbg,Ωᵃ)
  Ωᵃ,dΩᵃ,cell_to_is_active
end

function _triangulation_and_measure(Ωbg::DistributedTriangulation,quad::Tuple)
  name, _args, kwargs = quad
  phi, args = _args
  ltrians = local_views(Ωbg)
  dΩbg = _measure(Ωbg,(name,args,kwargs),phi)
  cell_to_is_active = map(lm->is_cell_active(lm),dΩbg)
  gids = local_views(get_cell_gids(get_background_model(Ωbg)))
  Ωᵃ = map((lt,ca)->Triangulation(lt,ca),ltrians,cell_to_is_active)
  dΩᵃ = map((db,at,gs)->restrict_measure(db,at,gs),dΩbg,Ωᵃ,gids)
  Mbg = get_background_model(Ωbg)
  DΩᵃ = DistributedTriangulation(Ωᵃ,Mbg)
  DΩᵃ,DistributedMeasure(dΩᵃ,DΩᵃ),cell_to_is_active
end

function _triangulation_and_measure(Ωbg::Triangulation,
                cell_quad::AbstractArray{<:Quadrature},
                cell_to_is_active::AbstractArray{Bool},
                cell_to_has_non_empty_quad::AbstractArray{Bool})
  dds = PhysicalDomain(); ids = PhysicalDomain()
  dΩbg = Measure(Ωbg,cell_quad,dds,ids)
  Ωᵃ = Triangulation(Ωbg,cell_to_is_active)
  dΩᵃ = restrict_measure(dΩbg,Triangulation(Ωbg,cell_to_has_non_empty_quad))
  Ωᵃ,dΩᵃ,cell_to_is_active
end

function _triangulation_and_measure(Ωbg::DistributedTriangulation,
                           cell_quad::AbstractArray,
                           cell_to_is_active::AbstractArray,
                           cell_to_has_non_empty_quad::AbstractArray)
  ltrians = local_views(Ωbg)
  dds = PhysicalDomain(); ids = PhysicalDomain()
  dΩbg = map((lt,cq)->Measure(lt,cq,dds,ids),ltrians,cell_quad)
  gids = local_views(get_cell_gids(get_background_model(Ωbg)))
  Ωᵃ = map((lt,ca)->Triangulation(lt,ca),ltrians,cell_to_is_active)
  Ωᵖ = map((lt,cp)->Triangulation(lt,cp),ltrians,cell_to_has_non_empty_quad)
  dΩᵃ = map((db,at,gs)->restrict_measure(db,at,gs),dΩbg,Ωᵖ,gids)
  Mbg = get_background_model(Ωbg)
  DΩᵃ = DistributedTriangulation(Ωᵃ,Mbg)
  DΩᵃ,DistributedMeasure(dΩᵃ,DΩᵃ),cell_to_is_active
end

function TriangulationAndMeasure(Ωbg,quad::Tuple)
  msg = "TriangulationAndMeasure can only receive the background triangulation"
  @notimplementedif num_cells(get_background_model(Ωbg)) != num_cells(Ωbg) msg
  _triangulation_and_measure(Ωbg,quad)
end

function TriangulationAndMeasure(Ωbg,cell_quad,cell_to_is_active,
      cell_to_has_non_empty_quad=cell_to_is_active)
  msg = "TriangulationAndMeasure can only receive the background triangulation"
  @notimplementedif num_cells(get_background_model(Ωbg)) != num_cells(Ωbg) msg
  _triangulation_and_measure(Ωbg,
                             cell_quad,
                             cell_to_is_active,
                             cell_to_has_non_empty_quad)
end

using Gridap.Geometry: get_cell_to_parent_cell
using Gridap.CellData: get_cell_quadrature

function compute_closest_point_projections(Ω::Triangulation,φ;
    cppdegree::Int=2,trim::Bool=false,limitstol::Float64=1.0e-8)
  compute_closest_point_projections(get_active_model(Ω),φ,
    cppdegree=cppdegree,trim=trim,limitstol=limitstol)
end

function compute_closest_point_projections(model::CartesianDiscreteModel,
                                           φ::AlgoimCallLevelSetFunction;
                                           cppdegree::Int=2,
                                           trim::Bool=false,
                                           limitstol::Float64=1.0e-8)
  cdesc = get_cartesian_descriptor(model)
  partition = Int32[cdesc.partition...]
  xmin = cdesc.origin
  xmax = xmin + Point(cdesc.sizes .* partition)
  fill_cpp_data(φ,partition,xmin,xmax,cppdegree,trim,limitstol)
end

function compute_closest_point_projections(fespace::FESpace,
                                           φ::AlgoimCallLevelSetFunction,
                                           order::Int;
                                           cppdegree::Int=2,
                                           trim::Bool=false,
                                           limitstol::Float64=1.0e-8)
  trian = get_triangulation(fespace)
  model = get_active_model(trian)
  # TO-ANSWER: Do I need the rmodel in this scope or not?
  rmodel = refine(model,order)
  cps = compute_closest_point_projections(
    rmodel,φ,order,cppdegree=cppdegree,trim=trim,limitstol=limitstol)
  msg = "Is the FE space order the same as the input order?"
  @assert length(cps) == num_free_dofs(fespace) msg
  cps = node_to_dof_order(cps,fespace,rmodel,order)
end

function compute_closest_point_projections(model::AdaptedDiscreteModel,
                                           φ::AlgoimCallLevelSetFunction,
                                           order::Int;
                                           cppdegree::Int=2,
                                           trim::Bool=false,
                                           limitstol::Float64=1.0e-8)
  reffe = ReferenceFE(lagrangian,Float64,order)
  rfespace = TestFESpace(model,reffe)
  _rφ = interpolate_everywhere(φ.φ,rfespace)
  rφ = AlgoimCallLevelSetFunction(_rφ,∇(_rφ))
  cdesc = get_cartesian_descriptor(get_model(model))
  partition = Int32[cdesc.partition...]
  xmin = cdesc.origin
  xmax = xmin + Point(cdesc.sizes .* partition)
  fill_cpp_data(rφ,partition,xmin,xmax,cppdegree,trim,limitstol)
end

function node_to_dof_order(cps,
                           fespace::FESpace,
                           rmodel::AdaptedDiscreteModel,
                           order::Int)
  D = num_dims(rmodel)
  cdesc = get_cartesian_descriptor(get_model(rmodel))
  partition = cdesc.partition
  orders = tfill(order,Val{D}())
  ones = tfill(1,Val{D}())
  range = CartesianIndices(orders.+1) .- CartesianIndex(ones) # 0-based
  ldof_to_lnode = get_ldof_to_lnode(orders,D)
  o2n_faces_map = rmodel.glue.o2n_faces_map
  node_partition = partition .+ 1
  ncells = num_cells(rmodel.parent)
  cell_node_ids = lazy_map(1:ncells) do cellid
    anchor_node = o2n_faces_map[cellid][1]
    node_ijk = CartesianIndices(partition)[anchor_node]
    range_cis = node_ijk .+ range
    node_ids = LinearIndices(node_partition)[range_cis]
    node_ids[ldof_to_lnode]
  end
  cell_dofs_ids = fespace.cell_dofs_ids
  c1 = array_cache(fespace.cell_dofs_ids)
  c2 = array_cache(cell_node_ids)
  ncps = similar(cps)
  for cellid in 1:ncells
    dof_ids = getindex!(c1,cell_dofs_ids,cellid)
    node_ids = getindex!(c2,cell_node_ids,cellid)
    ncps[dof_ids] = cps[node_ids]
  end
  ncps
end

function get_ldof_to_lnode(orders,D)

  # Generate indices of n-faces and order s.t.
  # (1) dimension-increasing (2) lexicographic
  bin_rang_nfaces = tfill(0:1,Val{D}())
  bin_ids_nfaces = vec(collect(Iterators.product(bin_rang_nfaces...)))
  sum_bin_ids_nfaces = sum.(bin_ids_nfaces)
  bin_ids_nfaces = permute!(bin_ids_nfaces,sortperm(sum_bin_ids_nfaces))

  # Generate LIs of basis funs s.t. order by n-faces
  lids_b = LinearIndices(Tuple([orders[i]+1 for i=1:D]))

  eet = eltype(eltype(bin_ids_nfaces))
  f(x) = Tuple( x[i] == one(eet) ? (0:0) : (1:orders[i]:(orders[i]+1)) for i in 1:length(x) )
  g(x) = Tuple( x[i] == one(eet) ? (2:orders[i]) : (0:0) for i in 1:length(x) )
  rang_nfaces = map(f,bin_ids_nfaces)
  rang_own_dofs = map(g,bin_ids_nfaces)

  perm = Int64[]
  for i = 1:length(bin_ids_nfaces)
    cis_nfaces = CartesianIndices(rang_nfaces[i])
    cis_own_dofs = CartesianIndices(rang_own_dofs[i])
    for ci in cis_nfaces
      ci = ci .+ cis_own_dofs
      perm = vcat(perm,reshape(lids_b[ci],length(ci)))
    end
  end

  perm
end

function compute_normal_displacement(
    cps::AbstractVector{<:Point},
    phi::AlgoimCallLevelSetFunction,
    fun,
    dt::Float64,
    Ω::Triangulation)
  # Note that cps must be (scalar) DoF-numbered, not lexicographic-numbered
  searchmethod = KDTreeSearch()
  cache1 = _point_to_cell_cache(searchmethod,Ω)
  x_to_cell(x) = _point_to_cell!(cache1, x)
  point_to_cell = lazy_map(x_to_cell, cps)
  cell_to_points, _ = make_inverse_table(point_to_cell, num_cells(Ω))
  cell_to_xs = lazy_map(Broadcasting(Reindex(cps)), cell_to_points)
  cell_point_xs = CellPoint(cell_to_xs, Ω, PhysicalDomain())
  fun_xs = evaluate(fun,cell_point_xs)
  nΓ_xs = evaluate(normal(phi,Ω),cell_point_xs)
  cell_point_disp = lazy_map(Broadcasting(⋅),fun_xs,nΓ_xs)
  cache_vals = array_cache(cell_point_disp)
  cache_ctop = array_cache(cell_to_points)
  disps = zeros(Float64,length(cps))
  for cell in 1:length(cell_to_points)
    pts = getindex!(cache_ctop,cell_to_points,cell)
    vals = getindex!(cache_vals,cell_point_disp,cell)
    for (i,pt) in enumerate(pts)
      val = vals[i]
      disps[pt] = dt * val
    end
  end
  disps
end

function compute_normal_displacement!(
    cache,
    cps::AbstractVector{<:Point},
    phi::AlgoimCallLevelSetFunction,
    fun,
    dt::Float64,
    Ω::Triangulation)
  # Note that cps must be (scalar) DoF-numbered, not lexicographic-numbered
  if isnothing(cache)
    searchmethod = KDTreeSearch()
    cache = _point_to_cell_cache(searchmethod,Ω)
  end
  x_to_cell(x) = _point_to_cell!(cache, x)
  point_to_cell = lazy_map(x_to_cell, cps)
  cell_to_points, _ = make_inverse_table(point_to_cell, num_cells(Ω))
  cell_to_xs = lazy_map(Broadcasting(Reindex(cps)), cell_to_points)
  cell_point_xs = CellPoint(cell_to_xs, Ω, PhysicalDomain())
  fun_xs = evaluate(fun,cell_point_xs)
  nΓ_xs = evaluate(normal(phi,Ω),cell_point_xs)
  cell_point_disp = lazy_map(Broadcasting(⋅),fun_xs,nΓ_xs)
  cache_vals = array_cache(cell_point_disp)
  cache_ctop = array_cache(cell_to_points)
  disps = zeros(Float64,length(cps))
  for cell in 1:length(cell_to_points)
    pts = getindex!(cache_ctop,cell_to_points,cell)
    vals = getindex!(cache_vals,cell_point_disp,cell)
    for (i,pt) in enumerate(pts)
      val = vals[i]
      disps[pt] = dt * val
    end
  end
  disps, cache
end

signed_distance(φ::Function,x,y) = sign(φ(y))*norm(x-y)

signed_distance(φ::T,x,y) where {T<:Number} = sign(φ)*norm(x-y)

function _compute_signed_distance(
    φ::AlgoimCallLevelSetFunction{<:Function,<:Function},
    cps::Vector{<:Point{N,T}},cos::Array{<:Point{N,T},N}) where {N,T}
  _dist(x,y) = signed_distance(φ.φ,x,y)
  lazy_map(_dist,cps,cos)
end

function _compute_signed_distance(
    φ::AlgoimCallLevelSetFunction{<:CellField,<:CellField},
    cps::Vector{<:Point{N,T}},cos::Array{<:Point{N,T},N}) where {N,T}
  φs = get_free_dof_values(φ.φ)
  lazy_map(signed_distance,φs,cps,cos)
end

function compute_distance_fe_function(
    bgmodel::CartesianDiscreteModel,
    fespace::FESpace,
    φ::AlgoimCallLevelSetFunction,
    order::Int;
    cppdegree::Int=2)
  cps = compute_closest_point_projections(
    fespace,φ,order,cppdegree=cppdegree)
  rmodel = refine(bgmodel,order)
  cos = get_node_coordinates(rmodel)
  cos = node_to_dof_order(cos,fespace,rmodel,order)
  dists = _compute_signed_distance(φ,cps,cos) 
  FEFunction(fespace,dists)
end

# --- Octree (p4est) level-set methods ----------------------------------------------
# Ported from upstream GridapEmbedded, branch distributed_agfem_with_algoim_quads @ b3658c1
# (src/AlgoimUtils/AlgoimUtils.jl), unchanged. Algoim's closest-point code needs a uniform
# grid, so φ is sampled on a virtual uniform grid at the finest level
# (n_c 2^L order + 1 nodes per direction, n_c the coarse partition, L = max_refinement_level),
# and the closest points are computed at the given coordinates (mesh nodes or free dofs).

function compute_closest_point_projections(model::OctreeDistributedDiscreteModel,
                                           φ::AlgoimCallLevelSetFunction,
                                           max_refinement_level::Int;
                                           cppdegree::Int=2,
                                           trim::Bool=false,
                                           limitstol::Float64=1.0e-8)

  # Uniform partition at the maximum refinement level
  coarse_desc = get_cartesian_descriptor(model.coarse_model)
  xmin = coarse_desc.origin
  coarse_partition = Int32[coarse_desc.partition...]
  fine_partition = coarse_partition .* Int32(2^max_refinement_level)
  xmax = xmin + Point(coarse_desc.sizes .* coarse_partition)

  # The level set values at the maximum refinement level
  fsizes = coarse_desc.sizes ./ Int32(2^max_refinement_level)
  fcells = fine_partition.+1
  fgrid = vec(map(i->xmin+Point((Tuple(i).-1).*fsizes),
              CartesianIndices((fcells...,))))
  fvals = φ.(fgrid)

  cpps = map(local_views(model)) do m

    # Coordinates of each local partition
    coords = get_node_coordinates(m)
    coords = map(i->get_array(i),coords)
    coords = reduce(vcat,coords)

    fill_cpp_data(fvals,fine_partition,xmin,xmax,coords,
                  cppdegree,trim,limitstol)
  end

  node_gids = get_face_gids(model,0)
  PVector(cpps,partition(node_gids)) |> consistent! |> wait
  cpps
end

function compute_closest_point_projections(model::OctreeDistributedDiscreteModel,
                                           V::DistributedFESpace,
                                           φ::AlgoimCallLevelSetFunction,
                                           order::Int,
                                           max_refinement_level::Int;
                                           cppdegree::Int=2,
                                           trim::Bool=false,
                                           limitstol::Float64=1.0e-8)

  # Uniform partition at the maximum refinement level
  coarse_desc = get_cartesian_descriptor(model.coarse_model)
  xmin = coarse_desc.origin
  coarse_partition = Int32[coarse_desc.partition...]
  fine_partition =
   coarse_partition .* ( Int32(2^max_refinement_level)*Int32(order) )
  xmax = xmin + Point(coarse_desc.sizes .* coarse_partition)

  # The level set values at the maximum refinement level
  fsizes =
    coarse_desc.sizes ./ ( Int32(2^max_refinement_level) * Int32(order) )
  fcells = fine_partition.+1
  fgrid = vec(map(i->xmin+Point((Tuple(i).-1).*fsizes),
              CartesianIndices((fcells...,))))
  fvals = φ.(fgrid)

  # Coordinates of free DoFs on each local partition
  coords = interpolate_everywhere(identity,V)
  coords_vals = get_free_dof_values(coords)

  cpps = map(local_views(model),local_views(coords_vals)) do m,coords
    fill_cpp_data(fvals,fine_partition,xmin,xmax,coords,
                  cppdegree,trim,limitstol)
  end

  FEFunction(V,PVector(cpps,partition(V.gids)))
end

function compute_closest_point_projections(
    model::OctreeDistributedDiscreteModel,
    topo::MPIArray{<:Gridap.Geometry.GridTopology},
    V::DistributedFESpace,
    φ::DistributedAlgoimCallLevelSetFunction,
    order::Int,
    max_refinement_level::Int;
    cppdegree::Int=2,
    trim::Bool=false,
    limitstol::Float64=1.0e-8)

  # Uniform partition at the maximum refinement level
  coarse_desc = get_cartesian_descriptor(model.coarse_model)
  xmin = coarse_desc.origin
  coarse_partition = Int32[coarse_desc.partition...]
  fine_partition =
   coarse_partition .* ( Int32(2^max_refinement_level) * Int32(order) )
  xmax = xmin + Point(coarse_desc.sizes .* coarse_partition)

  # The level set values at the maximum refinement level
  fsizes =
    coarse_desc.sizes ./ ( Int32(2^max_refinement_level) * Int32(order) )
  fcells = fine_partition .+ 1
  fgrid = vec(map(i->xmin+Point((Tuple(i).-1).*fsizes),
              CartesianIndices((fcells...,))))
  iφ = map(local_views(φ.values),topo) do lφ,topo
    vertex_to_cells = get_faces(topo,0,num_cell_dims(model))
    sm = Gridap.CellData.KDTreeSearch(vertex_to_cells=vertex_to_cells)
    Gridap.CellData.Interpolable(lφ,searchmethod=sm)
  end |> GridapDistributed.DistributedInterpolable
  # Computational hotspot of procedure follows
  # Point-wise evaluation (faster than vector of points)
  fvals = iφ(fgrid)

  # Coordinates of free DoFs on each local partition
  coords = interpolate_everywhere(identity,V)
  coords_vals = get_free_dof_values(coords)

  cpps = map(local_views(model),local_views(coords_vals)) do m,coords
    fill_cpp_data(fvals,fine_partition,xmin,xmax,coords,
                  cppdegree,trim,limitstol)
  end

  FEFunction(V,PVector(cpps,partition(V.gids)))
end

function compute_normal_displacement!(
    cache,
    uₕ::CellField,
    cpₕ::FEFunction,
    phi::AlgoimCallLevelSetFunction,
    dt::Float64,
    Ω::Triangulation;
    searchmethod=KDTreeSearch())
  # Note that cps must be (scalar) DoF-numbered, not lexicographic-numbered
  # searchmethod: point location of the closest points in Ω (used only when cache === nothing);
  # on an octree pass KDTreeSearch(vertex_to_cells=...) built from the non-conforming topology.
  if isnothing(cache)
    cache = _point_to_cell_cache(searchmethod,Ω)
  end
  x_to_cell(x) = _point_to_cell!(cache, x)
  cps = map(Point,eachcol(reshape(get_free_dof_values(cpₕ),num_dims(Ω),:)))
  # Locate each point once (a lazy array would be re-evaluated many times by the sort inside
  # make_inverse_table, repeating the KD-tree search and inverse-map solve per access).
  point_to_cell = collect(map(x_to_cell, cps))
  cell_to_points, _ = make_inverse_table(point_to_cell, num_cells(Ω))
  cell_to_xs = lazy_map(Broadcasting(Reindex(cps)), cell_to_points)
  cell_point_xs = CellPoint(cell_to_xs, Ω, PhysicalDomain())
  fun_xs = evaluate(uₕ,cell_point_xs)
  nΓ_xs = evaluate(normal(phi,Ω),cell_point_xs)
  cell_point_disp = lazy_map(Broadcasting(⋅),fun_xs,nΓ_xs)
  cache_vals = array_cache(cell_point_disp)
  cache_ctop = array_cache(cell_to_points)
  disps = zeros(Float64,length(cps))
  for cell in 1:length(cell_to_points)
    pts = getindex!(cache_ctop,cell_to_points,cell)
    vals = getindex!(cache_vals,cell_point_disp,cell)
    for (i,pt) in enumerate(pts)
      val = vals[i]
      disps[pt] = dt * val
    end
  end
  disps, cache
end

# Octree normal displacement, d_i = dt (u⋅n)(c_i) at the free dofs of φ's scalar space, 1 rank.
# - Ω: the background triangulation of the octree model. The closest points c_i are located in it
#   with the non-conforming vertex→cells map of `topo` (NonConformingGridTopology, or the conforming
#   topology on a never-adapted octree), which also finds the coarse neighbour across a hanging
#   face. uₕ may live on a sub-triangulation of Ω (e.g. the narrow band): it is evaluated at the
#   located background cells, as the serial uniform-mesh method does.
# - cpₕ: closest points as a vector FE function (free values interleaved D per node, in the order
#   of the scalar free dofs), from compute_closest_point_projections(model,[topo,]V,φ,order,L).
# Returns the per-rank displacement vectors (scalar free-dof order) and the per-rank caches.
# Several ranks would need the closest points owned by another rank to be exchanged (see the
# upstream distributed Cartesian compute_normal_displacement); not implemented.
function compute_normal_displacement!(
    cache,
    uₕ::DistributedCellField,
    cpₕ::DistributedCellField,
    phi::DistributedAlgoimCallLevelSetFunction,
    dt::Float64,
    Ω::DistributedTriangulation,
    topo::MPIArray{<:Gridap.Geometry.GridTopology})
  @notimplementedif length(local_views(Ω)) != 1 "Octree normal displacement: only 1 rank is implemented"
  caches = isnothing(cache) ? map(_->nothing,local_views(Ω)) : cache
  out = map(caches,local_views(uₕ),local_views(cpₕ),local_views(phi),local_views(Ω),topo
           ) do c,u,cp,φ,Ωl,tp
    sm = KDTreeSearch(vertex_to_cells=get_faces(tp,0,num_cell_dims(Ωl)))
    compute_normal_displacement!(c,u,cp,φ,dt,Ωl;searchmethod=sm)
  end
  map(first,out), map(last,out)
end

@inline signed_distance(φ::AlgoimCallLevelSetFunction,x,y) = signed_distance(φ.φ,x,y)

# Closest points and coordinates as flat vectors of length D*ndists (free-dof order).
function _compute_signed_distance(
    φ::AlgoimCallLevelSetFunction{<:Function,<:Function},
    cps::Vector{T},cos::Vector{T},ndists::Int,D::Int ) where {T}
  map(1:ndists) do i
    c = view(cos,D*(i-1)+1:D*i)
    p = view(cps,D*(i-1)+1:D*i)
    signed_distance(φ,p,c)
  end
end

function _compute_signed_distance(
    φ::AlgoimCallLevelSetFunction{<:CellField,<:CellField},
    cps::Vector{T},cos::Vector{T},ndists::Int,D::Int ) where {T}
  φs = get_free_dof_values(φ.φ)
  msg = """\n
    Check that FE space for LS function
    and closest point projections match"""
  @assert length(φs) == ndists msg
  map(1:ndists) do i
    c = view(cos,D*(i-1)+1:D*i)
    p = view(cps,D*(i-1)+1:D*i)
    signed_distance(φs[i],p,c)
  end
end

function compute_distance_fe_function(
    bgmodel::OctreeDistributedDiscreteModel,
    fespace_scalar_type::DistributedFESpace,
    fespace_vector_type::DistributedFESpace,
    φ::AlgoimCallLevelSetFunction,
    order::Int,
    max_refinement_level::Int;
    cppdegree::Int=2)

  cps = compute_closest_point_projections(
    bgmodel,fespace_vector_type,φ,order,
    max_refinement_level,cppdegree=cppdegree)
  cps_vals = get_free_dof_values(cps)

  coords = interpolate_everywhere(identity,fespace_vector_type)
  coords_vals = get_free_dof_values(coords)

  _dists = map(local_views(fespace_scalar_type),
               local_views(cps_vals),
               local_views(coords_vals)) do fs,cp,cos
    _compute_signed_distance(φ,cp,cos,num_free_dofs(fs),
      num_dims(bgmodel.coarse_model))
  end
  dists = PVector(_dists,partition(fespace_scalar_type.gids))
  FEFunction(fespace_scalar_type,dists)
end

function compute_distance_fe_function(
    bgmodel::OctreeDistributedDiscreteModel,
    topo::MPIArray{<:Gridap.Geometry.GridTopology},
    fespace_scalar_type::DistributedFESpace,
    fespace_vector_type::DistributedFESpace,
    φ::DistributedAlgoimCallLevelSetFunction,
    order::Int,
    max_refinement_level::Int;
    cppdegree::Int=2)

  cps = compute_closest_point_projections(
    bgmodel,topo,fespace_vector_type,φ,order,
    max_refinement_level,cppdegree=cppdegree)
  cps_vals = get_free_dof_values(cps)

  coords = interpolate_everywhere(identity,fespace_vector_type)
  coords_vals = get_free_dof_values(coords)

  _dists = map(local_views(fespace_scalar_type),
               local_views(cps_vals),
               local_views(coords_vals),
               local_views(φ)) do fs,cp,cos,φl
    _compute_signed_distance(φl,cp,cos,num_free_dofs(fs),
      num_dims(bgmodel.coarse_model))
  end
  dists = PVector(_dists,partition(fespace_scalar_type.gids))
  FEFunction(fespace_scalar_type,dists)
end

function compute_distance_fe_function(
    fespace_scalar_type::FESpace,
    fespace_vector_type::FESpace,
    closest_point_projections::FEFunction,
    φ::AlgoimCallLevelSetFunction)
  model = get_background_model(get_triangulation(fespace_scalar_type))
  D = num_dims(model)
  cos = get_free_dof_values(interpolate(identity,fespace_vector_type))
  cps = get_free_dof_values(closest_point_projections)
  ndists = num_free_dofs(fespace_scalar_type)
  dists = _compute_signed_distance(φ,cps,cos,ndists,D)
  FEFunction(fespace_scalar_type,dists)
end

abstract type QhullType end

struct DelaunayTrian <: QhullType end
const delaunaytrian = DelaunayTrian()

struct ConvexHull <: QhullType end
const convexhull = ConvexHull()

get_flags(::QhullType) = @abstractmethod
get_flags(::DelaunayTrian) = "qhull d Qt Qbb Qc Qz"
get_flags(::ConvexHull) = "qhull Qt Qc"

get_dimension(::QhullType,dim) = @abstractmethod
get_dimension(::DelaunayTrian,dim) = dim
get_dimension(::ConvexHull,dim) = dim-1

import Gridap.Visualization: visualization_data

function visualization_data(meas::Measure,filename;cellfields=Dict(),qhulltype=DelaunayTrian())
  node_coordinates = collect(Iterators.flatten(meas.quad.cell_point.values))
  grid = _to_grid(node_coordinates,qhulltype)
  ndata = Dict()
  for (k,v) in cellfields
    pts = get_cell_points(meas)
    eval = evaluate(v,pts)
    ndata[k] = collect(Iterators.flatten(eval))
  end
  visualization_data(grid,filename,nodaldata=ndata)
end

function visualization_data(meas::Vector{<:Measure},filename;cellfields=Dict(),qhulltype=DelaunayTrian())
  node_coordinates = vcat(map(m->collect(Iterators.flatten(m.quad.cell_point.values)),meas)...)
  grid = _to_grid(node_coordinates,qhulltype)
  ndata = Dict()
  for (k,v) in cellfields
    pts = map(m->get_cell_points(m),meas)
    eval = map(p->evaluate(v,p),pts)
    ndata[k] = vcat(map(e->collect(Iterators.flatten(e)),eval)...)
  end
  visualization_data(grid,filename,nodaldata=ndata)
end

function _to_grid(node_coordinates::Vector{<:Point{Dp,Tp}},qhulltype) where {Dp,Tp}
  d = get_dimension(qhulltype,Dp)
  connectivity = delaunay(reinterpret(node_coordinates),get_flags(qhulltype))[1:(d+1),:]
  cell_node_ids = Table(collect(eachcol(connectivity)))
  reffes = [LagrangianRefFE(Float64,Simplex(Val{d}()),1)]
  cell_types = collect(Fill(Int8(1),length(cell_node_ids)))
  UnstructuredGrid(node_coordinates,cell_node_ids,reffes,cell_types)
end

end # module