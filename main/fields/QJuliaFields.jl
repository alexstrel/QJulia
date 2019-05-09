module QJuliaFields

#load path to qjulia home directory
push!(LOAD_PATH, joinpath(@__DIR__, "../..", "core"))
push!(LOAD_PATH, @__DIR__)

import QJuliaRegisters
import QJuliaEnums
import QJuliaGrid


const complex_length = 2

abstract type QJuliaGenericField_qj end

mutable struct QJuliaLatticeField_qj <: QJuliaGenericField_qj
  # Field Matrix{T}, e.g., a single color spinor, a block color spinor (an eigenvector set), an SU(N) field etc.:
  v::Any

  # Lattice field structure
  field_desc::QJuliaGrid.QJuliaLatticeFieldDescr_qj

  QJuliaLatticeField_qj() = new( missing )

  function QJuliaLatticeField_qj(fdesc::QJuliaGrid.QJuliaLatticeFieldDescr_qj)

    # Since the field descriptor is an external argument (and data are not protected in Julia) we need this check again:
    if fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY && fdesc.nSpin == nothing ; error("Spin dof is not defined for the field with scalar geometry."); end

    if fdesc.grid.location == QJuliaEnums.QJULIA_CPU_FIELD_LOCATION
      # Set the field array total elements:
      tot_elems = complex_length*fdesc.siteSubset*(fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? fdesc.nSpin : fdesc.nColor)*fdesc.nColor*fdesc.grid.grid_volumeCB

      v = Matrix{fdesc.grid.register_type}(undef, tot_elems, Int(fdesc.geom)*fdesc.nBlock)
    else
      # Ok, set the field as a "missing" one. It will be inititalized later.
      v = missing
    end

    # call constructor
    new(v, fdesc)

  end #QJuliaLatticeField_qj constructor

  function QJuliaLatticeField_qj(grid::QJuliaGrid.QJuliaGridDescr_qj, geom::QJuliaEnums.QJuliaFieldGeometry_qj, parity::QJuliaEnums.QJuliaParity_qj; NSpin = nothing, NColor = 3, NBlock = 1)

    fdesc = QJuliaLatticeFieldDescr_qj{NSpin, NColor, NBlock}(grid::QJuliaGridDescr_qj, geom::QJuliaEnums.QJuliaFieldGeometry_qj, parity::QJuliaEnums.QJuliaParity_qj)

    if fdesc.grid.location == QJuliaEnums.QJULIA_CPU_FIELD_LOCATION
      # Set the field array total elements:
      tot_elems = complex_length*fdesc.siteSubset*(fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? fdesc.nSpin : fdesc.nColor)*fdesc.nColor*fdesc.grid.grid_volumeCB

      v = Matrix{fdesc.grid.register_type}(undef, tot_elems, Int(fdesc.geom)*fdesc.nBlock)
    else
      # Ok, set the field as a "missing" one. It will be inititalized later.
      v = missing
    end

    # call constructor
    new(v, fdesc)

  end #QJuliaLatticeField_qj constructor

  function QJuliaLatticeField_qj(field::QJuliaLatticeField_qj, parity::QJuliaEnums.QJuliaParity_qj)

     if field.v === missing; error("Cannot reference a missing field"); end
     if field.field_desc.siteSubset == 1; error("Cannot reference a parity field from the parity field"); end

     fdesc = deepcopy(field.field_desc)

     griddsc   = fdesc.grid
     griddsc.X = NTuple{QJuliaGrid.QJULIA_MAX_DIMS, Int}(ntuple(i->(i == 1 ? field.field_desc.grid.X[i] >> 1 : field.field_desc.grid.X[i]), QJuliaGrid.QJULIA_MAX_DIMS))

     griddsc.volume        = field.field_desc.grid.volumeCB
     griddsc.volumeCB      = field.field_desc.grid.volumeCB
     griddsc.grid_volume   = field.field_desc.grid.grid_volumeCB
     griddsc.grid_volumeCB = field.field_desc.grid.grid_volumeCB

     fdesc.parity     = parity
     fdesc.siteSubset = 1

     parity_volume = complex_length*griddsc.grid_volumeCB*(fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? fdesc.nSpin : fdesc.nColor)*fdesc.nColor

     offset = parity == QJuliaEnums.QJULIA_EVEN_PARITY ? 1 : parity_volume+1

     v = view(field.v, offset:offset+parity_volume-1, :)

     new(v, fdesc)

  end

end #QJuliaLatticeField_qj

# Pointer helpers:
@inline function GetPtr(field::QJuliaGenericField_qj); return field.v; end

@inline function GetParityPtr(field::QJuliaGenericField_qj, parity::QJuliaEnums.QJuliaParity_qj)
   if(field.field_desc.siteSubset == 1); return field.v; end

   parity_volume = complex_length*field.field_desc.real_volumeCB*(field.field_desc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? field.field_desc.nSpin : field.field_desc.nColor)*field.field_desc.nColor
   offset        = parity == QJuliaEnums.QJULIA_EVEN_PARITY ? 1 : parity_volume+1

   return view(field.v, offset:offset+parity_volume-1, :)
end

function field_info(field::QJuliaGenericField_qj; vrbs = true)

  field_desc = field.field_desc
  println(" ")
  println("General field info for ", typeof(field), ": ")

  if vrbs == true; QJuliaGrid.field_descr_info(field_desc); end

  println("Field data reference type : ", typeof(field.v))
  println("Field object pointer: ", pointer_from_objref(field))

  if field.v === missing
    println("Warning: data array was not allocated.")
  elseif field_desc.grid.location == QJuliaEnums.QJULIA_CUDA_FIELD_LOCATION
    println("Warning: data array was allocated on GPU")
    println("Physical length : ", length(field.v))
  else
    println("Row data pointer: ", pointer(field.v))
    println("Bytes : ", length(field.v))
  end

  println(" ")

end


# Specialized outer constructors:

# WARNING: transfered into CUDA impl
#CreateGenericField(fdesc::QJuliaGrid.QJuliaLatticeFieldDescr_qj) = QJuliaLatticeField_qj(fdesc)

CreateColorSpinor(grid::QJuliaGrid.QJuliaGridDescr_qj; parity::QJuliaEnums.QJuliaParity_qj = QJuliaEnums.QJULIA_INVALID_PARITY) = QJuliaLatticeField_qj(grid, QJuliaEnums.QJULIA_SCALAR_GEOMETRY, parity)

CreateBlockColorSpinor(grid::QJuliaGrid.QJuliaGridDescr_qj, NBlock; parity::QJuliaEnums.QJuliaParity_qj = QJuliaEnums.QJULIA_INVALID_PARITY) = QJuliaLatticeField_qj(grid, QJuliaEnums.QJULIA_SCALAR_GEOMETRY, parity; nBlock = NBlock)

CreateGaugeField(grid::QJuliaGrid.QJuliaGridDescr_qj) = QJuliaLatticeField_qj(grid, QJuliaEnums.QJULIA_VECTOR_GEOMETRY, QJuliaEnums.QJULIA_INVALID_PARITY)

#Create references to parity fields
Even(field::QJuliaGenericField_qj) = QJuliaLatticeField_qj(field, QJuliaEnums.QJULIA_EVEN_PARITY)
Odd(field::QJuliaGenericField_qj)  = QJuliaLatticeField_qj(field, QJuliaEnums.QJULIA_ODD_PARITY)

function (field::QJuliaLatticeField_qj)(i::Int)
  return field.v[i]
end

# Field Accessors
# input : field object
# out   : tuple of data array and args (which is a tuple of dof and offsets)

struct AccessorArgs
  #
  nParity::Int
  #
  nSpin::Int
  #
  nColor::Int
  #
  nVec::Int
  #
  length::Int
  #
  offset_cb::Int
  #
  field_order_id::Int
  # 1 for matter fields, 4 for gauge fields
  geom::Int
end


function create_field_accessor(field::QJuliaGenericField_qj; verbosity = true)

  nParity = field.field_desc.siteSubset
  nSpin   = field.field_desc.nSpin == nothing ? 0 : field.field_desc.nSpin
  nColor  = field.field_desc.nColor
  nVecs   = field.field_desc.nBlock

  field_order_id = 0 # currently default field order id : color inside spin inside space-time for spinors, color cols inside color rows inside space-time for gauge fields

  stride  = complex_length*field.field_desc.grid.grid_volume #include padding

  if     field.field_desc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY
    field_order_shape  = (stride, nColor, nSpin, nVecs)
    geom = 1
  elseif field.field_desc.geom == QJuliaEnums.QJULIA_VECTOR_GEOMETRY
    field_order_shape  = (stride, nColor, nColor, 4) #4 directions
    geom = 4
  elseif field.field_desc.geom == QJuliaEnums.QJULIA_COARSE_GEOMETRY
    field_order_shape  = (stride, nColor, nColor, nSpin, nSpin, 4) #4 directions
    geom = 8
  else
    error("Field geometry is not supported.")
  end

  if verbosity == true ; println("Create field accessor with dims ", field_order_shape);end

  field_order = reshape(field.v, field_order_shape)
  #
  offset_cb   = Int(length(field.v) / field.field_desc.siteSubset)
  #
  args        = AccessorArgs(nParity, nSpin, nColor, nVecs, stride, offset_cb, field_order_id, geom)
  #
  return (field_order, args)
end

end # module QJuliaFields
