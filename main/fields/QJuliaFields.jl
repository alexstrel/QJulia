module QJuliaFields

#load path to qjulia home directory
push!(LOAD_PATH, joinpath(@__DIR__, "../..", "core"))
push!(LOAD_PATH, @__DIR__)

import QJuliaRegisters
import QJuliaEnums
import QJuliaGrid

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

    # Set the field array total elements:
    tot_elems = fdesc.siteSubset*(fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? fdesc.nSpin : fdesc.NColor)*fdesc.nColor*fdesc.grid.grid_volumeCB

    if fdesc.grid.location == QJuliaEnums.QJULIA_CPU_FIELD_LOCATION
      v = Matrix{fdesc.grid.register_type}(undef, tot_elems, Int(fdesc.geom)*fdesc.nBlock)
    else
      v = missing
    end

    # call constructor
    new(v, fdesc)

  end #QJuliaLatticeField_qj constructor

  function QJuliaLatticeField_qj(grid::QJuliaGrid.QJuliaGridDescr_qj, geom::QJuliaEnums.QJuliaFieldGeometry_qj, parity::QJuliaEnums.QJuliaParity_qj; NSpin = nothing, NColor = 3, NBlock = 1)

    fdesc = QJuliaLatticeFieldDescr_qj{NSpin, NColor, NBlock}(grid::QJuliaGridDescr_qj, geom::QJuliaEnums.QJuliaFieldGeometry_qj, parity::QJuliaEnums.QJuliaParity_qj)

    # Set the field array total elements:
    tot_elems = fdesc.siteSubset*(fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? fdesc.nSpin : fdesc.nColor)*fdesc.nColor*fdesc.grid.grid_volumeCB

    if fdesc.grid.location == QJuliaEnums.QJULIA_CPU_FIELD_LOCATION
      v = Matrix{fdesc.grid.register_type}(undef, tot_elems, Int(fdesc.geom)*fdesc.nBlock)
    else
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

     parity_volume = griddsc.grid_volumeCB*(fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? fdesc.nSpin : fdesc.nColor)*fdesc.nColor

     offset = parity == QJuliaEnums.QJULIA_EVEN_PARITY ? 1 : parity_volume+1

     v = view(field.v, offset:offset+parity_volume-1, :)

     new(v, fdesc)

  end

end #QJuliaLatticeField_qj

# Pointer helpers:
@inline function GetPtr(field::QJuliaGenericField_qj); return field.v; end

@inline function GetParityPtr(field::QJuliaGenericField_qj, parity::QJuliaEnums.QJuliaParity_qj)
   if(field.field_desc.siteSubset == 1); return field.v; end

   parity_volume = field.field_desc.real_volumeCB*(field.field_desc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? field.field_desc.nSpin : field.field_desc.nColor)*field.field_desc.nColor
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
    println("Data array was allocated on GPU")
  else
    println("Row data pointer: ", pointer(field.v))
  end

  println(" ")

end


# Specialized outer constructors:

CreateGenericField(fdesc::QJuliaGrid.QJuliaLatticeFieldDescr_qj) = QJuliaLatticeField_qj(fdesc)

CreateColorSpinor(grid::QJuliaGrid.QJuliaGridDescr_qj, parity::QJuliaEnums.QJuliaParity_qj) = QJuliaLatticeField_qj(grid, QJuliaEnums.QJULIA_SCALAR_GEOMETRY, parity)

CreateBlockColorSpinor(grid::QJuliaGrid.QJuliaGridDescr_qj, NBlock, parity::QJuliaEnums.QJuliaParity_qj) = QJuliaLatticeField_qj(grid, QJuliaEnums.QJULIA_SCALAR_GEOMETRY, parity; nBlock = NBlock)

CreateGaugeField(grid::QJuliaGrid.QJuliaGridDescr_qj) = QJuliaLatticeField_qj(grid, QJuliaEnums.QJULIA_VECTOR_GEOMETRY, QJuliaEnums.QJULIA_INVALID_PARITY)

#Create references to parity fields
Even(field::QJuliaGenericField_qj) = QJuliaLatticeField_qj(field, QJuliaEnums.QJULIA_EVEN_PARITY)
Odd(field::QJuliaGenericField_qj)  = QJuliaLatticeField_qj(field, QJuliaEnums.QJULIA_ODD_PARITY)

end # module QJuliaFields
