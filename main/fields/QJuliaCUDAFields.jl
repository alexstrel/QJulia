module QJuliaCUDAFields

#load path to qjulia home directory
push!(LOAD_PATH, joinpath(@__DIR__, "../..", "core"))
push!(LOAD_PATH, @__DIR__)

import QJuliaRegisters
import QJuliaEnums
import QJuliaGrid
import QJuliaFields

using CUDAnative
using CuArrays

function CreateGenericCUDAField(fdesc::QJuliaGrid.QJuliaLatticeFieldDescr_qj)

  generic_field = QJuliaFields.CreateGenericField(fdesc)

  fdesc = generic_field.field_desc

  # Set the field array total elements:
  tot_elems = fdesc.siteSubset*(fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? fdesc.nSpin : fdesc.NColor)*fdesc.nColor*fdesc.grid.grid_volumeCB

  if fdesc.grid.location == QJuliaEnums.QJULIA_CUDA_FIELD_LOCATION
    generic_field.v = CuArray{fdesc.grid.register_type, 2}(undef, tot_elems, Int(fdesc.geom)*fdesc.nBlock)
  end

  return generic_field
end

end # module QJuliaCUDAFields
