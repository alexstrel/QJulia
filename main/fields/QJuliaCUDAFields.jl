#!/usr/bin/env julia

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

const complex_length = 2

function CreateGenericField(fdesc::QJuliaGrid.QJuliaLatticeFieldDescr_qj)

  generic_field = QJuliaFields.QJuliaLatticeField_qj(fdesc)

  if fdesc.grid.location == QJuliaEnums.QJULIA_CUDA_FIELD_LOCATION
    # Get reference to field parameters:
    fdesc = generic_field.field_desc
    # Set the field array total elements:
    tot_elems = complex_length*fdesc.siteSubset*(fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? fdesc.nSpin : fdesc.nColor)*fdesc.nColor*fdesc.grid.grid_volumeCB

    generic_field.v = CuArray{fdesc.grid.register_type, 2}(undef, tot_elems, Int(fdesc.geom)*fdesc.nBlock)

    #address_space = AS.Global or AS.Generic etc.
    #dev_ptr = CUDAnative.DevicePtr{fdesc.grid.register_type,address_space}(tot_elems*Int(fdesc.geom)*fdesc.nBlock)
    #generic_field.v = CuDeviceArray{fdesc.grid.register_type,2,address_space}((tot_elems, Int(fdesc.geom)*fdesc.nBlock), dev_ptr)
  end

  return generic_field
end


end # module QJuliaCUDAFields
