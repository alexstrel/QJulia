module QJuliaFieldUtils

using QJuliaFields
#using QJuliaUtils
using QJuliaGaugeUtils
using QJuliaEnums
using QJuliaInterface

@inline function gen_random_spinor!(field::QJuliaFields.QJuliaGenericField_qj) 

  if(field.field_desc.geom != QJuliaEnums.QJULIA_SCALAR_GEOMETRY); error("Cannot apply on fields with non-scalar geometry.");end
  if(field.field_desc.register_type != ComplexF64 && field.field_desc.register_type != ComplexF32 )
    error("Register type ", field.field_desc.register_type, " is currently not supported.")
  end

  for i in LinearIndices(field.v); field.v[i] = field.field_desc.register_type(rand(), rand()); end  

end #gen_random_spinor!

function construct_gauge_field!(field::QJuliaFields.QJuliaGenericField_qj, gtype, param::QJuliaInterface.QJuliaGaugeParam_qj)

  if(field.field_desc.geom != QJuliaEnums.QJULIA_VECTOR_GEOMETRY); error("Cannot apply on fields with non-vector geometry.");end
  if(field.field_desc.register_type != ComplexF64 && field.field_desc.register_type != ComplexF32 )
    error("Register type ", field.field_desc.register_type, " is currently not supported.")
  end
  QJuliaGaugeUtils.construct_gauge_field!(field.v, gtype, param)

end


end #QJuliaFieldUtils


