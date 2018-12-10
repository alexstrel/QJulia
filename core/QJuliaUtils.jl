module QJuliaUtils

export gridsize_from_cmdline

gridsize_from_cmdline = Array{Cint, 1}(undef, 4);
rank_order            = Int32(0)
device_id 	      = Int32(0)
rank_offset	      = Int32(0)	

function lex_rank_from_coords_t_(comms, fdata)::Cint
	rank = gridsize_from_cmdline[1]
     
	for i in 1:4
          rank = gridsize_from_cmdline[i] * rank + gridsize_from_cmdline[i]
          println("Communication direction: ", i , ", rank : ", rank , ", grid point ", gridsize_from_cmdline[i] ) 
	end
	return rank
end #lex_rank_from_coords_t_
lex_rank_from_coords_t_c = @cfunction(lex_rank_from_coords_t_, Cint, (Ptr{Cint}, Ptr{Cvoid}));

function lex_rank_from_coords_x_(comms, fdata)::Cint
	rank = gridsize_from_cmdline[4]
	for i in 4:-1:1
          rank = gridsize_from_cmdline[i] * rank + gridsize_from_cmdline[i]
          println("Communication direction: ", i , ", rank : ", rank , ", grid point ", gridsize_from_cmdline[i] ) 
	end
	return rank
end #lex_rank_from_coords_x_
lex_rank_from_coords_x_c = @cfunction(lex_rank_from_coords_x_, Cint, (Ptr{Cint}, Ptr{Cvoid}));

function get_rank_order(rorder::String) 
  if rorder == "col" 
    println("Setting column major rank order t->z->y->x, t is fastest")
    rank_order = Int32(0)
  elseif rorder == "row"
    println("Setting row major rank order x->y->z->t, x is fastest")
    rank_order = Int32(1)
  else
    throw("func 'get_rank_order':: Rank order is not supported.")
  end 
end #get_rank_order

@inline function gen_random_spinor!(spin::Vector{Complex{T}}) where T <: AbstractFloat
  for i in 1:length(spin);  spin[i] = Complex{T}(rand(), rand()); end
end #gen_random_spinor!

@inline function gen_const_spinor!(spin::Vector{Complex{T}}, rea = 0.0, img = 0.0) where T <: AbstractFloat
  for i in 1:length(spin);  spin[i] = Complex{T}(rea, img); end
end #gen_random_spinor!

@inline function print_spinor(spin::Vector{Complex{T}}, len) where T <: AbstractFloat
  for i in 1:len; println("Real = ", real(spin[i]), " Imaginary = ", imag(spin[i])); end
end #print_spinor

using QJuliaFields
using QJuliaEnums

@inline function gen_random_spinor!(field::QJuliaFields.QJuliaGenericField_qj) 

  if(field.field_desc.geom != QJuliaEnums.QJULIA_SCALAR_GEOMETRY); error("Cannot apply on fields with non-scalar geometry.");end
  if(field.field_desc.register_type != ComplexF64 && field.field_desc.register_type != ComplexF32 )
    error("Register type ", field.field_desc.register_type, " is currently not supported.")
  end

  for i in LinearIndices(field.v); field.v[i] = field.field_desc.register_type(rand(), rand()); end  

end #gen_random_spinor!

end #QJuliaUtils


