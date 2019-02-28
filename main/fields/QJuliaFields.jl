module QJuliaFields

#load path to qjulia home directory
push!(LOAD_PATH, joinpath(@__DIR__, "../..", "core"))

import QJuliaRegisters
import QJuliaEnums

#create type alias
#SSE
m128d   = QJuliaRegisters.m128d
m128    = QJuliaRegisters.m128
#AVX/AVX2
m256d   = QJuliaRegisters.m256d
m256    = QJuliaRegisters.m256
#AVX3
m512d   = QJuliaRegisters.m512d
m512    = QJuliaRegisters.m512

const QJULIA_MAX_DIMS = 6

#QJuliaLatticeFieldDesc_qj
mutable struct QJuliaLatticeFieldDescr_qj{T<:Any}
  # Register type
  register_type::Any

  # Field precision (Float16/32/64)
  prec::DataType

  # Field geometry (spinor, vector, tensor etc.)
  geom::QJuliaEnums.QJuliaFieldGeometry_qj

  # Number of field dimensions
  nDim::Int

  # The local space-time dimensions (without checkboarding)
  X::NTuple{QJULIA_MAX_DIMS, Int}

  # Padding parameter
  pad::NTuple{4, Int}

  # Local volume virtual decomposition (used for vectorization),
  # supported formats (in x,y,z,t order) are (see P.Boyle et al paper, arXiv:1512.03487):
  # 1. SP scalar type (1,1,1,1), 2. m128  type (1,1,1,2), 3. m256  type (1,1,2,2), 4. m512  type (1,2,2,2)
  # 4. DP scalar type (1,1,1,1), 5. m128d type (1,1,1,1), 6. m256d type (1,1,1,2), 7. m512d type (1,1,2,2)
  # 8. QUDA grid (all precisions) (X[1], X[2], X[3], X[4])
  D::NTuple{4, Int}

  # Field volume
  volume::Int

  # Field checkerboarded volume
  volumeCB::Int

  # Field volume after simdization and padding
  real_volume::Int

  # Field checkerboarded volume after simdization and padding
  real_volumeCB::Int

  # The number of dimensions we partition for communication
  nDimComms::Int

  # Whether the field is full or single parity (2 : full, 1 : single parity )
  siteSubset::Int

  # Parity type
  parity::QJuliaEnums.QJuliaParity_qj

  # Field location
  location::QJuliaEnums.QJuliaFieldLocation_qj

  # Defualt constructor
  QJuliaLatticeFieldDescr_qj{T}() where {T<:Any} = new(T, Float64, QJuliaEnums.QJULIA_INVALID_GEOMETRY, 4,
                                NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->1, QJULIA_MAX_DIMS)),
                                NTuple{4, Int}(ntuple(i->0, 4)),
                                NTuple{4, Int}(ntuple(i->1, 4)),
                                1, 1, 1, 1, 0, 2,
				QJuliaEnums.QJULIA_INVALID_PARITY,
				QJuliaEnums.QJULIA_CPU_FIELD_LOCATION)


  function QJuliaLatticeFieldDescr_qj{T}(geom::QJuliaEnums.QJuliaFieldGeometry_qj, parity::QJuliaEnums.QJuliaParity_qj, is_quda_grid::Bool, ndimscomm::Int, X::NTuple) where T
     # Check lattice dimensions
     if sizeof(X) > (sizeof(Int)*QJULIA_MAX_DIMS); error("Requested lattice dimensions are not supported"); end
     # Set lattice dimensions
     nDims = length(X)
     #
     xx = NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->(i<=length(X) ? X[i] : 1), QJULIA_MAX_DIMS))
     #
     simd_scale = 0
     #
     if(is_quda_grid == true && T !=  m128d && T !=  m128 && T !=  m256d && T !=  m256 && T !=  m512d && T !=  m512)
       d=(X[1], X[2], X[3], X[4]) #higher dims are not supported
     elseif(T == Complex{Float32} || T == Complex{Float64} || T == Complex{BigFloat} || T == m128d)
       d=(1, 1, 1, 1)
     elseif(T == m128 || T ==  m256d)
       d=(1, 1, 1, 2)
       simd_scale = 1
     elseif(T == m256 || T ==  m512d)
       d=(1, 1, 2, 2)
       simd_scale = 2
     elseif(T == m512)
       d=(1, 2, 2, 2)
       simd_scale = 3
     else
       error("Requested data type ", T , " is not supported")
     end

     # Set up padding
     pad = NTuple{4, Int}(ntuple(i->0, 4))

     # Set field precision
     if (T == Complex{Float64} || T == m128d || T == m256d || T == m512d )
       prec = Float64
     elseif (T == Complex{BigFloat})
       prec = BigFloat
     elseif (T == Complex{Float16})
       prec = Float16
     else
       prec = Float32
     end

     # Set field volume:
     volume = 1
     [volume *= i for i in X]

     # Set cb volume:
     volumeCB =  volume >> 1

     # Set field real volume
     real_volume = volume >> simd_scale

     # Set cb real volume
     real_volumeCB = volumeCB >> simd_scale

     # Set parity flags
     siteSubset = parity == QJuliaEnums.QJULIA_INVALID_PARITY ? 2 : 1

     # Set location
     location = is_quda_grid == true ? QJuliaEnums.QJULIA_CUDA_FIELD_LOCATION : QJuliaEnums.QJULIA_CPU_FIELD_LOCATION

     # call constructor
     new(T, prec, geom, nDims, xx, pad, d, volume, volumeCB, real_volume, real_volumeCB, ndimscomm, siteSubset, parity, location)

  end #QJuliaLatticeFieldDescr_qj constructor

end

function field_desc_info(field_desc::QJuliaLatticeFieldDescr_qj)

  println("Register type : ", field_desc.register_type)
  println("Precision : ", field_desc.prec )
  println("Field geometry : ", field_desc.geom )
  print("Field dimensions : ", field_desc.nDim, ", ("); [print(field_desc.X[i], ", ") for i in 1:field_desc.nDim]; println(")")
  println("Volume : ", field_desc.volume)
  println("VolumeCB : ", field_desc.volumeCB)
  println("Register elements for volume : ", field_desc.real_volume)
  println("Register elements for volumeCB : ", field_desc.real_volumeCB)
  println("Site subset : ", field_desc.siteSubset == 1 ? "parity field" : "full field.")
  println("Parity type : ", field_desc.parity)

end


abstract type QJuliaGenericField_qj end

mutable struct QJuliaLatticeField_qj{NSpin<:Any, NColor<:Any, NBlock<:Any} <: QJuliaGenericField_qj
  # Lattice field structure
  field_desc::QJuliaLatticeFieldDescr_qj

  # Spin dof
  nSpin::Int

  # Color dof
  nColor::Int

  # Block size
  nBlock::Int

  # Field Matrix{T}, e.g., a single color spinor, a block color spinor (an eigenvector set), an SU(N) field etc.:
  v::Any

  function QJuliaLatticeField_qj{NSpin,NColor,NBlock}(fdesc::QJuliaLatticeFieldDescr_qj) where NSpin where NColor where NBlock
     # Check spin dof:
     if (NSpin != 0 && NSpin != 1 && NSpin != 2 && NSpin != 4)
	error("NSpin parameter is incorrect, must be 0 (gauge field), 1 (fine staggered), 2 (coarse spinor) or 4 (fine spinor) ")
     end

     if fdesc.geom != QJuliaEnums.QJULIA_SCALAR_GEOMETRY
       if (NSpin  != 0); error("Spin dof is not allowed for this type of the field"); end
       if (NBlock != 1); error("Block is not supported for this type of the field");  end
     else
       if (NSpin  == 0); error("NSpin = 0 is not allowed for ", fdesc.geom, ". Check the field descriptor."); end
     end

     # Check NColor:
     if (NColor < 1) ; error("NColor parameter is incorrect, must be non-zero."); end

     # Check NBlock:
     if (NBlock < 1) ; error("NBlock parameter is incorrect, must be non-zero."); end

     # Set the field array total elements:
     tot_elems = fdesc.siteSubset*(fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? NSpin : NColor)*NColor*fdesc.real_volumeCB

     if fdesc.location == QJuliaEnums.QJULIA_CPU_FIELD_LOCATION
       v = Matrix{fdesc.register_type}(undef, tot_elems, Int(fdesc.geom)*NBlock)
     else
       v = nothing
     end

     # call constructor
     new(fdesc, NSpin, NColor, NBlock, v)

  end #QJuliaLatticeField_qj constructor

  function QJuliaLatticeField_qj{NSpin,NColor,NBlock}(field::QJuliaLatticeField_qj, parity::QJuliaEnums.QJuliaParity_qj) where NSpin where NColor where NBlock

     if field.v == nothing; error("Cannot reference an empty field"); end
     if field.field_desc.siteSubset == 1; error("Cannot reference a parity field from a parity field"); end

     pfdesc = deepcopy(field.field_desc)

     pfdesc.X        = NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->(i == 1 ? field.field_desc.X[i] >> 1 : field.field_desc.X[i]), QJULIA_MAX_DIMS))

     pfdesc.volume        = field.field_desc.volumeCB
     pfdesc.volumeCB      = field.field_desc.volumeCB
     pfdesc.real_volume   = field.field_desc.real_volumeCB
     pfdesc.real_volumeCB = field.field_desc.real_volumeCB

     pfdesc.parity     = parity
     pfdesc.siteSubset = 1

     parity_volume = field.field_desc.real_volumeCB*(pfdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? field.nSpin : field.nColor)*field.nColor

     offset = parity == QJuliaEnums.QJULIA_EVEN_PARITY ? 1 : parity_volume+1

     v = view(field.v, offset:offset+parity_volume-1, :)

     new(pfdesc, field.nSpin, field.nColor, field.nBlock, v)

  end

end #QJuliaLatticeField_qj

CreateBlockColorSpinor(fdesc::QJuliaLatticeFieldDescr_qj; NSpin::Int = 4, NColor::Int = 3, NBlock::Int = 1) = QJuliaLatticeField_qj{NSpin, NColor, NBlock}(fdesc)

# Specialized constructors and helper methods

# Pure single color spinor field
QJuliaLatticeField_qj{NSpin, NColor}(fdesc::QJuliaLatticeFieldDescr_qj) where NSpin where NColor =  QJuliaLatticeField_qj{NSpin, NColor, 1}(fdesc::QJuliaLatticeFieldDescr_qj)
#
CreateColorSpinor(fdesc::QJuliaLatticeFieldDescr_qj; NSpin::Int = 4, NColor::Int = 3) = QJuliaLatticeField_qj{NSpin, NColor}(fdesc)

# Gauge field
QJuliaLatticeField_qj{NColor}(fdesc::QJuliaLatticeFieldDescr_qj) where NColor  =  QJuliaLatticeField_qj{0, NColor, 1}(fdesc::QJuliaLatticeFieldDescr_qj)
#
CreateGaugeField(fdesc::QJuliaLatticeFieldDescr_qj; NColor::Int = 3) = QJuliaLatticeField_qj{NColor}(fdesc)

#Create references to parity fields
Even(field::QJuliaGenericField_qj) = QJuliaLatticeField_qj{field.nSpin, field.nColor, field.nBlock}(field, QJuliaEnums.QJULIA_EVEN_PARITY)
Odd(field::QJuliaGenericField_qj)  = QJuliaLatticeField_qj{field.nSpin, field.nColor, field.nBlock}(field, QJuliaEnums.QJULIA_ODD_PARITY)

# Pointer helpers:
@inline function GetParityPtr(field::QJuliaGenericField_qj, parity::QJuliaEnums.QJuliaParity_qj)
   if(field.field_desc.siteSubset == 1); return field.v; end

   parity_volume = field.field_desc.real_volumeCB*(field.field_desc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? field.nSpin : field.nColor)*field.nColor

   offset = parity == QJuliaEnums.QJULIA_EVEN_PARITY ? 1 : parity_volume+1

   return view(field.v, offset:offset+parity_volume-1, :)
end

function field_info(field::QJuliaGenericField_qj)

  field_desc = field.field_desc
  println(" ")
  println("General field info for ", typeof(field), ": ")
  field_desc_info(field_desc)
  if( field_desc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ); println("NSpin : ", field.nSpin )  ; end
  println("NColor : ", field.nColor )
  if( field_desc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ); println("NBlock : ", field.nBlock )  ; end
  println("Reference type : ", typeof(field.v))
  println("Field object pointer: ", pointer_from_objref(field))
  println("Row data pointer: ", pointer(field.v))
  println(" ")

end

# DO TEST

  spinor_field_desc = QJuliaLatticeFieldDescr_qj{m256d}(QJuliaEnums.QJULIA_SCALAR_GEOMETRY, QJuliaEnums.QJULIA_INVALID_PARITY, false, 0, (8,8,8,8,1))

  println(typeof(spinor_field_desc))

  println("Size of ", typeof(spinor_field_desc), " : ", sizeof(spinor_field_desc) )

  spinor_field = CreateColorSpinor(spinor_field_desc; NSpin=4)

  field_info(spinor_field)

  println("Size of ", typeof(spinor_field), " : ", sizeof(spinor_field), " ( ", sizeof(spinor_field.field_desc), " + ", sizeof(spinor_field.nSpin), " + ", sizeof(spinor_field.nColor), " + ", sizeof(spinor_field.nBlock), " + ", sizeof(spinor_field.v) ,")" )

  gauge_field_desc = QJuliaLatticeFieldDescr_qj{ComplexF32}(QJuliaEnums.QJULIA_VECTOR_GEOMETRY, QJuliaEnums.QJULIA_INVALID_PARITY, false, 0, (8,8,8,8))

  gauge_field = CreateGaugeField(gauge_field_desc)

  field_info(gauge_field)

  println("Test types: ", (typeof(spinor_field) == typeof(gauge_field)))

  even_gauge_field = Even(gauge_field)
  field_info(even_gauge_field)

  odd_spinor_field = Odd(spinor_field)
  println("Size of full spinor field in bytes : ", sizeof(spinor_field.v), " , size of (odd) parity field in bytes : ", sizeof(odd_spinor_field.v))

  field_info(odd_spinor_field)

  odd_spinor_field.v[1] = spinor_field.field_desc.register_type(ntuple(i->11.2018, 4))

  println("  ", spinor_field.v[1+spinor_field.field_desc.real_volumeCB*spinor_field.nSpin*spinor_field.nColor])

# END DO TEST
end #QJuliaFields
