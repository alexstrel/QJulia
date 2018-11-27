module QJuliaFields

#load path to qjulia home directory
push!(LOAD_PATH, string(ENV["QJULIA_HOME"],"/core"))

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
  QJuliaLatticeFieldDescr_qj{T}() where {T<:Any} = new(Float64, QJuliaEnums.QJULIA_INVALID_GEOMETRY, 4,
                                NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->1, QJULIA_MAX_DIMS)),
                                NTuple{4, Int}(ntuple(i->0, 4)),
                                NTuple{4, Int}(ntuple(i->1, 4)),
                                1, 1, 1, 1, 0, 2,
				QJuliaEnums.QJULIA_INVALID_PARITY,
				QJuliaEnums.QJULIA_CPU_FIELD_LOCATION) 


  function QJuliaLatticeFieldDescr_qj{T}(geom::QJuliaEnums.QJuliaFieldGeometry_qj, parity::QJuliaEnums.QJuliaParity_qj, is_quda_grid::Bool, ndimscomm::Int, X::NTuple{4,Int64}) where T
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
     new(prec, geom, nDims, xx, pad, d, volume, volumeCB, real_volume, real_volumeCB, ndimscomm, siteSubset, parity, location)

  end #QJuliaLatticeFieldDescr_qj constructor

end

abstract type QJuliaGenericField_qj end

mutable struct QJuliaLatticeField_qj{T<:Any, NSpin<:Any, NColor<:Any, NBlock<:Any} <: QJuliaGenericField_qj
  # Lattice field structure
  field_desc::QJuliaLatticeFieldDescr_qj{T}

  # Spin dof
  nSpin::Int

  # Color dof
  nColor::Int

  # Field array(s), e.g., single color spinor, block color spinor (eigenvector set), SU(N) field etc.:
  v::Matrix{T}

  function QJuliaLatticeField_qj{T,NSpin,NColor,NBlock}(fdesc::QJuliaLatticeFieldDescr_qj) where T where NSpin where NColor where NBlock
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
     tot_elems = fdesc.siteSubset*(fdesc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ? NSpin : 1)*NColor*fdesc.real_volumeCB

     v = Matrix{T}(undef, tot_elems, Int(fdesc.geom)*NBlock) 

     # call constructor
     new(fdesc, NSpin, NColor, v)

  end #QJuliaLatticeField_qj constructor

end #QJuliaLatticeField_qj

CreateBlockColorSpinor(fdesc::QJuliaLatticeFieldDescr_qj, T::Any, NSpin::Int, NColor::Int, NBlock::Int) = QJuliaLatticeField_qj{T, NSpin, NColor, NBlock}(fdesc)

# Specialized constructors and helper methods

# Pure single color spinor field
QJuliaLatticeField_qj{T, NSpin, NColor}(fdesc::QJuliaLatticeFieldDescr_qj) where T where NSpin where NColor =  QJuliaLatticeField_qj{T, NSpin, NColor, 1}(fdesc::QJuliaLatticeFieldDescr_qj)
#
CreateColorSpinor(fdesc::QJuliaLatticeFieldDescr_qj, T::Any, NSpin::Int, NColor::Int) = QJuliaLatticeField_qj{T, NSpin, NColor}(fdesc)

# Gauge field 
QJuliaLatticeField_qj{T, NColor}(fdesc::QJuliaLatticeFieldDescr_qj) where T where NColor                    =  QJuliaLatticeField_qj{T, 0, NColor, 1}(fdesc::QJuliaLatticeFieldDescr_qj)
#
CreateGaugeField(fdesc::QJuliaLatticeFieldDescr_qj, T::Any, NColor::Int) = QJuliaLatticeField_qj{T, NColor}(fdesc)

function field_info(field::QJuliaGenericField_qj)

  field_desc = field.field_desc
  println(" ")
  println("General field info for ", typeof(field), ": ")
  println("Size of ", typeof(field_desc), " : ", sizeof(field_desc) )
  println("Precision : ", field_desc.prec )
  println("Field geometry : ", field_desc.geom )
  if( field_desc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ); println("NSpin : ", field.nSpin )  ; end
  println("NColor : ", field.nColor )  
  if( field_desc.geom == QJuliaEnums.QJULIA_SCALAR_GEOMETRY ); println("NBlock : ", size(field.v)[2] )  ; end
  print("Field dimensions : ", field_desc.nDim, ", ("); [ if i != 1;print(i, ", ");end for i in field_desc.X]; println(")")
  println("Register type : ", typeof(field.v[1]))
  println("Volume : ", field_desc.volume)
  println("VolumeCB : ", field_desc.volumeCB)
  println("Register elements for volume : ", field_desc.real_volume)
  println("Register elements for volumeCB : ", field_desc.real_volumeCB)
  println("Site subset : ", field_desc.siteSubset == 1 ? "parity field" : "full field.")
  println("Parity type : ", field_desc.parity)
  println(" ")

end

# DO TEST

  test_spinor_field_desc = QJuliaLatticeFieldDescr_qj{m256d}(QJuliaEnums.QJULIA_SCALAR_GEOMETRY, QJuliaEnums.QJULIA_INVALID_PARITY, false, 0, (8,8,8,8))

  println(typeof(test_spinor_field_desc))

  test_spinor_field = CreateColorSpinor(test_spinor_field_desc, m256d, 4, 3)

  field_info(test_spinor_field)

  test_gauge_field_desc = QJuliaLatticeFieldDescr_qj{m256d}(QJuliaEnums.QJULIA_VECTOR_GEOMETRY, QJuliaEnums.QJULIA_INVALID_PARITY, false, 0, (8,8,8,8))

  test_gauge_field = CreateGaugeField(test_gauge_field_desc, m256d, 3)  

  field_info(test_gauge_field)

  println("Test types: ", (typeof(test_spinor_field) == typeof(test_gauge_field))) 

# END DO TEST
end #QJuliaFields


