module QJuliaFields

#load path to qjulia home directory
push!(LOAD_PATH, ENV["QJULIA_HOME"])

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

mutable struct QJuliaLatticeField_qj
  # Field precision (Float16/32/64)
  prec::DataType 

  # Field geometry (spinor, vector, tensor etc.)
  fgeom::QJuliaEnums.QJuliaFieldGeometry_qj

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

  # Field volume after simdization
  real_volume::Int

  # Field checkerboarded volume after simdization
  real_volumeCB::Int

  # The number of dimensions we partition for communication 
  nDimComms::Int

  # Whether the field is full or single parity (2 : full, 1 : single parity ) 
  siteSubset::Int

  # Parity type
  parity::QJuliaEnums.QJuliaParity_qj

  #defualt constructor (remarkable: no need to indicate enum name!)
  QJuliaLatticeField_qj() = new(Float64, QJuliaEnums.QJULIA_INVALID_GEOMETRY, 4,
                                NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->1, QJULIA_MAX_DIMS)),
                                NTuple{4, Int}(ntuple(i->1, 4)),
                                NTuple{4, Int}(ntuple(i->1, 4)),
                                1, 1, 1, 1, 0, 2,
				QJuliaEnums.QJULIA_INVALID_PARITY)

end

abstract type QJuliaGenericField_qj end

# Suppoerted field types: Complex{Float16}, Complex{Float32}, Complex{Float64}, Complex{BigFloat}, m128(d), m256(d), m512(d)
# Format: mySpinorFiled = QJuliaColorSpinorField_qj{Type} (nSpin, nColor, ...)

mutable struct QJuliaColorSpinorField_qj{T<:Any} <: QJuliaGenericField_qj
  # Lattice field structure
  field_desc::QJuliaLatticeField_qj

  # Spin 
  nSpin::Int

  # Color
  nColor::Int

  # Field array:
  v::Vector{T}

  function QJuliaColorSpinorField_qj{T}(nspin::Int, ncolor::Int, parity::QJuliaEnums.QJuliaParity_qj, is_quda_grid::Bool, ndimscomm::Int, X::NTuple{4,Int64}) where T
     # Setup lattice parameters
     field_desc = QJuliaLatticeField_qj()
     # Set field geometry
     field_desc.fgeom = QJuliaEnums.QJULIA_SCALAR_GEOMETRY
     # Check lattice dimensions
     if sizeof(X) > (sizeof(Int)*QJULIA_MAX_DIMS); error("Requested lattice dimensions are not supported"); end
     # Set lattice dimensions
     field_desc.nDim = length(X)
     #
     field_desc.X = NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->(i<=length(X) ? X[i] : 1), QJULIA_MAX_DIMS))
     #
     simd_scale = 0
     #
     if(is_quda_grid == true && T !=  m128d && T !=  m128 && T !=  m256d && T !=  m256 && T !=  m512d && T !=  m512)
       field_desc.D=(X[1], X[2], X[3], X[4]) #higher dims are not supported
     elseif(T == Complex{Float32} || T == Complex{Float64} || T == Complex{BigFloat} || T == m128d)
       field_desc.D=(1, 1, 1, 1) 
     elseif(T == m128 || T ==  m256d)
       field_desc.D=(1, 1, 1, 2)
       simd_scale = 1
     elseif(T == m256 || T ==  m512d)
       field_desc.D=(1, 1, 2, 2)
       simd_scale = 2
     elseif(T == m512)
       field_desc.D=(1, 2, 2, 2)
       simd_scale = 3
     else
       error("Requested data type ", T , " is not supported")
     end

     # Set field precision
     if (T == Complex{Float64} || T == m128d || T == m256d || T == m512d )
       field_desc.prec = Float64
     elseif (T == Complex{BigFloat})
       field_desc.prec = BigFloat
     elseif (T == Complex{Float16})
       field_desc.prec = Float16
     else
       field_desc.prec = Float32
     end

     # Set field volume:
     field_desc.volume = 1
     [field_desc.volume *= i for i in X]

     # Set cb volume:
     field_desc.volumeCB =  (field_desc.volume >> 1)

     # Set field real volume
     field_desc.real_volume = field_desc.volume >> simd_scale

     # Set cb real volume
     field_desc.real_volumeCB = field_desc.volumeCB >> simd_scale

     # Set parity flags
     field_desc.siteSubset = parity == QJuliaEnums.QJULIA_INVALID_PARITY ? 2 : 1

     # Parity type:
     field_desc.parity = parity 

     # Set the field array
     v = Vector{T}(undef, field_desc.siteSubset*nspin*ncolor*field_desc.real_volumeCB) 

     # call constructor
     new(field_desc, nspin, ncolor, v)

  end #QJuliaColorSpinorField_qj constructor

end #QJuliaColorSpinorField_qj



mutable struct QJuliaGaugeField_qj{T<:Any} <: QJuliaGenericField_qj
  # Lattice field structure
  field_desc::QJuliaLatticeField_qj
  # Color
  nColor::Int
  # Field array:
  v::Matrix{T}

  function QJuliaGaugeField_qj{T}(ncolor::Int, parity::QJuliaEnums.QJuliaParity_qj, is_quda_grid::Bool, ndimscomm::Int, X::NTuple{4,Int64}) where T
     # Setup lattice parameters
     field_desc = QJuliaLatticeField_qj()
     # Set field geometry
     field_desc.fgeom = QJuliaEnums.QJULIA_VECTOR_GEOMETRY
     # Check lattice dimensions
     if sizeof(X) > (sizeof(Int)*QJULIA_MAX_DIMS); error("Requested lattice dimensions are not supported"); end
     # Set lattice dimensions
     field_desc.nDim = length(X)
     #
     field_desc.X = NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->(i<=length(X) ? X[i] : 1), QJULIA_MAX_DIMS))
     #
     simd_scale = 0
     #
     if(is_quda_grid == true && T !=  m128d && T !=  m128 && T !=  m256d && T !=  m256 && T !=  m512d && T !=  m512)
       field_desc.D=(X[1], X[2], X[3], X[4]) #higher dims are not supported
     elseif(T == Complex{Float32} || T == Complex{Float64} || T == Complex{BigFloat} || T == m128d)
       field_desc.D=(1, 1, 1, 1) 
     elseif(T == m128 || T ==  m256d)
       field_desc.D=(1, 1, 1, 2)
       simd_scale = 1
     elseif(T == m256 || T ==  m512d)
       field_desc.D=(1, 1, 2, 2)
       simd_scale = 2
     elseif(T == m512)
       field_desc.D=(1, 2, 2, 2)
       simd_scale = 3
     else
       error("Requested data type ", T , " is not supported")
     end

     # Set field precision
     if (T == Complex{Float64} || T == m128d || T == m256d || T == m512d )
       field_desc.prec = Float64
     elseif (T == Complex{BigFloat})
       field_desc.prec = BigFloat
     elseif (T == Complex{Float16})
       field_desc.prec = Float16
     else
       field_desc.prec = Float32
     end

     # Set field volume:
     field_desc.volume = 1
     [field_desc.volume *= i for i in X]

     # Set cb volume:
     field_desc.volumeCB =  (field_desc.volume >> 1)

     # Set field real volume
     field_desc.real_volume = field_desc.volume >> simd_scale

     # Set cb real volume
     field_desc.real_volumeCB = field_desc.volumeCB >> simd_scale

     # Set parity flags
     field_desc.siteSubset = parity == QJuliaEnums.QJULIA_INVALID_PARITY ? 2 : 1

     # Parity type:
     field_desc.parity = parity 

     # Set the field array
     v = Matrix{T}(undef, field_desc.siteSubset*ncolor*field_desc.real_volumeCB, 4) 

     # call constructor
     new(field_desc, ncolor, v)

  end #QJuliaGaugeField_qj constructor

end #QJuliaGaugeField_qj

function general_field_info(field::QJuliaGenericField_qj)

  field_desc = field.field_desc
  println(" ")
  println("General field info for ", typeof(field), ": ")
  println("Size of ", typeof(field_desc), " : ", sizeof(field_desc) )
  println("Precision : ", field_desc.prec )
  println("Field geometry : ", field_desc.fgeom ) 
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

nColors(field::QJuliaGenericField_qj)   = println("NColor : ", field.nColor)
nSpin(field::QJuliaColorSpinorField_qj) = println("NSpin : ", field.nSpin)

# DO TEST
  my_spinor_field = QJuliaColorSpinorField_qj{m256d}(4, 3, QJuliaEnums.QJULIA_INVALID_PARITY, false, 0, (8,8,8,8))

  general_field_info(my_spinor_field)

  my_gauge_field = QJuliaGaugeField_qj{m256d}(3, QJuliaEnums.QJULIA_INVALID_PARITY, false, 0, (8,8,8,8))

  general_field_info(my_gauge_field)
 
# END DO TEST
end #QJuliaFields


