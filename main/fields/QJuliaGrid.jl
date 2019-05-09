module QJuliaGrid

push!(LOAD_PATH, joinpath(@__DIR__, "../..", "core"))

import QJuliaRegisters
import QJuliaEnums

# Generic reg types
half2    = QJuliaRegisters.half2
float2   = QJuliaRegisters.float2
double2  = QJuliaRegisters.double2
float4   = QJuliaRegisters.float4
double4  = QJuliaRegisters.double4
float8   = QJuliaRegisters.float8
double8  = QJuliaRegisters.double8
float16  = QJuliaRegisters.float16

complex16   = ComplexF16
complex32   = ComplexF32
complex64   = ComplexF64
complexBig  = Complex{BigFloat}

const QJULIA_MAX_DIMS = 6

abstract type QJuliaGrid_qj{T} end

mutable struct QJuliaGridDescr_qj{T<:Any} <: QJuliaGrid_qj{T}
  # Register type
  register_type::DataType

  # Number of field dimensions
  nDim::Int

  # The local space-time dimensions (without checkboarding)
  X::NTuple{QJULIA_MAX_DIMS, Int}

  # Padding parameter
  pad::NTuple{QJULIA_MAX_DIMS, Int}

  # Local volume virtual decomposition (used for vectorization),
  # supported formats (in x,y,z,t order) are (see P.Boyle et al paper, arXiv:1512.03487):
  # 1. SP scalar type (1,1,1,1), 2. m128  type (1,1,1,2), 3. m256  type (1,1,2,2), 4. m512  type (1,2,2,2)
  # 4. DP scalar type (1,1,1,1), 5. m128d type (1,1,1,1), 6. m256d type (1,1,1,2), 7. m512d type (1,1,2,2)
  # 8. QUDA grid (all precisions) (X[1], X[2], X[3], X[4])
  D::NTuple{4, Int}

  # The number of dimensions we partition for communication
  nDimComms::Int

  # Field volume
  volume::Int

  # Field checkerboarded volume
  volumeCB::Int

  # Field volume after simdization and padding
  grid_volume::Int

  # Field checkerboarded volume after simdization and padding
  grid_volumeCB::Int

  # Grid location
  location::QJuliaEnums.QJuliaFieldLocation_qj

  # Defualt constructor
  QJuliaGridDescr_qj{T}() where {T<:Any} = new(T,  4,
                                NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->1, QJULIA_MAX_DIMS)),
                                NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->0, QJULIA_MAX_DIMS)),
                                NTuple{4, Int}(ntuple(i->1, 4)),
                                0, 1, 1, 1, 1,
				QJuliaEnums.QJULIA_CPU_FIELD_LOCATION)


  function QJuliaGridDescr_qj{T}(location::QJuliaEnums.QJuliaFieldLocation_qj, ndimscomm::Int, X::NTuple) where T
     # Check lattice dimensions
     if sizeof(X) > (sizeof(Int)*QJULIA_MAX_DIMS); error("Requested lattice dimensions are not supported"); end
     # Set lattice dimensions
     nDims = length(X)
     #
     xx = NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->(i<=length(X) ? X[i] : 1), QJULIA_MAX_DIMS))
     #
     if ( location == QJuliaEnums.QJULIA_CUDA_FIELD_LOCATION)
       d=NTuple{4, Int}(ntuple(i->X[i], 4)) #WARNING: higher dims are cuurently not supported
     elseif(T == float4 || T ==  double4)
       d=(1, 1, 1, 2)
     elseif(T == float8 || T ==  double8)
       d=(1, 1, 2, 2)
     elseif(T == float16)
       d=(1, 2, 2, 2)
     else
       d=(1, 1, 1, 1) #(T == half2 || T == float2 || T == double2 || T == Complex{BigFloat} || T == ComplexF64 || T == ComplexF32 || T == ComplexF16)
     end

     simd_length = QJuliaRegisters.register_size((T <: NTuple || T == Complex{BigFloat} || T == ComplexF64 || T == ComplexF32 || T == ComplexF16) ? T : T())

     # Set up padding
     pad = NTuple{QJULIA_MAX_DIMS, Int}(ntuple(i->0, QJULIA_MAX_DIMS))

     # Set (complex) field volume:
     volume = 1
     [volume *= (i[1]+i[2]) for i in zip(xx,pad)]
     # Set cb volume:
     volumeCB =  volume >> 1

     # Set field (cb) grid volume
     grid_volume = volume / simd_length; grid_volumeCB = volumeCB / simd_length

     # call constructor
     new(T, nDims, xx, pad, d, ndimscomm, volume, volumeCB, grid_volume, grid_volumeCB, location)

  end #QJuliaLatticeFieldDescr_qj constructor

end

function grid_descr_info(grid::QJuliaGridDescr_qj)
  println("Register type : ", grid.register_type)
  print("Grid dimensions : ", grid.nDim, ", ("); [print(grid.X[i], ", ") for i in 1:grid.nDim]; println(")")
  println("Lattice volume : ", grid.volume)
  println("Lattice volume (CB) : ", grid.volumeCB)
  println("Grid volume : ", grid.grid_volume)
  println("Grid volume (CB) : ", grid.grid_volumeCB)
end

#QJuliaLatticeFieldDesc_qj
mutable struct QJuliaLatticeFieldDescr_qj{NSpin<:Any, NColor<:Any, NBlock<:Any}

  # Grid descriptor
  grid::Any

  # Field precision (Float16/32/64 or BigFloat)
  prec::DataType

  # Field geometry (spinor, vector, tensor etc.)
  geom::QJuliaEnums.QJuliaFieldGeometry_qj

  # Spin dof
  nSpin::Any

  # Color dof
  nColor::Int

  # Block size
  nBlock::Int

  # Whether the field is full or single parity (2 : full, 1 : single parity )
  siteSubset::Int

  # Parity type
  parity::QJuliaEnums.QJuliaParity_qj

  # Defualt constructor
  QJuliaLatticeFieldDescr_qj{NSpin, NColor, NBlock}() where NSpin where NColor where NBlock  = new( missing, Float64,
                                      QJuliaEnums.QJULIA_INVALID_GEOMETRY,
                                      NSpin, NColor, NBlock,
				      2,
				      QJuliaEnums.QJULIA_INVALID_PARITY)


  function QJuliaLatticeFieldDescr_qj{NSpin, NColor, NBlock}(grid::QJuliaGrid_qj, geom::QJuliaEnums.QJuliaFieldGeometry_qj, parity::QJuliaEnums.QJuliaParity_qj) where NSpin where NColor where NBlock
     # Check spin dof
     if (sum([nothing,1,2,4] .!= NSpin) == 4)
        error("NSpin parameter is incorrect, must be 0 (gauge field), 1 (fine staggered), 2 (coarse spinor) or 4 (fine spinor) ")
     end
     # Check NColor:
     if (NColor < 1) ; error("NColor parameter is incorrect, must be non-zero."); end

     # Check NBlock:
     if (NBlock < 1) ; error("NBlock parameter is incorrect, must be non-zero."); end

     if geom == QJuliaEnums.QJULIA_VECTOR_GEOMETRY
       #
       if (NSpin  != nothing); error("Spin dof is not allowed for this type of the field geometry"); end
       #
       if (NBlock != 1); error("Block is not supported for this type of the field geometry");  end
     else
       if (NSpin  == nothing); error("NSpin = 0 is not allowed for ", geom,); end
     end

     # Set field precision
     grt  = grid.register_type
     prec = QJuliaRegisters.register_type((grt <: NTuple || grt == Complex{BigFloat} || grt == ComplexF64 || grt == ComplexF32 || grt == ComplexF16) ? grt : grt())

     # Set parity flags
     siteSubset = parity == QJuliaEnums.QJULIA_INVALID_PARITY ? 2 : 1

     # call constructor
     new(grid, prec, geom, NSpin, NColor, NBlock, siteSubset, parity)

  end #QJuliaLatticeFieldDescr_qj constructor

end

function field_descr_info(field_desc::QJuliaLatticeFieldDescr_qj; grid_info = true)

  println("Grid info : ", grid_info)

  if grid_info == true; grid_descr_info(field_desc.grid); end

  println("Precision : ", field_desc.prec )
  println("Field geometry : ", field_desc.geom )
  println("Site subset : ", field_desc.siteSubset == 1 ? "parity field" : "full field.")

  if field_desc.siteSubset == 1 ; println("Parity type : ", field_desc.parity); end

  print("Spin : "); show(field_desc.nSpin)
  print( ", Color : "); show(field_desc.nColor)
  print(", Block size : "); show(field_desc.nBlock)

  println("\n")

end

# Collection of outer constructors:

CreateColorSpinorParams(grid_dscr::QJuliaGridDescr_qj, parity::QJuliaEnums.QJuliaParity_qj; NSpin::Int = 4, NColor::Int = 3, NBlock::Int = 1) = QJuliaLatticeFieldDescr_qj{NSpin, NColor, NBlock}(grid_dscr, QJuliaEnums.QJULIA_SCALAR_GEOMETRY, parity)

CreateGaugeParams(grid_dscr::QJuliaGridDescr_qj; NColor::Int = 3) = QJuliaLatticeFieldDescr_qj{nothing, NColor, 1}(grid_dscr, QJuliaEnums.QJULIA_VECTOR_GEOMETRY, QJuliaEnums.QJULIA_INVALID_PARITY)

end #QJuliaGrid
