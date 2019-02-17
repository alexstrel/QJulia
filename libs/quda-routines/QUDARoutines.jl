module QUDARoutines

import QJuliaInterface
import QJuliaEnums

################# Interface functions, found in quda.h #######################
#be sure that a directory with libquda.so is in LD_LIBRARY_PATH

   initCommsGridQuda_qj(len, gridsize, lex_rank_from_coords, fdata ) = ccall((:initCommsGridQuda, "libquda"), Cvoid, (Int32, Ptr{Int32}, Ptr{Cvoid}, Ptr{Cvoid}, ), len, gridsize, lex_rank_from_coords, fdata )

   initQudaDevice_qj(dev) = ccall((:initQudaDevice, "libquda"), Cvoid, (Int32,), dev)

   initQudaMemory_qj() = ccall((:initQudaMemory, "libquda"), Cvoid, (), )

   initQuda_qj(dev) = ccall((:initQuda, "libquda"), Cvoid, (Int32,), dev)

   endQuda_qj() = ccall((:endQuda, "libquda"), Cvoid, (), )

   printQudaGaugeParam_qj(param) = ccall((:printQudaGaugeParam, "libquda"), Cvoid, (Ref{QJuliaInterface.QJuliaGaugeParam_qj}, ), param)

   printQudaInvertParam_qj(param) = ccall((:printQudaInvertParam, "libquda"), Cvoid, (Ref{QJuliaInterface.QJuliaInvertParam_qj}, ), param)

   loadGaugeQuda_qj(gptr, param) = ccall((:loadGaugeQuda, "libquda"), Cvoid, (Ptr{Cvoid}, Ref{QJuliaInterface.QJuliaGaugeParam_qj}, ), gptr, param)

   freeGaugeQuda_qj() = ccall((:freeGaugeQuda, "libquda"), Cvoid, (), )

   saveGaugeQuda_qj(gptr, param) = ccall((:saveGaugeQuda, "libquda"), Cvoid, (Ptr{Cvoid}, Ref{QJuliaInterface.QJuliaGaugeParam_qj}, ), gptr, param)

   loadCloverQuda_qj(clvptr, clvptr_inv, param) = ccall((:loadCloverQuda, "libquda"), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ref{QJuliaInterface.QJuliaInvertParam_qj}, ), clvptr, clvptr_inv, param)

   freeCloverQuda_qj() = ccall((:freeCloverQuda, "libquda"), Cvoid, (), )

   plaqQuda_qj(plaq) = ccall((:plaqQuda, "libquda"), Cvoid, (Ref{Cdouble}, ), plaq)

   invertQuda_qj(xptr, yptr, param) = ccall((:invertQuda, "libquda"), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ref{QJuliaInterface.QJuliaInvertParam_qj}, ), xptr, yptr, param)

   newMultigridQuda_qj(param) = ccall((:newMultigridQuda, "libquda"), Ptr{Cvoid}, (Ref{QJuliaInterface.QJuliaMultigridParam_qj}, ), param)

   destroyMultigridQuda_qj(mg_instance) = ccall((:destroyMultigridQuda, "libquda"), Cvoid, (Ptr{Cvoid},), mg_instance)

   dslashQuda_qj(outptr, inptr, param, parity) = ccall((:dslashQuda, "libquda"), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ref{QJuliaInterface.QJuliaInvertParam_qj}, Ref{QJuliaEnums.QJuliaParity_qj},), outptr, inptr, param, parity)


   MatQuda_qj(outptr, inptr, param) = ccall((:MatQuda, "libquda"), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ref{QJuliaInterface.QJuliaInvertParam_qj},), outptr, inptr, param)

   MatDagMatQuda_qj(outptr, inptr, param) = ccall((:MatDagMatQuda, "libquda"), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ref{QJuliaInterface.QJuliaInvertParam_qj},), outptr, inptr, param)


   createGaugeFieldQuda_qj(gptr, geom, param) = ccall((:createGaugeFieldQuda, "libquda"), Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Ref{QJuliaInterface.QJuliaGaugeParam_qj}, ), gptr, geom, param)

   saveGaugeFieldQuda_qj(ogptr, igptr, param) = ccall((:saveGaugeFieldQuda, "libquda"), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ref{QJuliaInterface.QJuliaGaugeParam_qj}, ), ogptr, igptr, param)


   destroyGaugeFieldQuda_qj() = ccall((:destroyGaugeFieldQuda, "libquda"), Cvoid, (Ptr{Cvoid},), gptr )

   gaussGaugeQuda_qj(seed) = ccall((:gaussGaugeQuda, "libquda"), Cvoid, (Clong,), seed)

   commDim_qj(idx)         = ccall((:commDim, "libquda"), Cint, (Cint,), idx)

   commCoords_qj(idx)      = ccall((:commDim, "libquda"), Cint, (Cint,), idx)

#   read_gauge_field_qj(filename, gauge, prec, X, argc, argv) = ccall((:read_gauge_field, "libqiowrap"), Cvoid, (Ptr{Cchar}, Ptr{Cvoid}, Cint, Ptr{Cint}, Cint, Ptr{Ptr{Cchar}}), filename, gauge, prec, X, argc, argv)

#   QMPInitComms_qj(argc, argv, commDims) = ccall((:QMPInitComms, "libqiowrap"), Cvoid, (Cint, Ptr{Ptr{Cchar}}, Ptr{Cint}), argc, argv, commDims)

#   QMPFinalizeComms_qj() = ccall((:QMPFinalizeComms, "libqiowrap"), Cvoid, (), )

end #QUDARoutines
