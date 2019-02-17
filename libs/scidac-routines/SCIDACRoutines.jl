module SCIDACRoutines

import QJuliaInterface
import QJuliaEnums

################# Interface functions, found in quda.h #######################
#be sure that a directory with libscidacwrap.so is in LD_LIBRARY_PATH

#QIO routines (based on QUDA implementation):
   read_gauge_field_qj(filename, gauge, prec, X, argc, argv) = ccall((:read_gauge_field, "libscidacwrap"), Cvoid, (Ptr{Cchar}, Ptr{Cvoid}, Cint, Ptr{Cint}, Cint, Ptr{Ptr{Cchar}}), filename, gauge, prec, X, argc, argv)

#QMP routines:
   QMPInitComms_qj(argc, argv, commDims) = ccall((:QMPInitComms, "libscidacwrap"), Cvoid, (Cint, Ptr{Ptr{Cchar}}, Ptr{Cint}), argc, argv, commDims)

   QMPFinalizeComms_qj() = ccall((:QMPFinalizeComms, "libscidacwrap"), Cvoid, (), )

end #QUDARoutines
