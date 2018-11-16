module QMPRoutines

import QJuliaInterface
import QJuliaEnums

################# Interface functions, found in qump.h #######################
#be sure that a directory with libqmp.so is in LD_LIBRARY_PATH
 
   QMP_is_initialized_qj() = ccall((:QMP_is_initialized, "libqmp"), Cint, (), )

   QMP_finalize_msg_passing_qj() = ccall((:QMP_finalize_msg_passing, "libqmp"), Cvoid, (), ) 

   QMP_init_msg_passing_qj(argc, argv, required, provided) = ccall((:QMP_init_msg_passing, "libqmp"), Ref{QJuliaEnums.QJuliaQMPstatus_qj}, (Ptr{Cint}, Ptr{Ptr{Ptr{Cchar}}}, Ref{QJuliaEnums.QJuliaQMPthreadLevel_qj},  Ptr{QJuliaEnums.QJuliaQMPthreadLevel_qj}), argc, argv, required, provided)

   QMP_declare_logical_topology_map_qj(dims, ndim, map, mapdim) = ccall((:QMP_declare_logical_topology_map, "libqmp"), Ref{QJuliaEnums.QJuliaQMPstatus_qj}, (Ptr{Cint}, Cint, Ptr{Cint}, Cint), dims, ndim, map, mapdim)

 
end #QMPRoutines
