module QJuliaComms

#list of error codes (opempi3.1.2)

const QJULIA_MPI_SUCCESS                   =0
const QJULIA_MPI_ERR_BUFFER                =1
const QJULIA_MPI_ERR_COUNT                 =2
const QJULIA_MPI_ERR_TYPE                  =3
const QJULIA_MPI_ERR_TAG                   =4
const QJULIA_MPI_ERR_COMM                  =5
const QJULIA_MPI_ERR_RANK                  =6
const QJULIA_MPI_ERR_REQUEST               =7
const QJULIA_MPI_ERR_ROOT                  =8
const QJULIA_MPI_ERR_GROUP                 =9
const QJULIA_MPI_ERR_OP                    =10
const QJULIA_MPI_ERR_TOPOLOGY              =11
const QJULIA_MPI_ERR_DIMS                  =12
const QJULIA_MPI_ERR_ARG                   =13
const QJULIA_MPI_ERR_UNKNOWN               =14
const QJULIA_MPI_ERR_TRUNCATE              =15
const QJULIA_MPI_ERR_OTHER                 =16
const QJULIA_MPI_ERR_INTERN                =17
const QJULIA_MPI_ERR_IN_STATUS             =18
const QJULIA_MPI_ERR_PENDING               =19
const QJULIA_MPI_ERR_ACCESS                =20
const QJULIA_MPI_ERR_AMODE                 =21
const QJULIA_MPI_ERR_ASSERT                =22
const QJULIA_MPI_ERR_BAD_FILE              =23
const QJULIA_MPI_ERR_BASE                  =24
const QJULIA_MPI_ERR_CONVERSION            =25
const QJULIA_MPI_ERR_DISP                  =26
const QJULIA_MPI_ERR_DUP_DATAREP           =27
const QJULIA_MPI_ERR_FILE_EXISTS           =28
const QJULIA_MPI_ERR_FILE_IN_USE           =29
const QJULIA_MPI_ERR_FILE                  =30
const QJULIA_MPI_ERR_INFO_KEY              =31
const QJULIA_MPI_ERR_INFO_NOKEY            =32
const QJULIA_MPI_ERR_INFO_VALUE            =33
const QJULIA_MPI_ERR_INFO                  =34
const QJULIA_MPI_ERR_IO                    =35
const QJULIA_MPI_ERR_KEYVAL                =36
const QJULIA_MPI_ERR_LOCKTYPE              =37
const QJULIA_MPI_ERR_NAME                  =38
const QJULIA_MPI_ERR_NO_MEM                =39
const QJULIA_MPI_ERR_NOT_SAME              =40
const QJULIA_MPI_ERR_NO_SPACE              =41
const QJULIA_MPI_ERR_NO_SUCH_FILE          =42
const QJULIA_MPI_ERR_PORT                  =43
const QJULIA_MPI_ERR_QUOTA                 =44
const QJULIA_MPI_ERR_READ_ONLY             =45
const QJULIA_MPI_ERR_RMA_CONFLICT          =46
const QJULIA_MPI_ERR_RMA_SYNC              =47
const QJULIA_MPI_ERR_SERVICE               =48
const QJULIA_MPI_ERR_SIZE                  =49
const QJULIA_MPI_ERR_SPAWN                 =50
const QJULIA_MPI_ERR_UNSUPPORTED_DATAREP   =51
const QJULIA_MPI_ERR_UNSUPPORTED_OPERATION =52
const QJULIA_MPI_ERR_WIN                   =53
const QJULIA_MPI_T_ERR_MEMORY              =54
const QJULIA_MPI_T_ERR_NOT_INITIALIZED     =55
const QJULIA_MPI_T_ERR_CANNOT_INIT         =56
const QJULIA_MPI_T_ERR_INVALID_INDEX       =57
const QJULIA_MPI_T_ERR_INVALID_ITEM        =58
const QJULIA_MPI_T_ERR_INVALID_HANDLE      =59
const QJULIA_MPI_T_ERR_OUT_OF_HANDLES      =60
const QJULIA_MPI_T_ERR_OUT_OF_SESSIONS     =61
const QJULIA_MPI_T_ERR_INVALID_SESSION     =62
const QJULIA_MPI_T_ERR_CVAR_SET_NOT_NOW    =63
const QJULIA_MPI_T_ERR_CVAR_SET_NEVER      =64
const QJULIA_MPI_T_ERR_PVAR_NO_STARTSTOP   =65
const QJULIA_MPI_T_ERR_PVAR_NO_WRITE       =66
const QJULIA_MPI_T_ERR_PVAR_NO_ATOMIC      =67
const QJULIA_MPI_ERR_RMA_RANGE             =68
const QJULIA_MPI_ERR_RMA_ATTACH            =69
const QJULIA_MPI_ERR_RMA_FLAVOR            =70
const QJULIA_MPI_ERR_RMA_SHARED            =71
const QJULIA_MPI_T_ERR_INVALID             =72
const QJULIA_MPI_T_ERR_INVALID_NAME        =73

#MPI communicator handles:
const QJULIA_MPI_COMM_WORLD = 0
const QJULIA_MPI_COMM_SELF  = 1
const QJULIA_MPI_COMM_NULL  = 2


mutable struct Comm
    val::Cint
    Comm(val::Integer) = new(val)
end

const COMM_NULL  = Comm(QJULIA_MPI_COMM_NULL)
const COMM_SELF  = Comm(QJULIA_MPI_COMM_SELF)
const COMM_WORLD = Comm(QJULIA_MPI_COMM_WORLD)

##### MPI & other communication calls ######
   MPI_init_qj(argc, argv)  = ccall((:MPI_Init, "libmpi"), Cvoid, (Int32, Ptr{Ptr{Cchar}}), argc, argv)

   MPI_initialized_qj(flag) = ccall((:MPI_Initialized, "libmpi"), Cint, (Ptr{Cint}, ), flag)

   MPI_rank_qj(comm, rank)  = ccall((:MPI_Comm_rank, "libmpi"), Cint, (Ref{Cint}, Ptr{Cint}), comm, rank)

   MPI_size_qj(comm, size)  = ccall((:MPI_Comm_size, "libmpi"), Cint, (Ref{Cint}, Ptr{Cint}), comm, size)

   MPI_finalize_qj()        = ccall((:MPI_Finalize, "libmpi"), Cvoid, (), )

##### High-level wrappers

function MPI_Initialized_qj()
   flag  = Ref{Cint}()
   error_code = MPI_initialized_qj(flag)
   if error_code != QJULIA_MPI_SUCCESS
     println("WARNING: MPI_Initialized_qj returned MPI error code ", error_code)
   end 
   flag[] != 0
end

function MPI_Comm_rank_qj(comm::Comm)
   rank = Ref{Cint}()
   error_code = MPI_rank_qj(comm.val, rank)
   if error_code != QJULIA_MPI_SUCCESS
     println("WARNING: MPI_Comm_rank_qj returned MPI error code ", error_code)
   end 
   Int(rank[])
end

function MPI_Comm_size_qj(comm::Comm)
   size = Ref{Cint}()
   error_code = MPI_size_qj(comm.val, size)
   if error_code != QJULIA_MPI_SUCCESS
     println("WARNING: MPI_Comm_size_qj returned MPI error code ", error_code)
   end 
   Int(size[])
end


end #QJuliaComms 
