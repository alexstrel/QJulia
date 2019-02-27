module QJuliaEnums

  QJULIA_INVALID_ENUM = typemin(Int32)

  @enum QJuliaMemoryType_qj begin
    QJULIA_MEMORY_DEVICE
    QJULIA_MEMORY_PINNED
    QJULIA_MEMORY_MAPPED
    QJULIA_MEMORY_INVALID = QJULIA_INVALID_ENUM
  end

  #
  # Types used in QJuliaGaugeParam
  #

  @enum QJuliaLinkType_qj begin
    QJULIA_SU3_LINKS
    QJULIA_GENERAL_LINKS
    QJULIA_THREE_LINKS
    QJULIA_MOMENTUM_LINKS
    QJULIA_COARSE_LINKS
    QJULIA_SMEARED_LINKS
    QJULIA_WILSON_LINKS
    QJULIA_ASQTAD_FAT_LINKS
    QJULIA_ASQTAD_LONG_LINKS
    QJULIA_ASQTAD_MOM_LINKS
    QJULIA_ASQTAD_GENERAL_LINKS
    QJULIA_INVALID_LINKS = QJULIA_INVALID_ENUM
  end

  @enum QJuliaGaugeFieldOrder_qj begin
    QJULIA_FLOAT_GAUGE_ORDER  = 1
    QJULIA_FLOAT2_GAUGE_ORDER = 2
    QJULIA_FLOAT4_GAUGE_ORDER = 4
    QJULIA_QDP_GAUGE_ORDER
    QJULIA_QDPJIT_GAUGE_ORDER
    QJULIA_CPS_WILSON_GAUGE_ORDER
    QJULIA_MILC_GAUGE_ORDER
    QJULIA_MILC_SITE_GAUGE_ORDER
    QJULIA_BQCD_GAUGE_ORDER
    QJULIA_TIFR_GAUGE_ORDER
    QJULIA_TIFR_PADDED_GAUGE_ORDER
    QJULIA_INVALID_GAUGE_ORDER = QJULIA_INVALID_ENUM
  end

  @enum QJuliaTboundary_qj begin
    QJULIA_ANTI_PERIODIC_T = -1
    QJULIA_PERIODIC_T      =  1
    QJULIA_INVALID_T_BOUNDARY = QJULIA_INVALID_ENUM
  end

  @enum QJuliaPrecision_qj begin
    QJULIA_QUARTER_PRECISION = 1
    QJULIA_HALF_PRECISION    = 2
    QJULIA_SINGLE_PRECISION  = 4
    QJULIA_DOUBLE_PRECISION  = 8
    QJULIA_INVALID_PRECISION = QJULIA_INVALID_ENUM
  end

  @enum QJuliaReconstructType_qj begin
    QJULIA_RECONSTRUCT_NO = 18
    QJULIA_RECONSTRUCT_12 = 12
    QJULIA_RECONSTRUCT_8  = 8
    QJULIA_RECONSTRUCT_9  = 9
    QJULIA_RECONSTRUCT_13 = 13
    QJULIA_RECONSTRUCT_10 = 10
    QJULIA_RECONSTRUCT_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaGaugeFixed_qj begin
    QJULIA_GAUGE_FIXED_NO
    QJULIA_GAUGE_FIXED_YES
    QJULIA_GAUGE_FIXED_INVALID = QJULIA_INVALID_ENUM
  end

  #
  # Types used in QJuliaInvertParam
  #

  @enum QJuliaDslashType_qj begin
    QJULIA_WILSON_DSLASH
    QJULIA_CLOVER_WILSON_DSLASH
    QJULIA_DOMAIN_WALL_DSLASH
    QJULIA_DOMAIN_WALL_4D_DSLASH
    QJULIA_MOBIUS_DWF_DSLASH
    QJULIA_STAGGERED_DSLASH
    QJULIA_ASQTAD_DSLASH
    QJULIA_TWISTED_MASS_DSLASH
    QJULIA_TWISTED_CLOVER_DSLASH
    QJULIA_LAPLACE_DSLASH
    QJULIA_COVDEV_DSLASH
    QJULIA_INVALID_DSLASH = QJULIA_INVALID_ENUM
  end

  @enum QJuliaInverterType_qj begin
    QJULIA_CG_INVERTER
    QJULIA_BICGSTAB_INVERTER
    QJULIA_GCR_INVERTER
    QJULIA_MR_INVERTER
    QJULIA_MPBICGSTAB_INVERTER
    QJULIA_SD_INVERTER
    QJULIA_XSD_INVERTER
    QJULIA_PCG_INVERTER
    QJULIA_MPCG_INVERTER
    QJULIA_EIGCG_INVERTER
    QJULIA_INC_EIGCG_INVERTER
    QJULIA_GMRESDR_INVERTER
    QJULIA_GMRESDR_PROJ_INVERTER
    QJULIA_GMRESDR_SH_INVERTER
    QJULIA_FGMRESDR_INVERTER
    QJULIA_MG_INVERTER
    QJULIA_BICGSTABL_INVERTER
    QJULIA_CGNE_INVERTER
    QJULIA_CGNR_INVERTER
    QJULIA_CG3_INVERTER
    QJULIA_CG3NE_INVERTER
    QJULIA_CG3NR_INVERTER
    QJULIA_PIPEPCG_INVERTER
    QJULIA_PIPECG_INVERTER    
    QJULIA_PIPE2PCG_INVERTER
    QJULIA_SRE_PCG_INVERTER
    QJULIA_LRE_PCG_INVERTER
    QJULIA_CA_CG_INVERTER
    QJULIA_CA_GCR_INVERTER
    QJULIA_LANMR_INVERTER
    QJULIA_FCG_INVERTER
    QJULIA_INVALID_INVERTER = QJULIA_INVALID_ENUM
  end

  @enum QJuliaSolutionType_qj begin
    QJULIA_MAT_SOLUTION
    QJULIA_MATDAG_MAT_SOLUTION
    QJULIA_MATPC_SOLUTION
    QJULIA_MATPC_DAG_SOLUTION
    QJULIA_MATPCDAG_MATPC_SOLUTION
    QJULIA_MATPCDAG_MATPC_SHIFT_SOLUTION
    QJULIA_INVALID_SOLUTION = QJULIA_INVALID_ENUM
  end

#QJULIA_NORMEQ_SOLVE    = QJULIA_NORMOP_SOLVE
#QJULIA_NORMEQ_PC_SOLVE = QJULIA_NORMOP_PC_SOLVE
  @enum QJuliaSolveType_qj begin
    QJULIA_DIRECT_SOLVE
    QJULIA_NORMOP_SOLVE
    QJULIA_DIRECT_PC_SOLVE
    QJULIA_NORMOP_PC_SOLVE
    QJULIA_NORMERR_SOLVE
    QJULIA_NORMERR_PC_SOLVE
    QJULIA_NORMEQ_SOLVE
    QJULIA_NORMEQ_PC_SOLVE
    QJULIA_INVALID_SOLVE = QJULIA_INVALID_ENUM
  end

  @enum QJuliaMultigridCycleType_qj begin
    QJULIA_MG_CYCLE_VCYCLE
    QJULIA_MG_CYCLE_FCYCLE
    QJULIA_MG_CYCLE_WCYCLE
    QJULIA_MG_CYCLE_RECURSIVE
    QJULIA_MG_CYCLE_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaSchwarzType_qj begin
    QJULIA_ADDITIVE_SCHWARZ
    QJULIA_MULTIPLICATIVE_SCHWARZ
    QJULIA_INVALID_SCHWARZ = QJULIA_INVALID_ENUM
  end

  @enum QJuliaResidualType_qj begin
    QJULIA_L2_RELATIVE_RESIDUAL = 1
    QJULIA_L2_ABSOLUTE_RESIDUAL = 2
    QJULIA_HEAVY_QUARK_RESIDUAL = 4
    QJULIA_INVALID_RESIDUAL = QJULIA_INVALID_ENUM
  end

  # Whether the preconditioned matrix is (1-k^2 Deo Doe) or (1-k^2 Doe Deo)
  #
  # For the clover-improved Wilson Dirac operator QJULIA_MATPC_EVEN_EVEN
  # defaults to the "symmetric" form (1 - k^2 A_ee^-1 D_eo A_oo^-1 D_oe)
  # and likewise for QJULIA_MATPC_ODD_ODD.
  #
  # For the "asymmetric" form (A_ee - k^2 D_eo A_oo^-1 D_oe) select
  # QJULIA_MATPC_EVEN_EVEN_ASYMMETRIC.
  #

  @enum QJuliaMatPCType_qj begin
    QJULIA_MATPC_EVEN_EVEN
    QJULIA_MATPC_ODD_ODD
    QJULIA_MATPC_EVEN_EVEN_ASYMMETRIC
    QJULIA_MATPC_ODD_ODD_ASYMMETRIC
    QJULIA_MATPC_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaDagType_qj begin
    QJULIA_DAG_NO
    QJULIA_DAG_YES
    QJULIA_DAG_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaMassNormalization_qj begin
    QJULIA_KAPPA_NORMALIZATION
    QJULIA_MASS_NORMALIZATION
    QJULIA_ASYMMETRIC_MASS_NORMALIZATION
    QJULIA_INVALID_NORMALIZATION = QJULIA_INVALID_ENUM
  end

  @enum QJuliaSolverNormalization_qj begin
    QJULIA_DEFAULT_NORMALIZATION
    QJULIA_SOURCE_NORMALIZATION
  end

  @enum QJuliaPreserveSource_qj begin
    QJULIA_PRESERVE_SOURCE_NO
    QJULIA_PRESERVE_SOURCE_YES
    QJULIA_PRESERVE_SOURCE_INVALID
  end

  @enum QJuliaDiracFieldOrder_qj begin
    QJULIA_INTERNAL_DIRAC_ORDER
    QJULIA_DIRAC_ORDER
    QJULIA_QDP_DIRAC_ORDER
    QJULIA_QDPJIT_DIRAC_ORDER
    QJULIA_CPS_WILSON_DIRAC_ORDER
    QJULIA_LEX_DIRAC_ORDER
    QJULIA_TIFR_PADDED_DIRAC_ORDER
    QJULIA_INVALID_DIRAC_ORDER = QJULIA_INVALID_ENUM
  end

  @enum QJuliaCloverFieldOrder_qj begin
    QJULIA_FLOAT_CLOVER_ORDER  = 1
    QJULIA_FLOAT2_CLOVER_ORDER = 2
    QJULIA_FLOAT4_CLOVER_ORDER = 4
    QJULIA_PACKED_CLOVER_ORDER
    QJULIA_QDPJIT_CLOVER_ORDER
    QJULIA_BQCD_CLOVER_ORDER
    QJULIA_INVALID_CLOVER_ORDER = QJULIA_INVALID_ENUM
  end

  @enum QJuliaVerbosity_qj begin
    QJULIA_SILENT
    QJULIA_SUMMARIZE
    QJULIA_VERBOSE
    QJULIA_DEBUG_VERBOSE
    QJULIA_INVALID_VERBOSITY = QJULIA_INVALID_ENUM
  end

  @enum QJuliaTune_qj begin
    QJULIA_TUNE_NO
    QJULIA_TUNE_YES
    QJULIA_TUNE_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaPreserveDirac_qj begin
    QJULIA_PRESERVE_DIRAC_NO
    QJULIA_PRESERVE_DIRAC_YES
    QJULIA_PRESERVE_DIRAC_INVALID = QJULIA_INVALID_ENUM
  end

  #
  # Type used for "parity" argument to dslashQJulia()
  #

  @enum QJuliaParity_qj begin
    QJULIA_EVEN_PARITY
    QJULIA_ODD_PARITY
    QJULIA_INVALID_PARITY = QJULIA_INVALID_ENUM
  end

  #
  # Types used only internally
  #

  @enum QJuliaDiracType_qj begin
    QJULIA_WILSON_DIRAC
    QJULIA_WILSONPC_DIRAC
    QJULIA_CLOVER_DIRAC
    QJULIA_CLOVERPC_DIRAC
    QJULIA_DOMAIN_WALL_DIRAC
    QJULIA_DOMAIN_WALLPC_DIRAC
    QJULIA_DOMAIN_WALL_4DPC_DIRAC
    QJULIA_MOBIUS_DOMAIN_WALL_DIRAC
    QJULIA_MOBIUS_DOMAIN_WALLPC_DIRAC
    QJULIA_STAGGERED_DIRAC
    QJULIA_STAGGEREDPC_DIRAC
    QJULIA_ASQTAD_DIRAC
    QJULIA_ASQTADPC_DIRAC
    QJULIA_TWISTED_MASS_DIRAC
    QJULIA_TWISTED_MASSPC_DIRAC
    QJULIA_TWISTED_CLOVER_DIRAC
    QJULIA_TWISTED_CLOVERPC_DIRAC
    QJULIA_COARSE_DIRAC
    QJULIA_COARSEPC_DIRAC
    QJULIA_GAUGE_LAPLACE_DIRAC
    QJULIA_GAUGE_LAPLACEPC_DIRAC
    QJULIA_GAUGE_COVDEV_DIRAC
    QJULIA_INVALID_DIRAC = QJULIA_INVALID_ENUM
  end

  # Where the field is stored
  @enum QJuliaFieldLocation_qj begin
    QJULIA_CPU_FIELD_LOCATION = 1
    QJULIA_CUDA_FIELD_LOCATION = 2
    QJULIA_INVALID_FIELD_LOCATION = QJULIA_INVALID_ENUM
  end

  # Which sites are included
  @enum QJuliaSiteSubset_qj begin
    QJULIA_PARITY_SITE_SUBSET = 1
    QJULIA_FULL_SITE_SUBSET = 2
    QJULIA_INVALID_SITE_SUBSET = QJULIA_INVALID_ENUM
  end

  # Site ordering (always t-z-y-x with rightmost varying fastest)
  @enum QJuliaSiteOrder_qj begin
    QJULIA_LEXICOGRAPHIC_SITE_ORDER
    QJULIA_EVEN_ODD_SITE_ORDER
    QJULIA_ODD_EVEN_SITE_ORDER
    QJULIA_INVALID_SITE_ORDER
  end

  # Degree of freedom ordering
  @enum QJuliaFieldOrder_qj begin
    QJULIA_FLOAT_FIELD_ORDER  = 1
    QJULIA_FLOAT2_FIELD_ORDER = 2
    QJULIA_FLOAT4_FIELD_ORDER = 4
    QJULIA_SPACE_SPIN_COLOR_FIELD_ORDER
    QJULIA_SPACE_COLOR_SPIN_FIELD_ORDER
    QJULIA_QDPJIT_FIELD_ORDER
    QJULIA_QOP_DOMAIN_WALL_FIELD_ORDER
    QJULIA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER
    QJULIA_INVALID_FIELD_ORDER = QJULIA_INVALID_ENUM
  end

  @enum QJuliaFieldCreate_qj begin
    QJULIA_NULL_FIELD_CREATE
    QJULIA_ZERO_FIELD_CREATE
    QJULIA_COPY_FIELD_CREATE
    QJULIA_REFERENCE_FIELD_CREATE
    QJULIA_INVALID_FIELD_CREATE = QJULIA_INVALID_ENUM
  end

  @enum QJuliaGammaBasis_qj begin
    QJULIA_DEGRAND_ROSSI_GAMMA_BASIS
    QJULIA_UKQCD_GAMMA_BASIS
    QJULIA_CHIRAL_GAMMA_BASIS
    QJULIA_INVALID_GAMMA_BASIS = QJULIA_INVALID_ENUM
  end

  @enum QJuliaSourceType_qj begin
    QJULIA_POINT_SOURCE
    QJULIA_RANDOM_SOURCE
    QJULIA_CONSTANT_SOURCE
    QJULIA_SINUSOIDAL_SOURCE
    QJULIA_CORNER_SOURCE
    QJULIA_INVALID_SOURCE = QJULIA_INVALID_ENUM
  end

  @enum QJuliaNoiseType_qj begin
    QJULIA_NOISE_GAUSS
    QJULIA_NOISE_UNIFORM
    QJULIA_NOISE_INVALID = QJULIA_INVALID_ENUM
  end

  # used to select projection method for deflated solvers
  @enum QJuliaProjectionType_qj begin
      QJULIA_MINRES_PROJECTION
      QJULIA_GALERKIN_PROJECTION
      QJULIA_INVALID_PROJECTION = QJULIA_INVALID_ENUM
  end

  # used to select preconditioning method in domain-wall fermion
  @enum QJuliaDWFPCType_qj begin
    QJULIA_5D_PC
    QJULIA_4D_PC
    QJULIA_PC_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaTwistFlavorType_qj begin
    QJULIA_TWIST_SINGLET        = 1
    QJULIA_TWIST_NONDEG_DOUBLET = +2
    QJULIA_TWIST_DEG_DOUBLET    = -2
    QJULIA_TWIST_NO             = 0
    QJULIA_TWIST_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaTwistDslashType_qj begin
    QJULIA_DEG_TWIST_INV_DSLASH
    QJULIA_DEG_DSLASH_TWIST_INV
    QJULIA_DEG_DSLASH_TWIST_XPAY
    QJULIA_NONDEG_DSLASH
    QJULIA_DSLASH_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaTwistCloverDslashType_qj begin
    QJULIA_DEG_CLOVER_TWIST_INV_DSLASH
    QJULIA_DEG_DSLASH_CLOVER_TWIST_INV
    QJULIA_DEG_DSLASH_CLOVER_TWIST_XPAY
    QJULIA_TC_DSLASH_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaTwistGamma5Type_qj begin
    QJULIA_TWIST_GAMMA5_DIRECT
    QJULIA_TWIST_GAMMA5_INVERSE
    QJULIA_TWIST_GAMMA5_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaUseInitGuess_qj begin
    QJULIA_USE_INIT_GUESS_NO
    QJULIA_USE_INIT_GUESS_YES
    QJULIA_USE_INIT_GUESS_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaComputeNullVector_qj begin
    QJULIA_COMPUTE_NULL_VECTOR_NO
    QJULIA_COMPUTE_NULL_VECTOR_YES
    QJULIA_COMPUTE_NULL_VECTOR_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaSetupType_qj begin
    QJULIA_NULL_VECTOR_SETUP
    QJULIA_TEST_VECTOR_SETUP
    QJULIA_INVALID_SETUP_TYPE = QJULIA_INVALID_ENUM
  end

  @enum QJuliaBoolean_qj begin
    QJULIA_BOOLEAN_NO  = 0
    QJULIA_BOOLEAN_YES = 1
    QJULIA_BOOLEAN_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaDirection_qj begin
    QJULIA_BACKWARDS = -1
    QJULIA_FORWARDS  = +1
    QJULIA_BOTH_DIRS = 2
  end

  @enum QJuliaLinkDirection_qj begin
    QJULIA_LINK_BACKWARDS
    QJULIA_LINK_FORWARDS
    QJULIA_LINK_BIDIRECTIONAL
  end

  @enum QJuliaFieldGeometry_qj begin
    QJULIA_SCALAR_GEOMETRY = 1
    QJULIA_VECTOR_GEOMETRY = 4
    QJULIA_TENSOR_GEOMETRY = 6
    QJULIA_COARSE_GEOMETRY = 8
    QJULIA_INVALID_GEOMETRY = QJULIA_INVALID_ENUM
  end

  @enum QJuliaGhostExchange_qj begin
    QJULIA_GHOST_EXCHANGE_NO
    QJULIA_GHOST_EXCHANGE_PAD
    QJULIA_GHOST_EXCHANGE_EXTENDED
    QJULIA_GHOST_EXCHANGE_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaStaggeredPhase_qj begin
    QJULIA_STAGGERED_PHASE_NO   = 0
    QJULIA_STAGGERED_PHASE_MILC = 1
    QJULIA_STAGGERED_PHASE_CPS  = 2
    QJULIA_STAGGERED_PHASE_TIFR = 3
    QJULIA_STAGGERED_PHASE_INVALID = QJULIA_INVALID_ENUM
  end

  @enum QJuliaContractType_qj begin
    QJULIA_CONTRACT
    QJULIA_CONTRACT_PLUS
    QJULIA_CONTRACT_MINUS
    QJULIA_CONTRACT_GAMMA5
    QJULIA_CONTRACT_GAMMA5_PLUS
    QJULIA_CONTRACT_GAMMA5_MINUS
    QJULIA_CONTRACT_TSLICE
    QJULIA_CONTRACT_TSLICE_PLUS
    QJULIA_CONTRACT_TSLICE_MINUS
    QJULIA_CONTRACT_INVALID = QJULIA_INVALID_ENUM
  end

  #Allows to choose an appropriate external library
  @enum QJuliaExtLibType_qj begin
    QJULIA_CUSOLVE_EXTLIB
    QJULIA_EIGEN_EXTLIB
    QJULIA_MAGMA_EXTLIB
    QJULIA_EXTLIB_INVALID = QJULIA_INVALID_ENUM
  end

  #QMP stuff
  @enum QJuliaQMPstatus_qj begin
    QJULIA_QMP_SUCCESS = 0
    QJULIA_QMP_ERROR = 4097
    QJULIA_QMP_NOT_INITED
    QJULIA_QMP_RTENV_ERR
    QJULIA_QMP_CPUINFO_ERR
    QJULIA_QMP_NODEINFO_ERR
    QJULIA_QMP_NOMEM_ERR
    QJULIA_QMP_MEMSIZE_ERR
    QJULIA_QMP_HOSTNAME_ERR
    QJULIA_QMP_INITSVC_ERR
    QJULIA_QMP_TOPOLOGY_EXISTS
    QJULIA_QMP_CH_TIMEOUT
    QJULIA_QMP_NOTSUPPORTED
    QJULIA_QMP_SVC_BUSY
    QJULIA_QMP_BAD_MESSAGE
    QJULIA_QMP_INVALID_ARG
    QJULIA_QMP_INVALID_TOPOLOGY
    QJULIA_QMP_NONEIGHBOR_INFO
    QJULIA_QMP_MEMSIZE_TOOBIG
    QJULIA_QMP_BAD_MEMORY
    QJULIA_QMP_NO_PORTS
    QJULIA_QMP_NODE_OUTRANGE
    QJULIA_QMP_CHDEF_ERR
    QJULIA_QMP_MEMUSED_ERR
    QJULIA_QMP_INVALID_OP
    QJULIA_QMP_TIMEOUT
    QJULIA_QMP_MAX_STATUS
  end

  @enum QJuliaQMPictype_qj begin
    QJULIA_QMP_SWITCH = 0
    QJULIA_QMP_GRID = 1
#    QJULIA_QMP_MESH = 1
    QJULIA_QMP_FATTREE = 2
  end

  @enum QJuliaQMPthreadLevel_qj begin
    QJULIA_QMP_THREAD_SINGLE
    QJULIA_QMP_THREAD_FUNNELED
    QJULIA_QMP_THREAD_SERIALIZED
    QJULIA_QMP_THREAD_MULTIPLE
  end

  @enum QJuliaQMPclear2send_qj begin
    QJULIA_QMP_CTS_DISABLED  = -1
    QJULIA_QMP_CTS_NOT_READY = 0
    QJULIA_QMP_CTS_READY     = 1
  end

end
