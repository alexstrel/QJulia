module QJuliaInterface

  import QJuliaEnums

#WARNING: static array dims MUST coincide with corresponding ones from QUDA 
  const QJULIA_MAX_DWF_LS      = 32
  const QJULIA_MAX_MULTI_SHIFT = 32

  mutable struct QJuliaGaugeParam_qj 

    # The location of the gauge field
    location::QJuliaEnums.QJuliaFieldLocation_qj 

    # The local space-time dimensions (without checkboarding)
    X::NTuple{4, Cint}            

    # Used for Wilson and Wilson-clover
    anisotropy::Cdouble   
    # Used for staggered only 
    tadpole_coeff::Cdouble
    # Used by staggered long links
    scale::Cdouble

    # The link type of the gauge field (e.g., Wilson, fat, long, etc.)
    gtype::QJuliaEnums.QJuliaLinkType_qj 
    # The ordering on the input gauge field
    gauge_order::QJuliaEnums.QJuliaGaugeFieldOrder_qj 
    # The temporal boundary condition that will be used for fermion fields
    t_boundary::QJuliaEnums.QJuliaTboundary_qj 

    # The precision used by the caller
    cpu_prec::QJuliaEnums.QJuliaPrecision_qj 

    # The precision of the cuda gauge field
    cuda_prec::QJuliaEnums.QJuliaPrecision_qj 
    # The reconstruction type of the cuda gauge field
    reconstruct::QJuliaEnums.QJuliaReconstructType_qj 

    # The precision of the sloppy gauge field
    cuda_prec_sloppy::QJuliaEnums.QJuliaPrecision_qj 
    # The recontruction type of the sloppy gauge field
    reconstruct_sloppy::QJuliaEnums.QJuliaReconstructType_qj 

    # The precision of the sloppy gauge field for the refinement step in multishift
    cuda_prec_refinement_sloppy::QJuliaEnums.QJuliaPrecision_qj 
    # The recontruction type of the sloppy gauge field for the refinement step in multishift
    reconstruct_refinement_sloppy::QJuliaEnums.QJuliaReconstructType_qj 

    # The precision of the preconditioner gauge field
    cuda_prec_precondition::QJuliaEnums.QJuliaPrecision_qj 
    # The recontruction type of the preconditioner gauge field 
    reconstruct_precondition::QJuliaEnums.QJuliaReconstructType_qj 

    # Whether the input gauge field is in the axial gauge or not
    gauge_fix::QJuliaEnums.QJuliaGaugeFixed_qj 

    # The pad size that the cudaGaugeField will use (default=0)
    ga_pad::Cint       

    # Used by link fattening and the gauge and fermion forces
    site_ga_pad::Cint

    # Used by link fattening
    staple_pad::Cint
    # Used by link fattening
    llfat_ga_pad::Cint
    # Used by the gauge and fermion forces
    mom_ga_pad::Cint

    # Set the staggered phase type of the links
    staggered_phase_type::QJuliaEnums.QJuliaStaggeredPhase_qj 
    # Whether the staggered phase has already been applied to the links
    staggered_phase_applied::Cint

    # Imaginary chemical potential
    i_mu::Cdouble 

    # Width of overlapping domains
    overlap::Cint

    # When computing momentum, should we overwrite it or accumulate to 
    overwrite_mom::Cint

    # Use the resident gauge field as input 
    use_resident_gauge::Cint
    # Use the resident momentum field as input
    use_resident_mom::Cint
    # Make the result gauge field resident
    make_resident_gauge::Cint
    # Make the result momentum field resident
    make_resident_mom::Cint  
    # Return the result gauge field 
    return_result_gauge::Cint
    # Return the result momentum field 
    return_result_mom::Cint

    # Offset into MILC site struct to the gauge field (only if gauge_order=MILC_SITE_GAUGE_ORDER)
    gauge_offset::Csize_t
    # Offset into MILC site struct to the momentum field (only if gauge_order=MILC_SITE_GAUGE_ORDER) 
    mom_offset::Csize_t
    # Size of MILC site struct (only if gauge_order=MILC_SITE_GAUGE_ORDER)
    site_size::Csize_t 

    #defualt constructor (remarkable: no need to indicate enum name!)
    QJuliaGaugeParam_qj() = new(QJuliaEnums.QJULIA_CPU_FIELD_LOCATION, 
			     NTuple{4, Cint}(ntuple(i->16, 4)),
                             1.0,
			     0.0,
			     1.0,
                             QJuliaEnums.QJULIA_WILSON_LINKS,
			     QJuliaEnums.QJULIA_QDP_GAUGE_ORDER,
			     QJuliaEnums.QJULIA_PERIODIC_T,
			     QJuliaEnums.QJULIA_DOUBLE_PRECISION,
			     QJuliaEnums.QJULIA_SINGLE_PRECISION,
			     QJuliaEnums.QJULIA_RECONSTRUCT_NO,
			     QJuliaEnums.QJULIA_SINGLE_PRECISION,
			     QJuliaEnums.QJULIA_RECONSTRUCT_NO,
			     QJuliaEnums.QJULIA_SINGLE_PRECISION,
			     QJuliaEnums.QJULIA_RECONSTRUCT_NO,
			     QJuliaEnums.QJULIA_SINGLE_PRECISION,
			     QJuliaEnums.QJULIA_RECONSTRUCT_NO,
			     QJuliaEnums.QJULIA_GAUGE_FIXED_NO,
			     0,
			     0,
			     0,
			     0,
			     0,
                             QJuliaEnums.QJULIA_STAGGERED_PHASE_NO,
                             0, 
                             0.0,
                             0,
                             0,
			     0,
			     0,
			     0,
			     0,
			     0,
			     0, 
                             0,
			     0,
			     0)

  end #QJuliaGaugeParam


  mutable struct QJuliaInvertParam_qj 

    # The location of the input field
    input_location::QJuliaEnums.QJuliaFieldLocation_qj 
    # The location of the output field
    output_location::QJuliaEnums.QJuliaFieldLocation_qj 

    # The Dirac Dslash type that is being used
    dslash_type::QJuliaEnums.QJuliaDslashType_qj 
    # Which linear solver to use 
    inv_type::QJuliaEnums.QJuliaInverterType_qj 

    # Used for staggered only
    mass::Cdouble
    # Used for Wilson and Wilson-clover  
    kappa::Cdouble 
    # Domain wall height
    m5::Cdouble    
    # Extent of the 5th dimension (for domain wall)
    Ls::Cint       

    # MDWF coefficients
    b_5::NTuple{QJULIA_MAX_DWF_LS, Cdouble} 
    # will be used only for the mobius type of Fermion 
    c_5::NTuple{QJULIA_MAX_DWF_LS, Cdouble}   

    # Twisted mass parameter
    mu::Cdouble    
    # Twisted mass parameter
    epsilon::Cdouble 

    # Twisted mass flavor
    twist_flavor::QJuliaEnums.QJuliaTwistFlavorType_qj  

    # Solver tolerance in the L2 residual norm
    tol::Cdouble    
    # Solver tolerance in the L2 residual norm (used to restart InitCG)
    tol_restart::Cdouble   
    # Solver tolerance in the heavy quark residual norm
    tol_hq::Cdouble 

    # Whether to compute the true residual post solve
    compute_true_res::Cint
    # Actual L2 residual norm achieved in solver
    true_res::Cdouble
    # Actual heavy quark residual norm achieved in solver
    true_res_hq::Cdouble 
    # Maximum number of iterations in the linear solver
    maxiter::Cint 
    # Reliable update tolerance
    reliable_delta::Cdouble 
    # Reliable update tolerance used in post multi-shift solver refinement
    reliable_delta_refinement::Cdouble 
    # Whether to keep the partial solution accumuator in sloppy precision
    use_sloppy_partial_accumulator::Cint 

    solution_accumulator_pipeline::Cint

    max_res_increase::Cint

    max_res_increase_total::Cint

    # After how many iterations shall the heavy quark residual be updated 
    heavy_quark_check::Cint
    # Whether to use a pipelined solver with less global sums
    pipeline::Cint
    # Number of offsets in the multi-shift solver
    num_offset::Cint
    # Number of sources in the multiple source solver 
    num_src::Cint

    # Width of domain overlaps 
    overlap::Cint

    # Offsets for multi-shift solver 
    offset::NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}

    # Solver tolerance for each offset 
    tol_offset::NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}

    # Solver tolerance for each shift when refinement is applied using the heavy-quark residual
    tol_hq_offset::NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}

    # Actual L2 residual norm achieved in solver for each offset
    true_res_offset::NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}

    # Iterated L2 residual norm achieved in multi shift solver for each offset 
    iter_res_offset::NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}

    # Actual heavy quark residual norm achieved in solver for each offset 
    true_res_hq_offset::NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}

    # Residuals in the partial faction expansion 
    residue::NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}

    # Whether we should evaluate the action after the linear solver*/
    compute_action::Cint

    action::NTuple{2, Cdouble}

    # Type of system to solve
    solution_type::QJuliaEnums.QJuliaSolutionType_qj  
    # How to solve it
    solve_type::QJuliaEnums.QJuliaSolveType_qj        
    # The preconditioned matrix type
    matpc_type::QJuliaEnums.QJuliaMatPCType_qj        
    # Whether we are using the Hermitian conjugate system or not
    dagger::QJuliaEnums.QJuliaDagType_qj              
    # The mass normalization is being used by the caller
    mass_normalization::QJuliaEnums.QJuliaMassNormalization_qj 
    # The normalization desired in the solver
    solver_normalization::QJuliaEnums.QJuliaSolverNormalization_qj 
    # Preserve the source or not in the linear solver (deprecated)
    preserve_source::QJuliaEnums.QJuliaPreserveSource_qj       

    # The precision used by the input fermion fields
    cpu_prec::QJuliaEnums.QJuliaPrecision_qj                
    # The precision used by the QJULIA solver
    cuda_prec::QJuliaEnums.QJuliaPrecision_qj               
    # The precision used by the QJULIA sloppy operator
    cuda_prec_sloppy::QJuliaEnums.QJuliaPrecision_qj        
    # The precision of the sloppy gauge field for the refinement step in multishift
    cuda_prec_refinement_sloppy::QJuliaEnums.QJuliaPrecision_qj 
    # The precision used by the QJULIA preconditioner
    cuda_prec_precondition::QJuliaEnums.QJuliaPrecision_qj  

    # The order of the input and output fermion fields
    dirac_order::QJuliaEnums.QJuliaDiracFieldOrder_qj       
    # Gamma basis of the input and output host fields
    gamma_basis::QJuliaEnums.QJuliaGammaBasis_qj            

    # The location of the clover field
    clover_location::QJuliaEnums.QJuliaFieldLocation_qj     
    # The precision used for the input clover field       
    clover_cpu_prec::QJuliaEnums.QJuliaPrecision_qj         
    # The precision used for the clover field in the QJULIA solver
    clover_cuda_prec::QJuliaEnums.QJuliaPrecision_qj        
    # The precision used for the clover field in the QJULIA solver
    clover_cuda_prec_sloppy::QJuliaEnums.QJuliaPrecision_qj 
    # The precision of the sloppy clover field for the refinement step in multishift
    clover_cuda_prec_refinement_sloppy::QJuliaEnums.QJuliaPrecision_qj 
    # The precision used for the clover field in the QJULIA preconditioner
    clover_cuda_prec_precondition::QJuliaEnums.QJuliaPrecision_qj 

    # The order of the input clover field
    clover_order::QJuliaEnums.QJuliaCloverFieldOrder_qj 
    # Whether to use an initial guess in the solver or not
    use_init_guess::QJuliaEnums.QJuliaUseInitGuess_qj       

    # Coefficient of the clover term
    clover_coeff::Cdouble                   
    # Real number added to the clover diagonal (not to inverse)
    clover_rho::Cdouble                     

    # Whether to compute the trace log of the clover term
    compute_clover_trlog::Cint              
    # The trace log of the clover term (even/odd computed separately)
    trlogA::NTuple{2, Cdouble}                      

    # Whether to compute the clover field
    compute_clover::Cint                   
    # Whether to compute the clover inverse field 
    compute_clover_inverse::Cint           
    # Whether to copy back the clover matrix field 
    return_clover::Cint                     
    # Whether to copy back the inverted clover matrix field
    return_clover_inverse::Cint             
    # The verbosity setting to use in the solver 
    verbosity::QJuliaEnums.QJuliaVerbosity_qj               
    # The padding to use for the fermion fields
    sp_pad::Cint                           
    # The padding to use for the clover fields 
    cl_pad::Cint                            
    # The number of iterations performed by the solver
    iter::Cint                              
    # The Gflops rate of the solver
    gflops::Cdouble                         
    # The time taken by the solver
    secs::Cdouble                           
    # Enable auto-tuning? (default = QJULIA_TUNE_YES)
    tune::QJuliaEnums.QJuliaTune_qj                          

    # Number of steps in s-step algorithms
    Nsteps::Cint

    # Maximum size of Krylov space used by solver 
    gcrNkrylov::Cint

    inv_type_precondition::QJuliaEnums.QJuliaInverterType_qj 

    # Preconditioner instance, e.g., multigrid 
    preconditioner::Ptr{Cvoid}

    # Deflation instance
    deflation_op::Ptr{Cvoid}

    # Dirac Dslash used in preconditioner
    dslash_type_precondition::QJuliaEnums.QJuliaDslashType_qj 
    # Verbosity of the inner Krylov solver
    verbosity_precondition::QJuliaEnums.QJuliaVerbosity_qj 
    # Tolerance in the inner solver
    tol_precondition::Cdouble

    # Maximum number of iterations allowed in the inner solver
    maxiter_precondition::Cint

    # Relaxation parameter used in GCR-DD (default = 1.0)
    omega::Cdouble

    # Number of preconditioner cycles to perform per iteration 
    precondition_cycle::Cint

    # Whether to use additive or multiplicative Schwarz preconditioning 
    schwarz_type::QJuliaEnums.QJuliaSchwarzType_qj 

    residual_type::QJuliaEnums.QJuliaResidualType_qj 

    #Parameters for deflated solvers
    # The precision of the Ritz vectors
    cuda_prec_ritz::QJuliaEnums.QJuliaPrecision_qj 
    # How many vectors to compute after one solve
    #  for eigCG recommended values 8 or 16
    
    nev::Cint
    # EeigCG  : Search space dimension
    # GMResDR : Krylov subspace dimension
    max_search_dim::Cint
    #For systems with many RHS: current RHS index 
    rhs_idx::Cint
    # Specifies deflation space volume: total number of eigenvectors is nev*deflation_grid 
    deflation_grid::Cint
    # eigCG: selection criterion for the reduced eigenvector set 
    eigenval_tol::Cdouble
    # mixed precision eigCG tuning parameter:  minimum search vector space restarts 
    eigcg_max_restarts::Cint
    #initCG tuning parameter:  maximum restarts
    max_restart_num::Cint
    # initCG tuning parameter:  tolerance for cg refinement corrections in the deflation stage 
    inc_tol::Cdouble

    # Whether to make the solution vector(s) after the solve 
    make_resident_solution::Cint

    # Whether to use the resident solution vector(s) 
    use_resident_solution::Cint

    # Whether to use the solution vector to augment the chronological basis 
    chrono_make_resident::Cint

    # Whether the solution should replace the last entry in the chronology 
    chrono_replace_last::Cint

    # Whether to use the resident chronological basis 
    chrono_use_resident::Cint

    # The maximum length of the chronological history to store 
    chrono_max_dim::Cint

    # The index to indicate which chrono history we are augmenting 
    chrono_index::Cint

    # Precision to store the chronological basis in 
    chrono_precision::QJuliaEnums.QJuliaPrecision_qj 

    # Which external library to use in the linear solvers (MAGMA or Eigen) 
    extlib_type::QJuliaEnums.QJuliaExtLibType_qj 

    #defualt constructor
    QJuliaInvertParam_qj() = new(QJuliaEnums.QJULIA_CPU_FIELD_LOCATION,
				 QJuliaEnums.QJULIA_CPU_FIELD_LOCATION,
				 QJuliaEnums.QJULIA_WILSON_DSLASH,
				 QJuliaEnums.QJULIA_CG_INVERTER,
				 0.0,
				 0.0,
				 0.0,
				 1,
				 NTuple{QJULIA_MAX_DWF_LS, Cdouble}(ntuple(i->0, QJULIA_MAX_DWF_LS)),
				 NTuple{QJULIA_MAX_DWF_LS, Cdouble}(ntuple(i->0, QJULIA_MAX_DWF_LS)),
				 0.0,
				 0.0,
				 QJuliaEnums.QJULIA_TWIST_INVALID,
				 1e-10,
				 5e-3,
				 1e-2,
				 1,
				 0.0,
				 0.0,
				 256, 
				 1e-2, 
				 1e-2,
				 0,
				 1,
				 10,
				 10,
				 64,
				 1,
				 8,
				 32,
				 0,
				 NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}(ntuple(i->0, QJULIA_MAX_MULTI_SHIFT)),
				 NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}(ntuple(i->0, QJULIA_MAX_MULTI_SHIFT)),
				 NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}(ntuple(i->0, QJULIA_MAX_MULTI_SHIFT)),
				 NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}(ntuple(i->0, QJULIA_MAX_MULTI_SHIFT)),
				 NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}(ntuple(i->0, QJULIA_MAX_MULTI_SHIFT)),
				 NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}(ntuple(i->0, QJULIA_MAX_MULTI_SHIFT)),
				 NTuple{QJULIA_MAX_MULTI_SHIFT, Cdouble}(ntuple(i->0, QJULIA_MAX_MULTI_SHIFT)),
				 0,
				 NTuple{2, Cdouble}(ntuple(i->0, 2)),
				 QJuliaEnums.QJULIA_MAT_SOLUTION,
				 QJuliaEnums.QJULIA_NORMOP_PC_SOLVE,
				 QJuliaEnums.QJULIA_MATPC_EVEN_EVEN,
				 QJuliaEnums.QJULIA_DAG_NO,
				 QJuliaEnums.QJULIA_KAPPA_NORMALIZATION,
				 QJuliaEnums.QJULIA_DEFAULT_NORMALIZATION,
				 QJuliaEnums.QJULIA_PRESERVE_SOURCE_YES,
				 QJuliaEnums.QJULIA_DOUBLE_PRECISION,
				 QJuliaEnums.QJULIA_SINGLE_PRECISION,
				 QJuliaEnums.QJULIA_SINGLE_PRECISION,
				 QJuliaEnums.QJULIA_SINGLE_PRECISION,
				 QJuliaEnums.QJULIA_SINGLE_PRECISION,
				 QJuliaEnums.QJULIA_DIRAC_ORDER,
				 QJuliaEnums.QJULIA_DEGRAND_ROSSI_GAMMA_BASIS,
				 QJuliaEnums.QJULIA_CPU_FIELD_LOCATION,
				 QJuliaEnums.QJULIA_INVALID_PRECISION,
				 QJuliaEnums.QJULIA_INVALID_PRECISION,
				 QJuliaEnums.QJULIA_INVALID_PRECISION,
				 QJuliaEnums.QJULIA_INVALID_PRECISION,
				 QJuliaEnums.QJULIA_INVALID_PRECISION,
				 QJuliaEnums.QJULIA_INVALID_CLOVER_ORDER,
				 QJuliaEnums.QJULIA_USE_INIT_GUESS_NO,
				 0.0,
				 0.0,
				 0,
				 NTuple{2, Cdouble}(ntuple(i->0, 2)),
				 0,
				 0,
				 0,
				 0,
				 QJuliaEnums.QJULIA_VERBOSE,
				 0,
				 0,
				 256,
				 0.0,
				 0.0,
				 QJuliaEnums.QJULIA_TUNE_YES,
				 8,
				 16,
				 QJuliaEnums.QJULIA_INVALID_INVERTER,
				 C_NULL, C_NULL,
				 QJuliaEnums.QJULIA_INVALID_DSLASH,
				 QJuliaEnums.QJULIA_SILENT,
				 0.1,
				 8,
				 1.0,
				 1,
				 QJuliaEnums.QJULIA_ADDITIVE_SCHWARZ,
				 QJuliaEnums.QJULIA_L2_RELATIVE_RESIDUAL,
				 QJuliaEnums.QJULIA_SINGLE_PRECISION,
				 8,
				 64,
				 0,
				 8,
				 1e-3,
				 4, 
				 3,
				 1e-2,
				 0,
				 0,
				 0,
				 0,
				 0,
				 0,
				 0, 
				 QJuliaEnums.QJULIA_INVALID_PRECISION,
				 QJuliaEnums.QJULIA_EIGEN_EXTLIB)
				 

  end #QJuliaInvertParam

end #QJuliaInterface 
