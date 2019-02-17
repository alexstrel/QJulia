#pragma once

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

  typedef enum Eigen4JuliaWrapperPrecondType {
    PRECOND_INVALID,
    PRECOND_ILU,
    PRECOND_ICC
  } PrecondType;

  typedef enum Eigen4JuliaWrapperPrecisionType {
    FLOAT_SINGLE = 4,
    FLOAT_DOUBLE = 8
  } Precision;

  typedef enum Eigen4JuliaWrapperCompressionType {
    CRS_FORMAT = 1,
    CCS_FORMAT = 2
  } CompressionType;

  typedef struct PrecondDescr_s{
    void *pc_args;
    PrecondType pctype;

    int  rows;
    int  cols;

    Precision   precision;
    int    is_complex;//0 for real and 1 for complex

    CompressionType ctype;
    int nnz;
    
    int is_init;

  } PrecondDescr;

  void CreatePreconditioner(void *vals, int *innerIndices, int *outerIndexPtr, PrecondDescr *descr);

  void DestroyPreconditioner(PrecondDescr *descr);

  void ApplyPreconditioner(void *out, void *in, PrecondDescr *descr);


  #ifdef __cplusplus
  }
  #endif
