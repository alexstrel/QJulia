#include <complex>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <eigenpc.h>


using namespace Eigen;

template <typename T, int CompressionType>
class ICFArgs{

    typedef typename NaturalOrdering<int>::PermutationType PermutationType;
    typedef typename PermutationType::StorageIndex StorageIndex;
    //
    typedef SparseMatrix<T,ColMajor,StorageIndex>          FactorType;
    typedef SparseMatrix<T,CompressionType,StorageIndex>   CompressedSpMat;
    typedef Matrix<T, Dynamic, 1>                          VectorType;

    IncompleteCholesky<T, Lower, NaturalOrdering<int> > ichol;
    FactorType Lfactor;

  public:
    ICFArgs(void *vals, int *innerIndices, int *outerIndexPtr, const int n, const int nnz) : ichol( Map<CompressedSpMat >(n,n,nnz,outerIndexPtr,innerIndices,static_cast<T*>(vals))){
       std::cout << "Creating ICC preconditioner.." << std::endl;
       Lfactor = ichol.matrixL();// The lower part stored in CSC
       return;
    }

    inline void	apply(T *out, T *in){
       Map<VectorType, Unaligned> out_(out, Lfactor.rows());
       Map<const VectorType, Unaligned> in_(in,Lfactor.cols());

       if(out == in) return; //nothing to do

       out_ = Lfactor.template triangularView<Lower>().solve(in_);
       Lfactor.adjoint().template triangularView<Upper>().solveInPlace(out_);

       return;
    }

    ~ICFArgs() {}
};

template <typename T, int CompressType>
void createPreconditioner(void *vals, int *innerIndices, int *outerIndexPtr, PrecondDescr &descr) {
  if(descr.pctype == PRECOND_ICC) {
    descr.pc_args = static_cast<void*>( new ICFArgs<T, CompressType >(vals, innerIndices, outerIndexPtr, descr.rows, descr.nnz) );
  } else {
    printf("Precond type is not supported.\n"), exit(-1);
  }

  return;
}

template <typename T>
void createPreconditioner(void *vals, int *innerIndices, int *outerIndexPtr, PrecondDescr &descr) {
  if(descr.ctype == CRS_FORMAT){
    createPreconditioner<T, RowMajor>(vals, innerIndices, outerIndexPtr, descr);
  } else if(descr.ctype == CCS_FORMAT) {
    createPreconditioner<T, ColMajor>(vals, innerIndices, outerIndexPtr, descr);
  } else {
    printf("Compression type is not supported."), exit(-1);
  }
  return;
}

//Create preconditioner for a given problem matrix.
//we accept maptrix in CRS or CCS formats
void CreatePreconditioner(void *vals, int *innerIndices, int *outerIndexPtr, PrecondDescr *descr) {
  if (descr->rows != descr->cols) printf("Error: incorrect matrix dimensions (rows %d, cols %d)\n", descr->rows, descr->cols), exit(-1);

  std::cout << "Number of nnz " << descr->nnz << " problem size is : " << descr->rows << std::endl;
  std::cout << "Precision : " << descr->precision << " Complex : " << descr->is_complex << std::endl; 

  if(descr->precision == FLOAT_SINGLE){	  
    if(descr->is_complex) {
      createPreconditioner<std::complex<float>>(vals, innerIndices, outerIndexPtr, *descr);
    } else {
      createPreconditioner<float>(vals, innerIndices, outerIndexPtr, *descr);
    }
  } else {
    if(descr->is_complex) {
      createPreconditioner<std::complex<double>>(vals, innerIndices, outerIndexPtr, *descr);
    } else {
      createPreconditioner<double>(vals, innerIndices, outerIndexPtr, *descr);
    }
  }

  descr->is_init = 1;

  return;
}

template <typename T, int CompressType>
void DestroyPreconditioner(PrecondDescr &descr){
  if(descr.pctype == PRECOND_ICC) {
    typedef ICFArgs<T, CompressType > Preconditioner;
    delete static_cast<Preconditioner*> (descr.pc_args);
  } else {
    printf("Precond type is not supported.\n"), exit(-1);
  }

  return;
}

template <typename T>
void DestroyPreconditioner(PrecondDescr &descr){
  if(descr.ctype == CRS_FORMAT){
    DestroyPreconditioner<T, RowMajor>(descr);
  } else if(descr.ctype == CCS_FORMAT) {
    DestroyPreconditioner<T, ColMajor>(descr);
  } else {
    printf("Cannot destroy preconditioner with unsupported compression type."), exit(-1);
  }
  return;
}

//Create preconditioner:
void DestroyPreconditioner(PrecondDescr *descr){
  if (descr->is_init != 1) printf("Preconditioner was not initialized\n"), exit(-1);

  if(descr->precision == FLOAT_SINGLE){
    if(descr->is_complex) {
      DestroyPreconditioner<std::complex<float>>(*descr);
    } else {
      DestroyPreconditioner<float>(*descr);
    }
  } else {
    if(descr->is_complex) {
      DestroyPreconditioner<std::complex<double>>(*descr);
    } else {
      DestroyPreconditioner<double>(*descr);
    }
  }
  descr->pc_args = nullptr;
  descr->is_init = 0;

  return;
}

	template <typename T, int CompressType>
	void applyPreconditioner(void *out, void *in, PrecondDescr &descr)
  {
		 if(descr.pctype == PRECOND_ICC) {
			 typedef ICFArgs<T, CompressType > Preconditioner;
			 (static_cast<Preconditioner*> (descr.pc_args))->apply(static_cast<T*>(out), static_cast<T*>(in));
		 } else {
			 printf("Precond type is not supported.\n"), exit(-1);
		 }

		 return;
  }

	template <typename T>
	void applyPreconditioner(void *out, void *in, PrecondDescr &descr)
  {
		 if(descr.ctype == CRS_FORMAT){
		   applyPreconditioner<T, RowMajor>(out, in, descr);
		 } else if(descr.ctype == CCS_FORMAT) {
		   applyPreconditioner<T, ColMajor>(out, in, descr);
		 } else {
		   printf("Cannot destroy preconditioner with unsupported compression type."), exit(-1);
		 }
		 return;
  }

  //Create preconditioner:
  void ApplyPreconditioner(void *out, void *in, PrecondDescr *descr){
		 if(descr->precision == FLOAT_SINGLE){
			 if(descr->is_complex) {
				 applyPreconditioner<std::complex<float>>(out, in, *descr);
			 } else {
				 applyPreconditioner<float>(out, in, *descr);
			 }
		 } else {
			 if(descr->is_complex) {
				 applyPreconditioner<std::complex<double>>(out, in, *descr);
			 } else {
				 applyPreconditioner<double>(out, in, *descr);
			 }
		 }
     return;
  }
