module QJuliaComms

##### MPI & other communication calls ######
   MPI_init_qj(argc, argv) = ccall((:MPI_Init, "libmpi"), Cvoid, (Int32, Ptr{Ptr{Cchar}}), argc, argv)

   MPI_finalize_qj()       = ccall((:MPI_Finalize, "libmpi"), Cvoid, (), )

end #QJuliaComms 
