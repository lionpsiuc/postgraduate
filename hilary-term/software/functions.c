int MPI_Accumulate(const void* origin_addr, int origin_count,
                   MPI_Datatype origin_datatype, int target_rank,
                   MPI_Aint target_disp, int target_count,
                   MPI_Datatype target_datatype, MPI_Op op, MPI_Win win);

int MPI_Allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                  void* recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm);

int MPI_Allreduce(void* send_buffer, void* recv_buffer, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int MPI_Alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                 void* recvbuf, int recvcnt, MPI_Datatype recvtype,
                 MPI_Comm comm);

int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root,
              MPI_Comm comm);

int MPI_Bsend(const void* buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm);

int MPI_Cancel(MPI_Request* request);

int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);

int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
                    const int periods[], int reorder, MPI_Comm* comm_cart);

int MPI_Cart_shift(MPI_Comm comm, int direction, int displ, int* src,
                   int* dest);

int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm* newcomm);

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm* newcomm);

int MPI_Comm_group(MPI_Comm comm, MPI_Group* group);

int MPI_Comm_rank(MPI_Comm comm, int* rank);

int MPI_Comm_size(MPI_Comm comm, int* size);

int MPI_Comm_spawn(const char* command, char* argv[], int maxprocs,
                   MPI_Info info, int root, MPI_Comm comm, MPI_Comm* intercomm,
                   int array_of_errcodes[]);

int MPI_Comm_split(MPI_Comm comm, int colour, int key, MPI_Comm* newcomm);

int MPI_Dims_create(int nnodes, int ndims, int dims[]);

int MPI_Fetch_and_op(const void* origin_addr, void* result_addr,
                     MPI_Datatype datatype, int target_rank,
                     MPI_Aint target_disp, MPI_Op op, MPI_Win win);

int MPI_File_close(MPI_File* fh);

int MPI_File_open(MPI_Comm comm, const char* filename, int amode, MPI_Info info,
                  MPI_File* fh);

int MPI_File_read(MPI_File fh, void* buf, int count, MPI_Datatype datatype,
                  MPI_Status* status);

int MPI_File_read_all(MPI_File fh, void* buf, int count, MPI_Datatype datatype,
                      MPI_Status* status);

int MPI_File_read_at(MPI_File fh, MPI_Offset offset, void* buf, int count,
                     MPI_Datatype datatype, MPI_Status* status);

int MPI_File_seek(MPI_File fh, MPI_Offset offset, int whence);

int MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype,
                      MPI_Datatype filetype, const char* datarep,
                      MPI_Info info);

int MPI_File_write(MPI_File fh, const void* buf, int count,
                   MPI_Datatype datatype, MPI_Status* status);

int MPI_File_write_all(MPI_File fh, const void* buf, int count,
                       MPI_Datatype datatype, MPI_Status* status);

int MPI_File_write_at(MPI_File fh, MPI_Offset offset, const void* buf,
                      int count, MPI_Datatype datatype, MPI_Status* status);

int MPI_Finalize();

int MPI_Gather(void* send_buffer, int send_count, MPI_Datatype sendtype,
               void* recv_buffer, int recv_count, MPI_Datatype recvtype,
               int root, MPI_Comm comm);

int MPI_Get(void* origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count,
            MPI_Datatype target_datatype, MPI_Win win);

int MPI_Group_difference(MPI_Group group1, MPI_Group group2,
                         MPI_Group* newgroup);

int MPI_Group_excl(MPI_Group group, int n, const int ranks[],
                   MPI_Group* newgroup);

int MPI_Group_incl(MPI_Group group, int n, const int ranks[],
                   MPI_Group* newgroup);

int MPI_Group_intersection(MPI_Group group1, MPI_Group group2,
                           MPI_Group* newgroup);

int MPI_Group_rank(MPI_Group group, int* rank);

int MPI_Group_size(MPI_Group group, int* size);

int MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group* newgroup);

int MPI_Info_create(MPI_Info* info);

int MPI_Info_free(MPI_Info* info);

int MPI_Info_set(MPI_Info info, char* key, char* value);

int MPI_Init(int* argc, char*** argv);

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided);

int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request* request);

int MPI_Isend(void* buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request* request);

int MPI_Put(const void* origin_addr, int origin_count,
            MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
            int target_count, MPI_Datatype target_datatype, MPI_Win win);

int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status* status);

int MPI_Reduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm);

int MPI_Rsend(const void* buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm);

int MPI_Scatter(void* sendbuf, int sendcnt, MPI_Datatype sendtype,
                void* recvbuf, int recvcnt, MPI_Datatype recvtype, int root,
                MPI_Comm comm);

int MPI_Send(void* buf, int count, MPI_Datatype datatype, int dest, int tag,
             MPI_Comm comm);

int MPI_Sendrecv(void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
                 int sendtag, void* recvbuf, int recvcount,
                 MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                 MPI_Status* status);

int MPI_Ssend(const void* buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm);

int MPI_Test(MPI_Request* request, int* flag, MPI_Status* status);

int MPI_Test_cancelled(const MPI_Status* status, int* flag);

int MPI_Type_commit(MPI_Datatype* datatype);

int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype* newtype);

int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent,
                            MPI_Datatype* newtype);

int MPI_Type_create_struct(int count, int array_of_blocklengths[],
                           const MPI_Aint     array_of_displacements[],
                           const MPI_Datatype array_of_types[],
                           MPI_Datatype*      newtype);

int MPI_Type_free(MPI_Datatype* datatype);

int MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint* lb, MPI_Aint* extent);

int MPI_Type_size(MPI_Datatype datatype, int* size);

int MPI_Type_struct(int count, int* array_of_blocklengths,
                    MPI_Aint*     array_of_displacements,
                    MPI_Datatype* array_of_types, MPI_Datatype* newtype);

int MPI_Type_vector(int count, int blocklength, int stride,
                    MPI_Datatype oldtype, MPI_Datatype* newtype);

int MPI_Wait(MPI_Request* request, MPI_Status* status);

int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status array_of_statuses[]);

int MPI_Waitany(int count, MPI_Request array_of_requests[], int* indx,
                MPI_Status* status);

int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int* outcount,
                 int array_of_indices[], MPI_Status array_of_statuses[]);

int MPI_Win_complete(MPI_Win win);

int MPI_Win_create(void* base, MPI_Aint size, int disp_unit, MPI_Info info,
                   MPI_Comm comm, MPI_Win* win);

int MPI_Win_fence(int assert, MPI_Win win);

int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win);

int MPI_Win_post(MPI_Group group, int assert, MPI_Win win);

int MPI_Win_start(MPI_Group group, int assert, MPI_Win win);

int MPI_Win_unlock(int rank, MPI_Win win);

int MPI_Win_wait(MPI_Win win);
