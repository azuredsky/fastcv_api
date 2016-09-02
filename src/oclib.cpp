#if defined(FASTCV_USE_OCL)
namespace HPC { namespace fastcv {
 /*   class OCLContext {
        public:
            OCLContext(PlatformType_t ptype,
                    cl_device_type dtype,
                    int device) {
                this->ptype = ptype;
                this->dtype = dtype;
                this->device = device;
            }

            PlatformType_t ptype;
            cl_device_type dtype;
            int device;

            cl_context context;
            cl_command_queue queue;
    };

    static OCLContext* cont = NULL;

    HPCStatus_t initOCLib(PlatformType_t ptype,
            cl_device_type dtype, int device) {
        if(NULL == cont) {
            cont = new OCLContext(ptype, dtype, device);
            return HPC_SUCCESS;
        } else {
            return HPC_ALREADY_INITIALIZED;
        }
    }

    HPCStatus_t getOCLContext(cl_context *context) {
        if(NULL == cont) return HPC_NOT_INITIALIZED;
        if(NULL != context) *context = cont->context;
        return HPC_SUCCESS;
    }

    HPCStatus_t getOCLQueue(cl_command_queue *queue) {
        if(NULL == cont) return HPC_NOT_INITIALIZED;
        if(NULL != queue) *queue = cont->queue;
        return HPC_SUCCESS;
    }

    HPCStatus_t destroyOCLIb() {
        if(NULL == cont) {
            return HPC_NOT_INITIALIZED;
        } else {
            delete cont;
            cont = NULL;
            return HPC_SUCCESS;
        }
    }
*/
}};
#endif
