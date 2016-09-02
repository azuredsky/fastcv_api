#ifndef HPC_FASTCV_API_H_
#define HPC_FASTCV_API_H_

#include <stdlib.h>
#include <status.h>
#include <types.h>
#include <sys.h>
#include <string.h>


#if defined(FASTCV_USE_CUDA)
#include <cuda_runtime_api.h>
#include <half.h>
#endif

#if defined(FASTCV_USE_OCL)
#include <bcl.h>
#include <half.h>
#endif

#if __cplusplus >= 201103L
#define CHECK_CH_MACRO "the number of channels been computed is larger than input image or output image\n"
#endif
//using namespace uni::half;
namespace HPC { namespace fastcv {

#if defined(FASTCV_USE_OCL)
    class Opencl{
    public:
        Opencl(PlatformType_t typePlatform,cl_device_type typeDevice,int deviceId){
            platforms = NULL;
            devices = NULL;
            checkHPCError(bclGetPlatforms(&num_platforms,&platforms));
            checkHPCError(bclSelectPlatform(num_platforms, platforms,typePlatform, &pid));
            checkHPCError(bclPlatformGetDevices(platforms[pid],typeDevice,&num_devices,&devices));
            checkHPCError(bclCreateContext(platforms[pid],num_devices,devices,&context));
            checkHPCError(bclCreateCommandQueue(context,devices[deviceId],0,&command_queue));
        }

        cl_uint getNumPlatforms(){
            return num_platforms;
        }
        cl_platform_id* getPlatforms(){
            return platforms;
        }
        cl_uint getNumDevices(){
            return num_devices;
        }
        cl_device_id* getDeviceId(){
            return devices;
        }
        cl_context getContext(){
            return context;
        }
        cl_command_queue getCommandQueue(){
            return command_queue;
        }
        ~Opencl(){
            if(NULL != command_queue)checkHPCError(bclReleaseCommandQueue(command_queue));
            if(NULL != context)checkHPCError(bclReleaseContext(context));
            if(NULL != devices)free(devices);
            if(NULL != platforms)free(platforms);
        }
    private :
        cl_int ret;
        cl_uint pid;
        cl_uint num_platforms;
        cl_platform_id* platforms;
        cl_uint num_devices;
        cl_device_id* devices;
        cl_context context ;
        cl_command_queue command_queue ;
    };
    extern Opencl opencl;

#endif

    template <typename T> class Point2_;
    template <typename T> class Rect_;

    template<typename T> class Size_ {
    public:
        typedef T value_type;

        Size_() {
            this->width = 0;
            this->height = 0;
        }

        Size_(T _width, T _height) {
            this->width = _width;
            this->height = _height;
        }

        Size_(const Size_& sz) {
            if(&sz != this) {
                this->width = (T)sz.width;
                this->height = (T)sz.height;
            }
        }

        T w() const { return width;}
        T h() const { return height;}

        explicit Size_(const Point2_<T>& pt);
        Size_& operator = (const Size_& sz) {
            if(&sz != this) {
                this->width = (T)sz.width;
                this->height = (T)sz.height;
            }
        }
        //! the area (width*height)
        T area() const { return width*height;}

        //! conversion of another data type.
        template<typename T2> operator Size_<T2>() const {
            Size_<T2> ret;
            ret.width = (T2)width;
            ret.height = (T2)height;

            return ret;
        }

    protected:
        T width, height;
    };
    typedef Size_<int> Size2i;
    typedef Size2i Size;
    typedef Size_<float> Size2f;


    template<typename T> class  Point2_ {
    public:
        typedef T value_type;

        Point2_():x(0), y(0) {}

        // various constructors
        Point2_(T _x, T _y):x(_x), y(_y) {}

        Point2_(const Point2_& pt):x((T)pt.x), y((T)pt.y){}
        Point2_& operator = (const Point2_& pt){
            if(&pt != this) {
                x = (T) pt.x;
                y = (T) pt.y;
            }
            return *this;
        }

        //! conversion to another data type
        template<typename T2> operator Point2_<T2>() const {
            Point2_<T2> ret;
            ret.x = (T2)x;
            ret.y = (T2)y;
        }

        //!  dot product
        T dot(const Point2_& pt) const {
            T ret = pt.x*x + pt.y*y;
            return ret;
        }
        //!  cross-product
        T cross(const Point2_& pt) const {
            T ret = pt.x*y + pt.y*x;
            return ret;
        }
        //!  checks whether the point is inside the specified rectangle
        bool inside(const Rect_<T>& r) const;

        T operator[](size_t index) const {
            return index == 0 ? x : y;
        }

        T x, y;
    };
    typedef Point2_<int> Point2i;
    typedef Point2i Point;
    typedef Point2_<float> Point2f;
    typedef Point2_<double> Point2d;

    template<typename T> class  Point3_ {
    public:
        typedef T value_type;

        Point3_():x(0), y(0), z(0) {}

        // various constructors
        Point3_(T _x, T _y, T _z):x(_x), y(_y), z(_z) {}

        Point3_(const Point3_& pt):x((T)pt.x), y((T)pt.y), z((T)pt.z){}
        explicit Point3_(const Point2_<T>& pt):x(pt.x),y(pt.y),z(0){}
        Point3_& operator = (const Point3_& pt) {
            if(this != &pt) {
                x = (T)pt.x;
                y = (T)pt.y;
                z = (T)pt.z;
            }
            return *this;
        }

        //!  conversion to another data type
        template<typename T2> operator Point3_<T2>() const {
            Point3_<T2> ret;
            ret.x = (T2)x;
            ret.y = (T2)y;
            ret.z = (T2)z;
            return ret;
        }

        //!  dot product
        T dot(const Point3_& pt) const {
            return pt.x*x+pt.y*y+pt.z*z;
        }
        //!  cross product of the 2 3D points
        Point3_ cross(const Point3_& pt) const;

        T operator[](size_t index) const {
            switch(index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: break;
            }
        }

        T x, y, z;
    };
    typedef Point3_<int> Point3i;
    typedef Point3_<float> Point3f;
    typedef Point3_<double> Point3d;

    template<typename T> class Rect_ {
    public:
        typedef T value_type;

        Rect_();

        //! various constructors
        Rect_(T _x, T _y, T _width, T _height):x(_x),y(_y),width(_width), height(_height){}

        Rect_(const Rect_& r):x(r.x),y(r.y),width(r.width),height(r.height){}
        Rect_(const Point2_<T>& org, const Size_<T>& sz):x(org.x),y(org.y),width(sz.width),height(sz.height){}
        Rect_(const Point2_<T>& pt1, const Point2_<T>& pt2) {
            x = pt1.x;
            y = pt1.y;
            width = pt2.x-pt1.x;
            height = pt2.y-pt1.y;
        }
        //
        Rect_& operator = (const Rect_& r) {
            if(*this != r) {
                this->x = r.x;
                this->y = r.y;
                this->width = r.width;
                this->height = r.height;
            }
            return *this;
        }
        //! the top-left corner
        Point2_<T> tl() const{
            return Point2_<T>(x, y);
        }
        //! the bottom-right corner
        Point2_<T> br() const {
            return Point2_<T>(x+width, y+height);
        }

        //! size (width, height) of the rectangle
        Size_<T> size() const {
            return Size_<T>(width, height);
        }
        //!  area (width*height) of the rectangle
        T area() const {
            return width*height;
        }

        //!  conversion to another data type
        template<typename T2> operator Rect_<T2>() const {
            return Rect_<T2>((T2)x, (T2)y, (T2)width, (T2)height);
        }
        //!  checks whether the rectangle contains the point
        bool contains(const Point2_<T>& pt) const {
            bool flag = (x+width-pt.x)*(pt.x-x) > 0;
            flag = flag && ((y+height-pt.y)*(pt.y-y) > 0);
            return flag;
        }

        T x, y, width, height;
    };
    typedef Rect_<int> Recti;
    typedef Recti Rect;
    typedef Rect_<float> Rectf;

    class  Range {
    public:
        Range(int _start, int _end):start(_start), end(_end){}
        int size() const { return end-start;}
        bool empty() const { return start == end;}

        int start, end;
    };

    /*!
      A short numerical vector.

      This template class represents short numerical vectors (of 1, 2, 3, 4 ... elements)
      on which you can perform basic arithmetical operations, access individual elements using [] operator etc.

      The template takes 2 parameters:
      -# _Tp element type
      -# cn the number of elements

      In addition to the universal notation like Vec<float, 3>, you can use shorter aliases
      for the most popular specialized variants of Vec, e.g. Vec3f ~ Vec<float, 3>.
      */
    template<typename T, int cn> class Vec
    {
    public:
        typedef T value_type;
        //! default constructor
        Vec(){
            for(int i = 0; i < cn; i++) val[i] = T(0);
        }
        //!< 1-element vector constructor
        Vec(T v0){
            val[0] = v0;
            for(int i = 1; i < cn; i++) val[i] = T(0);
        }
        //!< 2-element vector constructor
        Vec(T v0, T v1){
            val[0] = v0; val[1] = v1;
            for(int i = 2; i < cn; i++) val[i] = T(0);
        }
        //!< 3-element vector constructor
        Vec(T v0, T v1, T v2) {
            val[0] = v0; val[1] = v1; val[2] = v2;
            for(int i = 3; i < cn; i++) val[i] = T(0);
        }
        //!< 4-element vector constructor
        Vec(T v0, T v1, T v2, T v3){
            val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
            for(int i = 4; i < cn; i++) val[i] = T(0);
        }

        //avoid implicit conversion
        explicit Vec(const T* values){
            for( int i = 0; i < cn; i++ ) val[i] = values[i];
        }

        Vec(const Vec<T, cn>& v){
            for( int i = 0; i < cn; i++ ) val[i] = v.val[i];
        }

        const int nSize() const{
            return cn;
        }

        static Vec all(T alpha){
            Vec<T, cn> M;
            for( int i = 0; i < cn; i++ ) M.val[i] = alpha;
            return M;
        }

        //! conversion to another data type
        template<typename T2> operator Vec<T2, cn>() const{
            Vec<T2, cn> v;
            for( int i = 0; i < cn; i++ )
                v.val[i] = (T2)(this->val[i]);
            return v;
        }

        /*! element access */
        const T& operator [](int i) const{
            return this->val[i];
        }
        T& operator[](int i){
            return this->val[i];
        }
        const T& operator ()(int i) const{
            return this->val[i];
        }
        T& operator ()(int i){
            return this->val[i];
        }
        T val[cn];
    };

    /* \typedef

       Shorter aliases for the most popular specializations of Vec<T,n>
       */
    typedef Vec<uchar, 2> Vec2b;
    typedef Vec<uchar, 3> Vec3b;
    typedef Vec<uchar, 4> Vec4b;

    typedef Vec<short, 2> Vec2s;
    typedef Vec<short, 3> Vec3s;
    typedef Vec<short, 4> Vec4s;

    typedef Vec<ushort, 2> Vec2w;
    typedef Vec<ushort, 3> Vec3w;
    typedef Vec<ushort, 4> Vec4w;

    typedef Vec<int, 2> Vec2i;
    typedef Vec<int, 3> Vec3i;
    typedef Vec<int, 4> Vec4i;

    typedef Vec<float, 2> Vec2f;
    typedef Vec<float, 3> Vec3f;
    typedef Vec<float, 4> Vec4f;

    typedef Vec<double, 2> Vec2d;
    typedef Vec<double, 3> Vec3d;
    typedef Vec<double, 4> Vec4d;

    //@{
    /*@brief m rows, n cols
     **/
    template<typename T, int m, int n> class MatX2D {
    public:
        typedef T value_type;

        MatX2D(){};
        T* ptr() { return mem;}
        const T* ptr() const { return mem;}
        T& at(int y, int x) {return mem[y*n + x];}

    private:
        T mem[m*n];
    };

    typedef MatX2D<float, 1, 2> MatX2D12f;
    typedef MatX2D<double, 1, 2> Matx12d;

    typedef MatX2D<float, 1, 6> Matx16f;
    typedef MatX2D<double, 1, 6> Matx16d;

    typedef MatX2D<float, 2, 1> Matx21f;
    typedef MatX2D<double, 2, 1> Matx21d;

    typedef MatX2D<float, 6, 1> Matx61f;
    typedef MatX2D<double, 6, 1> Matx61d;

    typedef MatX2D<float, 2, 2> Matx22f;
    typedef MatX2D<double, 2, 2> Matx22d;

    typedef MatX2D<float, 6, 6> Matx66f;
    typedef MatX2D<double, 6, 6> Matx66d;

    //@}
    /*  typedef enum {
        X86 = 1,
        ARM = 2,
        CUDA = 4,
        OCL = 8,
        } EcoEnv_t;*/

#if defined(FASTCV_USE_CUDA)
#define DEFAULT_TYPE EcoEnv_t::ECO_ENV_CUDA
#elif defined(FASTCV_USE_OCL)
#define DEFAULT_TYPE EcoEnv_t::ECO_ENV_OCL
#elif defined(FASTCV_USE_X86)
#define DEFAULT_TYPE EcoEnv_t::ECO_ENV_X86
#elif defined(FASTCV_USE_ARM)
#define DEFAULT_TYPE EcoEnv_t::ECO_ENV_ARM
#endif

    template<typename T, int numChannels, EcoEnv_t type = DEFAULT_TYPE> class Mat {
    public:
        typedef T value_type;
        enum { AUTO_STEP = 0 };
        Mat() {
            ndims = 0;
            nElements = 0;
            memSize = 0;
            step = AUTO_STEP;
            mem = NULL;
        }

        Mat(int _cols) {
            int size[] = {_cols};
            init(1, size);
        }

        Mat(int _rows, int _cols) {
            int size[] = {_cols, _rows};
            init(2, size);
        }

        Mat(int _rows, int _cols, size_t _step){
            int size[] = {_cols, _rows};
            init_step(2, size, _step);
        }

        Mat(int _rows, int _cols, T* data_ptr, size_t _step = AUTO_STEP) {
            int size[] = {_cols, _rows};
            init_data(2, size, data_ptr, _step);
        }

        Mat(int _depth, int _rows, int _cols) {
            int size[] = {_cols, _rows, _depth};
            init(3, size);
        }

        explicit Mat(Size _size) {
            //int s[] = {_size.h(), _size.w()};
            int s[] = {_size.w(), _size.h()};
            init(2, s);
        }
        //! n-dim array constructor
        Mat(int ndims, const int* dims) {
            init(ndims, dims);
        }

        int width() const{
            return this->dims[0];
        }
        int height() const {
            return this->dims[1];
        }
        size_t widthStep() const {
            return this->step;
        }
        /*
           T* getPtr() const {
           return this->mem;
           }
           */

        Mat(const Mat<T, numChannels, type>& m) = delete;

        //! deep copy constructor
        HPCStatus_t deepCopy(const Mat<T, numChannels, type> & m) {
            //must be allocated
            if(0 == ndims || NULL == mem || this->nElements != m.nElements) {
                return HPC_OP_NOT_PERMITED;
            }

            size_t s = m.nElements*sizeof(T)*numChannels;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                if(ndims==2){
                    int rows = this->dims[1];
                    int cols = this->dims[0];
                    if( rows > 0 && cols > 0 ){
                        T* dptr = (T*)this->mem;
                        const T* sptr = (T*)m.mem;

                        size_t len = cols * numChannels * sizeof(T);
                        int sstep = m.step / sizeof(T);
                        int dstep = this->step / sizeof(T);
                        for( ; rows--; sptr += sstep, dptr += dstep ){
                            memcpy( dptr, sptr, len );
                        }
                    }
                }
                else
                    memcpy(this->mem, m.mem, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                if(ndims==2){
                    int rows = this->dims[1];
                    int cols = this->dims[0];
                    if( rows > 0 && cols > 0 ){
                        T* dptr = (T*)this->mem;
                        const T* sptr = (T*)m.mem;

                        size_t len = cols * numChannels * sizeof(T);
                        int sstep = m.step / sizeof(T);
                        int dstep = this->step / sizeof(T);
                        for( ; rows--; sptr += sstep, dptr += dstep ){
                            memcpy( dptr, sptr, len );
                        }
                    }
                }
                else
                    memcpy(this->mem, m.mem, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                if(ndims==2){
                    int rows = this->dims[1];
                    int cols = this->dims[0];
                    if( rows > 0 && cols > 0 ){
                        T* dptr = (T*)this->mem;
                        const T* sptr = (T*)m.mem;

                        size_t len = cols * numChannels * sizeof(T);
                        int sstep = m.step / sizeof(T);
                        int dstep = this->step / sizeof(T);
                        for( ; rows--; sptr += sstep, dptr += dstep ){
                            cudaMemcpy(dptr, sptr, len, cudaMemcpyDefault);
                        }
                    }
                    else
                        cudaMemcpy(this->mem, m.mem, s, cudaMemcpyDefault);
                }
#endif
            }
            else if(type ==EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
#endif
            }
            return HPC_SUCCESS;
        }

        HPCStatus_t setSize(Size2i size) {
            return setSize(size.w(), size.h());
        }

        HPCStatus_t setSize(int width, int height) {
            if(0 != ndims || NULL != mem)
                return HPC_OP_NOT_PERMITED;

            this->ndims = 2;
            this->dims[0] = width;
            this->dims[1] = height;
            this->nElements = width*height;
            this->step = width * numChannels * sizeof(T);
            this->own = true;

            size_t s = width*height*numChannels*sizeof(T);
            if(type ==EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                x86Malloc((void**)&mem, s);
#endif
            }
            else if(type ==EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                armMalloc((void**)&mem, s);
#endif
            }
            else if(type ==EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                cudaMalloc((void**)&mem, s);
#endif
            }
            else if(type ==EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //#error malloc buffer
#endif
            }
            return HPC_SUCCESS;
        }

        HPCStatus_t shareData(int width, int height, void *data, size_t _step = AUTO_STEP) {
            if(0 != ndims || NULL != mem)
                return HPC_OP_NOT_PERMITED;
            size_t minStep = width * numChannels * sizeof(T);
            if(_step < minStep && _step != AUTO_STEP)
                return HPC_OP_NOT_PERMITED;

            this->ndims = 2;
            this->dims[0] = width;
            this->dims[1] = height;
            this->nElements = width*height;
            if(step == AUTO_STEP)
                this->step = width * numChannels * sizeof(T);
            else
                this->step = step;

            this->mem = data;
            this->own = false;

            return HPC_SUCCESS;
        }

        //! copy host data to Mat
        HPCStatus_t fromHost(const T* _data) {
            size_t s = memSize;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                memcpy(mem, _data, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                memcpy(mem, _data, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                cudaMemcpy(mem, _data, s, cudaMemcpyDefault);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                checkHPCError(bclEnqueueWriteBuffer(opencl.getCommandQueue(),(cl_mem)mem,CL_TRUE,0,s,_data,0,NULL,NULL));
#endif
            }
            return HPC_SUCCESS;
        }
        //! copy data to host
        HPCStatus_t toHost(T* _data) {
            size_t s = memSize;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                memcpy(_data, mem, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                memcpy(_data, mem, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                cudaMemcpy(_data, mem, s, cudaMemcpyDefault);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                checkHPCError(bclEnqueueReadBuffer(opencl.getCommandQueue(),(cl_mem)mem,CL_TRUE,0,s,_data,0,NULL,NULL));
#endif
            }
            return HPC_SUCCESS;
        }

        Mat& operator = (const Mat& m) = delete;
        //! set all the elements to s.
        Mat& operator = (const T& s) = delete;

        //! returns reference to the specified element (1D case)
        T operator ()(int idx0) {
            if(type == EcoEnv_t::ECO_ENV_X86 || type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_X86) || defined(FASTCV_USE_ARM)
                T* ret = (T*)mem;
                return ret[idx0];
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                T ret;
                cudaMemcpy(&ret, (T*)mem+idx0, sizeof(T), cudaMemcpyDefault);
                return ret;
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
#endif
            }
        }

        //! returns reference to the specified element (2D case)
        T operator ()(int idx0, int idx1);
        //! returns reference to the specified element (3D case)
        T operator ()(int idx0, int idx1, int idx2);

        //! the same as above, with the pointer dereferencing
        //! dereference to the specified element (1D case or 2D case)
        template<typename _Tp> _Tp& at(int i0=0){
            int cols = dims[0];
            int i = i0/cols, j = i0 - i*cols;
            return ((_Tp*)((T*)mem + i*step/sizeof(T)))[j];
        }

        //! dereference to the specified element (2D case)
        template<typename _Tp> _Tp& at(int i0, int i1){
            return ((_Tp*)((T*)mem + i0*step/sizeof(T)))[i1];
        }

        //! dereference to the specified element (3D case)
        template<typename _Tp> _Tp& at(int i0, int i1, int i2){
            return *((_Tp*)((T*)mem + i0*dims[0]*dims[1]*numChannels +
                    i1*dims[0]*numChannels + i2*numChannels));
        }

        const int& numDimensions() const { return ndims;}
        const ::size_t numElements() const { return nElements;}
        const int* dimensions() const { return dims;}
        HPCStatus_t depth(int* dth) const {
            if(ndims < 3) return HPC_OP_NOT_PERMITED;
            *dth = dims[2];
            return HPC_SUCCESS;
        }
        HPCStatus_t numRows(int *rows) const {
            if(ndims < 2) return HPC_OP_NOT_PERMITED;
            *rows = dims[1];
            return HPC_SUCCESS;
        }
        HPCStatus_t numColumns(int *cols) const {
            if(ndims < 1) return HPC_OP_NOT_PERMITED;
            *cols = dims[0];
            return HPC_SUCCESS;
        }

        void* ptr() const {
            return mem;
        }

        ~Mat() {
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                if(NULL != mem && own) { x86Free(mem); mem = NULL;}
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                if(NULL != mem && own) armFree(mem);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                if(NULL != mem && own) cudaFree(mem);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //#error fill
                if(NULL != mem && own) checkHPCError(bclReleaseMem((cl_mem)mem));
#endif
            }
        }

    private:
        int ndims;
        int dims[4];
        size_t step;
        ::size_t nElements;
        ::size_t memSize;
        void* mem;
        bool own;

        HPCStatus_t init(int ndims, const int* dims) {
            this->ndims = ndims;
            memcpy(this->dims, dims, ndims*sizeof(int));
            this->step = dims[0] * numChannels * sizeof(T);
            size_t ds = 1;
            for(int i = 0; i < ndims; i++) {
                ds *= dims[i];
            }
            this->nElements = ds;
            this->own = true;

            size_t s = ds*sizeof(T)*numChannels;
            this->memSize = s;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                x86Malloc((void**)&mem, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                armMalloc((void**)&mem, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                cudaMalloc((void**)&mem, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //bclCreateBuff;
                checkHPCError(bclCreateBuffer(opencl.getContext(),CL_MEM_READ_WRITE,s,NULL,(cl_mem *)(&mem)));
#endif
            }
            return HPC_SUCCESS;
        }

        HPCStatus_t init_step(int ndims, const int* dims, size_t _step) {
            this->ndims = ndims;
            memcpy(this->dims, dims, ndims*sizeof(int));
            this->step = _step;
            size_t ds = 1;
            for(int i = 0; i < ndims; i++) {
                ds *= dims[i];
            }
            this->nElements = ds;

            ds = this->step;
            for(int i = 1; i < ndims; i++) {
                ds *= dims[i];
            }
            this->memSize = ds;
            this->own = true;

            size_t s = ds;
            this->memSize = s;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                x86Malloc((void**)&mem, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                armMalloc((void**)&mem, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                cudaMalloc((void**)&mem, s);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //bclCreateBuff;
                checkHPCError(bclCreateBuffer(opencl.getContext(),CL_MEM_READ_WRITE,s,NULL,(cl_mem*)(&mem)));
#endif
            }
           return HPC_SUCCESS;
        }

        HPCStatus_t init_data(int ndims, const int* dims, void* data_ptr, size_t _step) {
            this->ndims = ndims;
            memcpy(this->dims, dims, ndims*sizeof(int));
            size_t minStep = this->dims[0] * numChannels * sizeof(T);
            if(_step < minStep && _step != AUTO_STEP)
                return HPC_OP_NOT_PERMITED;
            if(_step == AUTO_STEP)
                this->step = dims[0] * numChannels * sizeof(T);
            else
                this->step = _step;
            size_t ds = 1;
            for(int i = 0; i < ndims; i++) {
                ds *= dims[i];
            }
            this->nElements = ds;
            ds = this->step;
            for(int i = 1; i < ndims; i++) {
                ds *= dims[i];
            }
            this->memSize = ds;
            this->own = false;
            //size_t s = ds*sizeof(T)*numChannels;
            mem = data_ptr;
        }
    };

    ///  we only supports 1, 3, 4 channels image

    /**
     * @brief load image name to m
     *
     * we only support uchar data type currently. and we want to support
     * float in the near future
     *
     * @notes : T supports uchar and float, float means its data is
     * normalized
     *
     * @warning, this functions is not threadSafe currently
     **/
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t imread(const char* name, Mat<T, nc, type> *m);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t imdecode(unsigned char* databuffer, const unsigned int dataSize,  Mat<T, nc, type> *m);
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t imencode(const char* ext,unsigned char** buffer, const Mat<T, nc, type> *m);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t imwrite(const char* name, const Mat<T, nc, type> *m);

    typedef enum {
        INTERPOLATION_TYPE_NEAREST_POINT = 0,
        INTERPOLATION_TYPE_LINEAR = 1,
        INTERPOLATION_TYPE_INTER_AREA = 3
    } InterpolationType;

    //@{
    /**
     * @brief should support 3 or 4 channels of image
     *
     * @param T             datatype of image
     * @param ncSrc         the number of channels of src
     * @param ncDst         the number of channels of dst
     *
     * @notes: if ncSrc or ncDst is larger than nc, let the last channnel
     * unchanged.
     * @notes: Tsrc, Tdst supports uchar and float
     **/
    template<InterpolationType ip, typename Tsrc, int ncSrc, typename Tdst, int
        ncDst, int nc, EcoEnv_t type>
        HPCStatus_t resize(const Mat<Tsrc, ncSrc, type>& src
            , Mat<Tdst, ncDst, type> *dst);

    template<InterpolationType ip, typename T, int nc, EcoEnv_t type>
        HPCStatus_t resize(const Mat<T, nc, type>& src, Mat<T, nc, type> *dst) {
            return resize<ip, T, nc, T, nc, nc, type>(src, dst);
        }
    //@}

    //@{
    /**
     * @brief crop dst image from src start at p(left top corner)
     *
     * @notes: Tsrc, Tdst supports uchar and float
     *
     **/
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t crop(const Mat<Tsrc, ncSrc, type>& src, Point2i p, Tdst ratio, Mat<Tdst, ncDst, type> *dst);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t crop(const Mat<T, nc, type>& src, Point2i p, T ratio, Mat<T, nc, type> *dst){
            return crop<T, nc, T, nc, nc, type>(src, p, ratio, dst);
        }
    //@}

    typedef enum {
        BGR2NV21    = 0,
        NV212BGR    = 1,
        NV212RGB    = 2,
        RGB2NV21    = 3,
        BGR2NV12    = 4,
        NV122BGR    = 5,
        YCrCb2BGR   = 6,
        BGR2YCrCb   = 7,
        BGR2HSV     = 8,
        HSV2BGR     = 9,
        BGR2GRAY    = 10,
        GRAY2BGR    = 11,
        BGR2BGRA    = 12,
        BGRA2BGR    = 13,
        BGR2RGB     = 14,
        BGR2LAB     = 15,
        LAB2BGR     = 16,
        BGR2I420    = 17,
        I4202BGR    = 18,
        YUV2GRAY_420    = 19,
        RGB2NV12    = 20,
        NV122RGB    = 21
    } ColorCvtType;
    //@{
    /**
     * @brief should support 3 or 4 channels of image
     *
     * @param T             datatype of image
     * @param ncSrc         the number of channels of src
     * @param ncDst         the number of channels of dst
     *
     * @notes if ncSrc or ncDst is larger than 3, let the last channnel
     * unchanged. ncDst should be 3 for most case.
     **/
    template<ColorCvtType ct, typename Tsrc, int ncSrc, typename Tdst, int ncDst, EcoEnv_t type>
        HPCStatus_t cvtColor(const Mat<Tsrc, ncSrc, type>& src, Mat<Tdst, ncDst, type> *dst);

    template<ColorCvtType ct, typename T, int nc, EcoEnv_t type>
        HPCStatus_t cvtColor(const Mat<T, nc, type>& src, Mat<T, nc, type> *dst) {
            return cvtColor<ct, T, nc, T, nc, type>(src, dst);
        }
    //@}

    //@{
    /**
     * @brief should support 3 or 4 channels of image
     *
     * @param T             datatype of image
     * @param ncSrc         the number of channels of src
     * @param ncDst         the number of channels of dst
     * @param nc            the number of channels need to be processed
     *
     * @notes if ncSrc or ncDst is larger than 3, let the last channnel
     * unchanged. ncDst should be 3 for most case.
     *
     **/
    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t warpAffine(const Mat<T, ncSrc, type>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<T, ncDst, type> *dst);
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t warpAffine(const Mat<T, nc, type>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<T, nc, type> *dst) {
            return warpAffine<T, nc, nc, nc, type>(src, map, mode, dst);
        }
    //@}

    //@{
    /**
     * @brief should support 3 or 4 channels of image
     *
     * @param T             datatype of image
     * @param ncSrc         the number of channels of src
     * @param ncDst         the number of channels of dst
     * @param nc            the number of channels need to be processed
     *
     * @notes if ncSrc or ncDst is larger than 3, let the last channnel
     * unchanged. ncDst should be 3 for most case.
     *
     **/
    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t remap(const Mat<T, ncSrc, type>& src, const Mat<float, 1, type>& mapX,const Mat<float, 1, type>& mapY , InterpolationType mode, Mat<T, ncDst, type> *dst);
    //@}

    //@{
    /**
     * @brief should support 1, 3 or 4 channels of image
     *
     * @param T             datatype of image
     * @param ncSrc         the number of channels of src
     * @param ncDst         the number of channels of dst
     * @param nc            the number of channels need to be processed
     *
     * @notes if ncSrc or ncDst is larger than 3, let the last channnel
     * unchanged. ncDst should be 3 for most case.
     *
     **/
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t convertTo(const Mat<Tsrc, ncSrc, type>& src, float ratio, Mat<Tdst, ncDst, type> *dst);

    template<typename Tsrc, typename Tdst, int nc, EcoEnv_t type>
        HPCStatus_t convertTo(const Mat<Tsrc, nc, type>& src, float ratio, Mat<Tdst, nc, type> *dst) {
            return convertTo<Tsrc, nc, Tdst, nc, nc, type>(src, ratio, dst);
        }
    //@}

    //@{
    /**
     * @brief gaussian blur, should support 1, 3, 4 channels of image
     *
     * @param T             datatype of image
     * @param ncSrc         the number of channels of src
     * @param ncDst         the number of channels of dst
     * @param nc            the number of channels need to be processed
     *
     * @notes
     **/
    template<typename T, int filterSize, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t gaussianBlur(const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type> *dst);

    template<typename T, int filterSize, int nc, EcoEnv_t type>
        HPCStatus_t gaussianBlur(const Mat<T, nc, type>& src, Mat<T, nc, type> *dst) {
            return gaussianBlur<T, filterSize, nc, nc, nc, type>(src, dst);
        }
    //@}

    //@{
    /**
     * @brief split a multi channel mat to multi mat with one channel
     *split support contiguous memory and uncontinous memory storage
     when dst->height() == src.height(), it is the uncontinous memory storage,
     when dst->height() == src.height() * nc, it is the continous memory storage.

     * @param T             datatype of image
     * @param nc            the number of channels of src for split(dst
     * for merge)
     *
     **/
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t merge(const Mat<T, 1, type> *src, Mat<T, nc, type> *dst);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t split(const Mat<T, nc, type>& src, Mat<T, 1, type> *dst);
    //@}

    //@{
    /**
     * @brief integral image
     *
     **/
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t integral(const Mat<Tsrc, ncSrc, type>& src, Mat<Tdst, ncDst, type> *dst);

    template<typename Tsrc, typename Tdst, int nc, EcoEnv_t type>
        HPCStatus_t integral(const Mat<Tsrc, nc, type>& src, Mat<Tdst, nc, type> *dst) {
            return integral<Tsrc, nc, Tdst, nc, nc, type>(src, dst);
        }

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t integral(const Mat<T, nc, type>& src, Mat<T, nc, type> *dst) {
            return integral<T, T, nc, type>(src, dst);
        }
    //@}

    //@{
    /**
     * @brief standard rotate image from center, only support degree 90, 180, 270;
     **/
    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t rotate(const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type>* dst, float degree);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t rotate(const Mat<T, nc, type>& src, Mat<T, nc, type>* dst, float degree) {
            return rotate<T, nc, nc, nc, type>(src, dst, degree);
        }
    /**YUV420 family type**/
    typedef enum{
        YUV420_NV12 = 0,
        YUV420_NV21 = 1,
        YUV420_I420 = 2
    }YUV420Type;
    //@{
    /**
     * @brief rotate YUV420 image from center, only support degree 90, 180, 270;
     **/
    template<YUV420Type yt, typename T, EcoEnv_t type>
        HPCStatus_t rotate_YUV420(const Mat<T, 1,type>& src, Mat<T, 1,type>* dst, float degree);
    //@}

    //@{
    /**
     * @brief set the content of matrix to fixed value
     *
     *
     **/
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t setTo(T value, Mat<T, nc, type> *mat);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t setTo(const T value[nc], Mat<T, nc, type> *mat);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t zeros(Mat<T, nc, type> *mat) { return setTo((T)0, mat);}

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t ones(Mat<T, nc, type> *mat) { return setTo((T)1, mat);}
    //@}

    //@{
    /**
     * @brief multiply two mat
     *
     *
     *
     **/
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t add(const Mat<T, nc, type>& a, const Mat<T, nc, type>& b, Mat<T, nc, type> *c);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t add(const Mat<T, nc, type>& a, T b, Mat<T, nc, type> *c);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t mul(const Mat<T, nc, type>& a, const Mat<T, nc, type>& b, Mat<T, nc, type> *c);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t mul(const Mat<T, nc, type>& a, const T b, Mat<T, nc, type> *c);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t mls(const Mat<T, nc, type>& a, const Mat<T, nc, type>& b, Mat<T, nc, type> *c);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t mla(const Mat<T, nc, type>& a, const Mat<T, nc, type>& b, Mat<T, nc, type> *c);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t div(const Mat<T, nc, type>& a, const Mat<T, nc, type>& b, Mat<T, nc, type> *c);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t div(const Mat<T, nc, type>& a, const T b, Mat<T, nc, type> *c) {
            return mul(a, (T)1/b, c);
        }
    //@}

    typedef enum {
        BORDER_CONSTANT = 0, //iiiiii|abcdefgh|iiiiiii with some specified i
        BORDER_REPLICATE = 1, // aaaaaa|abcdefgh|hhhhhhh
        BORDER_REFLECT = 2, // fedcba|abcdefgh|hgfedcb
        BORDER_WRAP    = 3, // cdefgh|abcdefgh|abcdefg
        BORDER_REFLECT101 = 4, // gfedcb|abcdefgh|gfedcba
        BORDER_DEFAULT = 4, // same as BORDER_REFLECT_101
        BORDER_TRANSPARENT = 5, // uvwxyz|absdefgh|i
        BORDER_ISOLATED = 6, //do not look outside of ROI
    } BorderType;

    //@{
    /**
     * @param depth         depth need to be computed
     * @param ks            kernel size
     *
     *
     **/
    template<BorderType bt, typename T, int filterSize, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t filter2D(const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type> *dst, const MatX2D<float, filterSize, filterSize>& kernel);

    template<BorderType bt, typename T, int filterSize, int nc, EcoEnv_t type>
        HPCStatus_t filter2D(const Mat<T, nc, type>& src, Mat<T, nc, type> *dst, const MatX2D<float, filterSize, filterSize>& kernel) {
            return filter2D<bt, T, filterSize, nc, nc, nc, type>(src, dst, kernel);
        }

    //@}

    //@{
    /**
     * @param depth         depth need to be computed
     * @param ks            kernel size
     *
     *
     **/
    template<BorderType bt, typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t boxFilter(const Mat<T, ncSrc, type>& src, Size2i ks, Point2i anchor, Mat<T, ncDst, type> *dst, bool normalized);

    template<BorderType bt, typename T, int nc, EcoEnv_t type>
        HPCStatus_t boxFilter(const Mat<T, nc, type>& src, Size2i ks, Point2i anchor, Mat<T, nc, type> *dst, bool normalized) {
            return boxFilter<bt, T, nc, nc, nc, type>(src, ks, anchor, dst, normalized);
        }

    template<BorderType bt, typename T, int nc, EcoEnv_t type>
        HPCStatus_t boxFilter(const Mat<T, nc, type>& src, Size2i ks, Mat<T, nc, type> *dst,  bool normalized) {
            Point2i anchor;
            anchor.x = -1;
            anchor.y = -1;
            return  boxFilter<bt, T, nc, nc, nc, type>(src, ks, anchor, dst, normalized);
        }

    //@{
    /**
     *brief blur 
     *
     *
     **/
    template<BorderType bt, typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t blur(const Mat<T, ncSrc, type>& src, Size2i ks, Point2i anchor, Mat<T, ncDst, type> *dst){
            return boxFilter<bt, T, ncSrc, ncDst, nc, type>(src, ks, anchor, dst, 1);
        }

    template<BorderType bt, typename T, int nc, EcoEnv_t type>
        HPCStatus_t blur(const Mat<T, nc, type>& src, Size2i ks, Point2i anchor, Mat<T, nc, type> *dst) {
            return boxFilter<bt, T, nc, nc, nc, type>(src, ks, anchor, dst, 1);
        }

    template<BorderType bt, typename T, int nc, EcoEnv_t type>
        HPCStatus_t blur(const Mat<T, nc, type>& src, Size2i ks, Mat<T, nc, type> *dst) {
            Point2i anchor;
            anchor.x = -1;
            anchor.y = -1;
            return  boxFilter<bt, T, nc, nc, nc, type>(src, ks, anchor, dst, 1);
        }

    //@}

    //@{
    /**
     *
     * @brief
     *
     * @param
     * @param
     *
     **/
    template<BorderType bt, typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t bilateralFilter(const Mat<T, ncSrc, type>& src, int diameter, double color, double space, Mat<T, ncDst, type> *dst);

    template<BorderType bt, typename T, int nc, EcoEnv_t type>
        HPCStatus_t bilateralFilter(const Mat<T, nc, type>& src, int diameter, double color, double space, Mat<T, nc, type> *dst) {
            return bilateralFilter<bt, T, nc, nc, nc, type>(src, diameter, color, space, dst);
        }
    //@}

    //@{
    /**
     *
     * @brief blur an image with fast bilateral filter
     *
     * @param src            input Mat
     * @param base           base Mat
     * @param space          space_sigma
     * @param color          color_sigma
     * @param dst            output Mat
     **/
    template<BorderType bt, typename T, int ncSrc, int ncBase, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t fastBilateralFilter(const Mat<T, ncSrc, type>& src, const Mat<T, ncBase, type>& base, double color, double space, Mat<T, ncDst, type> *dst);

    template<BorderType bt, typename T, int nc, EcoEnv_t type>
        HPCStatus_t fastBilateralFilter(const Mat<T, nc, type>& src, const Mat<T, nc, type>& base, double color, double space, Mat<T, nc, type> *dst) {
            return fastBilateralFilter<bt, T, nc, nc, nc, nc, type>(src, base, color, space, dst);
        }
    //@}

    //@{
    /**
     *
     * @Fast Guided Filter
     *
     * @param guided         guided image
     * @param src            input image
     * @param dst            output image
     * @param radius         radius of boxfilter
     * @param eps            epsilon
     * @param scale          resize scale
     **/
    template<BorderType bt, typename T, int ncGuided, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t fastGuidedFilter(const Mat<T, ncGuided, type>& guided, const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type> *dst, int radius, float eps, int scale);

    template<BorderType bt, typename T, int nc, EcoEnv_t type>
        HPCStatus_t fastGuidedFilter(const Mat<T, nc, type>& src, Mat<T, nc, type> *dst, float eps, int radius) {
            return fastGuidedFilter<bt, T, nc, nc, nc, nc, type>(src, src, dst, radius, eps, radius);
        }
    //@}

    //@{
    /**
     * @brief general filter algorithm, should support 1, 3, 4 channels of image
     *
     * @param T             datatype of image
     * @param ncSrc         the number of channels of src
     * @param ncDst         the number of channels of dst
     * @param nc            the number of channels need to be processed
     *
     * @notes
     **/
    template<typename T, int filterSize, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t filter(const Mat<T, ncSrc, type>& src, const MatX2D<T, filterSize, filterSize>& f, Mat<T, ncDst, type> *dst);

    template<typename T, int filterSize, int nc, EcoEnv_t type>
        HPCStatus_t filter(const Mat<T, nc, type>& src, const MatX2D<T, filterSize, filterSize>& f, Mat<T, nc, type> *dst) {
            return filter<T, filterSize, nc, nc, nc, type>(src, f, dst);
        }
    //@}

    //@{
    /**
     * @erode an image
     *
     * @param T             datatype of image
     * @param ncSrc         the number of channels of src
     * @param ncDst         the number of channels of dst
     * @param nc            the number of channels need to be processed
     *
     * @notes
     **/

    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t erode(const Mat<T, ncSrc, type>& src, const Mat<uchar, 1, type>& element, Mat<T, ncDst, type> *dst);

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t erode(const Mat<T, nc, type>& src, const Mat<uchar, 1, type>& element, Mat<T, nc, type> *dst) {
            return erode<T, nc, nc, nc, type>(src, element, dst);
        }
    //@}

    //@{
    /**
     * @dilate an image
     *
     * @param T             datatype of image
     * @param ncSrc         the number of channels of src
     * @param ncDst         the number of channels of dst
     * @param nc            the number of channels need to be processed
     *
     * @notes
     **/
     template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t dilate(const Mat<T, ncSrc, type>& src, const Mat<uchar, 1, type>& element, Mat<T, ncDst, type> *dst);
     template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t dilate(const Mat<T, nc, type>& src, const Mat<uchar, 1, type>& element, Mat<T, nc, type> *dst) {
            return dilate<T, nc, nc, nc, type>(src, element, dst);
        }
    //@}
        
}; };

#endif
