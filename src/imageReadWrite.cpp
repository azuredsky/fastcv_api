#include <fastcv.hpp>
#include <algorithm>
#include <string>

#ifdef FASTCV_USE_JPEG
extern "C" {
#include "jpeglib.h"
}
#ifndef FASTCV_USE_PNG
extern "C" {
#include "setjmp.h"
}
#endif
#endif

#ifdef FASTCV_USE_PNG
extern "C" {
#include "png.h"
}
#endif


namespace HPC { namespace fastcv {
    template<typename T>
        inline T ror(const T& a, const unsigned int n=1){
            return n?(T)((a>>n)|(a<<((sizeof(T)<<3) -n))):a;
        }
    inline float ror(const float a, const unsigned int n=1){
        return (float)ror((int)a, n);
    }
    inline double ror(const double a, const unsigned int n=1){
        return (double)ror((long)a, n);
    }

    typedef enum{
        FASTCV_PNG_8UC1=1,//|b|
        FASTCV_PNG_8UC2=2,//|ba|
        FASTCV_PNG_8UC3=3,//|bgr|
        FASTCV_PNG_8UC4=4,//|bgra|
        FASTCV_PNG_16UC1=11,//|b|
        FASTCV_PNG_16UC2=12,//|ba|
        FASTCV_PNG_16UC3=13,//|bgr|
        FASTCV_PNG_16UC4=14,//|bgra|
    }FASTCV_PNG_DEPTH;

    struct PaletteEntry
    {
        unsigned char b, g, r,a;
    };

    HPCStatus_t load_Bmp_(const char* name, unsigned char** m, int* numCols, int* numRows);
    HPCStatus_t write_Bmp_(const char* name, unsigned char* buffer, int numCols, int numRows, int nc_buffer);
    HPCStatus_t imdecode_Bmp_(unsigned char* databuffer,const unsigned int dataSize,unsigned char **m,int* numCols, int* numRows);
    HPCStatus_t encode_Bmp_(unsigned char **buffer, unsigned char* m_buffer, int numCols, int numRows, int channels);

#ifdef FASTCV_USE_JPEG
    HPCStatus_t load_Jpeg_(const char* name, unsigned char** m, int* numCols, int* numRows, int* numChannels);
    HPCStatus_t write_Jpeg_(const char* name, unsigned char* buffer, int numCols, int numRows, int nc, const unsigned int quality=100);
    HPCStatus_t imdecode_Jpeg_(unsigned char* buff,const unsigned int dataSize,unsigned char** m,int* numCols,int *numRows,int* numChannels);
	
    HPCStatus_t encode_Jpeg_(unsigned char** buffer,unsigned char* m_buffer, int numCols, int numRows, int nc, const unsigned int quality=100);

#endif
#ifdef FASTCV_USE_PNG
    HPCStatus_t load_Png_(const char* name, void** m, int* numCols, int* numRows, FASTCV_PNG_DEPTH *png_depth);
    HPCStatus_t imdecode_Png_(unsigned char* buffer ,const unsigned int dataSize,void **m,int* numCols,int* numRows,FASTCV_PNG_DEPTH *png_depth);
    template<typename T, int nc> HPCStatus_t write_Png_(const char* name, T* m, int numCols, int numRows, const unsigned int bytes_per_pixel=0);
    template<typename T,int nc> HPCStatus_t encode_Png_(unsigned char** buffer, T* m, int numCols, int numRows, const std::vector<int> params, const unsigned int bytes_per_pixel = 0);

#endif

     void icvCvt_BGR2RGB_8u_C3R(const uchar* bgr, int bgr_step, uchar* rgb,int rgb_step, Size size)
     {
         int i;
         size_t height = size.h();
         size_t width = size.w();

         for(;height>0;height--)
         {
             for(i = 0;i < width;i++,bgr += 3, rgb += 3)
             {
                 uchar t0 = bgr[0], t1 = bgr[1], t2 = bgr[2];
                 rgb[2] = t0;
                 rgb[1] = t1;
                 rgb[0] = t2;
             }
             bgr += bgr_step - width*3;
             rgb += rgb_step - width*3;
         }
     }

     void icvCvt_BGRA2BGR_8u_C4C3R(const uchar* bgra, int bgra_step,uchar* bgr, int bgr_step, Size size, int _swap_rg)
     {
         int i;
         size_t height = size.h(), width = size.w();
         int swap_rb = _swap_rg ? 2 : 0;
         for(;height--;)
         {
            for(i = 0;i<width;i++, bgr += 3, bgr += 4)
             {
                 uchar t0 = bgra[swap_rb], t1 = bgra[1];
                 bgr[0] = t0;
                 bgr[1] = t1;
                 t0 = bgra[swap_rb^2];
                 bgr[2] = t0;

             }
             bgr += bgr_step - width*3;
             bgra += bgra_step - width*4;
         }
     }
     template<typename T,int nc,  EcoEnv_t type>
        HPCStatus_t imdecodeBmp(unsigned char* databuffer,const unsigned int dataSize,Mat<T,nc,type> *m)
        {
            int numCols = 0,numRows = 0;
            unsigned char *data = NULL;
            HPCStatus_t st = imdecode_Bmp_(databuffer,dataSize,&data,&numCols,&numRows);
            if(st != HPC_SUCCESS)
            {
                if(data != NULL) free(data);
                return st;
            }
            if(m->ptr() == NULL){
                m->setSize(numCols, numRows);
            } else {
                if(m->height() != numRows || m->width() != numCols){
                    fprintf(stderr, "wrong dims for loading img");
                    if(data != NULL) free(data);
                    return HPC_OP_NOT_PERMITED;
                }
            }

            T *tmpbuffer = (T*)malloc(numCols * numRows * nc * sizeof(T));
            if(NULL==tmpbuffer) return HPC_ALLOC_FAILED;

            switch(nc){
                case 1:
                    {
                        int gray;
                        for(int i=0; i<numRows*numCols; i++){
                            gray = (data[i*3+2]*77 + data[i*3+1]*150 + data[i*3] * 29 + 128) >> 8;
                            gray = gray > 255 ? 255 : gray;
                            tmpbuffer[i] = (T)gray;
                        }
                        m->fromHost(tmpbuffer);
                    }
                    break;
                case 3:
                    {
                        for(int i=0; i<numRows*numCols*nc; i++)
                            tmpbuffer[i] = (T)data[i];
                        m->fromHost(tmpbuffer);
                    }
                    break;
                case 4:
                    {
                        for(int i=0; i<numRows*numCols; i++){
                            tmpbuffer[i*4] = (T)data[i*3];
                            tmpbuffer[i*4+1] = (T)data[i*3+1];
                            tmpbuffer[i*4+2] = (T)data[i*3+2];
                            tmpbuffer[i*4+3] = (T)255;
                        }
                        m->fromHost(tmpbuffer);
                    }
                    break;
                default:
                    if(data != NULL)free(data);
                    if(tmpbuffer != NULL)free(tmpbuffer);
                    return HPC_OP_NOT_PERMITED;
            }
            if(data != NULL)free(data);
            if(tmpbuffer != NULL)free(tmpbuffer);
            return HPC_SUCCESS;
        }


    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t loadBmp(const char* name, Mat<T, nc, type> *m) {
            int numCols=0, numRows=0;
            unsigned char *data = NULL;
            HPCStatus_t st = load_Bmp_(name, &data, &numCols, &numRows);
            if(st != HPC_SUCCESS){
                if(data != NULL) free(data);
                return st;
            }

            if(m->ptr() == NULL){
                m->setSize(numCols, numRows);
            } else {
                if(m->height() != numRows || m->width() != numCols){
                    fprintf(stderr, "wrong dims for loading img");
                    if(data != NULL) free(data);
                    return HPC_OP_NOT_PERMITED;
                }
            }

            T *buffer = (T*)malloc(numCols * numRows * nc * sizeof(T));
            if(NULL==buffer) return HPC_ALLOC_FAILED;

            switch(nc){
                case 1:
                    {
                        int gray;
                        for(int i=0; i<numRows*numCols; i++){
                            gray = (data[i*3+2]*77 + data[i*3+1]*150 + data[i*3] * 29 + 128) >> 8;
                            gray = gray > 255 ? 255 : gray;
                            buffer[i] = (T)gray;
                        }
                        m->fromHost(buffer);
                    }
                    break;
                case 3:
                    {
                        for(int i=0; i<numRows*numCols*nc; i++)
                            buffer[i] = (T)data[i];
                        m->fromHost(buffer);
                    }
                    break;
                case 4:
                    {
                        for(int i=0; i<numRows*numCols; i++){
                            buffer[i*4] = (T)data[i*3];
                            buffer[i*4+1] = (T)data[i*3+1];
                            buffer[i*4+2] = (T)data[i*3+2];
                            buffer[i*4+3] = (T)255;
                        }
                        m->fromHost(buffer);
                    }
                    break;
                default:
                    if(data != NULL)free(data);
                    if(buffer != NULL)free(buffer);
                    return HPC_OP_NOT_PERMITED;
            }
            if(data != NULL)free(data);
            if(buffer != NULL)free(buffer);
            return HPC_SUCCESS;
        }
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t imdecodeJpeg(unsigned char* databuffer,const unsigned int dataSize,Mat<T,nc,type> *m)
        {

#ifdef FASTCV_USE_JPEG
            int numCols=0, numRows=0, numChannels=0;
            unsigned char *data = NULL;
            HPCStatus_t st = imdecode_Jpeg_(databuffer, dataSize,&data, &numCols, &numRows, &numChannels);
            if(st != HPC_SUCCESS){
                if(data != NULL) free(data);
                return st;
            }

            if(m->ptr() == NULL){
                m->setSize(numCols, numRows);
            } else {
                if(m->height() != numRows || m->width() != numCols){
                    fprintf(stderr, "wrong dims for loading img");
                    if(data != NULL) free(data);
                    return HPC_OP_NOT_PERMITED;
                }
            }

            T *buffer = (T*)malloc(numCols * numRows * nc * sizeof(T));
            if(NULL==buffer) return HPC_ALLOC_FAILED;

            switch(nc){
                case 1:
                    {
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++)
                                buffer[i] = (T)data[i];
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3 || numChannels == 4){
                            int gray;
                            for(int i=0; i<numRows*numCols; i++){
                                gray = (data[i*numChannels+2]*77 + data[i*numChannels+1]*150 + data[i*numChannels] * 29 + 128) >> 8;
                                gray = gray > 255 ? 255 : gray;
                                buffer[i] = (T)gray;
                            }
                            m->fromHost(buffer);
                        }
                    }
                    break;
                case 3:
                    {
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++){
                                buffer[i*3] = (T)data[i];
                                buffer[i*3+1] = (T)data[i];
                                buffer[i*3+2] = (T)data[i];
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3){
                            for(int i=0; i<numRows*numCols*nc; i++)
                                buffer[i] = (T)data[i];
                            m->fromHost(buffer);
                        }
                        if(numChannels == 4){
                            for(int i=0; i<numRows*numCols; i++){
                                buffer[3*i] = (T)data[4*i];
                                buffer[3*i+1] = (T)data[4*i+1];
                                buffer[3*i+2] = (T)data[4*i+2];
                            }
                            m->fromHost(buffer);
                        }
                    }
                    break;
                case 4:
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++){
                                buffer[i*4] = (T)data[i];
                                buffer[i*4+1] = (T)data[i];
                                buffer[i*4+2] = (T)data[i];
                                buffer[i*4+3] = (T)data[i];
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3){
                            for(int i=0; i<numRows*numCols; i++){
                                buffer[i*4] = (T)data[i*3];
                                buffer[i*4+1] = (T)data[i*3+1];
                                buffer[i*4+2] = (T)data[i*3+2];
                                buffer[i*4+3] = (T)255;
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 4){
                            for(int i=0; i<numRows*numCols*4; i++){
                                buffer[i] = (T)data[i];
                            }
                            m->fromHost(buffer);
                        }
                    break;
                default:
                    if(data != NULL)free(data);
                    if(buffer != NULL)free(buffer);
                    return HPC_OP_NOT_PERMITED;
            }
            if(data != NULL)free(data);
            if(buffer != NULL)free(buffer);
            return HPC_SUCCESS;
#else
            return HPC_NOT_SUPPORTED;
#endif
        }


    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t loadJpeg(const char* name, Mat<T, nc, type> *m) {
#ifdef FASTCV_USE_JPEG
            int numCols=0, numRows=0, numChannels=0;
            unsigned char *data = NULL;
            HPCStatus_t st = load_Jpeg_(name, &data, &numCols, &numRows, &numChannels);
            if(st != HPC_SUCCESS){
                if(data != NULL) free(data);
                return st;
            }

            if(m->ptr() == NULL){
                m->setSize(numCols, numRows);
            } else {
                if(m->height() != numRows || m->width() != numCols){
                    fprintf(stderr, "wrong dims for loading img");
                    if(data != NULL) free(data);
                    return HPC_OP_NOT_PERMITED;
                }
            }

            T *buffer = (T*)malloc(numCols * numRows * nc * sizeof(T));
            if(NULL==buffer) return HPC_ALLOC_FAILED;

            switch(nc){
                case 1:
                    {
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++)
                                buffer[i] = (T)data[i];
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3 || numChannels == 4){
                            int gray;
                            for(int i=0; i<numRows*numCols; i++){
                                gray = (data[i*numChannels+2]*77 + data[i*numChannels+1]*150 + data[i*numChannels] * 29 + 128) >> 8;
                                gray = gray > 255 ? 255 : gray;
                                buffer[i] = (T)gray;
                            }
                            m->fromHost(buffer);
                        }
                    }
                    break;
                case 3:
                    {
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++){
                                buffer[i*3] = (T)data[i];
                                buffer[i*3+1] = (T)data[i];
                                buffer[i*3+2] = (T)data[i];
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3){
                            for(int i=0; i<numRows*numCols*nc; i++)
                                buffer[i] = (T)data[i];
                            m->fromHost(buffer);
                        }
                        if(numChannels == 4){
                            for(int i=0; i<numRows*numCols; i++){
                                buffer[3*i] = (T)data[4*i];
                                buffer[3*i+1] = (T)data[4*i+1];
                                buffer[3*i+2] = (T)data[4*i+2];
                            }
                            m->fromHost(buffer);
                        }
                    }
                    break;
                case 4:
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++){
                                buffer[i*4] = (T)data[i];
                                buffer[i*4+1] = (T)data[i];
                                buffer[i*4+2] = (T)data[i];
                                buffer[i*4+3] = (T)data[i];
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3){
                            for(int i=0; i<numRows*numCols; i++){
                                buffer[i*4] = (T)data[i*3];
                                buffer[i*4+1] = (T)data[i*3+1];
                                buffer[i*4+2] = (T)data[i*3+2];
                                buffer[i*4+3] = (T)255;
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 4){
                            for(int i=0; i<numRows*numCols*4; i++){
                                buffer[i] = (T)data[i];
                            }
                            m->fromHost(buffer);
                        }
                    break;
                default:
                    if(data != NULL)free(data);
                    if(buffer != NULL)free(buffer);
                    return HPC_OP_NOT_PERMITED;
            }
            if(data != NULL)free(data);
            if(buffer != NULL)free(buffer);
            return HPC_SUCCESS;
#else
            return HPC_NOT_SUPPORTED;
#endif

        }

   template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t imdecodePng(unsigned char* databuffer,const unsigned int dataSize, Mat<T, nc, type> *m) {
#ifdef FASTCV_USE_PNG
            int numCols=0, numRows=0, numChannels=0;
            FASTCV_PNG_DEPTH png_depth;
            void *data = NULL;
            HPCStatus_t st = imdecode_Png_(databuffer,dataSize, &data, &numCols, &numRows, &png_depth);
            if(st != HPC_SUCCESS){
                if(data != NULL) free(data);
                return st;
            }
            numChannels = (int)png_depth % 10;
            bool is_u8 = !((bool)(png_depth / 10));

            if(m->ptr() == NULL){
                m->setSize(numCols, numRows);
            } else {
                if(m->height() != numRows || m->width() != numCols){
                    fprintf(stderr, "wrong dims for loading img\n");
                    if(data != NULL) free(data);
                    return HPC_OP_NOT_PERMITED;
                }
            }

            T *buffer = (T*)malloc(numCols * numRows * nc * sizeof(T));
            if(NULL==buffer) return HPC_ALLOC_FAILED;

            switch(nc){
                case 1:
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++){
                                if(is_u8)
                                    buffer[i] = (T)(((uchar*)data)[i]);
                                else
                                    buffer[i] = (T)(((unsigned short*)data)[i]);
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3 || numChannels == 4){
                            int gray;
                            for(int i=0; i<numRows*numCols; i++){
                                if(is_u8)
                                    gray = (((uchar*)data)[i*numChannels+2]*77 + ((uchar*)data)[i*numChannels+1]*150 + ((uchar*)data)[i*numChannels] * 29 + 128) >> 8;
                                else
                                    gray = (((unsigned short*)data)[i*numChannels+2]*77 + ((unsigned short*)data)[i*numChannels+1]*150 + ((unsigned short*)data)[i*numChannels] * 29 + 128) >> 8;
                                gray = gray > 255 ? 255 : gray;
                                buffer[i] = (T)gray;
                            }
                            m->fromHost(buffer);
                        }
                    break;
                case 3:
                    {
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++){
                                if(is_u8){
                                    buffer[i*3] = (T)(((uchar*)data)[i]);
                                    buffer[i*3+1] = (T)(((uchar*)data)[i]);
                                    buffer[i*3+2] = (T)(((uchar*)data)[i]);
                                } else {
                                    buffer[i*3] = (T)(((unsigned short*)data)[i]);
                                    buffer[i*3+1] = (T)(((unsigned short*)data)[i]);
                                    buffer[i*3+2] = (T)(((unsigned short*)data)[i]);
                                }
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3){
                            for(int i=0; i<numRows*numCols*3; i++)
                                if(is_u8)
                                    buffer[i] = (T)(((uchar*)data)[i]);
                                else
                                    buffer[i] = (T)(((unsigned short*)data)[i]);
                            m->fromHost(buffer);
                        }
                        if(numChannels == 4 && is_u8){
                            for(int i=0; i<numRows*numCols; i++){
                                buffer[i*3] = (T)(((uchar*)data)[i*4]);
                                buffer[i*3+1] = (T)(((uchar*)data)[i*4+1]);
                                buffer[i*3+2] = (T)(((uchar*)data)[i*4+2]);
                            }
                            m->fromHost(buffer);
                        }
                    }
                    break;
                case 4:
                    {
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++){
                                if(is_u8){
                                    T data_ = (T)(((uchar*)data)[i]);
                                    buffer[i*4] = data_;
                                    buffer[i*4+1] = data_;
                                    buffer[i*4+2] = data_;
                                    buffer[i*4+3] = data_;
                                } else {
                                    T data_ = (T)(((unsigned short*)data)[i]);
                                    buffer[i*4] = data_;
                                    buffer[i*4+1] = data_;
                                    buffer[i*4+2] = data_;
                                    buffer[i*4+3] = data_;
                                }
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3){
                            for(int i=0; i<numRows*numCols; i++){
                                if(is_u8){
                                    buffer[i*4] = (T)(((uchar*)data)[i*3]);
                                    buffer[i*4+1] = (T)(((uchar*)data)[i*3+1]);
                                    buffer[i*4+2] = (T)(((uchar*)data)[i*3+2]);
                                    buffer[i*4+3] = (T)255;
                                } else {
                                    buffer[i*4] = (T)(((unsigned short*)data)[i*3]);
                                    buffer[i*4+1] = (T)(((unsigned short*)data)[i*3+1]);
                                    buffer[i*4+2] = (T)(((unsigned short*)data)[i*3+2]);
                                    buffer[i*4+3] = (T)255;
                                }
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 4){
                            for(int i=0; i<numRows*numCols*4; i++){
                                if(is_u8)
                                    buffer[i] = (T)(((uchar*)data)[i]);
                                else
                                    buffer[i] = (T)(((unsigned short*)data)[i]);
                            }
                            m->fromHost(buffer);
                        }
                    }
                    break;
                default:
                    if(data != NULL)free(data);
                    if(buffer != NULL)free(buffer);
                    return HPC_OP_NOT_PERMITED;
            }
            if(data != NULL)free(data);
            if(buffer != NULL)free(buffer);
            return HPC_SUCCESS;
#else
            return HPC_NOT_SUPPORTED;
#endif

        }


    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t loadPng(const char* name, Mat<T, nc, type> *m) {
#ifdef FASTCV_USE_PNG
            int numCols=0, numRows=0, numChannels=0;
            FASTCV_PNG_DEPTH png_depth;
            void *data = NULL;
            HPCStatus_t st = load_Png_(name, &data, &numCols, &numRows, &png_depth);
            if(st != HPC_SUCCESS){
                if(data != NULL) free(data);
                return st;
            }
            numChannels = (int)png_depth % 10;
            bool is_u8 = !((bool)(png_depth / 10));

            if(m->ptr() == NULL){
                m->setSize(numCols, numRows);
            } else {
                if(m->height() != numRows || m->width() != numCols){
                    fprintf(stderr, "wrong dims for loading img\n");
                    if(data != NULL) free(data);
                    return HPC_OP_NOT_PERMITED;
                }
            }

            T *buffer = (T*)malloc(numCols * numRows * nc * sizeof(T));
            if(NULL==buffer) return HPC_ALLOC_FAILED;

            switch(nc){
                case 1:
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++){
                                if(is_u8)
                                    buffer[i] = (T)(((uchar*)data)[i]);
                                else
                                    buffer[i] = (T)(((unsigned short*)data)[i]);
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3 || numChannels == 4){
                            int gray;
                            for(int i=0; i<numRows*numCols; i++){
                                if(is_u8)
                                    gray = (((uchar*)data)[i*numChannels+2]*77 + ((uchar*)data)[i*numChannels+1]*150 + ((uchar*)data)[i*numChannels] * 29 + 128) >> 8;
                                else
                                    gray = (((unsigned short*)data)[i*numChannels+2]*77 + ((unsigned short*)data)[i*numChannels+1]*150 + ((unsigned short*)data)[i*numChannels] * 29 + 128) >> 8;
                                gray = gray > 255 ? 255 : gray;
                                buffer[i] = (T)gray;
                            }
                            m->fromHost(buffer);
                        }
                    break;
                case 3:
                    {
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++){
                                if(is_u8){
                                    buffer[i*3] = (T)(((uchar*)data)[i]);
                                    buffer[i*3+1] = (T)(((uchar*)data)[i]);
                                    buffer[i*3+2] = (T)(((uchar*)data)[i]);
                                } else {
                                    buffer[i*3] = (T)(((unsigned short*)data)[i]);
                                    buffer[i*3+1] = (T)(((unsigned short*)data)[i]);
                                    buffer[i*3+2] = (T)(((unsigned short*)data)[i]);
                                }
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3){
                            for(int i=0; i<numRows*numCols*3; i++)
                                if(is_u8)
                                    buffer[i] = (T)(((uchar*)data)[i]);
                                else
                                    buffer[i] = (T)(((unsigned short*)data)[i]);
                            m->fromHost(buffer);
                        }
                        if(numChannels == 4 && is_u8){
                            for(int i=0; i<numRows*numCols; i++){
                                buffer[i*3] = (T)(((uchar*)data)[i*4]);
                                buffer[i*3+1] = (T)(((uchar*)data)[i*4+1]);
                                buffer[i*3+2] = (T)(((uchar*)data)[i*4+2]);
                            }
                            m->fromHost(buffer);
                        }
                    }
                    break;
                case 4:
                    {
                        if(numChannels == 1){
                            for(int i=0; i<numRows*numCols; i++){
                                if(is_u8){
                                    T data_ = (T)(((uchar*)data)[i]);
                                    buffer[i*4] = data_;
                                    buffer[i*4+1] = data_;
                                    buffer[i*4+2] = data_;
                                    buffer[i*4+3] = data_;
                                } else {
                                    T data_ = (T)(((unsigned short*)data)[i]);
                                    buffer[i*4] = data_;
                                    buffer[i*4+1] = data_;
                                    buffer[i*4+2] = data_;
                                    buffer[i*4+3] = data_;
                                }
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 3){
                            for(int i=0; i<numRows*numCols; i++){
                                if(is_u8){
                                    buffer[i*4] = (T)(((uchar*)data)[i*3]);
                                    buffer[i*4+1] = (T)(((uchar*)data)[i*3+1]);
                                    buffer[i*4+2] = (T)(((uchar*)data)[i*3+2]);
                                    buffer[i*4+3] = (T)255;
                                } else {
                                    buffer[i*4] = (T)(((unsigned short*)data)[i*3]);
                                    buffer[i*4+1] = (T)(((unsigned short*)data)[i*3+1]);
                                    buffer[i*4+2] = (T)(((unsigned short*)data)[i*3+2]);
                                    buffer[i*4+3] = (T)255;
                                }
                            }
                            m->fromHost(buffer);
                        }
                        if(numChannels == 4){
                            for(int i=0; i<numRows*numCols*4; i++){
                                if(is_u8)
                                    buffer[i] = (T)(((uchar*)data)[i]);
                                else
                                    buffer[i] = (T)(((unsigned short*)data)[i]);
                            }
                            m->fromHost(buffer);
                        }
                    }
                    break;
                default:
                    if(data != NULL)free(data);
                    if(buffer != NULL)free(buffer);
                    return HPC_OP_NOT_PERMITED;
            }
            if(data != NULL)free(data);
            if(buffer != NULL)free(buffer);
            return HPC_SUCCESS;
#else
            return HPC_NOT_SUPPORTED;
#endif

        }

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t loadTiff(const char* name, Mat<T, nc, type> *m) {
            return HPC_NOT_IMPLEMENTED;

        }

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t encodeBmp(unsigned char **buffer,const Mat<T, nc, type> *m)
        {
            if(!m->ptr())
                return HPC_POINTER_NULL;
            int numRows = m->height();
            int numCols = m->width();
            T* data = (T*)malloc(m->numElements() * nc * sizeof(T));
            if(NULL == data)
                return HPC_ALLOC_FAILED;
           ((Mat<T,nc>*)m)->toHost(data);
            int nc_buffer =( nc >= 3 ? 3 : nc);
            uchar *m_buffer = (uchar *)malloc(numCols * numRows * nc_buffer * sizeof(T));
            if(NULL == m_buffer) return HPC_ALLOC_FAILED;

            switch(nc){
                case 1:
                    for(int i=0; i<numRows * numCols; i++)
                        m_buffer[i] = (uchar)data[i];
                    break;
                case 2:
                    for(int i=0; i<numRows * numCols * 2; i++)
                        m_buffer[i] = (uchar)data[i];
                    break;
                case 3:
                    for(int i=0; i<numRows * numCols * 3; i++)
                        m_buffer[i] = (uchar)data[i];
                    break;
                case 4:
                    printf("warning: only three channels will be written.\n");
                    for(int i=0; i<numRows * numCols; i++){
                        m_buffer[i*3] = (uchar)data[i*4];
                        m_buffer[i*3+1] = (uchar)data[i*4+1];
                        m_buffer[i*3+2] = (uchar)data[i*4+2];
                    }
                    break;
                default:
                    if(data != NULL)free(data);
                    if(m_buffer != NULL)free(m_buffer);
                    return HPC_OP_NOT_PERMITED;
            }
            HPCStatus_t st = encode_Bmp_(buffer, m_buffer, numCols, numRows, nc_buffer);
 
            if(data != NULL )
                free(data);
            if(m_buffer != NULL)
                free(m_buffer);
            return st;

        }
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t writeBmp(const char* name, const Mat<T, nc, type> *m) {
            if(!name || !m)
                return HPC_POINTER_NULL;

            int numRows = m->height();
            int numCols = m->width();

            T* data = (T*)malloc(m->numElements() * nc * sizeof(T));
            if(NULL == data) return HPC_ALLOC_FAILED;
            ((Mat<T, nc>*)m)->toHost(data);

            int nc_buffer = nc >=3 ? 3 : nc;
            uchar *buffer = (uchar*)malloc(numCols * numRows * nc_buffer * sizeof(T));
            if(NULL == buffer) return HPC_ALLOC_FAILED;

            switch(nc){
                case 1:
                    for(int i=0; i<numRows * numCols; i++)
                        buffer[i] = (uchar)data[i];
                    break;
                case 2:
                    for(int i=0; i<numRows * numCols * 2; i++)
                        buffer[i] = (uchar)data[i];
                    break;
                case 3:
                    for(int i=0; i<numRows * numCols * 3; i++)
                        buffer[i] = (uchar)data[i];
                    break;
                case 4:
                    printf("warning: only three channels will be written.\n");
                    for(int i=0; i<numRows * numCols; i++){
                        buffer[i*3] = (uchar)data[i*4];
                        buffer[i*3+1] = (uchar)data[i*4+1];
                        buffer[i*3+2] = (uchar)data[i*4+2];
                    }
                    break;
                default:
                    if(data != NULL)free(data);
                    if(buffer != NULL)free(buffer);
                    return HPC_OP_NOT_PERMITED;
            }
            HPCStatus_t st = write_Bmp_(name, buffer, numCols, numRows, nc_buffer);
            if(data != NULL) free(data);
            if(buffer != NULL)free(buffer);
            return st;

        }


    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t encodeJpeg(unsigned char **buffer, const Mat<T, nc, type> *m)
        {
#ifdef FASTCV_USE_JPEG
            if(!m->ptr())
                return HPC_POINTER_NULL;
            int numRows = m->height();
            int numCols = m->width();
            T* data = (T*)malloc(m->numElements() * nc * sizeof(T));
            if(NULL == data)
                return HPC_ALLOC_FAILED;
            ((Mat<T, nc>*)m)->toHost(data);
            uchar *m_buffer = (uchar*)malloc(numCols * numRows * nc *sizeof(T));
            if(NULL == m_buffer)
                return HPC_ALLOC_FAILED;
            for(int i = 0;i < numRows * numCols * nc; i++)
            {
                m_buffer[i] =(uchar)data[i];
            }
            HPCStatus_t st = encode_Jpeg_(buffer, m_buffer, numCols, numRows, nc);
            if(data != NULL)
                free(data);
            if(m_buffer != NULL) 
                free(m_buffer);
            return st;
#else
            return HPC_NOT_SUPPORTED;
#endif
        }
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t writeJpeg(const char* name, const Mat<T, nc, type> *m) {
#ifdef FASTCV_USE_JPEG
            if(!name || !m)
                return HPC_POINTER_NULL;

            int numRows = m->height();
            int numCols = m->width();

            T* data = (T*)malloc(m->numElements() * nc * sizeof(T));
            if(NULL == data) return HPC_ALLOC_FAILED;
            ((Mat<T, nc>*)m)->toHost(data);

            uchar *buffer = (uchar*)malloc(numCols * numRows * nc * sizeof(T));
            if(NULL == buffer) return HPC_ALLOC_FAILED;
            for(int i=0; i<numRows * numCols * nc; i++){
                buffer[i] = data[i];
            }

            HPCStatus_t st = write_Jpeg_(name, buffer, numCols, numRows, nc);
            if(data != NULL) free(data);
            if(buffer != NULL)free(buffer);
            return st;
#else
            return HPC_NOT_SUPPORTED;
#endif

        }
        

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t encodePng(unsigned char **buffer, const Mat<T, nc, type> *m, const std::vector<int> &params = std::vector<int>())
        {
#ifdef FASTCV_USE_PNG
            if(!m)
                return HPC_POINTER_NULL;
            int numRows = m -> height();
            int numCols = m -> width();
            printf("encodepng");
            T *data = (T*)malloc(m->numElements() * nc * sizeof(T));
            if(NULL == data)
                return HPC_ALLOC_FAILED;
            ((Mat<T, nc>*)m) -> toHost(data);
            HPCStatus_t st = encode_Png_<T ,nc>(buffer, data, numCols, numRows, params);
            if(data != NULL) free(data);
            return st;

#else
            return HPC_NOT_SUPPORTED;
#endif

        }
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t writePng(const char* name, const Mat<T, nc, type> *m) {
#ifdef FASTCV_USE_PNG
            if(!name || !m)
                return HPC_POINTER_NULL;

            int numRows = m->height();
            int numCols = m->width();

            T* data = (T*)malloc(m->numElements() * nc * sizeof(T));
            if(NULL == data) return HPC_ALLOC_FAILED;
            ((Mat<T, nc>*)m)->toHost(data);

            HPCStatus_t st = write_Png_<T, nc>(name, data, numCols, numRows);
            if(data != NULL) free(data);
            return st;

#else
            return HPC_NOT_SUPPORTED;
#endif

        }

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t writeTiff(const char* name, const Mat<T, nc, type> *m) {
            return HPC_NOT_IMPLEMENTED;

        }

    inline const std::string getFilenameExtension(const char* name) {
        std::string s(name);
        size_t pos = s.find_last_of(".");
        return s.substr(pos+1);
    }

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t inline loadImageWithExtension(const std::string& ext, const char* name, Mat<T, nc, type> *m) {
            if("bmp" == ext) {
                return loadBmp<T, nc>(name, m);
            } else if("jpg" == ext || "jpeg" == ext) {
                return loadJpeg<T, nc>(name, m);
            } else if("png" == ext) {
                return loadPng<T, nc>(name, m);
            } else if("tif" == ext || "tiff" == ext) {
                return loadTiff<T, nc>(name, m);
            } else {
                return HPC_NOT_IMPLEMENTED;
            }
        }

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t inline writeImageWithExtension(const std::string& ext, const char* name, const Mat<T, nc, type> *m) {
            if("bmp" == ext) {
                return writeBmp<T, nc>(name, m);
            } else if("jpg" == ext || "jpeg" == ext) {
                return writeJpeg<T, nc>(name, m);
            } else if("png" == ext) {
                return writePng<T, nc>(name, m);
            } else if("tif" == ext || "tiff" == ext) {
                return writeTiff<T, nc>(name, m);
            } else {
                return HPC_NOT_IMPLEMENTED;
            }
        }
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t inline encodeImageWithExtension(const std::string& ext, unsigned char **buffer, const Mat<T, nc, type> *m) {
            if("bmp" == ext) {
                return encodeBmp<T, nc>(buffer, m);
            } else if("jpg" == ext || "jpeg" == ext) {
                return encodeJpeg<T, nc>(buffer, m);
            } else if("png" == ext) {
                printf("pn....");
                return encodePng<T, nc>(buffer, m);
            } else {
                return HPC_NOT_IMPLEMENTED;
            }
        }

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t imread(const char* name, Mat<T, nc, type> *m) {
            const std::string& ext = getFilenameExtension(name);
            std::string str = ext;
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            return loadImageWithExtension<T, nc>(ext, name, m);
        }
    
    template<typename T,int nc, EcoEnv_t type>
        HPCStatus_t imdecode(unsigned char* databuffer,const unsigned int dataSize,Mat<T,nc,type> *m)
        {
            if(NULL == databuffer)
               return HPC_POINTER_NULL;
            unsigned char* currentbuffer = databuffer;
            unsigned char pngCheck[8] = { 0 };
            memcpy(pngCheck,currentbuffer,8);

#if defined(FASTCV_USE_JPEG)
            if((int)currentbuffer[0] == 255 && (int)currentbuffer[1] ==216 && (int)currentbuffer[2] == 255)
            {
                return imdecodeJpeg(databuffer,dataSize,m);
            }
#endif
#if defined(FASTCV_USE_PNG)
            if(!png_sig_cmp(pngCheck, 0, 8))
            {
               return imdecodePng(databuffer,dataSize,m);
            }
#endif
            if(*currentbuffer == 'B' && *(currentbuffer+1) == 'M')
            {
               return imdecodeBmp(databuffer,dataSize,m);
            }
            else
            {
                return HPC_NOT_IMPLEMENTED;
            }
       }

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t imwrite(const char* name, const Mat<T, nc, type> *m) {
            const std::string& ext = getFilenameExtension(name);
            std::string str = ext;
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            return writeImageWithExtension<T, nc>(ext, name, m);
        }
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t imencode(const char* ext, unsigned char ** buffer, const Mat<T, nc, type> *m)
        {
            printf("imencode");
            std::string str(ext);
            printf("imencode 22");
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            return encodeImageWithExtension<T, nc>(str, buffer, m);
        }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t imread<uchar, 1, EcoEnv_t::ECO_ENV_X86>(const char* name, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imread<uchar, 3, EcoEnv_t::ECO_ENV_X86>(const char* name, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imread<uchar, 4, EcoEnv_t::ECO_ENV_X86>(const char* name, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imread<float, 1, EcoEnv_t::ECO_ENV_X86>(const char* name, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imread<float, 3, EcoEnv_t::ECO_ENV_X86>(const char* name, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imread<float, 4, EcoEnv_t::ECO_ENV_X86>(const char* name, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *m);

    template HPCStatus_t imdecode<uchar, 1, EcoEnv_t::ECO_ENV_X86>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imdecode<uchar, 3, EcoEnv_t::ECO_ENV_X86>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imdecode<uchar, 4, EcoEnv_t::ECO_ENV_X86>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imdecode<float, 1, EcoEnv_t::ECO_ENV_X86>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imdecode<float, 3, EcoEnv_t::ECO_ENV_X86>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imdecode<float, 4, EcoEnv_t::ECO_ENV_X86>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *m);

    template HPCStatus_t imwrite<uchar, 1, EcoEnv_t::ECO_ENV_X86>(const char* name, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imwrite<uchar, 3, EcoEnv_t::ECO_ENV_X86>(const char* name, const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imwrite<uchar, 4, EcoEnv_t::ECO_ENV_X86>(const char* name, const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imwrite<float, 1, EcoEnv_t::ECO_ENV_X86>(const char* name, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imwrite<float, 3, EcoEnv_t::ECO_ENV_X86>(const char* name, const Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imwrite<float, 4, EcoEnv_t::ECO_ENV_X86>(const char* name, const Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *m);

    template HPCStatus_t imencode<uchar, 1, EcoEnv_t::ECO_ENV_X86>(const char* ext, unsigned char** buffer, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *m);

    template HPCStatus_t imencode<uchar, 3, EcoEnv_t::ECO_ENV_X86>(const char* ext, unsigned char** buffer, const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imencode<uchar, 4, EcoEnv_t::ECO_ENV_X86>(const char* ext, unsigned char** buffer, const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imencode<float, 1, EcoEnv_t::ECO_ENV_X86>(const char* ext, unsigned char** buffer, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imencode<float, 3, EcoEnv_t::ECO_ENV_X86>(const char* ext, unsigned char** buffer, const Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *m);
    template HPCStatus_t imencode<float, 4, EcoEnv_t::ECO_ENV_X86>(const char* ext, unsigned char** buffer, const Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *m);




#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t imread<uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const char* name, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imread<uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const char* name, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imread<uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const char* name, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imread<float, 1, EcoEnv_t::ECO_ENV_ARM>(const char* name, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imread<float, 3, EcoEnv_t::ECO_ENV_ARM>(const char* name, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imread<float, 4, EcoEnv_t::ECO_ENV_ARM>(const char* name, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *m);

    template HPCStatus_t imdecode<uchar, 1, EcoEnv_t::ECO_ENV_ARM>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imdecode<uchar, 3, EcoEnv_t::ECO_ENV_ARM>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imdecode<uchar, 4, EcoEnv_t::ECO_ENV_ARM>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imdecode<float, 1, EcoEnv_t::ECO_ENV_ARM>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imdecode<float, 3, EcoEnv_t::ECO_ENV_ARM>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imdecode<float, 4, EcoEnv_t::ECO_ENV_ARM>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *m);

    template HPCStatus_t imwrite<uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const char* name, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imwrite<uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const char* name, const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imwrite<uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const char* name, const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imwrite<float, 1, EcoEnv_t::ECO_ENV_ARM>(const char* name, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imwrite<float, 3, EcoEnv_t::ECO_ENV_ARM>(const char* name, const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imwrite<float, 4, EcoEnv_t::ECO_ENV_ARM>(const char* name, const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *m);
    
    template HPCStatus_t imencode<uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const char* ext, unsigned char** buffer, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imencode<uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const char* ext, unsigned char** buffer, const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *m)
    template HPCStatus_t imencode<uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const char* ext, unsigned char** buffer, const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imencode<float, 1, EcoEnv_t::ECO_ENV_ARM>(const char* ext, unsigned char** buffer, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imencode<float, 3, EcoEnv_t::ECO_ENV_ARM>(const char* ext, unsigned char** buffer, const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *m);
    template HPCStatus_t imencode<float, 4, EcoEnv_t::ECO_ENV_ARM>(const char* ext, unsigned char** buffer, const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *m);


#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t imread<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const char* name, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imread<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const char* name, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imread<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const char* name, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imread<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const char* name, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imread<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const char* name, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imread<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const char* name, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *m);

    template HPCStatus_t imdecode<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imdecode<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imdecode<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imdecode<float, 1, EcoEnv_t::ECO_ENV_CUDA>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imdecode<float, 3, EcoEnv_t::ECO_ENV_CUDA>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imdecode<float, 4, EcoEnv_t::ECO_ENV_CUDA>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *m);

    template HPCStatus_t imwrite<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const char* name, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imwrite<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const char* name, const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imwrite<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const char* name, const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imwrite<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const char* name, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imwrite<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const char* name, const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imwrite<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const char* name, const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *m);

    template HPCStatus_t imencode<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const char* ext, unsigned char** buffer, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imencode<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const char* ext, unsigned char** buffer, const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *m)
    template HPCStatus_t imencode<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const char* ext, unsigned char** buffer, const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imencode<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const char* ext, unsigned char** buffer, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imencode<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const char* ext, unsigned char** buffer, const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *m);
    template HPCStatus_t imencode<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const char* ext, unsigned char** buffer, const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *m);


#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t imread<uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const char* name, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imread<uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const char* name, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imread<uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const char* name, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imread<float, 1, EcoEnv_t::ECO_ENV_OCL>(const char* name, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imread<float, 3, EcoEnv_t::ECO_ENV_OCL>(const char* name, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imread<float, 4, EcoEnv_t::ECO_ENV_OCL>(const char* name, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *m);

    template HPCStatus_t imdecode<uchar, 1, EcoEnv_t::ECO_ENV_OCL>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imdecode<uchar, 3, EcoEnv_t::ECO_ENV_OCL>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imdecode<uchar, 4, EcoEnv_t::ECO_ENV_OCL>(unsigned char* databuffer,const unsigned int dataSize, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imdecode<float, 1, EcoEnv_t::ECO_ENV_OCL>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imdecode<float, 3, EcoEnv_t::ECO_ENV_OCL>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imdecode<float, 4, EcoEnv_t::ECO_ENV_OCL>(unsigned char* databuffer,const unsigned int dataSize, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *m);

    template HPCStatus_t imwrite<uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const char* name, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imwrite<uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const char* name, const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imwrite<uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const char* name, const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imwrite<float, 1, EcoEnv_t::ECO_ENV_OCL>(const char* name, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imwrite<float, 3, EcoEnv_t::ECO_ENV_OCL>(const char* name, const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imwrite<float, 4, EcoEnv_t::ECO_ENV_OCL>(const char* name, const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *m);

    template HPCStatus_t imencode<uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const char* ext, unsigned char** buffer, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imencode<uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const char* ext, unsigned char** buffer, const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *m)
    template HPCStatus_t imencode<uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const char* ext, unsigned char** buffer, const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imencode<float, 1, EcoEnv_t::ECO_ENV_OCL>(const char* ext, unsigned char** buffer, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imencode<float, 3, EcoEnv_t::ECO_ENV_OCL>(const char* ext, unsigned char** buffer, const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *m);
    template HPCStatus_t imencode<float, 4, EcoEnv_t::ECO_ENV_OCL>(const char* ext, unsigned char** buffer, const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *m);

#endif


     HPCStatus_t imdecode_Bmp_(unsigned char* databuffer,const unsigned int dataSize,unsigned char **m,int* numCols, int* numRows)
    {
#define pixel(A,x,y,c) (A[y*dx*3+x*3+c])
        if(!databuffer||!m)
            return HPC_POINTER_NULL;
        unsigned char * currentbuffer = databuffer;
        if(*currentbuffer != 'B' || *(currentbuffer+1) != 'M')
        {
            fprintf(stderr, "imdecode_bmp():Invalid BMP buffer data\n");
            return HPC_OP_NOT_PERMITED;
        }

        int file_size = currentbuffer[0x02] + (currentbuffer[0x03]<<8) + (currentbuffer[0x04]<<16) + (currentbuffer[0x05]<<24);
        int offset = currentbuffer[0x0A] + (currentbuffer[0x0B]<<8) + (currentbuffer[0x0C]<<16) + (currentbuffer[0x0D]<<24);
        int header_size = currentbuffer[0x0E] + (currentbuffer[0x0F]<<8) + (currentbuffer[0x10]<<16) + (currentbuffer[0x11]<<24);
        int dx = currentbuffer[0x12] + (currentbuffer[0x13]<<8) + (currentbuffer[0x14]<<16) + (currentbuffer[0x15]<<24);
        int dy = currentbuffer[0x16] + (currentbuffer[0x17]<<8) + (currentbuffer[0x18]<<16) + (currentbuffer[0x19]<<24);
        int compression = currentbuffer[0x1E] + (currentbuffer[0x1F]<<8) + (currentbuffer[0x20]<<16) + (currentbuffer[0x21]<<24);
        int nb_colors = currentbuffer[0x2E] + (currentbuffer[0x2F]<<8) + (currentbuffer[0x30]<<16) + (currentbuffer[0x31]<<24);
        int bpp = currentbuffer[0x1C] + (currentbuffer[0x1D]<<8);
        if(file_size != dataSize)
        {
            fprintf(stderr,"imdecode_bmp:buffer data is not Continuous\n");
        }
        currentbuffer += 54;
        if(header_size > 40)
           currentbuffer += header_size - 40;
        const int dx_bytes = (bpp==1)?(dx/8 + (dx%8?1:0)):((bpp==4)?(dx/2 + (dx%2?1:0)):(dx*bpp/8)),
                     align_bytes = (4 - dx_bytes%4)%4,
                     buf_size = std::min(abs(dy)*(dx_bytes + align_bytes), file_size - offset);
        
        //colormap
        int *colormap = NULL;
        if (bpp<16){
            if (!nb_colors) nb_colors = 1<<bpp;
        } else nb_colors = 0;
        if (nb_colors){
            colormap = (int*)malloc(nb_colors*sizeof(int));
            if(NULL == colormap){
            
                return HPC_ALLOC_FAILED;
            }
           
            memcpy(colormap,currentbuffer,nb_colors*sizeof(int));
            currentbuffer += nb_colors*sizeof(int);
        }
        const int xoffset = offset - 14 - header_size - sizeof(int)*nb_colors;
        if (xoffset>0) currentbuffer += xoffset;

        //buffer
        unsigned char *tmpbuffer;
        tmpbuffer = (unsigned char *)malloc(buf_size);
        if(NULL == tmpbuffer)
        {
            if(NULL != colormap)
               free(colormap);
            return HPC_ALLOC_FAILED;
        }
       memcpy(tmpbuffer,currentbuffer,buf_size);
        unsigned char * ptrs = tmpbuffer;
        if(compression)
        {
            if(databuffer)
               fprintf(stderr,"imdecode_bmp(): Unable to load compressed data");
        }

        unsigned char * img = *m = (unsigned char *)malloc(dx*dy*3*sizeof(unsigned char));
        if(NULL == img)
        {
            free(colormap);
            return HPC_ALLOC_FAILED;
        }
        switch(bpp)
        {
           case 1:   //Monochrome
           {
               for (int y = dy -1;y>=0;--y)
               {
                   unsigned char mask = 0x80, val = 0;
                   for(int x = 0;x<dx;++x)
                   {
                       if(mask == 0x80) val = *(ptrs++);
                       const unsigned char *col = (unsigned char *)(colormap +(val&mask?1:0));
                       pixel(img, x, y, 0) = *(col++);
                       pixel(img, x, y, 1) = *(col++);
                       pixel(img, x, y, 2) = *(col++);
                       mask = ror(mask);
                   }
                   ptrs += align_bytes;
               }
           
           }break;
           case 4:  //16 colors
           {
               for (int y = dy-1;y>0;--y)
               {
                   unsigned char mask = 0xF0, val = 0;
                   for(int x = 0;x<dx;++x)
                   {
                       if (mask==0xF0) val = *(ptrs++);
                       const unsigned char color = (unsigned char)((mask<16)?(val&mask):((val&mask)>>4));
                       const unsigned char *col = (unsigned char*)(colormap + color);
                       pixel(img, x, y, 0) = *(col++);
                       pixel(img, x, y, 1) = *(col++);
                       pixel(img, x, y, 2) = *(col++);
                       mask = ror(mask,4);
                   }
                   ptrs += align_bytes;
               }

           }break;
           case 8:  //256 colors
           {
                for (int y = dy - 1; y>=0; --y) 
                {
               
                    for(int x = 0; x < dx; x++)
                    {
                        const unsigned char *col = (unsigned char*)(colormap + *(ptrs++));
                        pixel(img, x, y, 0) = *(col++);
                        pixel(img, x, y, 1) = *(col++);
                        pixel(img, x, y, 2) = *(col++);
                    }
                    ptrs+=align_bytes;
                }

           }break;
           case 16: //16 bits colors
           {
               for (int y = dy - 1; y>=0; --y) 
               {                             
                    for(int x=0; x < dx; x++){
                        const unsigned char c1 = *(ptrs++), c2 = *(ptrs++);
                        const unsigned short col = (unsigned short)(c1|(c2<<8));
                        pixel(img, x, y, 0) = (col&0x1F);
                        pixel(img, x, y, 1) = ((col>>5)&0x1F);
                        pixel(img, x, y, 2) = ((col>>10)&0x1F);
                    }
                    ptrs+=align_bytes;
               }
           }break;
           case 24:  //24 bit colors
           {
               for (int y = dy - 1; y>=0; --y) 
               {
                    for(int x=0; x<dx; x++){
                        pixel(img, x, y, 0) = *(ptrs++);
                        pixel(img, x, y, 1) = *(ptrs++);
                        pixel(img, x, y, 2) = *(ptrs++);
                    }
                    ptrs+=align_bytes;
                }
           }break;
           case 32: //32 bit colors
           {
                for (int y = dy - 1; y>=0; --y) 
                {
                             
                    for(int x=0; x<dx; x++){
                        pixel(img, x, y, 0) = *(ptrs++);
                        pixel(img, x, y, 1) = *(ptrs++);
                        pixel(img, x, y, 2) = *(ptrs++);
                        ++ptrs;
                    }
                    ptrs+=align_bytes;
                }
           }break;

        }

        *numCols = dx;
        *numRows = dy;
        if(nb_colors) free(colormap);
        free(tmpbuffer);
        return HPC_SUCCESS;
#undef pixel   

    }


    HPCStatus_t load_Bmp_(const char* name, unsigned char** m, int* numCols, int* numRows) {
#define pixel(A, x, y, c) (A[y*dx*3+x*3+c])
        if(!name || !m)
            return HPC_POINTER_NULL;
        FILE *const nfile = fopen(name, "rb");
        unsigned char header[64] = {0};
        size_t readSize = fread(header, sizeof(unsigned char), 54, nfile);
        if(readSize == 0){
            fclose(nfile);
            return HPC_OP_NOT_PERMITED;
        }
        if (*header!='B' || header[1]!='M') {
            if (!name) fclose(nfile);
            fprintf(stderr, "load_bmp(): Invalid BMP file\n");
            return HPC_OP_NOT_PERMITED;
        }

        // Read header and pixel buffer
        int
            file_size = header[0x02] + (header[0x03]<<8) + (header[0x04]<<16) + (header[0x05]<<24),
                      offset = header[0x0A] + (header[0x0B]<<8) + (header[0x0C]<<16) + (header[0x0D]<<24),
                      header_size = header[0x0E] + (header[0x0F]<<8) + (header[0x10]<<16) + (header[0x11]<<24),
                      dx = header[0x12] + (header[0x13]<<8) + (header[0x14]<<16) + (header[0x15]<<24),
                      dy = header[0x16] + (header[0x17]<<8) + (header[0x18]<<16) + (header[0x19]<<24),
                      compression = header[0x1E] + (header[0x1F]<<8) + (header[0x20]<<16) + (header[0x21]<<24),
                      nb_colors = header[0x2E] + (header[0x2F]<<8) + (header[0x30]<<16) + (header[0x31]<<24),
                      bpp = header[0x1C] + (header[0x1D]<<8);

        if (!file_size || file_size==offset) {
            fseek(nfile,0,SEEK_END);
            file_size = (int)ftell(nfile);
            fseek(nfile,54,SEEK_SET);
        }
        if (header_size>40) fseek(nfile, header_size - 40, SEEK_CUR);

        const int
            iobuffer = 24*1024*1024,
                     dx_bytes = (bpp==1)?(dx/8 + (dx%8?1:0)):((bpp==4)?(dx/2 + (dx%2?1:0)):(dx*bpp/8)),
                     align_bytes = (4 - dx_bytes%4)%4,
                     buf_size = std::min(abs(dy)*(dx_bytes + align_bytes), file_size - offset);

        //colormap
        int *colormap = NULL;
        if (bpp<16){
            if (!nb_colors) nb_colors = 1<<bpp;
        } else nb_colors = 0;
        if (nb_colors){
            colormap = (int*)malloc(nb_colors*sizeof(int));
            if(NULL == colormap){
                fclose(nfile);
                return HPC_ALLOC_FAILED;
            }
            readSize = fread(colormap, sizeof(int), nb_colors, nfile);
        }
        const int xoffset = offset - 14 - header_size - 4*nb_colors;
        if (xoffset>0) fseek(nfile, xoffset, SEEK_CUR);

        //buffer
        unsigned char *buffer;
        if (buf_size<iobuffer){
            buffer = (unsigned char*)malloc(buf_size);
            readSize = fread(buffer, sizeof(unsigned char), buf_size, nfile);
        }
        else buffer = (unsigned char*)malloc((dx_bytes + align_bytes) * sizeof(unsigned char));
        if(NULL == buffer){
            fclose(nfile);
            if(NULL != colormap)
                free(colormap);
            return HPC_ALLOC_FAILED;
        }
        unsigned char *ptrs = buffer;

        // Decompress buffer (if necessary)
        if (compression) {
            if (name)
                fprintf(stderr, "load_bmp(): Unable to load compressed data ");
            else {
                if (!name) fclose(nfile);
            }
        }

        // Read pixel data
        unsigned char* img = *m = (unsigned char*)malloc(dx*dy*3*sizeof(unsigned char));
        if(NULL == img){
            fclose(nfile);
            free(colormap);
            return HPC_ALLOC_FAILED;
        }
        switch (bpp) {
            case 1 : { // Monochrome
                         for (int y = dy - 1; y>=0; --y) {
                             if (buf_size>=iobuffer) {
                                 readSize = fread(ptrs=buffer, sizeof(unsigned char), dx_bytes, nfile);
                                 fseek(nfile, align_bytes, SEEK_CUR);
                             }
                             unsigned char mask = 0x80, val = 0;
                             for(int x = 0; x < dx; x++ ){
                                 if (mask==0x80) val = *(ptrs++);
                                 const unsigned char *col = (unsigned char*)(colormap + (val&mask?1:0));
                                 pixel(img, x, y, 0) = *(col++);
                                 pixel(img, x, y, 1) = *(col++);
                                 pixel(img, x, y, 2) = *(col++);
                                 mask = ror(mask);
                             }
                             ptrs+=align_bytes;
                         }
                     } break;
            case 4 : { // 16 colors
                         for (int y = dy - 1; y>=0; --y) {
                             if (buf_size>=iobuffer) {
                                 readSize = fread(ptrs=buffer, sizeof(unsigned char), dx_bytes, nfile);
                                 fseek(nfile,align_bytes,SEEK_CUR);
                             }
                             unsigned char mask = 0xF0, val = 0;
                             for(int x = 0; x < dx; x++){
                                 if (mask==0xF0) val = *(ptrs++);
                                 const unsigned char color = (unsigned char)((mask<16)?(val&mask):((val&mask)>>4));
                                 const unsigned char *col = (unsigned char*)(colormap + color);
                                 pixel(img, x, y, 0) = *(col++);
                                 pixel(img, x, y, 1) = *(col++);
                                 pixel(img, x, y, 2) = *(col++);
                                 mask = ror(mask,4);
                             }
                             ptrs+=align_bytes;
                         }
                     } break;
            case 8 : { //  256 colors
                         for (int y = dy - 1; y>=0; --y) {
                             if (buf_size>=iobuffer) {
                                 readSize = fread(ptrs=buffer, sizeof(unsigned char), dx_bytes, nfile);
                                 fseek(nfile,align_bytes,SEEK_CUR);
                             }
                             for(int x = 0; x < dx; x++){
                                 const unsigned char *col = (unsigned char*)(colormap + *(ptrs++));
                                 pixel(img, x, y, 0) = *(col++);
                                 pixel(img, x, y, 1) = *(col++);
                                 pixel(img, x, y, 2) = *(col++);
                             }
                             ptrs+=align_bytes;
                         }
                     } break;
            case 16 : { // 16 bits colors
                          for (int y = dy - 1; y>=0; --y) {
                              if (buf_size>=iobuffer) {
                                  readSize = fread(ptrs=buffer, sizeof(unsigned char), dx_bytes,nfile);
                                  fseek(nfile, align_bytes, SEEK_CUR);
                              }
                              for(int x=0; x < dx; x++){
                                  const unsigned char c1 = *(ptrs++), c2 = *(ptrs++);
                                  const unsigned short col = (unsigned short)(c1|(c2<<8));
                                  pixel(img, x, y, 0) = (col&0x1F);
                                  pixel(img, x, y, 1) = ((col>>5)&0x1F);
                                  pixel(img, x, y, 2) = ((col>>10)&0x1F);
                              }
                              ptrs+=align_bytes;
                          }
                      } break;
            case 24 : { // 24 bits colors
                          for (int y = dy - 1; y>=0; --y) {
                              if (buf_size>=iobuffer) {
                                  readSize = fread(ptrs=buffer, sizeof(unsigned char), dx_bytes,nfile);
                                  fseek(nfile,align_bytes,SEEK_CUR);
                              }
                              for(int x=0; x<dx; x++){
                                  pixel(img, x, y, 0) = *(ptrs++);
                                  pixel(img, x, y, 1) = *(ptrs++);
                                  pixel(img, x, y, 2) = *(ptrs++);
                              }
                              ptrs+=align_bytes;
                          }
                      } break;
            case 32 : { // 32 bits colors
                          for (int y = dy - 1; y>=0; --y) {
                              if (buf_size>=iobuffer) {
                                  readSize = fread(ptrs=buffer, sizeof(unsigned char), dx_bytes,nfile);
                                  fseek(nfile, align_bytes, SEEK_CUR);
                              }
                              for(int x=0; x<dx; x++){
                                  pixel(img, x, y, 0) = *(ptrs++);
                                  pixel(img, x, y, 1) = *(ptrs++);
                                  pixel(img, x, y, 2) = *(ptrs++);
                                  ++ptrs;
                              }
                              ptrs+=align_bytes;
                          }
                      } break;
        }
        *numCols = dx;
        *numRows = dy;
        //if (dy<0) mirror('y');
        if (!nfile) fclose(nfile);
        if (nb_colors) free(colormap);
        free(buffer);
        return HPC_SUCCESS;
#undef pixel
    }

#ifdef FASTCV_USE_JPEG
    // Custom error handler for libjpeg.
    struct _fastcv_error_mgr {
      struct jpeg_error_mgr original;
      jmp_buf setjmp_buffer;
      char message[JMSG_LENGTH_MAX];
    };



    typedef struct _fastcv_error_mgr *_fastcv_error_ptr;

    METHODDEF(void) _fastcv_jpeg_error_exit(j_common_ptr cinfo) {
      _fastcv_error_ptr c_err = (_fastcv_error_ptr) cinfo->err;  // Return control to the setjmp point
      (*cinfo->err->format_message)(cinfo,c_err->message);
      jpeg_destroy(cinfo);  // Clean memory and temp files.
      longjmp(c_err->setjmp_buffer,1);
    }


    HPCStatus_t imdecode_Jpeg_(unsigned char* buff,const unsigned int dataSize,unsigned char** m,int* numCols,int *numRows,int* numChannels)
    {
        if(!buff)
        {
            fprintf(stderr,"imdecode_Jpeg_(): Specified buff pointer is (null)\n");
            return HPC_POINTER_NULL;
        }
        struct jpeg_decompress_struct cinfo;
        struct _fastcv_error_mgr jerr;
        cinfo.err = jpeg_std_error(&jerr.original);

        jerr.original.error_exit = _fastcv_jpeg_error_exit;
        if(setjmp(jerr.setjmp_buffer))
        {
            fprintf(stderr,"imdecode_Jpeg_():Error message returned by libjpeg: %s\n",jerr.message);
            return HPC_OP_NOT_PERMITED;
        }
        jpeg_create_decompress(&cinfo);

        jpeg_mem_src(&cinfo,buff,dataSize);
        jpeg_read_header(&cinfo,TRUE);
        jpeg_start_decompress(&cinfo);
        
        if (cinfo.output_components!=1 && cinfo.output_components!=3 && cinfo.output_components!=4) 
        {
            fprintf(stderr, "imdecode_Jpeg_(): Failed to load JPEG data from memory buff\n");           
            return HPC_OP_NOT_PERMITED;
        }
        int dx=cinfo.output_width;
        int dy=cinfo.output_height;
        int nc=cinfo.output_components;
        //CImg<ucharT> buffer(cinfo.output_width*cinfo.output_components);
        unsigned char* img = *m = (unsigned char*)malloc(dx*dy*nc*sizeof(unsigned char));
        if(NULL == img) return HPC_ALLOC_FAILED;
        unsigned char* buffer = (unsigned char*)malloc(dx * nc * sizeof(unsigned char));
        if(NULL == buffer) return HPC_ALLOC_FAILED;

        JSAMPROW row_pointer[1];
        while (cinfo.output_scanline<cinfo.output_height) 
        {
            *row_pointer = buffer;
            if (jpeg_read_scanlines(&cinfo,row_pointer,1)!=1) 
            {
                fprintf(stderr, "imdecode_Jpeg_(): Incomplete data .\n");
                break;
            }
            const unsigned char *ptrs = buffer;
            switch (nc) {
                case 1 : {
                             for(int x=0; x<dx; x++){
                                 *(img++) = *(ptrs++);
                             }
                         } break;
                case 3 : {
                             for(int x=0; x<dx; x++){
                                 img[2] = *(ptrs++);
                                 img[1] = *(ptrs++);
                                 img[0] = *(ptrs++);
                                 img += 3;
                             }
                         } break;
                case 4 : {
                             for(int x=0; x<dx; x++) {
                                 img[2] = *(ptrs++);
                                 img[1] = *(ptrs++);
                                 img[0] = *(ptrs++);
                                 img[3] = *(ptrs++);
                                 img += 4;
                             }
                         } break;
            }
        }
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        *numCols = dx;
        *numRows = dy;
        *numChannels = nc;
       
        return HPC_SUCCESS;


    }



    HPCStatus_t load_Jpeg_(const char* name, unsigned char** m, int* numCols, int* numRows, int* numChannels){

        if (!name){
            fprintf(stderr, "load_jpeg(): Specified filename is (null).\n");
            return HPC_POINTER_NULL;
        }

        struct jpeg_decompress_struct cinfo;
        struct _fastcv_error_mgr jerr;
        cinfo.err = jpeg_std_error(&jerr.original);
        jerr.original.error_exit = _fastcv_jpeg_error_exit;

        if (setjmp(jerr.setjmp_buffer)) { // JPEG error
            fprintf(stderr, "load_jpeg(): Error message returned by libjpeg: %s\n", jerr.message);
            return HPC_OP_NOT_PERMITED;
        }

        FILE *const nfile = fopen(name,"rb");
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, nfile);
        jpeg_read_header(&cinfo, TRUE);
        jpeg_start_decompress(&cinfo);
        
        if (cinfo.output_components!=1 && cinfo.output_components!=3 && cinfo.output_components!=4) {
            fprintf(stderr, "load_jpeg(): Failed to load JPEG data from file\n");
            if(!nfile)
                fclose(nfile);
            return HPC_OP_NOT_PERMITED;
        }
        int dx=cinfo.output_width;
        int dy=cinfo.output_height;
        int nc=cinfo.output_components;
        //CImg<ucharT> buffer(cinfo.output_width*cinfo.output_components);
        unsigned char* img = *m = (unsigned char*)malloc(dx*dy*nc*sizeof(unsigned char));
        if(NULL == img) return HPC_ALLOC_FAILED;
        unsigned char* buffer = (unsigned char*)malloc(dx * nc * sizeof(unsigned char));
        if(NULL == buffer) return HPC_ALLOC_FAILED;

        JSAMPROW row_pointer[1];
        while (cinfo.output_scanline<cinfo.output_height) {
            *row_pointer = buffer;
            if (jpeg_read_scanlines(&cinfo,row_pointer,1)!=1) {
                fprintf(stderr, "load_jpeg(): Incomplete data in file.\n");
                break;
            }
            const unsigned char *ptrs = buffer;
            switch (nc) {
                case 1 : {
                             for(int x=0; x<dx; x++){
                                 *(img++) = *(ptrs++);
                             }
                         } break;
                case 3 : {
                             for(int x=0; x<dx; x++){
                                 img[2] = *(ptrs++);
                                 img[1] = *(ptrs++);
                                 img[0] = *(ptrs++);
                                 img += 3;
                             }
                         } break;
                case 4 : {
                             for(int x=0; x<dx; x++) {
                                 img[2] = *(ptrs++);
                                 img[1] = *(ptrs++);
                                 img[0] = *(ptrs++);
                                 img[3] = *(ptrs++);
                                 img += 4;
                             }
                         } break;
            }
        }
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        *numCols = dx;
        *numRows = dy;
        *numChannels = nc;
        if (!nfile) fclose(nfile);
        return HPC_SUCCESS;
    }
#endif

#ifdef FASTCV_USE_PNG

    struct ImageSource
    {
        unsigned char *data;
        int size;
        int offset;
        
    };

    static void pngReadCallback(png_structp png_ptr,png_bytep data,png_size_t length)
    {
        struct ImageSource *isource = (struct ImageSource*)png_get_io_ptr(png_ptr);
        if(isource->offset  + length <= isource ->size)
        {
            memcpy(data,isource->data+isource->offset,length);
            isource ->offset += length;
        }
        else 
        {
            png_error(png_ptr,"imdecode_Png_():Png input buffer is incomplete.\n");
        }
    }

    HPCStatus_t imdecode_Png_(unsigned char* buffer ,const unsigned int dataSize,void **m,int* numCols,int* numRows,FASTCV_PNG_DEPTH *png_depth)
    {
        if(!buffer)
        {
            fprintf(stderr,"imdecode_Png_():input data buffer is null.\n");
            return HPC_POINTER_NULL;
        }

         // Setup PNG structures for read
        png_voidp user_error_ptr = 0;
        png_error_ptr user_error_fn = 0, user_warning_fn = 0;
        png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,user_error_ptr,user_error_fn,user_warning_fn);
        if (!png_ptr) 
        {
            fprintf(stderr, "imdecode_Png_(): Failed to initialize 'png_ptr' structure .\n" );
            return HPC_OP_NOT_PERMITED;
        }
        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) 
        {           
            png_destroy_read_struct(&png_ptr,(png_infopp)0,(png_infopp)0);
            fprintf(stderr, "imdecode_Png_(): Failed to initialize 'info_ptr' structure.\n");
            return HPC_OP_NOT_PERMITED;
        }
        png_infop end_info = png_create_info_struct(png_ptr);
        if (!end_info) 
        {
            png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)0);
            fprintf(stderr, "imdecode_Png_(): Failed to initialize 'end_info' structure.\n");
            return HPC_OP_NOT_PERMITED;
        }

        // Error handling callback for png file reading
        if (setjmp(png_jmpbuf(png_ptr))) 
        {           
            png_destroy_read_struct(&png_ptr, &end_info, (png_infopp)0);
            fprintf(stderr, "imdecode_Png_(): Encountered unknown fatal error in libpng.\n");
            return HPC_OP_NOT_PERMITED;
        }
        struct ImageSource imgsource;
        imgsource.data = buffer;
        imgsource.size = dataSize;
        imgsource.offset = 0;
        png_set_read_fn(png_ptr,&imgsource,pngReadCallback);

         // Get PNG Header Info up to data block
        png_read_info(png_ptr,info_ptr);
        png_uint_32 W, H;
        int bit_depth, color_type, interlace_type;
        bool is_gray = false;
        png_get_IHDR(png_ptr,info_ptr,&W,&H,&bit_depth,&color_type,&interlace_type,(int*)0,(int*)0);

        // Transforms to unify image data
        if (color_type==PNG_COLOR_TYPE_PALETTE) 
        {
            png_set_palette_to_rgb(png_ptr);
            color_type = PNG_COLOR_TYPE_RGB;
            bit_depth = 8;
        }
        if (color_type==PNG_COLOR_TYPE_GRAY && bit_depth<8) 
        {
            png_set_expand_gray_1_2_4_to_8(png_ptr);
            is_gray = true;
            bit_depth = 8;
        }
        if (png_get_valid(png_ptr,info_ptr,PNG_INFO_tRNS))
        {
            png_set_tRNS_to_alpha(png_ptr);
            color_type |= PNG_COLOR_MASK_ALPHA;
        }
        if (color_type==PNG_COLOR_TYPE_GRAY || color_type==PNG_COLOR_TYPE_GRAY_ALPHA) 
        {
            png_set_gray_to_rgb(png_ptr);
            color_type |= PNG_COLOR_MASK_COLOR;
            is_gray = true;
        }
        if (color_type==PNG_COLOR_TYPE_RGB)
            png_set_filler(png_ptr, 0xffffU, PNG_FILLER_AFTER);
        png_set_interlace_handling(png_ptr);
        png_read_update_info(png_ptr, info_ptr);
        if (bit_depth!=8 && bit_depth!=16) 
        {          
            png_destroy_read_struct(&png_ptr, &end_info,(png_infopp)0);
            fprintf(stderr, "imdecode_Png_(): Invalid bit depth %u in file.\n", bit_depth);
            return HPC_OP_NOT_PERMITED;
        }
        const int byte_depth = bit_depth>>3;

        // Allocate Memory for Image Read
        png_bytep *const imgData = new png_bytep[H];
        for (unsigned int row = 0; row<H; ++row)
            imgData[row] = new png_byte[byte_depth*4*W];
        png_read_image(png_ptr,imgData);
        png_read_end(png_ptr,end_info);

        // Read pixel data
        if (color_type!=PNG_COLOR_TYPE_RGB && color_type!=PNG_COLOR_TYPE_RGB_ALPHA) 
        {            
            png_destroy_read_struct(&png_ptr,&end_info,(png_infopp)0);
            fprintf(stderr, "imdecode_Png_(): Invalid color coding type %u.\n", color_type);
            return HPC_OP_NOT_PERMITED;
        }
        const bool is_alpha = (color_type==PNG_COLOR_TYPE_RGBA);
        *m = (void*)malloc(W * H * ((is_gray?1:3)+(is_alpha?1:0)) * byte_depth);
        if(NULL == *m) return HPC_ALLOC_FAILED;
        switch (bit_depth) {
            case 8 : {
                         unsigned char *img = (unsigned char*)(*m);
                         if(is_gray){
                             if(is_alpha){
                                 *png_depth = FASTCV_PNG_8UC2;
                             }else{
                                 *png_depth = FASTCV_PNG_8UC1;
                             }
                         }else{
                             if(is_alpha){
                                 *png_depth = FASTCV_PNG_8UC4;
                             }else{
                                 *png_depth = FASTCV_PNG_8UC3;
                             }
                         }
                         int img_nc = (int)(*png_depth) % 10;
                         for(unsigned int y=0; y<H; y++) {
                             const unsigned char *ptrs = (unsigned char*)imgData[y];
                             for(unsigned int x=0; x<W; x++) {
                                 switch (*png_depth){
                                     case FASTCV_PNG_8UC1:
                                         {
                                             img[0] = ptrs[0];
                                         }break;
                                     case FASTCV_PNG_8UC2:
                                         {
                                             img[0] = ptrs[0];
                                             img[1] = ptrs[3];
                                         }break;
                                     case FASTCV_PNG_8UC3:
                                         {
                                             img[0] = ptrs[2];
                                             img[1] = ptrs[1];
                                             img[2] = ptrs[0];
                                         }break;
                                     case FASTCV_PNG_8UC4:
                                         {
                                             img[0] = ptrs[2];
                                             img[1] = ptrs[1];
                                             img[2] = ptrs[0];
                                             img[3] = ptrs[3];
                                         }break;
                                     default:
                                         break;
                                 }
                                 img += img_nc;
                                 ptrs += 4;
                             }
                         }
                     } break;
            case 16 : {
                         unsigned short *img = (unsigned short*)(*m);
                         if(is_gray){
                             if(is_alpha){
                                 *png_depth = FASTCV_PNG_16UC2;
                             }else{
                                 *png_depth = FASTCV_PNG_16UC1;
                             }
                         }else{
                             if(is_alpha){
                                 *png_depth = FASTCV_PNG_16UC4;
                             }else{
                                 *png_depth = FASTCV_PNG_16UC3;
                             }
                         }
                         int img_nc = (int)(*png_depth) % 10;
                         for(unsigned int y=0; y<H; y++) {
                             const unsigned short *ptrs = (unsigned short*)imgData[y];
                             for(unsigned int x=0; x<W; x++) {
                                 switch (*png_depth){
                                     case FASTCV_PNG_16UC1:
                                         {
                                             img[0] = ptrs[0];
                                         }break;
                                     case FASTCV_PNG_16UC2:
                                         {
                                             img[0] = ptrs[0];
                                             img[1] = ptrs[3];
                                         }break;
                                     case FASTCV_PNG_16UC3:
                                         {
                                             img[0] = ptrs[2];
                                             img[1] = ptrs[1];
                                             img[2] = ptrs[0];
                                         }break;
                                     case FASTCV_PNG_16UC4:
                                         {
                                             img[0] = ptrs[2];
                                             img[1] = ptrs[1];
                                             img[2] = ptrs[0];
                                             img[3] = ptrs[3];
                                         }break;
                                     default:
                                         break;
                                 }
                                 img += img_nc;
                                 ptrs += 4;
                             }
                         }
                     } break;
        }
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        *numCols = W;
        *numRows = H;

        for(unsigned int n=0; n<H; n++)
            delete[] imgData[n];
        delete[] imgData;
      
        return HPC_SUCCESS;

    }

    HPCStatus_t load_Png_(const char* name, void** m, int* numCols, int* numRows, FASTCV_PNG_DEPTH *png_depth){
        if (!name){
            fprintf(stderr, "load_png(): Specified filename is (null).\n");
            return HPC_POINTER_NULL;
        }

        const char *volatile nfilename = name;
        FILE *volatile nfile = fopen(nfilename, "rb");

        unsigned char pngCheck[8] = { 0 };
        int readSize = fread(pngCheck, sizeof(unsigned char), 8, (FILE*)nfile);
        if(readSize == 0){
            fclose(nfile);
            return HPC_OP_NOT_PERMITED;
        }
        if (png_sig_cmp(pngCheck, 0, 8)) {
            fclose(nfile);
            fprintf(stderr, "load_png(): Invalid PNG file.\n");
            return HPC_OP_NOT_PERMITED;
        }

        // Setup PNG structures for read
        png_voidp user_error_ptr = 0;
        png_error_ptr user_error_fn = 0, user_warning_fn = 0;
        png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,user_error_ptr,user_error_fn,user_warning_fn);
        if (!png_ptr) {
            fclose(nfile);
            fprintf(stderr, "load_png(): Failed to initialize 'png_ptr' structure .\n" );
            return HPC_OP_NOT_PERMITED;
        }
        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) {
            fclose(nfile);
            png_destroy_read_struct(&png_ptr,(png_infopp)0,(png_infopp)0);
            fprintf(stderr, "load_png(): Failed to initialize 'info_ptr' structure.\n");
            return HPC_OP_NOT_PERMITED;
        }
        png_infop end_info = png_create_info_struct(png_ptr);
        if (!end_info) {
            fclose(nfile);
            png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)0);
            fprintf(stderr, "load_png(): Failed to initialize 'end_info' structure.\n");
            return HPC_OP_NOT_PERMITED;
        }

        // Error handling callback for png file reading
        if (setjmp(png_jmpbuf(png_ptr))) {
            fclose(nfile);
            png_destroy_read_struct(&png_ptr, &end_info, (png_infopp)0);
            fprintf(stderr, "load_png(): Encountered unknown fatal error in libpng.\n");
            return HPC_OP_NOT_PERMITED;
        }
        png_init_io(png_ptr, nfile);
        png_set_sig_bytes(png_ptr, 8);

        // Get PNG Header Info up to data block
        png_read_info(png_ptr,info_ptr);
        png_uint_32 W, H;
        int bit_depth, color_type, interlace_type;
        bool is_gray = false;
        png_get_IHDR(png_ptr,info_ptr,&W,&H,&bit_depth,&color_type,&interlace_type,(int*)0,(int*)0);

        // Transforms to unify image data
        if (color_type==PNG_COLOR_TYPE_PALETTE) {
            png_set_palette_to_rgb(png_ptr);
            color_type = PNG_COLOR_TYPE_RGB;
            bit_depth = 8;
        }
        if (color_type==PNG_COLOR_TYPE_GRAY && bit_depth<8) {
            png_set_expand_gray_1_2_4_to_8(png_ptr);
            is_gray = true;
            bit_depth = 8;
        }
        if (png_get_valid(png_ptr,info_ptr,PNG_INFO_tRNS)) {
            png_set_tRNS_to_alpha(png_ptr);
            color_type |= PNG_COLOR_MASK_ALPHA;
        }
        if (color_type==PNG_COLOR_TYPE_GRAY || color_type==PNG_COLOR_TYPE_GRAY_ALPHA) {
            png_set_gray_to_rgb(png_ptr);
            color_type |= PNG_COLOR_MASK_COLOR;
            is_gray = true;
        }
        if (color_type==PNG_COLOR_TYPE_RGB)
            png_set_filler(png_ptr, 0xffffU, PNG_FILLER_AFTER);

        png_read_update_info(png_ptr, info_ptr);
        if (bit_depth!=8 && bit_depth!=16) {
            fclose(nfile);
            png_destroy_read_struct(&png_ptr, &end_info,(png_infopp)0);
            fprintf(stderr, "load_png(): Invalid bit depth %u in file.\n", bit_depth);
            return HPC_OP_NOT_PERMITED;
        }
        const int byte_depth = bit_depth>>3;

        // Allocate Memory for Image Read
        png_bytep *const imgData = new png_bytep[H];
        for (unsigned int row = 0; row<H; ++row)
            imgData[row] = new png_byte[byte_depth*4*W];
        png_read_image(png_ptr,imgData);
        png_read_end(png_ptr,end_info);

        // Read pixel data
        if (color_type!=PNG_COLOR_TYPE_RGB && color_type!=PNG_COLOR_TYPE_RGB_ALPHA) {
            fclose(nfile);
            png_destroy_read_struct(&png_ptr,&end_info,(png_infopp)0);
            fprintf(stderr, "load_png(): Invalid color coding type %u.\n", color_type);
            return HPC_OP_NOT_PERMITED;
        }
        const bool is_alpha = (color_type==PNG_COLOR_TYPE_RGBA);
        *m = (void*)malloc(W * H * ((is_gray?1:3)+(is_alpha?1:0)) * byte_depth);
        if(NULL == *m) return HPC_ALLOC_FAILED;
        switch (bit_depth) {
            case 8 : {
                         unsigned char *img = (unsigned char*)(*m);
                         if(is_gray){
                             if(is_alpha){
                                 *png_depth = FASTCV_PNG_8UC2;
                             }else{
                                 *png_depth = FASTCV_PNG_8UC1;
                             }
                         }else{
                             if(is_alpha){
                                 *png_depth = FASTCV_PNG_8UC4;
                             }else{
                                 *png_depth = FASTCV_PNG_8UC3;
                             }
                         }
                         int img_nc = (int)(*png_depth) % 10;
                         for(unsigned int y=0; y<H; y++) {
                             const unsigned char *ptrs = (unsigned char*)imgData[y];
                             for(unsigned int x=0; x<W; x++) {
                                 switch (*png_depth){
                                     case FASTCV_PNG_8UC1:
                                         {
                                             img[0] = ptrs[0];
                                         }break;
                                     case FASTCV_PNG_8UC2:
                                         {
                                             img[0] = ptrs[0];
                                             img[1] = ptrs[3];
                                         }break;
                                     case FASTCV_PNG_8UC3:
                                         {
                                             img[0] = ptrs[2];
                                             img[1] = ptrs[1];
                                             img[2] = ptrs[0];
                                         }break;
                                     case FASTCV_PNG_8UC4:
                                         {
                                             img[0] = ptrs[2];
                                             img[1] = ptrs[1];
                                             img[2] = ptrs[0];
                                             img[3] = ptrs[3];
                                         }break;
                                     default:
                                         break;
                                 }
                                 img += img_nc;
                                 ptrs += 4;
                             }
                         }
                     } break;
            case 16 : {
                         unsigned short *img = (unsigned short*)(*m);
                         if(is_gray){
                             if(is_alpha){
                                 *png_depth = FASTCV_PNG_16UC2;
                             }else{
                                 *png_depth = FASTCV_PNG_16UC1;
                             }
                         }else{
                             if(is_alpha){
                                 *png_depth = FASTCV_PNG_16UC4;
                             }else{
                                 *png_depth = FASTCV_PNG_16UC3;
                             }
                         }
                         int img_nc = (int)(*png_depth) % 10;
                         for(unsigned int y=0; y<H; y++) {
                             const unsigned short *ptrs = (unsigned short*)imgData[y];
                             for(unsigned int x=0; x<W; x++) {
                                 switch (*png_depth){
                                     case FASTCV_PNG_16UC1:
                                         {
                                             img[0] = ptrs[0];
                                         }break;
                                     case FASTCV_PNG_16UC2:
                                         {
                                             img[0] = ptrs[0];
                                             img[1] = ptrs[3];
                                         }break;
                                     case FASTCV_PNG_16UC3:
                                         {
                                             img[0] = ptrs[2];
                                             img[1] = ptrs[1];
                                             img[2] = ptrs[0];
                                         }break;
                                     case FASTCV_PNG_16UC4:
                                         {
                                             img[0] = ptrs[2];
                                             img[1] = ptrs[1];
                                             img[2] = ptrs[0];
                                             img[3] = ptrs[3];
                                         }break;
                                     default:
                                         break;
                                 }
                                 img += img_nc;
                                 ptrs += 4;
                             }
                         }
                     } break;
        }
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        *numCols = W;
        *numRows = H;

        for(unsigned int n=0; n<H; n++)
            delete[] imgData[n];
        delete[] imgData;
        fclose(nfile);
        return HPC_SUCCESS;


    }
#endif

    void FillGrayPalette(PaletteEntry* palette, int bpp, bool negative)
    {
        int i,length = 1 << bpp;
        int xor_mask = negative ? 255 : 0;
        for(i = 0; i < length; i++)
        {
            int val = (i*255/(length - 1)) ^ xor_mask;
            palette[i].b = palette[i].g = palette[i].r = (uchar)val;
            palette[i].a = 0;
        }
    }
    HPCStatus_t encode_Bmp_(unsigned char **buffer, unsigned char* m_buffer, int numCols, int numRows, int channels)
    {
        int width = numCols, height = numRows;
        int fileStep = (width*channels + 3)&-4;
        uchar zeropad[] = "\0\0\0\0";
        int bitmapHeaderSize = 40;
        int paletteSize = channels >1 ? 0 : 1024;
        int headerSize = 14 + bitmapHeaderSize + paletteSize;
        int fileSize = fileStep*height + headerSize;
        PaletteEntry palette[256];
        (*buffer) = (unsigned char *)malloc(fileSize);
        unsigned char * current = (*buffer);
        unsigned char header[54] = {0};
        header[0] = 'B';
        header[1] = 'M';
        header[0x02] = fileSize & 0xFF;
        header[0x03] = (fileSize >> 8) & 0xFF;
        header[0x04] = (fileSize >> 16) & 0xFF;
        header[0x05] = (fileSize >> 24) & 0xFF;
        header[0x0A] = headerSize & 0xFF;
        header[0x0B] = (headerSize >> 8) & 0xFF;
        header[0x0C] = (headerSize >> 16) & 0xFF;
        header[0x0D] = (headerSize >> 24) & 0xFF;
        header[0x0E] = 0x28;
        header[0x12] = width & 0xFF;
        header[0x13] = (width >> 8) & 0xFF;
        header[0x14] = (width >> 16) & 0xFF;
        header[0x15] = (width >> 24) & 0xFF;
        header[0x16] = height & 0xFF;
        header[0x17] = (height >> 8) & 0xFF;
        header[0x18] = (height >> 16) & 0xFF;
        header[0x19] = (height >> 24) & 0xFF;
        header[0x1A] = 1;
        header[0x1B] = 0;
        header[0x1C] = channels << 3;
        header[0x1D] = 0;

        memcpy(current, header, 54);
        current += 54;
        if(channels == 1)
        {
            FillGrayPalette(palette, 8, false);
            memcpy(current, palette, sizeof(palette));
            current += sizeof(palette);
        }
        for(int y = height - 1; y >= 0; y--)
        {
            memcpy(current, m_buffer + y*channels*width, width*channels);
            current += width*channels;
            if(fileStep > width*channels)
            {
                int padding_size = fileStep - width*channels;
                memcpy(current, zeropad,padding_size);
                current += padding_size;

            }

        }

        return HPC_SUCCESS;

    }
    HPCStatus_t write_Bmp_(const char* name, unsigned char* buffer, int numCols, int numRows, int nc_buffer){
        if (!name || !buffer){
            fprintf(stderr, "write_bmp(): Specified filename is (null).\n");
            return HPC_POINTER_NULL;
        }

        FILE *const nfile = fopen(name, "wb");
        unsigned char header[54] = { 0 }, align_buf[4] = { 0 };
        unsigned int _width = numCols;
        unsigned int _height = numRows;
        const unsigned int
            align = (4 - (3 * _width)%4)%4,
            buf_size = (3 * _width + align) * _height,
            file_size = 54 + buf_size;
        header[0] = 'B'; header[1] = 'M';
        header[0x02] = file_size&0xFF;
        header[0x03] = (file_size>>8)&0xFF;
        header[0x04] = (file_size>>16)&0xFF;
        header[0x05] = (file_size>>24)&0xFF;
        header[0x0A] = 0x36;
        header[0x0E] = 0x28;
        header[0x12] = _width&0xFF;
        header[0x13] = (_width>>8)&0xFF;
        header[0x14] = (_width>>16)&0xFF;
        header[0x15] = (_width>>24)&0xFF;
        header[0x16] = _height&0xFF;
        header[0x17] = (_height>>8)&0xFF;
        header[0x18] = (_height>>16)&0xFF;
        header[0x19] = (_height>>24)&0xFF;
        header[0x1A] = 1;
        header[0x1B] = 0;
        header[0x1C] = 24;
        header[0x1D] = 0;
        header[0x22] = buf_size&0xFF;
        header[0x23] = (buf_size>>8)&0xFF;
        header[0x24] = (buf_size>>16)&0xFF;
        header[0x25] = (buf_size>>24)&0xFF;
        header[0x27] = 0x1;
        header[0x2B] = 0x1;
        fwrite(header, sizeof(uchar), 54, nfile);


        uchar *buffer_ptr = buffer + (numRows-1) * numCols * nc_buffer;
        switch (nc_buffer) {
            case 1 : {
                         for(int y=0; y<numRows; y++) {
                             for(int x=0; x<numCols; x++) {
                                 const unsigned char val = *(buffer_ptr++);
                                 fputc(val, nfile); fputc(val, nfile); fputc(val, nfile);
                             }
                             fwrite(align_buf, sizeof(uchar), align, nfile);
                             buffer_ptr -= 2*numCols;
                         }
                     } break;
            case 2 : {
                         for(int y=0; y<numRows; y++) {
                             for(int x=0; x<numCols; x++) {
                                 fputc(0,nfile);
                                 fputc((*(buffer_ptr++)),nfile);
                                 fputc((*(buffer_ptr++)),nfile);
                             }
                             fwrite(align_buf, sizeof(uchar), align, nfile);
                             buffer_ptr -= 2 * numCols * nc_buffer;
                         }
                     } break;
            case 3 : {
                          for(int y=0; y<numRows; y++) {
                              for(int x=0; x<numCols; x++) {
                                  fputc((unsigned char)(*(buffer_ptr++)),nfile);
                                  fputc((unsigned char)(*(buffer_ptr++)),nfile);
                                  fputc((unsigned char)(*(buffer_ptr++)),nfile);
                              }
                              fwrite(align_buf, sizeof(uchar), align, nfile);
                              buffer_ptr -= 2 * numCols * nc_buffer;
                          }
                      }
        }
        fclose(nfile);
        return HPC_SUCCESS;
    }



#ifdef FASTCV_USE_JPEG
    ////////////////////////////JpegEncoder//////////////////////////////
    
    std::vector<uchar> jpeg_buf;
	
    std::vector<uchar> *m_buf = &jpeg_buf;

	struct JpegErrorMgr
	{
		struct jpeg_error_mgr pub;
		jmp_buf setjmp_buffer;
	};
    struct JpegDestination
    {
        struct jpeg_destination_mgr pub;
        std::vector<uchar> *buf, *dst;
    };

    METHODDEF(void)
    stub(j_compress_ptr)
    {
    }

    METHODDEF(void)
    term_destination (j_compress_ptr cinfo)
    {
        JpegDestination* dest = (JpegDestination*)cinfo->dest;
        size_t sz = dest->dst->size();
		size_t bufsz = dest->buf->size() - dest->pub.free_in_buffer;
        if( bufsz > 0 )
        {
            dest->dst->resize(sz + bufsz);
            memcpy( &(*dest->dst)[0] + sz, &(*dest->buf)[0], bufsz);
                                               
        }
    }
    METHODDEF(boolean)
    empty_output_buffer (j_compress_ptr cinfo)
    {
        JpegDestination* dest = (JpegDestination*)cinfo->dest;
		if(dest==NULL)
			return FALSE;

		
        size_t sz = dest->dst->size();
        size_t bufsz = dest->buf->size();
        dest->dst->resize(sz + bufsz);
        memcpy( &(*dest->dst)[0] + sz, &(*dest->buf)[0], bufsz);

        dest->pub.next_output_byte = &(*dest->buf)[0];
        dest->pub.free_in_buffer = bufsz;
        return TRUE;
    }
    METHODDEF(void)
    error_exit( j_common_ptr cinfo )
    {
         JpegErrorMgr* err_mgr = (JpegErrorMgr*)(cinfo->err);

         /* Return control to the setjmp point */
         longjmp( err_mgr->setjmp_buffer, 1 );
    }

    static void jpeg_buffer_dest(j_compress_ptr cinfo, JpegDestination* destination)
    {
        cinfo->dest = &destination->pub;

        destination->pub.init_destination = stub;
        destination->pub.empty_output_buffer = empty_output_buffer;
        destination->pub.term_destination = term_destination;
    }

    HPCStatus_t encode_Jpeg_(unsigned char** buffer,unsigned char* m_buffer, int numCols, int numRows, int nc, const unsigned int quality)
    {
        if(!m_buffer)
            return HPC_POINTER_NULL;
        HPCStatus_t st = HPC_OTHER_ERROR;
        int width = numCols, height = numRows;
        std::vector<uchar> out_buf(1<<12);
        std::vector<uchar> _buffer;
        uchar *tempBuffer = NULL;
		m_buf->clear();

        struct jpeg_compress_struct cinfo;
        JpegErrorMgr jerr;
        JpegDestination dest;
       // struct jpeg_error_mgr jerr;
        //cinfo.err = jpeg_std_error(&jerr);
        
        cinfo.err = jpeg_std_error(&jerr.pub);
        jerr.pub.error_exit = error_exit;
        jpeg_create_compress(&cinfo);
        dest.dst = m_buf;
        dest.buf = &out_buf;

        jpeg_buffer_dest(&cinfo, &dest);
        dest.pub.next_output_byte = &out_buf[0];
        dest.pub.free_in_buffer = out_buf.size();

        if(setjmp(jerr.setjmp_buffer) == 0)
        {
            cinfo.image_width = width;
            cinfo.image_height = height;

            int _channels = nc;
            int channels = _channels > 1 ? 3 : 1;
            cinfo.input_components = channels;
            cinfo.in_color_space = channels > 1 ?JCS_RGB :JCS_GRAYSCALE;
            int temp_quality = quality<100 ? quality : 100;
            jpeg_set_defaults(&cinfo);
            jpeg_set_quality(&cinfo, temp_quality,TRUE);
            jpeg_start_compress(&cinfo, TRUE);
           // if(channels > 1)
             //   _buffer.get_allocator().allocate(width*channels);

            if(channels > 1)
                tempBuffer = (unsigned char *)malloc(width*channels);
            //tempBuffer = reinterpret_cast<unsigned char *>(_buffer.data());;
            for(int y = 0;y < height; y++)
            {
                uchar *data = m_buffer + width*y;
                uchar *ptr = data;
                if(_channels == 3)
                {
                    Size size(width,1);
                   // printf("tempbuffer:%c\n",*tempBuffer);
                    icvCvt_BGR2RGB_8u_C3R(data, 0, tempBuffer, 0, size);
                    ptr = tempBuffer;
                }
                else if(_channels == 4)
                {
                    Size size(width,1);
                    icvCvt_BGRA2BGR_8u_C4C3R(data, 0, tempBuffer, 0 ,size,2);
                    ptr = tempBuffer;
                }
                jpeg_write_scanlines(&cinfo, &ptr, 1);

            }

            jpeg_finish_compress(&cinfo);
            *buffer = (uchar *)malloc(m_buf->size()+1);
            memcpy(*buffer, m_buf->data(), m_buf->size());
			(*buffer)[m_buf->size()+1]='\0';
            st = HPC_SUCCESS;
        }

            jpeg_destroy_compress(&cinfo);
			if(tempBuffer != NULL)
				free(tempBuffer);
		
            return st;
    }

    HPCStatus_t write_Jpeg_(const char* name, unsigned char* data, int numCols, int numRows, int nc, const unsigned int quality){
        if (!name && !data){
            fprintf(stderr, "write_Jpeg(): Specified filename is (null).\n");
            return HPC_POINTER_NULL;
        }

        unsigned int
            dimbuf = 0,
            _width = numCols,
            _height = numRows;

        J_COLOR_SPACE colortype = JCS_RGB;

        switch(nc) {
            case 1 : dimbuf = 1; colortype = JCS_GRAYSCALE; break;
            case 2 : dimbuf = 3; colortype = JCS_RGB; break;
            case 3 : dimbuf = 3; colortype = JCS_RGB; break;
            default : dimbuf = 4; colortype = JCS_CMYK; break;
        }

        // Call libjpeg functions
        struct jpeg_compress_struct cinfo;
        struct jpeg_error_mgr jerr;
        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        FILE *const nfile = fopen(name, "wb");
        jpeg_stdio_dest(&cinfo,nfile);
        cinfo.image_width = _width;
        cinfo.image_height = _height;
        cinfo.input_components = dimbuf;
        cinfo.in_color_space = colortype;
        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo,quality<100?quality:100,TRUE);
        jpeg_start_compress(&cinfo,TRUE);

        JSAMPROW row_pointer[1];
        uchar *buffer = (uchar*)malloc((unsigned long)_width*dimbuf);
        if(NULL == buffer) return HPC_ALLOC_FAILED;

        while (cinfo.next_scanline<cinfo.image_height) {
            unsigned char *ptrd = buffer;

            // Fill pixel buffer
            switch (nc) {
                case 1 : { // Greyscale images
                             const uchar *ptr_data = data + cinfo.next_scanline * _width;
                             for(unsigned int b = 0; b < cinfo.image_width; b++)
                                 *(ptrd++) = *(ptr_data++);
                         } break;
                case 2 : { // RG images
                             const uchar *ptr_data = data + cinfo.next_scanline * _width * nc;
                             for(unsigned int b = 0; b < cinfo.image_width; ++b) {
                                 *(ptrd++) = ptr_data[0];
                                 *(ptrd++) = ptr_data[1];
                                 *(ptrd++) = 0;
                                 ptr_data += 2;
                             }
                         } break;
                case 3 : { // RGB images
                             const uchar *ptr_data = data + cinfo.next_scanline * _width * nc;
                             for(unsigned int b = 0; b < cinfo.image_width; ++b) {
                                 *(ptrd++) = ptr_data[2];
                                 *(ptrd++) = ptr_data[1];
                                 *(ptrd++) = ptr_data[0];
                                 ptr_data += 3;
                             }
                         } break;
                default : { // CMYK images
                              const uchar *ptr_data = data + cinfo.next_scanline * _width * 4;
                              for(unsigned int b = 0; b < cinfo.image_width; ++b) {
                                  *(ptrd++) = ptr_data[2];
                                  *(ptrd++) = ptr_data[1];
                                  *(ptrd++) = ptr_data[0];
                                  *(ptrd++) = ptr_data[3];
                                  ptr_data += 3;
                              }
                          }
            }
            *row_pointer = buffer;
            jpeg_write_scanlines(&cinfo,row_pointer,1);
        }
        jpeg_finish_compress(&cinfo);
        fclose(nfile);
        jpeg_destroy_compress(&cinfo);
        return HPC_SUCCESS;

    }
#endif

#ifdef FASTCV_USE_PNG
    std::vector<unsigned char> *buf_p = NULL;

    static void PngencodeCallback(void *_png_ptr, uchar* data, size_t size)
    {
        if(size == 0)
            return;
        png_structp png_ptr = (png_structp)_png_ptr;
        
        buf_p = (std::vector<unsigned char> *)png_get_io_ptr(png_ptr);
        size_t cursz = buf_p->size();
        buf_p->resize(cursz + size);
        memcpy(&(*buf_p)[cursz], data, size);
    }

    template<typename T, int nc>
    HPCStatus_t encode_Png_(unsigned char **buffer, T* m, int numCols, int numRows, const std::vector<int> params, const unsigned int bytes_per_pixel = 0)
    {
        if(!m)
            return HPC_POINTER_NULL;
    
        printf("encode_Png_\n");
        std::vector<unsigned char> m_buf;
        buf_p = &m_buf;
        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
        png_infop info_ptr = 0;
        std::vector<unsigned char> out;

       // int y, width = numCols, height = numRows;
        //int channels = nc;
        if(!png_ptr)
        {
            fprintf(stderr, "encode_Png_():  Failed to initialize 'png_ptr'.\n");
            return HPC_OP_NOT_PERMITED;

        }
        info_ptr = png_create_info_struct(png_ptr);
        if(!info_ptr)
        {
            png_destroy_write_struct(&png_ptr,(png_infopp)0);
            fprintf(stderr, "encode_Png_(): Failed to initialize 'info_ptr'structure\n");
            return HPC_OP_NOT_PERMITED;
        }

        if (setjmp(png_jmpbuf(png_ptr))) 
        {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            fprintf(stderr, "encod_Png_(): Encountered unknown fatal error in libpng.\n");
            return HPC_OP_NOT_PERMITED;
        }
        printf("bufp");
        png_set_write_fn(png_ptr, &out, (png_rw_ptr)PngencodeCallback, NULL);

        float stmax=0;
        for(int i=0; i<numRows*numCols*nc; i++){
            stmax = float(m[i])>stmax ? float(m[i]) : stmax;
        }
        const int bit_depth = bytes_per_pixel?(bytes_per_pixel*8):(stmax>=256?16:8);

        int color_type;
        switch (nc) {
            case 1 : color_type = PNG_COLOR_TYPE_GRAY; break;
            case 2 : color_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
            case 3 : color_type = PNG_COLOR_TYPE_RGB; break;
            default : color_type = PNG_COLOR_TYPE_RGB_ALPHA;
        }
        const int interlace_type = PNG_INTERLACE_NONE;
        const int compression_type = PNG_COMPRESSION_TYPE_DEFAULT;
        const int filter_method = PNG_FILTER_TYPE_DEFAULT;

        png_set_IHDR(png_ptr,info_ptr,numCols,numRows,bit_depth,color_type,interlace_type,compression_type,filter_method);
        png_write_info(png_ptr,info_ptr);

        const int byte_depth = bit_depth>>3;
        const int numChan = nc>4?4:nc;
        const int pixel_bit_depth_flag = numChan * (bit_depth - 1);

        // Allocate Memory for Image Save and Fill pixel data
        png_bytep *const imgData = new png_byte*[numRows];
        for (int row = 0; row<numRows; ++row) imgData[row] = new png_byte[byte_depth*numChan*numCols];
        const T *pC0 = m;
        switch (pixel_bit_depth_flag) {
            case 7 :  { // Gray 8-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned char *ptrd = imgData[y];
                              for(int x=0; x<numCols; x++)
                                  *(ptrd++) = (unsigned char)*(pC0++);
                          }
                      } break;
            case 14 : { // Gray w/ Alpha 8-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned char *ptrd = imgData[y];
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned char)*(pC0++);
                                  *(ptrd++) = (unsigned char)*(pC0++);
                              }
                          }
                      } break;
            case 21 :  { // RGB 8-bit
                           for(int y=0; y<numRows; y++) {
                               unsigned char *ptrd = imgData[y];
                               for(int x=0; x<numCols; x++){
                                   *(ptrd++) = (unsigned char)*(pC0+2);
                                   *(ptrd++) = (unsigned char)*(pC0+1);
                                   *(ptrd++) = (unsigned char)*(pC0);
                                   pC0 += 3;
                               }
                           }
                       } break;
            case 28 : { // RGB x/ Alpha 8-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned char *ptrd = imgData[y];
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned char)*(pC0+2);
                                  *(ptrd++) = (unsigned char)*(pC0+1);
                                  *(ptrd++) = (unsigned char)*(pC0);
                                  *(ptrd++) = (unsigned char)*(pC0+3);
                                  pC0 += 4;
                              }
                          }
                      } break;
            case 15 : { // Gray 16-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned short *ptrd = (unsigned short*)(imgData[y]);
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned short)*(pC0++);
                              }
                              //if (!cimg::endianness()) cimg::invert_endianness((unsigned short*)imgData[y],_width);
                          }
                      } break;
            case 30 : { // Gray w/ Alpha 16-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned short *ptrd = (unsigned short*)(imgData[y]);
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned short)*(pC0++);
                                  *(ptrd++) = (unsigned short)*(pC0++);
                              }
                              //if (!cimg::endianness()) cimg::invert_endianness((unsigned short*)imgData[y],2*_width);
                          }
                      } break;
            case 45 : { // RGB 16-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned short *ptrd = (unsigned short*)(imgData[y]);
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned short)*(pC0+2);
                                  *(ptrd++) = (unsigned short)*(pC0+1);
                                  *(ptrd++) = (unsigned short)*(pC0);
                                  pC0 += 3;
                              }
                              //if (!cimg::endianness()) cimg::invert_endianness((unsigned short*)imgData[y],3*_width);
                          }
                      } break;
            case 60 : { // RGB w/ Alpha 16-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned short *ptrd = (unsigned short*)(imgData[y]);
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned short)*(pC0+2);
                                  *(ptrd++) = (unsigned short)*(pC0+1);
                                  *(ptrd++) = (unsigned short)*(pC0);
                                  *(ptrd++) = (unsigned short)*(pC0+3);
                                  pC0 += 4;
                              }
                              //if (!cimg::endianness()) cimg::invert_endianness((unsigned short*)imgData[y],4*_width);
                          }
                      } break;
            default :
                      fprintf(stderr, "encode_png_(): Encountered unknown fatal error in libpng.\n");
                      for(int n=0; n<numRows; n++)
                          delete[] imgData[n];
                      delete[] imgData;
                      return HPC_OP_NOT_PERMITED;
        }
        png_write_image(png_ptr,imgData);
        png_write_end(png_ptr,info_ptr);
        png_destroy_write_struct(&png_ptr, &info_ptr);

        unsigned char * vec_ptr = reinterpret_cast<unsigned char *>(buf_p->data());
        *buffer = (unsigned char *)malloc(buf_p->size());
        memcpy(*buffer, vec_ptr, buf_p->size());
        printf("%c\n",*(*buffer));

        // Deallocate Image Write Memory
        for(int n=0; n<numRows; n++)
            delete[] imgData[n];
        delete[] imgData;
        return HPC_SUCCESS;

        }

    template<typename T, int nc> HPCStatus_t write_Png_(const char* name, T* m, int numCols, int numRows, const unsigned int bytes_per_pixel=0){
        if (!name && !m){
            fprintf(stderr, "write_png(): Specified filename is (null).\n");
            return HPC_POINTER_NULL;
        }

        const char *volatile nfilename = name; // two 'volatile' here to remove a g++ warning due to 'setjmp'.
        FILE *volatile nfile = fopen(nfilename,"wb");

        // Setup PNG structures for write
        png_voidp user_error_ptr = 0;
        png_error_ptr user_error_fn = 0, user_warning_fn = 0;
        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,user_error_ptr, user_error_fn,
                user_warning_fn);
        if(!png_ptr){
            fprintf(stderr, "save_png(): Failed to initialize 'png_ptr' structure when saving file.\n");
            fclose(nfile);
            return HPC_OP_NOT_PERMITED;
        }
        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) {
            png_destroy_write_struct(&png_ptr,(png_infopp)0);
            fclose(nfile);
            fprintf(stderr, "save_png(): Failed to initialize 'info_ptr' structure when saving file.\n");
            return HPC_OP_NOT_PERMITED;
        }
        if (setjmp(png_jmpbuf(png_ptr))) {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            fclose(nfile);
            fprintf(stderr, "save_png(): Encountered unknown fatal error in libpng when saving file.\n");
            return HPC_OP_NOT_PERMITED;
        }
        png_init_io(png_ptr, nfile);

        float stmax=0;
        for(int i=0; i<numRows*numCols*nc; i++){
            stmax = float(m[i])>stmax ? float(m[i]) : stmax;
        }
        const int bit_depth = bytes_per_pixel?(bytes_per_pixel*8):(stmax>=256?16:8);

        int color_type;
        switch (nc) {
            case 1 : color_type = PNG_COLOR_TYPE_GRAY; break;
            case 2 : color_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
            case 3 : color_type = PNG_COLOR_TYPE_RGB; break;
            default : color_type = PNG_COLOR_TYPE_RGB_ALPHA;
        }
        const int interlace_type = PNG_INTERLACE_NONE;
        const int compression_type = PNG_COMPRESSION_TYPE_DEFAULT;
        const int filter_method = PNG_FILTER_TYPE_DEFAULT;
        png_set_IHDR(png_ptr,info_ptr,numCols,numRows,bit_depth,color_type,interlace_type,compression_type,filter_method);
        png_write_info(png_ptr,info_ptr);
        const int byte_depth = bit_depth>>3;
        const int numChan = nc>4?4:nc;
        const int pixel_bit_depth_flag = numChan * (bit_depth - 1);

        // Allocate Memory for Image Save and Fill pixel data
        png_bytep *const imgData = new png_byte*[numRows];
        for (int row = 0; row<numRows; ++row) imgData[row] = new png_byte[byte_depth*numChan*numCols];
        const T *pC0 = m;
        switch (pixel_bit_depth_flag) {
            case 7 :  { // Gray 8-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned char *ptrd = imgData[y];
                              for(int x=0; x<numCols; x++)
                                  *(ptrd++) = (unsigned char)*(pC0++);
                          }
                      } break;
            case 14 : { // Gray w/ Alpha 8-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned char *ptrd = imgData[y];
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned char)*(pC0++);
                                  *(ptrd++) = (unsigned char)*(pC0++);
                              }
                          }
                      } break;
            case 21 :  { // RGB 8-bit
                           for(int y=0; y<numRows; y++) {
                               unsigned char *ptrd = imgData[y];
                               for(int x=0; x<numCols; x++){
                                   *(ptrd++) = (unsigned char)*(pC0+2);
                                   *(ptrd++) = (unsigned char)*(pC0+1);
                                   *(ptrd++) = (unsigned char)*(pC0);
                                   pC0 += 3;
                               }
                           }
                       } break;
            case 28 : { // RGB x/ Alpha 8-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned char *ptrd = imgData[y];
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned char)*(pC0+2);
                                  *(ptrd++) = (unsigned char)*(pC0+1);
                                  *(ptrd++) = (unsigned char)*(pC0);
                                  *(ptrd++) = (unsigned char)*(pC0+3);
                                  pC0 += 4;
                              }
                          }
                      } break;
            case 15 : { // Gray 16-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned short *ptrd = (unsigned short*)(imgData[y]);
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned short)*(pC0++);
                              }
                              //if (!cimg::endianness()) cimg::invert_endianness((unsigned short*)imgData[y],_width);
                          }
                      } break;
            case 30 : { // Gray w/ Alpha 16-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned short *ptrd = (unsigned short*)(imgData[y]);
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned short)*(pC0++);
                                  *(ptrd++) = (unsigned short)*(pC0++);
                              }
                              //if (!cimg::endianness()) cimg::invert_endianness((unsigned short*)imgData[y],2*_width);
                          }
                      } break;
            case 45 : { // RGB 16-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned short *ptrd = (unsigned short*)(imgData[y]);
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned short)*(pC0+2);
                                  *(ptrd++) = (unsigned short)*(pC0+1);
                                  *(ptrd++) = (unsigned short)*(pC0);
                                  pC0 += 3;
                              }
                              //if (!cimg::endianness()) cimg::invert_endianness((unsigned short*)imgData[y],3*_width);
                          }
                      } break;
            case 60 : { // RGB w/ Alpha 16-bit
                          for(int y=0; y<numRows; y++) {
                              unsigned short *ptrd = (unsigned short*)(imgData[y]);
                              for(int x=0; x<numCols; x++){
                                  *(ptrd++) = (unsigned short)*(pC0+2);
                                  *(ptrd++) = (unsigned short)*(pC0+1);
                                  *(ptrd++) = (unsigned short)*(pC0);
                                  *(ptrd++) = (unsigned short)*(pC0+3);
                                  pC0 += 4;
                              }
                              //if (!cimg::endianness()) cimg::invert_endianness((unsigned short*)imgData[y],4*_width);
                          }
                      } break;
            default :
                      fclose(nfile);
                      fprintf(stderr, "save_png(): Encountered unknown fatal error in libpng when saving file.\n");
                      for(int n=0; n<numRows; n++)
                          delete[] imgData[n];
                      delete[] imgData;
                      return HPC_OP_NOT_PERMITED;
        }
        png_write_image(png_ptr,imgData);
        png_write_end(png_ptr,info_ptr);
        png_destroy_write_struct(&png_ptr, &info_ptr);

        // Deallocate Image Write Memory
        for(int n=0; n<numRows; n++)
            delete[] imgData[n];
        delete[] imgData;

        fclose(nfile);
        return HPC_SUCCESS;

    }
#endif

}};

