#include <sys/time.h>

#define roll \
    for(int kk=0; kk<10; kk++) \

#define ss \
    gettimeofday(&start_, NULL);

#define ee {\
    gettimeofday(&end_, NULL);\
    fprintf(stderr, "s_%f\n", timingExec(start_, end_));\
}

#define eee {\
    gettimeofday(&end_, NULL);\
    fprintf(stderr, "m_%f\n", timingExec(start_, end_)/10);\
}

#define eeo {\
    gettimeofday(&end_, NULL);\
    fprintf(stderr, "o_s_%f\n", timingExec(start_, end_));\
}

#define eeeo {\
    gettimeofday(&end_, NULL);\
    fprintf(stderr, "o_m_%f\n", timingExec(start_, end_)/10);\
}


#if defined(USE_CUDA)
#define gettime(st){\
    gettimeofday(&start_, NULL);\
    if(st != HPC_SUCCESS)printf("%s, %d\n", __FILE__, __LINE__);\
    cudaDeviceSynchronize();\
    gettimeofday(&end_, NULL);\
    fprintf(stderr, "s_%f\n", timingExec(start_, end_));\
    gettimeofday(&start_, NULL);\
    for(int kk=0; kk<10; kk++){\
        if(st != HPC_SUCCESS)printf("%s, %d\n", __FILE__, __LINE__);\
        cudaDeviceSynchronize();\
    }\
    gettimeofday(&end_, NULL);\
    fprintf(stderr, "m_%f\n", timingExec(start_, end_)/10);\
}
#else
#define gettime(st){\
    gettimeofday(&start_, NULL);\
    if(st != HPC_SUCCESS)printf("%s, %d\n", __FILE__, __LINE__);\
    gettimeofday(&end_, NULL);\
    fprintf(stderr, "s_%f\n", timingExec(start_, end_));\
    gettimeofday(&start_, NULL);\
    for(int kk=0; kk<10; kk++){\
        if(st != HPC_SUCCESS)printf("%s, %d\n", __FILE__, __LINE__);\
    }\
    gettimeofday(&end_, NULL);\
    fprintf(stderr, "m_%f\n", timingExec(start_, end_)/10);\
}
#endif



#define gettime_opencv(st){\
    gettimeofday(&start_, NULL);\
    st ;\
    gettimeofday(&end_, NULL);\
    fprintf(stderr, "o_s_%f\n", timingExec(start_, end_));\
    gettimeofday(&start_, NULL);\
    for(int kk=0; kk<10; kk++){\
        st ;\
    }\
    gettimeofday(&end_, NULL);\
    fprintf(stderr, "o_m_%f\n", timingExec(start_, end_)/10);\
}

#define show(name) {\
    fprintf(stderr, #name);\
    fprintf(stderr, "\n");\
}

#define show_(name) {\
    fprintf(stderr, "%d\n", name);\
}


inline double timingExec(struct timeval start, struct timeval end){
    double timeuse = 1000.0*(end.tv_sec-start.tv_sec) + (end.tv_usec - start.tv_usec)/1000.0;
    return timeuse;
}

struct timeval start_, end_;

