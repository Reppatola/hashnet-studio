// Эталон хэша из kernels.cu (тот же код, что в CUDA) для сверки с Rust: gcc ng_hash_ref.c && ./a.out
#include <stdio.h>
static int ng_hash(const unsigned char* tok,int r,int p,int n,int buckets){
    unsigned int h=2166136261u;
    for(int i=0;i<n;i++){unsigned int b=(p>=i)?(unsigned int)tok[r-i]:0u;h=(h^b)*16777619u;}
    h^=h>>15;h*=2246822519u;h^=h>>13;return (int)(h%(unsigned int)buckets);}
int main(){const unsigned char w[]="abcde";printf("%d %d\n",ng_hash(w,4,4,3,4096),ng_hash(w,4,4,5,4096));return 0;}
