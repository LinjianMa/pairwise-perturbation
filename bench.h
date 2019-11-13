#ifndef __BENCH_ALS_H__
#define __BENCH_ALS_H__

#include <ctf.hpp>
using namespace CTF;

int bench_contraction(int n, int niter, char const *iA, char const *iB,
                      char const *iC, CTF_World &dw);

int bench_contraction_no_dist(int n, int niter, char const *iA, char const *iB,
                              char const *iC, CTF_World &dw);

#endif
