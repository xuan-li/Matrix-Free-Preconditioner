#pragma once

#include <limits>
#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>


template <class T, class TM, class TV>
class ConjugateGradient {
    /** All notations adopted from Wikipedia,
         * q denotes A*p in general */
    TV r, p, q, temp;
    TV mr, s;

public:
    T relative_tolerance;
    int max_iterations;
    ConjugateGradient(const int max_it_input);
    ~ConjugateGradient();
    void setRelativeTolerance(const T tolerance_input = 1);
    void reinitialize(const TV& b);
    T dotProduct(const TV& A, const TV& B);
    int solve(TM& A, TV& x, const TV& b, const bool verbose = false);
};


template <class T, class TM, class TV>
ConjugateGradient<T, TM, TV>::ConjugateGradient(const int max_it_input)
    : max_iterations(max_it_input)
{
    setRelativeTolerance(1);
}

template <class T, class TM, class TV>
ConjugateGradient<T, TM, TV>::~ConjugateGradient()
{
}

template <class T, class TM, class TV>
void ConjugateGradient<T, TM, TV>::setRelativeTolerance(const T tolerance_input)
{
    relative_tolerance = tolerance_input;
}

template <class T, class TM, class TV>
void ConjugateGradient<T, TM, TV>::reinitialize(const TV& b)
{
    r = b; // r.resizeLike(b);
    p = b; // p.resizeLike(b);
    q = b; // q.resizeLike(b);
    temp = b; // temp.resizeLike(b);

    mr = b; // mr.resizeLike(b);
    s = b; // s.resizeLike(b);
}

template <class T, class TM, class TV>
T ConjugateGradient<T, TM, TV>::dotProduct(const TV& A, const TV& B)
{
    return (A.array() * B.array()).sum();
}

template <class T, class TM, class TV>
int ConjugateGradient<T, TM, TV>::solve(TM& A, TV& x, const TV& b, const bool verbose)
{
    reinitialize(x);
    int cnt = 0;
    T alpha, beta, residual, zTrk, zTrk_last;

    //NOTE: requires that the input x has been projected
    A.multiply(x, temp);
    r = b - temp;
    A.project(r);
    A.precondition(r, q); //NOTE: requires that preconditioning matrix is projected
    p = q;
    zTrk = std::abs(dotProduct(r, q));
    residual = r.norm();
    T local_tolerance = relative_tolerance * residual;
    std::cout << "CG target residual: " << local_tolerance << std::endl;
    for (cnt = 0; cnt < max_iterations; ++cnt) {
        if (residual <= local_tolerance) {
            std::cout << "CG terminates at " << cnt << " residual = " << residual << std::endl;
            return cnt;
        }

        if (cnt % 50 == 0) {
            std::cout << "CG iter " << cnt << "; residual = " << residual << std::endl;
            // logging::logger().info("CG iter {}; (preconditioned norm) residual = {}", cnt, residual);
        }

        A.multiply(p, temp);
        A.project(temp);
        alpha = zTrk / dotProduct(temp, p);

        x = x + alpha * p;
        r = r - alpha * temp;
        A.precondition(r, q); //NOTE: requires that preconditioning matrix is projected

        zTrk_last = zTrk;
        zTrk = dotProduct(q, r);
        beta = zTrk / zTrk_last;

        p = q + beta * p;

        residual = r.norm();
    }
    std::cout << "ConjugateGradient max iterations reached " << max_iterations;
    return max_iterations;
}