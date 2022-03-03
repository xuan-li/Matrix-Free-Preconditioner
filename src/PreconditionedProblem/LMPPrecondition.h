#pragma once
#include "Oracle.h"
#include <Eigen/Cholesky>
#include <iostream>

class LMPPreconditionedOracle: public Oracle
{
public:
    using Oracle::Vec;
    using Mat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Matd = Eigen::MatrixXd;
    using Base = Oracle;


    int k, n;
    Matd L11, L11T;
    Mat L21;
    Vec d1, d2;

    void initialize(const Mat& A) override {
        Base::initialize(A);

        k = 10;
        n = A.cols();

        Matd H11, H21;
        H11.resize(k, k);
        H21.resize(n - k, k);
        for (int i = 0; i < k; ++i)
        {
            // get first k cols
            Vec ei = Vec::Zero(n);
            ei(i) = 1.0;
            Vec hi;
            this->multiply(ei, hi);
            H11.col(i) = hi.topRows(k);
            H21.col(i) = hi.bottomRows(n - k);
        }

        Eigen::LDLT<Matd> ldlt;
        ldlt.compute(H11);
        d1 = ldlt.vectorD();
        L11 = ldlt.matrixL();
        L11T = ldlt.matrixU();
        Matd D1 = d1.asDiagonal();
        Matd D1_inv = D1;
        for (int i = 0; i < D1_inv.rows(); ++i)
        {
            D1_inv(i, i) = 1.0 / D1_inv(i, i);
        }


        L21 = (H21 * L11T.triangularView<Eigen::Upper>().solve(D1_inv)).sparseView();
        std::cout << L21.rows() << " " << n << " " << k << std::endl;
        d2 = Vec::Zero(n - k);
        for (int i = 0; i < n - k; ++i)
        {
            double sum = 0;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L21, i); it; ++it)
            {
                sum += it.value() * it.value() * D1(it.col());
            }
            d2(i) = this->diagonal_entries[i + k] - sum;
        }
    }

    void precondition(const Vec& v, Vec& Pv) const override {
        // Pv = v;
        // for (int i = 0; i < this->diagonal_entries.size(); ++i) {
        //     if (abs(this->diagonal_entries[i]) > 0)
        //         Pv[i] /= this->diagonal_entries[i];
        // }

        // [L11T  L21T]^(-1) [D1   ]^(-1) [L11   ]^(-1) 
        // [         I]      [   D2]      [L21  I]
        Pv = Vec::Zero(n);
        Vec x = v.segment(0, k);
        Vec y = v.segment(k, n - k);
        x = L11.triangularView<Eigen::Lower>().solve(x);
        y = y - L21 * x;
        for (int i = 0; i < k; ++i)
        {
            x(i) /= d1(i);
        }
        for (int i = 0; i < n - k; ++i)
        {
            y(i) /= d2(i);
        }
        x = L11T.triangularView<Eigen::Upper>().solve(x - L21.transpose() * y);
        Pv.segment(0, k) += x;
        Pv.segment(k, n - k) += y;
    }
};