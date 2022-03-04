#pragma once
#include "Oracle.h"
#include <Eigen/Cholesky>
#include <iostream>
#include "../KLargest.h"

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
    Eigen::VectorXi perm;

    void multiply(const Vec& x, Vec& Ax) const override {
        auto inv_perm_x = inverse_permute(x);
        Base::multiply(inv_perm_x, Ax);
        Ax = permute(Ax);
    }

    Vec diagonal() const override {
        return permute(diagonal_entries);
    }

    void initialize(const Mat& A, int k) {
        Base::initialize(A);
        this->k = k;
        n = A.cols();
        // permutation
        perm.resize(n);
        double pivot = kthLargest(this->diagonal_entries, k+1);
        Eigen::VectorXi flag = Eigen::VectorXi::Zero(n);
        int i = 0;
        for (int j = 0; j < n; ++j) {
            if (this->diagonal_entries[j] > pivot) {
                perm[j] = i++;
                flag(j) = 1;
            }
        }
        for (int j = 0; j < n; ++j) {
            if (flag(j) == 0)
                perm[j] = i++;
        }

        Matd H11, H21;
        H11.resize(k, k);
        H21.resize(n - k, k);
        for (int i = 0; i < k; ++i)
        {
            // get first k cols
            Vec ei = Vec::Zero(n);
            ei(i) = 1.0;
            Vec hi;
            multiply(ei, hi);
            H11.col(i) = hi.topRows(k);
            H21.col(i) = hi.bottomRows(n - k);
        }

        // !!!!!!!! Eigen solves PAPT = LDLT.... !!!!!!!
        Eigen::LDLT<Matd> ldlt;
        ldlt.compute(H11);
        H11 = ldlt.matrixLDLT();
        d1 = ldlt.vectorD();
        L11 = ldlt.matrixL();
        L11T = ldlt.matrixU();
        auto perm2 = Eigen::PermutationMatrix<Eigen::Dynamic>(ldlt.transpositionsP()).indices();
        for (int i = 0; i < n; ++ i) {
            if (perm(i) < k)
                perm(i) = perm2(perm(i));
        }
        // permute H21's cols
        Matd new_H21;
        for (int i = 0; i < k; ++i) {
            new_H21.col(perm2(i)) = H21.col(i);
        }
        H21 = new_H21;
        Matd D1 = d1.asDiagonal();
        Matd D1_inv = D1;
        for (int i = 0; i < D1_inv.rows(); ++i)
        {
            D1_inv(i, i) = 1.0 / D1(i, i);
        }

        L21 = (H21 * L11T.triangularView<Eigen::Upper>().solve(D1_inv)).sparseView();
        d2 = Vec::Zero(n - k);
        Vec diag = diagonal();
        for (int i = 0; i < n - k; ++i)
        {
            double sum = 0;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L21, i); it; ++it)
            {
                sum += it.value() * it.value() * D1(it.col(), it.col());
            }
            d2(i) = diag[i + k] - sum;
        }
    }

    void precondition(const Vec& v, Vec& Pv) const override {
        // Pv = v;
        // Vec diag = diagonal();
        // for (int i = 0; i < this->diagonal_entries.size(); ++i) {
        //     if (abs(diag[i]) > 0)
        //         Pv[i] /= diag[i];
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
        Pv.segment(0, k) = x;
        Pv.segment(k, n - k) = y;
    }

    Vec permute(const Vec& v_in) const
    {
        Vec v_out = v_in;
        for (int i = 0; i < n; ++i) 
            v_out(perm(i)) = v_in(i);
        return v_out;
    }
    Vec inverse_permute(const Vec& v_in) const
    {
        Vec v_out = v_in;
        for (int i = 0; i < n; ++i) 
            v_out(i) = v_in(perm(i));
        return v_out;
    }
};