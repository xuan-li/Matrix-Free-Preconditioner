#include <Eigen/Eigen>
#include "IO/MatrixBinaryIO.h"
#include "IO/MarketIO.h"
#include "PreconditionedProblem/LMPPrecondition.h"
#include "Spectra/GenEigsSolver.h"
#include "common.h"
#include "KLargest.h"

using namespace Spectra;
class ProblemWrapper
{
    const LMPPreconditionedOracle& A;
public:
    using Scalar = double;  // A typedef named "Scalar" is required
    bool precond = false;
    ProblemWrapper(const LMPPreconditionedOracle& A): A(A) {}
    int rows() const{ return A.n; }
    int cols() const{ return A.n; }
    // y_out = M * x_in
    void perform_op(const double *x_in, double *y_out) const
    {
        Eigen::Map<const Eigen::VectorXd> v_in(x_in, rows());
        Eigen::VectorXd temp(rows());
        A.multiply(v_in, temp);
        Eigen::VectorXd result(rows());
        if (precond)
            A.precondition(temp, result);
        else
            result = temp;
        memcpy(y_out, result.data(), sizeof(double) * rows());
    }
};

int main(int argc, char** argv)
{
    // Eigen::VectorXd vec(5);
    // vec << 1,2,3,4,5 ;
    // SpMat A = Eigen::MatrixXd(vec.asDiagonal()).sparseView();
    SpMat A; // row major to use eigen build-in parallelization
    Eigen::loadMarket(A, DATA_DIR"bcsstk01.mtx", true);
    // Eigen::SparseMatrix<double, Eigen::RowMajor> A; // row major to use eigen build-in parallelization
    // Eigen::read_binary_sparse(DATA_DIR"system_matrix_random_F.bin", A);
    LMPPreconditionedOracle oracle;
    oracle.initialize(A, 10);
    ProblemWrapper op(oracle);
    op.precond = true;
    GenEigsSolver<ProblemWrapper> eigs(op, 1, 5);
    eigs.init();
    eigs.compute(SortRule::LargestMagn);
    if (eigs.info() == CompInfo::Successful)
    {
        std::cout << "Precond largest Eigenvalues found:\n" << eigs.eigenvalues().transpose().real() << std::endl;
    }
    op.precond = false;
    eigs.init();
    eigs.compute(SortRule::LargestMagn);
    if (eigs.info() == CompInfo::Successful)
    {
        Eigen::VectorXd evalues = eigs.eigenvalues().real();
        std::cout << "Nonprecond largest Eigenvalues found:\n" << evalues.transpose() << std::endl;
    }
}