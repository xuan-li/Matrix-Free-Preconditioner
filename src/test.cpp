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
        Eigen::Map<const Eigen::VectorXd> v_in(x_in, rows(), rows());
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
    SpMat A; // row major to use eigen build-in parallelization
    Eigen::loadMarket(A, DATA_DIR"494_bus.mtx");
    LMPPreconditionedOracle oracle;
    oracle.initialize(A);
    ProblemWrapper op(oracle);
    op.precond = true;
    GenEigsSolver<ProblemWrapper> eigs(op, 20, 40);
    eigs.init();
    eigs.compute(SortRule::LargestMagn);
    if (eigs.info() == CompInfo::Successful)
    {
        Eigen::VectorXd evalues = eigs.eigenvalues().real();
        std::cout << "Precond largest Eigenvalues found:\n" << evalues.transpose() << std::endl;
    }
    eigs.compute(SortRule::SmallestMagn);
    if (eigs.info() == CompInfo::Successful)
    {
        Eigen::VectorXd evalues = eigs.eigenvalues().real();
        std::cout << "Precond smallest Eigenvalues found:\n" << evalues.transpose() << std::endl;
    }
    op.precond = false;
    eigs.compute(SortRule::LargestMagn);
    if (eigs.info() == CompInfo::Successful)
    {
        Eigen::VectorXd evalues = eigs.eigenvalues().real();
        std::cout << "Nonprecond largest Eigenvalues found:\n" << evalues.transpose() << std::endl;
    }
    eigs.compute(SortRule::SmallestMagn);
    if (eigs.info() == CompInfo::Successful)
    {
        Eigen::VectorXd evalues = eigs.eigenvalues().real();
        std::cout << "Nonprecond smallest Eigenvalues found:\n" << evalues.transpose() << std::endl;
    }

}