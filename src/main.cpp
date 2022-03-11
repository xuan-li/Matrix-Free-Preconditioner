#include <Eigen/Eigen>
#include "IO/MatrixBinaryIO.h"
#include "PreconditionedProblem/DiagonalPrecondition.h"
#include "PreconditionedProblem/LMPPrecondition.h"
#include "Solver/ConjugateGradient.h"
#include "Solver/Minres.h"

int main(int argc, char** argv)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A; // row major to use eigen build-in parallelization
    Eigen::read_binary_sparse(DATA_DIR"system_matrix_random_F.bin", A);
    Eigen::VectorXd rhs = A * Eigen::VectorXd::Ones(A.rows());
    std::cout << "Size of A :" << A.rows() << std::endl;
    LMPPreconditionedOracle oracle;
    oracle.initialize(A, 20);
    Minres<double, Oracle, Eigen::VectorXd> cg(100000);
    cg.setRelativeTolerance(1e-5);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(rhs.size());
    rhs = oracle.permute(rhs);
    cg.solve(oracle, x, rhs);
    x = oracle.inverse_permute(x);
    std::cout << "Relative error to ground truth: " << Eigen::VectorXd((x.array() - 1)/x.size()).norm() << std::endl;
    return 0;
}