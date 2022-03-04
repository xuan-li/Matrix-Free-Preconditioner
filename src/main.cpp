#include <Eigen/Eigen>
#include "IO/MatrixBinaryIO.h"
#include "PreconditionedProblem/DiagonalPrecondition.h"
#include "PreconditionedProblem/LMPPrecondition.h"
#include "Solver/ConjugateGradient.h"
#include "Solver/Minres.h"

int main(int argc, char** argv)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A; // row major to use eigen build-in parallelization
    Eigen::VectorXd rhs;
    Eigen::read_binary_sparse(DATA_DIR"system_matrix_random_F.bin", A);
    Eigen::read_binary(DATA_DIR"rhs_random_F.bin", rhs);
    LMPPreconditionedOracle oracle;
    oracle.initialize(A, 50);
    Minres<double, Oracle, Eigen::VectorXd> cg(100000);
    cg.setRelativeTolerance(1e-7);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(rhs.size());
    rhs = oracle.permute(rhs);
    cg.solve(oracle, x, rhs);
    x = oracle.inverse_permute(x);
    return 0;
}