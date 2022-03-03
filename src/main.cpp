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
    Eigen::read_binary_sparse(DATA_DIR"system_matrix_random.bin", A);
    Eigen::read_binary(DATA_DIR"rhs_random.bin", rhs);
    LMPPreconditionedOracle oracle;
    oracle.initialize(A);
    Minres<double, Oracle, Eigen::VectorXd> cg(100000);
    cg.setRelativeTolerance(1e-7);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(rhs.size());
    cg.solve(oracle, x, rhs);
    return 0;
}