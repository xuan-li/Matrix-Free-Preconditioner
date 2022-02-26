#include <Eigen/Eigen>

class Oracle
{
private:
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
public:
    using Vec = Eigen::VectorXd;
    Vec diagonal_entries;
    virtual void initialize(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A);
    Oracle() {}
    virtual ~Oracle() = default;
    virtual void precondition(const Vec& v, Vec& precond_v);
    virtual void multiply(const Vec& x, Vec& Ax) {Ax = A * x;}
    virtual void project(Vec& v) {};
};

void Oracle::initialize(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A) {
    this->A = A;
    diagonal_entries.resize(A.cols());
    for (int i = 0; i < A.cols(); ++i) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it) {
            if (it.col() == it.row()) {
                diagonal_entries[it.col()] = it.value();
            }
        }
    }
}

void Oracle::precondition(const Vec& v, Vec& precond_v)
{
    precond_v = v;
}