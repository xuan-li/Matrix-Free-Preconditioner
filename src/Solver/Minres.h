#pragma once

#include <Eigen/Eigen>

template <class T>
class GivensRotation;

template <class T, class TM, class TV>
class Minres {

    using TM2 = Eigen::Matrix<T, 2, 2>;
    using TV2 = Eigen::Matrix<T, 2, 1>;

    GivensRotation<T> Gk, Gkm1, Gkm2; //The Q in the QR of Hk: Givens rotations

    T gamma, delta, epsilon; //These are the newest entries in R from the QR of Hk
    T beta_kp1, alpha_k, beta_k, sk; //This is the last column in Hk
    TV mk, mkm1, mkm2; //mk is the newest search direction in the memory friendly basis for the Krylov space, you need the other two to generate mk
    TV z, qkp1, qk, qkm1; //These are the Lanczos vectors needed at each iteration, qk is denoted as vk in the notes
    T tk; //This is the step length in the direction of mk
    TV2 last_two_components_of_givens_transformed_least_squares_rhs; //This will track the residual with just a constant number of flops per iteration
    T rhsNorm2;

    int max_iterations;
    T relative_tolerance;

public:
    Minres(const int max_it_input)
        : Gk(0, 1)
        , Gkm1(0, 1)
        , Gkm2(0, 1)
        , max_iterations(max_it_input)
    {
    }

    ~Minres() { }

    void setRelativeTolerance(const T tolerance_input)
    {
        relative_tolerance = tolerance_input;
    }

    void reinitialize(const TV& b)
    {
        mk.resizeLike(b);
        mkm1.resizeLike(b);
        mkm2.resizeLike(b);
        z.resizeLike(b);
        qkm1.resizeLike(b);
        qk.resizeLike(b);
        qkp1.resizeLike(b);
        mk.setZero();
        mkm1.setZero();
        mkm2.setZero();
        qkm1.setZero();
        qk.setZero();
        qkp1.setZero();
        gamma = 0;
        delta = 0;
        epsilon = 0;
        beta_kp1 = 0;
        alpha_k = 0;
        beta_k = 0;
        tk = 0;
        Gk.setIdentity();
        Gkm1.setIdentity();
        Gkm2.setIdentity();

        rhsNorm2 = b.squaredNorm();
    }

    T dotProduct(const TV& A, const TV& B)
    {
        return (A.array() * B.array()).sum();
    }

    int solve(TM& A, TV& x, const TV& b, const bool verbose = false)
    {
        assert(x.size() == b.size());
        reinitialize(b);

        //qkp1 = b - A * x;
        A.multiply(x, qkp1);
        qkp1 = b - qkp1;

        A.project(qkp1);
        A.precondition(qkp1, z); //z= M_inv*qkp1

        T residual_preconditioned_norm = std::sqrt(dotProduct(z, qkp1));
        T residual = qkp1.norm();
        beta_kp1 = residual_preconditioned_norm;
        T local_tolerance = relative_tolerance * residual;

        std::cout << "Minres target residual: " << local_tolerance << std::endl;

        if (residual_preconditioned_norm > 0) {
            qkp1 /= beta_kp1;
            z /= beta_kp1;
        }
        last_two_components_of_givens_transformed_least_squares_rhs = TV2(residual_preconditioned_norm, 0);

        for (int k = 0; k < max_iterations; k++) {
            if (residual <= local_tolerance) {
                std::cout << "Minres terminates at " << k << " residual = " << residual << std::endl;
                return k; //Output the number of iterations.
            }
            if (k % 50 == 0) {
                std::cout << "Minres iter " << k << " residual = " << residual << std::endl;
            }

            //use mk to store zk to save storage
            mkm2.swap(mkm1);
            mkm1.swap(mk);
            mk = z;

            beta_k = beta_kp1;

            // Save the last two Lanczos vectors.
            // Equivalent to qkm1 = qk; qk = qkp1; And qkp1 will be overwritten right after anyway.
            qkm1.swap(qkp1);
            qkm1.swap(qk);

            A.multiply(mk, qkp1); //qpk1 = A * zk; Get the important part of the next Lanczos vector: q_k+1
            A.project(qkp1);
            alpha_k = dotProduct(mk, qkp1);

            qkp1 -= alpha_k * qk;
            qkp1 -= beta_k * qkm1;

            A.precondition(qkp1, z);
            beta_kp1 = std::sqrt(std::max((T)0, dotProduct(z, qkp1)));

            if (beta_kp1 > 0) {
                qkp1 /= beta_kp1;
                z /= beta_kp1;
            }

            residual_preconditioned_norm = applyAllPreviousGivensRotationsAndDetermineNewGivens(); //This determines the newest Givens rotation and applies the previous two where appropriate
            {
                TV temp;
                A.multiply(x, temp);
                A.project(temp);
                residual = (b - temp).norm();
            }

            //Three term recurence for the m's, mk stores the old zk which is set at beginning of the iteration
            mk = (mk - delta * mkm1 - epsilon * mkm2) / gamma;
            x += tk * mk;
        }

        std::cout << "Minres max iterations reached " << max_iterations;
        return max_iterations;
    }

    T applyAllPreviousGivensRotationsAndDetermineNewGivens()
    {
        //QR the LHS: gamma, delta, epsilon
        Gkm2 = Gkm1;
        Gkm1 = Gk;
        TV2 epsilon_k_and_phi_k(0, beta_k);
        Gkm2.rowRotation(epsilon_k_and_phi_k);

        epsilon = epsilon_k_and_phi_k(0);
        TV2 delta_k_and_zsi_k(epsilon_k_and_phi_k(1), alpha_k);
        Gkm1.rowRotation(delta_k_and_zsi_k);
        delta = delta_k_and_zsi_k(0);
        TV2 temp(delta_k_and_zsi_k(1), beta_kp1);

        Gk.compute(temp(0), temp(1));
        Gk.rowRotation(temp);
        gamma = temp(0);

        //Now deal with the RHS: tk and residual (two norm)
        Gk.rowRotation(last_two_components_of_givens_transformed_least_squares_rhs);
        tk = last_two_components_of_givens_transformed_least_squares_rhs(0);
        T residual = last_two_components_of_givens_transformed_least_squares_rhs(1); //This is the two norm of the residual.
        last_two_components_of_givens_transformed_least_squares_rhs = TV2(residual, 0); //Set up for the next iteration
        if (residual < 0)
            return -residual;
        else
            return residual;
    }
};


/**
    Class for givens rotation.
    Row rotation G*A corresponds to something like
    c -s  0
    ( s  c  0 ) A
    0  0  1
    Column rotation A G' corresponds to something like
    c -s  0
    A ( s  c  0 )
    0  0  1

    c and s are always computed so that
    ( c -s ) ( a )  =  ( * )
    s  c     b       ( 0 )

    Assume rowi<rowk.
    */
template <class T>
class GivensRotation {
public:
    int rowi;
    int rowk;
    T c;
    T s;

    GivensRotation(int rowi_in, int rowk_in)
        : rowi(rowi_in)
        , rowk(rowk_in)
        , c(1)
        , s(0)
    {
    }

    GivensRotation(T a, T b, int rowi_in, int rowk_in)
        : rowi(rowi_in)
        , rowk(rowk_in)
    {
        compute(a, b);
    }

    ~GivensRotation() { }

    void setIdentity()
    {
        c = 1;
        s = 0;
    }

    void transposeInPlace()
    {
        s = -s;
    }

    /**
        Compute c and s from a and b so that
        ( c -s ) ( a )  =  ( * )
        s  c     b       ( 0 )
        */
    void compute(const T a, const T b)
    {
        using std::sqrt;

        T d = a * a + b * b;
        c = 1;
        s = 0;
        T sqrtd = sqrt(d);
        //T t = MATH_TOOLS::rsqrt(d);
        if (sqrtd) {
            T t = 1 / sqrtd;
            c = a * t;
            s = -b * t;
        }
    }

    /**
        This function computes c and s so that
        ( c -s ) ( a )  =  ( 0 )
        s  c     b       ( * )
        */
    void computeUnconventional(const T a, const T b)
    {
        using std::sqrt;

        T d = a * a + b * b;
        c = 0;
        s = 1;
        T sqrtd = sqrt(d);
        //T t = MATH_TOOLS::rsqrt(d);
        if (sqrtd) {
            T t = 1 / sqrtd;
            s = a * t;
            c = b * t;
        }
    }

    /**
      Fill the R with the entries of this rotation
        */
    template <class MatrixType>
    void fill(const MatrixType& R) const
    {
        MatrixType& A = const_cast<MatrixType&>(R);
        A = MatrixType::Identity();
        A(rowi, rowi) = c;
        A(rowk, rowi) = -s;
        A(rowi, rowk) = s;
        A(rowk, rowk) = c;
    }

    /**
        This function does something like Q^T A -> A 
        [ c -s  0 ]
        [ s  c  0 ] A -> A
        [ 0  0  1 ]
        It only affects row i and row k of A.
        */
    template <class MatrixType>
    void rowRotation(MatrixType& A) const
    {
        for (int j = 0; j < A.cols(); j++) {
            T tau1 = A(rowi, j);
            T tau2 = A(rowk, j);
            A(rowi, j) = c * tau1 - s * tau2;
            A(rowk, j) = s * tau1 + c * tau2;
        }
        //not type safe :/
    }

    /**
        This function does something like A Q -> A
           [ c  s  0 ]
        A  [-s  c  0 ]  -> A
           [ 0  0  1 ]
        It only affects column i and column k of A.
        */
    template <class MatrixType>
    void columnRotation(MatrixType& A) const
    {
        for (int j = 0; j < A.rows(); j++) {
            T tau1 = A(j, rowi);
            T tau2 = A(j, rowk);
            A(j, rowi) = c * tau1 - s * tau2;
            A(j, rowk) = s * tau1 + c * tau2;
        }
        //not type safe :/
    }

    /**
      Multiply givens must be for same row and column
      **/
    void operator*=(const GivensRotation<T>& A)
    {
        T new_c = c * A.c - s * A.s;
        T new_s = s * A.c + c * A.s;
        c = new_c;
        s = new_s;
    }

    /**
      Multiply givens must be for same row and column
      **/
    GivensRotation<T> operator*(const GivensRotation<T>& A) const
    {
        GivensRotation<T> r(*this);
        r *= A;
        return r;
    }
};