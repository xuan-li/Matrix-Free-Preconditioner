#include "Oracle.h"

class DiagonalPreconditionedOracle: public Oracle
{
public:
    using Oracle::Vec;
    void precondition(const Vec& v, Vec& Pv) const override {
        Pv = v;
        for (int i = 0; i < this->diagonal_entries.size(); ++i) {
            if (abs(this->diagonal_entries[i]) > 0)
                Pv[i] /= this->diagonal_entries[i];
        }
    }
};