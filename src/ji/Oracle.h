#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace ji {

using Vec = std::array<double, 6>;
using Mat = std::array<double, 36>; // row-major, six rows even for a three-row contact
inline constexpr uint32_t NoBody = std::numeric_limits<uint32_t>::max();

enum class Kind : uint8_t { Soft, Hard, Contact };

// A and b are the frozen six-coordinate discrete dynamics A*v = b + J^T*lambda.
// A is diagonal and positive for a moving body. A fixed endpoint uses NoBody.
struct Body {
    Vec a{}, b{}, v{};
    double beta = 0;
};

struct Side {
    uint32_t body = NoBody;
    Mat j{};
    Vec x{}, z{}, u{};
};

struct Constraint {
    Kind kind = Kind::Hard;
    uint32_t rows = 3;
    Side side[2];
    Vec error{}, impulse{}, velocity_bias{};
    // Soft rows: dt-scaled prescribed endpoint velocity, separate from position error.
    double stiffness = 0, damping = 0, friction = 0;
};

struct System {
    std::vector<Body> bodies;
    std::vector<Constraint> constraints;
};

struct Residual {
    double primal = 0;   // Eq. (16), infinity norm of x-z
    double dual = 0;     // Eq. (17), infinity norm of beta*sum(J^T*(z_old-z_new))
    double dynamics = 0; // Eq. (1), infinity norm of A*v-b-sum(J^T*lambda)
    double contact_law = 0; // strict normal-first/tangent-disk fixed-point residual, multiplier units
};

// Reject malformed or nonfinite frozen systems before either solver runs.
// Penalties may be absent only when the caller will initialize them first.
void ValidateSystem(const System &, bool require_penalties = true);

// Eq. (13): M*Tr(A_i)/Tr(C_i)+c, with C_i=sum(J_i,k^T*J_i,k).
// A body with no constraint rows receives c; its beta is unused.
void InitializePenalties(System &, double M = 16, double c = 1e5);

// Eq. (15), applied to every body. For zero residual denominators the ratio
// takes the corresponding clamp endpoint; both zero leaves beta unchanged.
void UpdatePenalties(System &, Residual, double alpha = 4);

// Strict SCC projection: clamp the normal first, then project tangentially
// onto the disk of radius mu*normal. This is not Euclidean cone projection.
std::array<double, 3> StrictContactProjection(std::array<double, 3>, double mu);
double ContactLawResidual(const System &, const Constraint &, const Vec &relative);

// One ordered pass of Eqs. (6)-(10), followed by independently assembled
// Eq. (1), (16) and (17) residuals. Geometry and Jacobians remain frozen.
Residual Iterate(System &);

struct Settings {
    uint32_t iterations = 100;
    uint32_t penalty_period = 0; // 0 disables Eq. (15)
    double alpha = 4;
    double tolerance = 0; // 0 runs exactly `iterations` passes
};
struct Result {
    Residual residual;
    uint32_t iterations = 0;
    bool converged = false;
};
Result Solve(System &, Settings = {});

} // namespace ji
