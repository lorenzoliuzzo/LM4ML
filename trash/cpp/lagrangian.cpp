#include <iostream>
#include <array>
#include <functional>
#include <cmath>

template <size_t N_BODIES, size_t N_DIM, typename... PotArgs>
class PotentialEnergy : public std::function<const double&(const std::array<std::array<double, N_DIM>, N_BODIES>& /* positions */,
                                                           const std::array<std::array<double, N_DIM>, N_BODIES>& /* velocities */,
                                                           const std::array<double, N_BODIES>&                    /* masses */)> {

    public:

        constexpr PotentialEnergy(const PotArgs&... pot_args) noexcept : pot_args{pot_args...} {
            std::cout << "PotentialEnergy constructor with some parameters... " << std::get<0>(this->pot_args) << '\n';
        }

        virtual constexpr double operator()(const std::array<std::array<double, N_DIM>, N_BODIES>& positions,
                                            const std::array<std::array<double, N_DIM>, N_BODIES>& velocities,
                                            const std::array<double, N_BODIES>& masses) const = 0; 

        std::tuple<PotArgs...> pot_args; 

};


template <size_t N_BODIES, size_t N_DIM>
class Gravity : public PotentialEnergy<N_BODIES, N_DIM, double> {

    public:
    
        constexpr Gravity(const double& g = 9.81) noexcept : PotentialEnergy<N_BODIES, N_DIM, double>(g) {
            std::cout << "Gravity constructor with some parameters... " << std::get<0>(this->pot_args) << '\n';
        }

        constexpr double operator()(const std::array<std::array<double, N_DIM>, N_BODIES>& positions,
                                    const std::array<std::array<double, N_DIM>, N_BODIES>& velocities,
                                    const std::array<double, N_BODIES>& masses) const override {
            const double g = std::get<0>(this->pot_args);

            double V = 0.0; 
            for (size_t i = 0; i < N_BODIES; ++i) 
                V += g * masses[i] * positions[i][N_DIM - 1];

            return V;
        }

};

template <size_t N_BODIES, size_t N_DIM>
class Elastic : public PotentialEnergy<N_BODIES, N_DIM, double, double, std::array<std::array<double, N_DIM>, N_BODIES>> {

    public:
    
        constexpr Elastic(const double& k, const double& rest_l = 0.0, const std::array<std::array<double, N_DIM>, N_BODIES>& fixed_pts = {}) noexcept 
            : PotentialEnergy<N_BODIES, N_DIM, double, double, std::array<std::array<double, N_BODIES>, N_DIM>>(k, rest_l, fixed_pts) {
            std::cout << "Elastic constructor with some parameters... " << std::get<0>(this->pot_args) << '\n';
        }

        constexpr double operator()(const std::array<std::array<double, N_DIM>, N_BODIES>& positions,
                                    const std::array<std::array<double, N_DIM>, N_BODIES>& velocities,
                                    const std::array<double, N_BODIES>& masses) const override {
            const double k = std::get<0>(this->pot_args);
            const double rl = std::get<1>(this->pot_args);
            const auto fp = std::get<0>(this->pot_args); 

            double V = 0.0; 
            for (size_t i = 0; i < N_BODIES; ++i) 
                for (size_t j = 0; j < N_DIM; ++j) {
                    const double enl = positions[i][j] - fp[i][j];
                    V += 0.5 * k * (std::pow(enl, 2) - rl);
                }

            return V;
        }

};

template <size_t N_BODIES, size_t N_DIM, typename... PotFns>
class Lagrangian : public std::function<const double&(const std::array<std::array<double, N_DIM>, N_BODIES>& /* positions */,
                                                      const std::array<std::array<double, N_DIM>, N_BODIES>& /* velocities */,
                                                      const std::array<double, N_BODIES>&                    /* masses */)> {

    public: 

        constexpr double operator()(const std::array<std::array<double, N_DIM>, N_BODIES>& positions,
                                    const std::array<std::array<double, N_DIM>, N_BODIES>& velocities,
                                    const std::array<double, N_BODIES>& masses) const {

            return this->kinetic_energy(velocities, masses) - this->potential_energy(positions, velocities, masses); 

        }

                                               
        constexpr double kinetic_energy(const std::array<std::array<double, N_DIM>, N_BODIES>& velocities,
                                        const std::array<double, N_BODIES>& masses) const {
            double kinetic_energy = 0.0;
            for (size_t i = 0; i < N_BODIES; ++i) 
                for (size_t j = 0; j < N_DIM; ++j) 
                    kinetic_energy += 0.5 * masses[i] * velocities[i][j] * velocities[i][j];
            return kinetic_energy;
        }

        constexpr double potential_energy(const std::array<std::array<double, N_DIM>, N_BODIES>& positions,
                                          const std::array<std::array<double, N_DIM>, N_BODIES>& velocities,
                                          const std::array<double, N_BODIES>& masses) const {
            double potential_energy = 0.0;
            std::apply(
                [&](const auto&... pot_fn) {
                    ((potential_energy += pot_fn(positions, velocities, masses)), ...);
                }, this->potentials
            );
            return potential_energy;        
        }
        
    private:
        std::tuple<PotFns...> potentials; 

}; 


int main() {

    constexpr size_t N_bodies = 1;
    constexpr size_t N_dim = 3;
    auto E = Elastic<N_bodies, N_dim>(10);

    Lagrangian<N_bodies, N_dim, Gravity<N_bodies, N_dim>, E> L{}; 

    // Example values
    std::array<std::array<double, N_dim>, N_bodies> positions = {1.0, 2.0, 5.0};  // Example position for each body
    std::array<std::array<double, N_dim>, N_bodies> velocities = {0.0, 0.0, 0.0}; // Example velocity for each body
    std::array<double, N_bodies> masses = {1.0}; // Example mass for each body

    // Calculate Lagrangian
    double lagrangian_value = L(positions, velocities, masses);

    std::cout << "Lagrangian Value: " << lagrangian_value << std::endl;

    return 0; 
}