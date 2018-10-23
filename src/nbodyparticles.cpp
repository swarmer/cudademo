#include <random>

#include "nbodyparticles.h"


vector<tuple<float, float, float, float>> CompositeParticleGenerator::get_particles() const
{
    vector<tuple<float, float, float, float>> particles;

    for (ParticleGenerator* generator : subgenerators) {
        vector<tuple<float, float, float, float>> subparticles = generator->get_particles();
        particles.insert(particles.end(), subparticles.cbegin(), subparticles.cend());
    }

    return particles;
}


vector<tuple<float, float, float, float>> UniformRandomParticleGenerator::get_particles() const
{
    vector<tuple<float, float, float, float>> particles;

    std::random_device random_device;
    std::mt19937 rng{random_device()};
    std::uniform_real_distribution<float> x_distr{x_from, x_to};
    std::uniform_real_distribution<float> y_distr{y_from, y_to};
    std::uniform_real_distribution<float> speedx_distr{-0.0, 0.0};
    std::uniform_real_distribution<float> speedy_distr{-0.0, 0.0};

    for (size_t i = 0; i < count; ++i) {
        float x = x_distr(rng);
        float y = y_distr(rng);
        float speedx = speedx_distr(rng);
        float speedy = speedy_distr(rng);
        particles.push_back(make_tuple(x, y, speedx, speedy));
    }

    return particles;
}
