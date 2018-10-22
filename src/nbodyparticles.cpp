#include <random>

#include "nbodyparticles.h"


vector<tuple<float, float>> CompositeParticleGenerator::get_particles() const
{
    vector<tuple<float, float>> particles;

    for (ParticleGenerator* generator : subgenerators) {
        vector<tuple<float, float>> subparticles = generator->get_particles();
        particles.insert(particles.end(), subparticles.cbegin(), subparticles.cend());
    }

    return particles;
}


vector<tuple<float, float>> UniformRandomParticleGenerator::get_particles() const
{
    vector<tuple<float, float>> particles;

    std::random_device random_device;
    std::mt19937 rng{random_device()};
    std::uniform_real_distribution<float> x_distr{x_from, x_to};
    std::uniform_real_distribution<float> y_distr{y_from, y_to};

    for (size_t i = 0; i < count; ++i) {
        float x = x_distr(rng);
        float y = y_distr(rng);
        particles.push_back(make_tuple(x, y));
    }

    return particles;
}
