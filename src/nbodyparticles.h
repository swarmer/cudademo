#ifndef NBODY_NBODYPARTICLES_H
#define NBODY_NBODYPARTICLES_H

#include <vector>
#include <tuple>

using std::make_tuple;
using std::vector;
using std::tuple;


struct ParticleGenerator {
    virtual vector<tuple<float, float>> get_particles() const = 0;
};


struct CompositeParticleGenerator : public ParticleGenerator {
    vector<ParticleGenerator*> subgenerators;

    vector<tuple<float, float>> get_particles() const override;
};


struct UniformRandomParticleGenerator : public ParticleGenerator {
    float x_from, x_to;
    float y_from, y_to;
    size_t count;

    UniformRandomParticleGenerator(
        float x_from_, float x_to_,
        float y_from_, float y_to_,
        size_t count_
    ) : x_from(x_from_), x_to(x_to_), y_from(y_from_), y_to(y_to_), count(count_)
    {}

    vector<tuple<float, float>> get_particles() const override;
};

#endif
