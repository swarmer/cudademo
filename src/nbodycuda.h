#ifndef NBODY_NBODYCUDA_H
#define NBODY_NBODYCUDA_H

#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>

#include "nbodyparticles.h"

using std::vector;
using std::tuple;


// Comment to disable CUDA acceleration
#define USE_CUDA


class NBodyRenderer {
    const size_t m_width, m_height;

    uint32_t* frame_buffer = nullptr;
    size_t particle_count = 0;
    float* particle_x_arr = nullptr;
    float* particle_y_arr = nullptr;
    float* particle_x_speed = nullptr;
    float* particle_y_speed = nullptr;

public:
    NBodyRenderer(size_t width = 1400, size_t height = 800);

    NBodyRenderer(const NBodyRenderer&) = delete;
    NBodyRenderer& operator=(const NBodyRenderer&) = delete;
    NBodyRenderer(NBodyRenderer&&) = default;
    NBodyRenderer& operator=(NBodyRenderer&&) = default;

    virtual ~NBodyRenderer();

    void update_software();
    void update_cuda();

    void update();

    int width();
    int height();
    size_t buffer_size() const;
    const uint32_t* get_buffer();
};

#endif
