#include <algorithm>
#include <cmath>

#include "nbodycuda.h"

using std::tie;


template<class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    if (v < lo)
        return lo;
    else if (v > hi)
        return hi;
    else
        return v;
}


constexpr inline float particle_render_kernel(float x1, float y1, float x2, float y2)
{
    constexpr float max_l1 = 100;
    if (abs(x1 - x2) > max_l1 || abs(y1 - y2) > max_l1)
        return 0.0f;

    constexpr float max_value = 250;
    float distance = std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
    float value = 1 / std::pow(distance, 1.3) * max_value;
    return clamp(value, 0.0f, max_value);
}


constexpr inline size_t coords2d_to_1d(size_t row_size, size_t x, size_t y)
{
    return row_size * y + x;
}


NBodyRenderer::NBodyRenderer(size_t width, size_t height)
    : m_width{width}, m_height{height}
{
    auto particle_generator = UniformRandomParticleGenerator{
        0.0f, (float)width,
        0.0f, (float)height,
        300u,
    };

    vector<tuple<float, float>> particles = particle_generator.get_particles();

#ifdef USE_CUDA
    cudaMallocManaged(&frame_buffer, m_width * m_height * sizeof(uint32_t));
    cudaMallocManaged(&particle_x_arr, particles.size() * sizeof(float));
    cudaMallocManaged(&particly_y_arr, particles.size() * sizeof(float));
#else
    frame_buffer = (uint32_t*)malloc(m_width * m_height * sizeof(uint32_t));
    particle_x_arr = (float*)malloc(particles.size() * sizeof(float));
    particly_y_arr = (float*)malloc(particles.size() * sizeof(float));
#endif

    particle_count = particles.size();
    for (size_t i = 0; i < particle_count; ++i) {
        float x, y;
        tie(x, y) = particles[i];

        particle_x_arr[i] = x;
        particly_y_arr[i] = y;
    }
}

NBodyRenderer::~NBodyRenderer()
{
#ifdef USE_CUDA
    cudaFree(frame_buffer);
    cudaFree(particle_x_arr);
    cudaFree(particly_y_arr);
#else
    free(frame_buffer);
    free(particle_x_arr);
    free(particly_y_arr);
#endif
}

void NBodyRenderer::update_software()
{
    // TODO: update particle positions

    for (size_t pixel_x = 0; pixel_x < m_width; ++pixel_x) {
        for (size_t pixel_y = 0; pixel_y < m_height; ++pixel_y) {
            float brightness = 0;

            for (size_t i = 0; i < particle_count; ++i) {
                float x = particle_x_arr[i];
                float y = particly_y_arr[i];

                brightness += particle_render_kernel(pixel_x, pixel_y, x, y);
            }

            uint32_t channel = clamp((int)brightness, 0, 255);
            uint32_t pixel = (
                (channel << 16)
                | (channel << 8)
                | channel
            );
            frame_buffer[coords2d_to_1d(m_width, pixel_x, pixel_y)] = pixel;
        }
    }
}

void NBodyRenderer::update_cuda()
{
    // TODO
}

void NBodyRenderer::update()
{
#ifdef USE_CUDA
    update_cuda();
#else
    update_software();
#endif
}

int NBodyRenderer::width() { return m_width; }
int NBodyRenderer::height() { return m_height; }
size_t NBodyRenderer::buffer_size() const { return m_width * m_height; }
const uint32_t* NBodyRenderer::get_buffer() { return frame_buffer; }
