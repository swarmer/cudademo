#include <algorithm>
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "nbodycuda.h"

using std::tie;


template<class T>
__device__ __host__ constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    if (v < lo)
        return lo;
    else if (v > hi)
        return hi;
    else
        return v;
}


__device__ __host__ constexpr float particle_render_kernel(
    float x1, float y1, float x2, float y2
)
{
    constexpr float max_l1 = 50;
    if (abs(x1 - x2) > max_l1 || abs(y1 - y2) > max_l1)
        return 0.0f;

    constexpr float max_value = 250;
    float distance = std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
    float value = 1 / std::pow(distance, 1.3) * max_value;
    return clamp(value, 0.0f, max_value);
}


__device__ __host__ constexpr inline size_t coords2d_to_1d(size_t row_size, size_t x, size_t y)
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
    cudaMallocManaged(&particle_y_arr, particles.size() * sizeof(float));
#else
    frame_buffer = (uint32_t*)malloc(m_width * m_height * sizeof(uint32_t));
    particle_x_arr = (float*)malloc(particles.size() * sizeof(float));
    particle_y_arr = (float*)malloc(particles.size() * sizeof(float));
#endif

    particle_count = particles.size();
    for (size_t i = 0; i < particle_count; ++i) {
        float x, y;
        tie(x, y) = particles[i];

        particle_x_arr[i] = x;
        particle_y_arr[i] = y;
    }
}

NBodyRenderer::~NBodyRenderer()
{
#ifdef USE_CUDA
    cudaFree(frame_buffer);
    cudaFree(particle_x_arr);
    cudaFree(particle_y_arr);
#else
    free(frame_buffer);
    free(particle_x_arr);
    free(particle_y_arr);
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
                float y = particle_y_arr[i];

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

__global__ void cuda_render(
        uint32_t frame_buffer[],
        size_t width, size_t height,
        float particle_x_arr[],
        float particle_y_arr[],
        size_t particle_count
)
{
    const size_t particle_index = blockIdx.x * 10 + threadIdx.z;
    if (particle_index >= particle_count)
        return;

    float px = particle_x_arr[particle_index];
    float py = particle_y_arr[particle_index];

    ssize_t center_pixel_x = int(px);
    ssize_t center_pixel_y = int(py);

    constexpr ssize_t window_size = 9;
    for (ssize_t shift_x = -4; shift_x <= 4; ++shift_x) {
        for (ssize_t shift_y = -4; shift_y <= 4; ++shift_y) {
            ssize_t pixel_x = center_pixel_x + (window_size * threadIdx.x) + shift_x;
            ssize_t pixel_y = center_pixel_y + (window_size * threadIdx.y) + shift_y;

            if (pixel_x < 0 || pixel_x >= width || pixel_y < 0 || pixel_y >= height) {
                // pixel out of bounds
                continue;
            }

            float dbrightness = particle_render_kernel((float)pixel_x, (float)pixel_y, px, py);
            int dchannel = clamp((uint32_t)dbrightness, (uint32_t)0, (uint32_t)255);
            uint32_t dpixel = (
                (dchannel << 16)
                | (dchannel << 8)
                | dchannel
            );
            frame_buffer[coords2d_to_1d(width, pixel_x, pixel_y)] += dpixel;
        }
    }
}

void NBodyRenderer::update_cuda()
{
    cudaMemset(frame_buffer, 0, m_width * m_height * sizeof(uint32_t));

    dim3 threads_per_block(5, 5, 10);  // 10 particles x 5 column blocks x 5 rows blocks
    dim3 blocks_count(ceil(particle_count / 10), 1, 1);
    cuda_render<<<blocks_count, threads_per_block>>>(
        frame_buffer, m_width, m_height,
        particle_x_arr, particle_y_arr, particle_count
    );

    cudaDeviceSynchronize();
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
