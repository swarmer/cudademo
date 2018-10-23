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
    constexpr float max_value = 250;
    float distance = std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
    float value = 1 / std::pow(distance, 1.7) * max_value;
    return value;
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
        1500u,
    };

    vector<tuple<float, float, float, float>> particles = particle_generator.get_particles();

#ifdef USE_CUDA
    cudaMallocManaged(&frame_buffer, m_width * m_height * sizeof(uint32_t));
    cudaMallocManaged(&particle_x_arr, particles.size() * sizeof(float));
    cudaMallocManaged(&particle_y_arr, particles.size() * sizeof(float));
    cudaMallocManaged(&particle_x_speed, particles.size() * sizeof(float));
    cudaMallocManaged(&particle_y_speed, particles.size() * sizeof(float));
#else
    frame_buffer = (uint32_t*)malloc(m_width * m_height * sizeof(uint32_t));
    particle_x_arr = (float*)malloc(particles.size() * sizeof(float));
    particle_y_arr = (float*)malloc(particles.size() * sizeof(float));
    particle_x_speed = (float*)malloc(particles.size() * sizeof(float));
    particle_y_speed = (float*)malloc(particles.size() * sizeof(float));
#endif

    particle_count = particles.size();
    for (size_t i = 0; i < particle_count; ++i) {
        particle_x_arr[i] = std::get<0>(particles[i]);
        particle_y_arr[i] = std::get<1>(particles[i]);
        particle_x_speed[i] = std::get<2>(particles[i]);
        particle_y_speed[i] = std::get<3>(particles[i]);
    }
}

NBodyRenderer::~NBodyRenderer()
{
#ifdef USE_CUDA
    cudaFree(frame_buffer);
    cudaFree(particle_x_arr);
    cudaFree(particle_y_arr);
    cudaFree(particle_x_speed);
    cudaFree(particle_y_speed);
#else
    free(frame_buffer);
    free(particle_x_arr);
    free(particle_y_arr);
    free(particle_x_speed);
    free(particle_y_speed);
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
            ssize_t pixel_x = center_pixel_x + (window_size * ((ssize_t)threadIdx.x - 2)) + shift_x;
            ssize_t pixel_y = center_pixel_y + (window_size * ((ssize_t)threadIdx.y - 2)) + shift_y;

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

            uint32_t *pixel_addr = &frame_buffer[coords2d_to_1d(width, pixel_x, pixel_y)];
            uint32_t current = *pixel_addr;
            uint32_t assumed = 0;
            do {
                assumed = current;
                current = atomicCAS(
                    (unsigned int*)pixel_addr, assumed,
                    clamp(assumed + dpixel, 0u, 0x00FFFFFFu)
                );
            } while (assumed != current);
        }
    }
}

__global__ void cuda_accelerate(
    float particle_x_arr[],
    float particle_y_arr[],
    float particle_x_speed[],
    float particle_y_speed[],
    size_t particle_count
)
{
    // target is one being accelerated, source is one applying force
    const size_t particle_target = blockIdx.x * 10 + threadIdx.x;
    const size_t particle_source = blockIdx.y * 10 + threadIdx.y;
    if (particle_target >= particle_count || particle_source >= particle_count)
        return;
    if (particle_target == particle_source)
        return;

    float targetx = particle_x_arr[particle_target];
    float targety = particle_y_arr[particle_target];
    float sourcex = particle_x_arr[particle_source];
    float sourcey = particle_y_arr[particle_source];

    constexpr float g = 0.001;
    constexpr float maxaccel = 0.001;
    float distance = std::sqrt(std::pow(sourcex - targetx, 2) + std::pow(sourcey - targety, 2));
    float accel = 1 / distance * g;
    float accelx = (sourcex - targetx) / distance * accel;
    float accely = (sourcey - targety) / distance * accel;
    accelx = clamp(accelx, -maxaccel, maxaccel);
    accely = clamp(accely, -maxaccel, maxaccel);

    atomicAdd(&particle_x_speed[particle_target], accelx);
    atomicAdd(&particle_y_speed[particle_target], accely);
}

__global__ void cuda_move(
    float particle_x_arr[],
    float particle_y_arr[],
    float particle_x_speed[],
    float particle_y_speed[],
    size_t particle_count
)
{
    const size_t particle_index = blockIdx.x * 100 + threadIdx.x;
    if (particle_index >= particle_count)
        return;

    float speedx = particle_x_speed[particle_index];
    float speedy = particle_y_speed[particle_index];

    atomicAdd(&particle_x_arr[particle_index], speedx);
    atomicAdd(&particle_y_arr[particle_index], speedy);
}

void NBodyRenderer::update_cuda()
{
    // clear the frame
    cudaMemset(frame_buffer, 0, m_width * m_height * sizeof(uint32_t));

    // accelerate particles
    dim3 athreads_per_block(10, 10, 1);  // 10 particles x 10 particles
    dim3 ablocks_count(ceil(particle_count / 10), ceil(particle_count / 10), 1);
    cuda_accelerate<<<ablocks_count, athreads_per_block>>>(
        particle_x_arr, particle_y_arr,
        particle_x_speed, particle_y_speed,
        particle_count
    );

    // move particles
    dim3 mthreads_per_block(100, 1, 1);
    dim3 mblocks_count(ceil(particle_count / 100), 1, 1);
    cuda_move<<<mblocks_count, mthreads_per_block>>>(
        particle_x_arr, particle_y_arr,
        particle_x_speed, particle_y_speed,
        particle_count
    );

    // render particles
    dim3 rthreads_per_block(5, 5, 10);  // 10 particles x 5 column blocks x 5 rows blocks
    dim3 rblocks_count(ceil(particle_count / 10), 1, 1);
    cuda_render<<<rblocks_count, rthreads_per_block>>>(
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
