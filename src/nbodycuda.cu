#include "nbodycuda.h"

using std::tie;


constexpr inline size_t coords2d_to_1d(size_t row_size, size_t x, size_t y)
{
    return row_size * y + x;
}


NBodyRenderer::NBodyRenderer(size_t width, size_t height)
    : m_width{width}, m_height{height}, framebuf((unsigned int)(width * height))
{
    auto particle_generator = UniformRandomParticleGenerator{
        0.0f, (float)width,
        0.0f, (float)height,
        3000u,
    };

    particles = particle_generator.get_particles();
}

void NBodyRenderer::update()
{
    // TODO: update particle positions

    for (auto &coords: particles) {
        float x, y;
        tie(x, y) = coords;

        framebuf[coords2d_to_1d(m_width, (size_t)x, (size_t)y)] = 0x00FFFFFF;
    }
}

int NBodyRenderer::width() { return m_width; }
int NBodyRenderer::height() { return m_height; }
size_t NBodyRenderer::buffer_size() const { return m_width * m_height; }
const vector<uint32_t>& NBodyRenderer::get_buffer() { return framebuf; }
