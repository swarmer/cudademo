#include "nbodycuda.h"


NBodyRenderer::NBodyRenderer(size_t width, size_t height)
    : m_width{width}, m_height{height}, framebuf((unsigned int)(width * height))
{
    // TODO
}

void NBodyRenderer::update()
{
    static long step_r = 0;
    static long step_g = 64;
    static long step_b = 128;

    uint32_t color = (
        ((step_r % 255) << 16)
        | ((step_g % 255) << 8)
        | (step_b % 255)
    );

    framebuf.assign(buffer_size(), color);

    ++step_r;
    ++step_g;
    ++step_b;
}

int NBodyRenderer::width() { return m_width; }
int NBodyRenderer::height() { return m_height; }
size_t NBodyRenderer::buffer_size() const { return m_width * m_height; }
const vector<uint32_t>& NBodyRenderer::get_buffer() { return framebuf; }
