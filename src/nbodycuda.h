#ifndef NBODY_NBODYCUDA_H
#define NBODY_NBODYCUDA_H

#include <cstdint>
#include <memory>
#include <vector>

using std::vector;


class NBodyRenderer {
    const size_t m_width, m_height;

    vector<uint32_t> framebuf;

public:
    NBodyRenderer(size_t width = 1400, size_t height = 800);

    NBodyRenderer(const NBodyRenderer&) = delete;
    NBodyRenderer& operator=(const NBodyRenderer&) = delete;
    NBodyRenderer(NBodyRenderer&&) = default;
    NBodyRenderer& operator=(NBodyRenderer&&) = default;

    void update();

    int width();
    int height();
    size_t buffer_size() const;
    const vector<uint32_t>& get_buffer();
};

#endif
