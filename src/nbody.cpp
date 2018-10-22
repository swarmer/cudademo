#include <cstdint>
#include <memory>

#include <SDL.h>

using std::unique_ptr;


class NBodyRenderer {
    const int m_width, m_height;

    uint32_t *pixels;

public:
    NBodyRenderer(int width = 640, int height = 480)
        : m_width{width}, m_height{height}
    {
        pixels = new uint32_t[width * height];
        memset(pixels, 0, width * height * sizeof(uint32_t));
    }

    NBodyRenderer(const NBodyRenderer&) = delete;
    NBodyRenderer& operator=(const NBodyRenderer&) = delete;
    NBodyRenderer(NBodyRenderer&&) = default;
    NBodyRenderer& operator=(NBodyRenderer&&) = default;

    void update() {
        // TODO
    }

    int width() { return m_width; }
    int height() { return m_height; }
    const uint32_t* get_buffer() { return pixels; }

    ~NBodyRenderer() {
        delete[] pixels;
    }
};


class NBodySDLWindow {
    const unique_ptr<NBodyRenderer> nbody_renderer;

    SDL_Window *sdl_window;
    SDL_Renderer *sdl_renderer;
    SDL_Texture *sdl_texture;

public:
    explicit NBodySDLWindow(unique_ptr<NBodyRenderer> nbody_renderer_)
        : nbody_renderer{std::move(nbody_renderer_)}
    {
        SDL_Init(SDL_INIT_VIDEO);
        sdl_window = SDL_CreateWindow(
            "NBody Simulation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            nbody_renderer->width(), nbody_renderer->height(), 0
        );
        sdl_renderer = SDL_CreateRenderer(sdl_window, -1, 0);
        sdl_texture = SDL_CreateTexture(
            sdl_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC,
            nbody_renderer->width(), nbody_renderer->height()
        );
    }

    NBodySDLWindow(const NBodySDLWindow&) = delete;
    NBodySDLWindow& operator=(const NBodySDLWindow&) = delete;
    NBodySDLWindow(NBodySDLWindow&&) = default;
    NBodySDLWindow& operator=(NBodySDLWindow&&) = default;

    void run_event_loop() {
        while (true)
        {
            // handle exit events
            SDL_Event event;
            SDL_WaitEvent(&event);
            if (event.type == SDL_QUIT) {
                break;
            }

            // run and render a simulation step
            nbody_renderer->update();

            // draw a newly rendered frame
            SDL_UpdateTexture(
                sdl_texture, NULL, nbody_renderer->get_buffer(),
                nbody_renderer->width() * sizeof(uint32_t)
            );
            SDL_RenderClear(sdl_renderer);
            SDL_RenderCopy(sdl_renderer, sdl_texture, NULL, NULL);
            SDL_RenderPresent(sdl_renderer);
        }
    }

    ~NBodySDLWindow() {
        SDL_DestroyTexture(sdl_texture);
        SDL_DestroyRenderer(sdl_renderer);
        SDL_DestroyWindow(sdl_window);
        SDL_Quit();
    }
};


int main(int argc, char *argv[]) {
    NBodySDLWindow nbody_window{std::make_unique<NBodyRenderer>()};
    nbody_window.run_event_loop();
    return 0;
}
