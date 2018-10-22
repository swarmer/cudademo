#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <SDL.h>

#include "nbodycuda.h"

using std::unique_ptr;
using std::vector;


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

    void run_event_loop()
    {
        while (true)
        {
            // handle exit events
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    return;
                }
            }

            auto t0 = std::chrono::system_clock::now();

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

            auto t1 = std::chrono::system_clock::now();
            std::chrono::duration<double> dt = t1 - t0;
            std::cout << "Frame update took " << dt.count() << "s\n";
        }
    }

    ~NBodySDLWindow()
    {
        SDL_DestroyTexture(sdl_texture);
        SDL_DestroyRenderer(sdl_renderer);
        SDL_DestroyWindow(sdl_window);
        SDL_Quit();
    }
};


int main(int argc, char *argv[])
{
    NBodySDLWindow nbody_window{std::make_unique<NBodyRenderer>()};
    nbody_window.run_event_loop();
    return 0;
}
