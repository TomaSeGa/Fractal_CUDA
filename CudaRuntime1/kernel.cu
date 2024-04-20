#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WIDTH 9600
#define HEIGHT 7200
#define MAX_ITER 1000
#define BLOCK_SIZE  256 // 256 threads per blocks

int GPU_or_CPU = 0; // 0 for GPU (API CUDA) , 1 for CPU

//structure to represent pixels
struct Pixel {
    uint8_t r, g, b;
};

#if (GPU_or_CPU == 0)

//functions for the GPU

__global__ void julia_kernel(double cx, double cy, Pixel* image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < WIDTH && y < HEIGHT) {
        double xf = 4.0 * x / WIDTH - 2.0;
        double yf = 4.0 * y / HEIGHT - 2.0;
        double zx = xf, zy = yf, tmp;
        int iter = 0;
        while (zx * zx + zy * zy < 4.0 && iter < MAX_ITER) {
            tmp = zx * zx - zy * zy + cx;
            zy = 2.0 * zx * zy + cy;
            zx = tmp;
            iter++;
        }
        
        double color = (double)iter / MAX_ITER;
        image[y * WIDTH + x].r = (uint8_t)((color) * 193);
        image[y * WIDTH + x].g = (uint8_t)((color) * 182);
        image[y * WIDTH + x].b = (uint8_t)((color) * 255);
    }
}


//image bitmap creation
void write_bitmap(const char* filename, Pixel* image) {
    std::ofstream fp(filename, std::ios::binary);
    if (!fp.is_open()) {
        std::cerr << "Error: Couldn't open file " << filename << " for writing" << std::endl;
        return;
    }

    int padding = (4 - (WIDTH * sizeof(Pixel)) % 4) % 4;
    int filesize = 54 + (WIDTH * sizeof(Pixel) + padding) * HEIGHT;
    uint8_t fileheader[14] = {
        'B', 'M',        // Signature
        (uint8_t)(filesize & 0xFF), // File size
        (uint8_t)((filesize >> 8) & 0xFF),
        (uint8_t)((filesize >> 16) & 0xFF),
        (uint8_t)((filesize >> 24) & 0xFF),
        0, 0, 0, 0,       // Reserved
        54, 0, 0, 0       // Offset to pixel data
    };
    fp.write(reinterpret_cast<char*>(fileheader), sizeof(uint8_t) * 14);

    uint8_t infoheader[40] = {
        40, 0, 0, 0,      // Info header size
        (uint8_t)(WIDTH & 0xFF),     // Image width
        (uint8_t)((WIDTH >> 8) & 0xFF),
        (uint8_t)((WIDTH >> 16) & 0xFF),
        (uint8_t)((WIDTH >> 24) & 0xFF),
        (uint8_t)(HEIGHT & 0xFF),    // Image height
        (uint8_t)((HEIGHT >> 8) & 0xFF),
        (uint8_t)((HEIGHT >> 16) & 0xFF),
        (uint8_t)((HEIGHT >> 24) & 0xFF),
        1, 0,             // Number of color planes
        24, 0,            // Bits per pixel
        0, 0, 0, 0,       // Compression
        0, 0, 0, 0,       // Image size
        0, 0, 0, 0,       // Horizontal resolution
        0, 0, 0, 0,       // Vertical resolution
        0, 0, 0, 0        // Colors in palette
    };
    fp.write(reinterpret_cast<char*>(infoheader), sizeof(uint8_t) * 40);

    for (int y = HEIGHT - 1; y >= 0; y--) {
        for (int x = 0; x < WIDTH; x++) {
            fp.write(reinterpret_cast<char*>(&image[y * WIDTH + x]), sizeof(Pixel));
        }
        uint8_t padding_byte = 0;
        fp.write(reinterpret_cast<char*>(&padding_byte), sizeof(uint8_t) * padding);
    }

    fp.close();
}



int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Pixel* image;
    cudaMallocManaged(&image, WIDTH * HEIGHT * sizeof(Pixel));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord(start);
    julia_kernel<<<gridSize, blockSize>>>(-0.8, 0.156, image); //kernel function call
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    write_bitmap("julia_fractal_GPU_time16.bmp", image);

    cudaFree(image);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds/1000;
    std::cout << "Execution time: " << seconds << std::endl;

    return 0;
}

#elif (GPU_or_CPU == 1)

// CODE C++ CPU (CUDA) 

//function to create a btimap from all the calculated pixels
void write_bitmap(const char* filename, Pixel** image) {
    std::ofstream fp(filename, std::ios::binary);
    if (!fp.is_open()) {
        std::cerr << "Error: Couldn't open file " << filename << " for writing" << std::endl;
        return;
    }

    int padding = (4 - (WIDTH * sizeof(Pixel)) % 4) % 4;
    int filesize = 54 + (WIDTH * sizeof(Pixel) + padding) * HEIGHT;
    uint8_t fileheader[14] = {
        'B', 'M',        // Signature
        (uint8_t)(filesize & 0xFF), // File size
        (uint8_t)((filesize >> 8) & 0xFF),
        (uint8_t)((filesize >> 16) & 0xFF),
        (uint8_t)((filesize >> 24) & 0xFF),
        0, 0, 0, 0,       // Reserved
        54, 0, 0, 0       // Offset to pixel data
    };
    fp.write(reinterpret_cast<char*>(fileheader), sizeof(uint8_t) * 14);

    uint8_t infoheader[40] = {
        40, 0, 0, 0,      // Info header size
        (uint8_t)(WIDTH & 0xFF),     // Image width
        (uint8_t)((WIDTH >> 8) & 0xFF),
        (uint8_t)((WIDTH >> 16) & 0xFF),
        (uint8_t)((WIDTH >> 24) & 0xFF),
        (uint8_t)(HEIGHT & 0xFF),    // Image height
        (uint8_t)((HEIGHT >> 8) & 0xFF),
        (uint8_t)((HEIGHT >> 16) & 0xFF),
        (uint8_t)((HEIGHT >> 24) & 0xFF),
        1, 0,             // Number of color planes
        24, 0,            // Bits per pixel
        0, 0, 0, 0,       // Compression
        0, 0, 0, 0,       // Image size
        0, 0, 0, 0,       // Horizontal resolution
        0, 0, 0, 0,       // Vertical resolution
        0, 0, 0, 0        // Colors in palette
    };
    fp.write(reinterpret_cast<char*>(infoheader), sizeof(uint8_t) * 40);

    for (int y = HEIGHT - 1; y >= 0; y--) {
        for (int x = 0; x < WIDTH; x++) {
            fp.write(reinterpret_cast<char*>(&image[y][x]), sizeof(Pixel));
        }
        uint8_t padding_byte = 0;
        fp.write(reinterpret_cast<char*>(&padding_byte), sizeof(uint8_t) * padding);
    }

    fp.close();
}

// function to implements the algorithm for calculating the Julia set for a given point (x, y)
int julia(double cx, double cy, double x, double y) {
    double zx = x, zy = y, tmp;
    int iter = 0;
    while (zx * zx + zy * zy < 4.0 && iter < MAX_ITER) {
        tmp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = tmp;
        iter++;
    }
    return iter;
}

int main() {
    clock_t start = clock();

    Pixel** image = new Pixel * [HEIGHT];
    for (int i = 0; i < HEIGHT; i++) {
        image[i] = new Pixel[WIDTH];
    }

    //-0.7 : 0.27  heavy
    //-0.8 : 0.156 light
    //-0.7269 : 0.1889 heavy 
    double init_x = -0.8;
    double int_y = 0.156;

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double xf = 4.0 * x / WIDTH - 2.0;
            double yf = 4.0 * y / HEIGHT - 2.0;
            int iter = julia(init_x, int_y, xf, yf);
            double color = (double)iter / MAX_ITER;

            //pink light : 255,182,193
            //bleu maya : 115, 194, 251
            //bleu saphir : 1, 49, 180

            image[y][x].r = (uint8_t)(color * 251);
            image[y][x].g = (uint8_t)(color * 194);
            image[y][x].b = (uint8_t)(color * 115);
        }
    }

    write_bitmap("julia_fractal_CPU_time_test.bmp", image);

    for (int i = 0; i < HEIGHT; i++) {
        delete[] image[i];
    }
    delete[] image;

    clock_t end = clock();
    double execution_time = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Execution time: " << execution_time << " seconds" << std::endl;
    return 0;
}
#endif

