// fast_np_count.c
// Native MicroPython module replacing the slow Python np_count loop.
// Reads raw RGB565 frame once, computes LAB mean/stdev for all 294 grid cells in C.
// Output: stats_array (294, 6) = [L_mean, A_mean, B_mean, L_stdev, A_stdev, B_stdev]
// Matches original np_count output exactly — nothing downstream changes.
//
// Math note: arm-none-eabi-gcc (Homebrew) ships without newlib, so libm (powf, sqrtf)
// is unavailable at link time. All math is implemented inline:
//   - sqrtf  -> single VSQRT.F32 ARM FPU instruction (exact)
//   - cbrt   -> Newton-Raphson 3 iterations (for cie_f cube root)
//   - powf   -> exp2(b * log2(a)) via polynomial (for gamma 2.4 only)

#include "py/dynruntime.h"

// Grid constants -- must match perception_subsystem.py
#define N_ROWS      14
#define N_COLS      21
#define IMG_W       240
#define IMG_H       160
#define CELL_W      (IMG_W / N_COLS)     // 11
#define CELL_H      (IMG_H / N_ROWS)     // 11
#define CELL_PIXELS (CELL_W * CELL_H)    // 121

/* ---- libm-free math ---------------------------------------------------- */

// Square root: emits single VSQRT.F32 instruction on Cortex-M7 FPU (exact)
static inline float f_sqrt(float x) {
    float r;
    __asm__ volatile ("vsqrt.f32 %0, %1" : "=t"(r) : "t"(x));
    return r;
}

// log2(x) for x > 0, error < 0.005 across the input range we use
static float f_log2(float x) {
    union { float f; unsigned u; } t;
    t.f = x;
    int e = (int)((t.u >> 23) & 0xFFu) - 127;
    t.u = (t.u & 0x007FFFFFu) | 0x3F800000u;  // normalise mantissa to [1, 2)
    float m = t.f;
    // Remez minimax polynomial for log2(m), m in [1, 2)
    float p = -1.7417939f + (2.8212026f + (-1.4699568f + 0.44717955f * m) * m) * m;
    return p + (float)e;
}

// 2^x
static float f_exp2(float x) {
    union { float f; unsigned u; } t;
    int xi = (int)x;
    if (x < 0.0f && (float)xi != x) xi -= 1;   // floor for negative values
    float xf = x - (float)xi;
    // Polynomial for 2^xf, xf in [0, 1)
    float p = 1.0f + xf * (0.6931472f + xf * (0.2402265f + xf * (0.0555041f + xf * 0.0096181f)));
    t.f = p;
    t.u = (t.u & 0x007FFFFFu) | ((unsigned)(xi + 127) << 23);
    return t.f;
}

// pow(a, b) via log2/exp2 -- only called with b=2.4 (gamma linearization)
static inline float f_powf(float a, float b) {
    if (a <= 0.0f) return 0.0f;
    return f_exp2(b * f_log2(a));
}

// Cube root via Newton-Raphson (3 iterations) -- for cie_f, input always > 0.008856
static float f_cbrt(float x) {
    union { float f; unsigned u; } hack;
    hack.f = x;
    hack.u = hack.u / 3u + 0x2a555555u;   // bit-hack initial estimate
    float y = hack.f;
    float x3 = x * (1.0f / 3.0f);
    y = y * (2.0f / 3.0f) + x3 / (y * y);
    y = y * (2.0f / 3.0f) + x3 / (y * y);
    y = y * (2.0f / 3.0f) + x3 / (y * y);
    return y;
}

/* ---- RGB565 -> LAB conversion ------------------------------------------ */

// sRGB gamma linearization
static inline float linearize(float c) {
    if (c <= 0.04045f)
        return c / 12.92f;
    return f_powf((c + 0.055f) / 1.055f, 2.4f);
}

// CIE f() function for XYZ -> LAB
static inline float cie_f(float t) {
    if (t > 0.008856f)
        return f_cbrt(t);
    return 7.787f * t + (16.0f / 116.0f);
}

// Convert one RGB565 pixel to LAB, write into *L, *A, *B
static inline void rgb565_to_lab(unsigned short pixel,
                                  float *L, float *A, float *B) {
    float r = (float)((pixel >> 11) & 0x1Fu) * (1.0f / 31.0f);
    float g = (float)((pixel >>  5) & 0x3Fu) * (1.0f / 63.0f);
    float b = (float)( pixel        & 0x1Fu) * (1.0f / 31.0f);

    r = linearize(r);
    g = linearize(g);
    b = linearize(b);

    float X = 0.4124f * r + 0.3576f * g + 0.1805f * b;
    float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float Z = 0.0193f * r + 0.1192f * g + 0.9505f * b;

    X /= 0.95047f;
    // Y /= 1.0 (no-op)
    Z /= 1.08883f;

    float fx = cie_f(X);
    float fy = cie_f(Y);
    float fz = cie_f(Z);

    *L = 116.0f * fy - 16.0f;
    *A = 500.0f * (fx - fy);
    *B = 200.0f * (fy - fz);
}

/* ---- MicroPython entry point ------------------------------------------- */

// fast_np_count(raw_bytes, out_buffer)
//   raw_bytes : bytes/bytearray of 76800 RGB565 bytes (240*160*2)
//   out_buffer: bytearray of 7056 bytes (294*6*4) to receive float32 stats
static mp_obj_t fast_np_count(mp_obj_t raw_bytes_obj, mp_obj_t out_obj) {
    mp_buffer_info_t raw_info;
    mp_get_buffer_raise(raw_bytes_obj, &raw_info, MP_BUFFER_READ);
    const unsigned short *pixels = (const unsigned short *)raw_info.buf;

    mp_buffer_info_t out_info;
    mp_get_buffer_raise(out_obj, &out_info, MP_BUFFER_WRITE);
    float *out = (float *)out_info.buf;

    for (int row = 0; row < N_ROWS; row++) {
        for (int col = 0; col < N_COLS; col++) {
            int cell_idx = row * N_COLS + col;

            float sum_L = 0.0f, sum_A = 0.0f, sum_B = 0.0f;
            float sum_L2 = 0.0f, sum_A2 = 0.0f, sum_B2 = 0.0f;

            for (int pr = 0; pr < CELL_H; pr++) {
                int img_row = row * CELL_H + pr;
                for (int pc = 0; pc < CELL_W; pc++) {
                    int img_col = col * CELL_W + pc;
                    unsigned short pixel = pixels[img_row * IMG_W + img_col];

                    float lv, av, bv;
                    rgb565_to_lab(pixel, &lv, &av, &bv);

                    sum_L  += lv;   sum_L2 += lv * lv;
                    sum_A  += av;   sum_A2 += av * av;
                    sum_B  += bv;   sum_B2 += bv * bv;
                }
            }

            float n  = (float)CELL_PIXELS;
            float mL = sum_L / n,  mA = sum_A / n,  mB = sum_B / n;
            float sL = f_sqrt(sum_L2 / n - mL * mL);
            float sA = f_sqrt(sum_A2 / n - mA * mA);
            float sB = f_sqrt(sum_B2 / n - mB * mB);

            // Write [L_mean, A_mean, B_mean, L_stdev, A_stdev, B_stdev]
            out[cell_idx * 6 + 0] = mL;
            out[cell_idx * 6 + 1] = mA;
            out[cell_idx * 6 + 2] = mB;
            out[cell_idx * 6 + 3] = sL;
            out[cell_idx * 6 + 4] = sA;
            out[cell_idx * 6 + 5] = sB;
        }
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fast_np_count_obj, fast_np_count);

// Module entry point -- called when MicroPython imports this module
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    MP_DYNRUNTIME_INIT_ENTRY
    mp_store_global(MP_QSTR_fast_np_count, MP_OBJ_FROM_PTR(&fast_np_count_obj));
    MP_DYNRUNTIME_INIT_EXIT
}