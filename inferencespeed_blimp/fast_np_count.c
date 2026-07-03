// fast_np_count.c  (v2 -- LUT optimized, const arrays for .mpy compatibility)
// Native MicroPython module replacing the slow Python np_count loop.
// Reads raw RGB565 frame once, computes LAB mean/stdev for all 294 grid cells in C.
// Output: stats_array (294, 6) = [L_mean, A_mean, B_mean, L_stdev, A_stdev, B_stdev]
//
// .mpy restriction: writable static (BSS) variables are not allowed.
// LUTs are therefore declared as static const (rodata) with values precomputed
// offline: linearize(i/31.0) for 5-bit R/B, linearize(i/63.0) for 6-bit G.
// This eliminates all f_powf calls from the hot loop.

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

// Cube root via Newton-Raphson (3 iterations) -- for cie_f
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

/* ---- Gamma linearization LUT (static const = rodata, allowed in .mpy) -- */
// Values = sRGB linearize(c) = c/12.92 if c<=0.04045, else ((c+0.055)/1.055)^2.4
// Precomputed in Python offline; no runtime math needed.

// lut_r5: linearize(i/31.0) for i=0..31  -- used for R and B channels (5-bit)
static const float lut_r5[32] = {
    0.000000000f,  // i=0
    0.002496754f,  // i=1
    0.005370537f,  // i=2
    0.009529479f,  // i=3
    0.015133507f,  // i=4
    0.022298883f,  // i=5
    0.031130090f,  // i=6
    0.041722574f,  // i=7
    0.054164573f,  // i=8
    0.068538413f,  // i=9
    0.084921457f,  // i=10
    0.103386834f,  // i=11
    0.124004009f,  // i=12
    0.146839231f,  // i=13
    0.171955903f,  // i=14
    0.199414890f,  // i=15
    0.229274770f,  // i=16
    0.261592050f,  // i=17
    0.296421353f,  // i=18
    0.333815575f,  // i=19
    0.373826021f,  // i=20
    0.416502533f,  // i=21
    0.461893590f,  // i=22
    0.510046406f,  // i=23
    0.561007014f,  // i=24
    0.614820343f,  // i=25
    0.671530283f,  // i=26
    0.731179749f,  // i=27
    0.793810735f,  // i=28
    0.859464367f,  // i=29
    0.928180948f,  // i=30
    1.000000000f   // i=31
};

// lut_g6: linearize(i/63.0) for i=0..63  -- used for G channel (6-bit)
static const float lut_g6[64] = {
    0.000000000f,  // i=0
    0.001228562f,  // i=1
    0.002457123f,  // i=2
    0.003725128f,  // i=3
    0.005260758f,  // i=4
    0.007113470f,  // i=5
    0.009299643f,  // i=6
    0.011834528f,  // i=7
    0.014732435f,  // i=8
    0.018006871f,  // i=9
    0.021670655f,  // i=10
    0.025735999f,  // i=11
    0.030214582f,  // i=12
    0.035117609f,  // i=13
    0.040455856f,  // i=14
    0.046239713f,  // i=15
    0.052479218f,  // i=16
    0.059184090f,  // i=17
    0.066363750f,  // i=18
    0.074027348f,  // i=19
    0.082183781f,  // i=20
    0.090841711f,  // i=21
    0.100009582f,  // i=22
    0.109695632f,  // i=23
    0.119907908f,  // i=24
    0.130654277f,  // i=25
    0.141942434f,  // i=26
    0.153779915f,  // i=27
    0.166174104f,  // i=28
    0.179132240f,  // i=29
    0.192661425f,  // i=30
    0.206768633f,  // i=31
    0.221460712f,  // i=32
    0.236744392f,  // i=33
    0.252626292f,  // i=34
    0.269112919f,  // i=35
    0.286210680f,  // i=36
    0.303925882f,  // i=37
    0.322264736f,  // i=38
    0.341233363f,  // i=39
    0.360837795f,  // i=40
    0.381083978f,  // i=41
    0.401977780f,  // i=42
    0.423524987f,  // i=43
    0.445731311f,  // i=44
    0.468602390f,  // i=45
    0.492143791f,  // i=46
    0.516361013f,  // i=47
    0.541259489f,  // i=48
    0.566844587f,  // i=49
    0.593121612f,  // i=50
    0.620095811f,  // i=51
    0.647772370f,  // i=52
    0.676156420f,  // i=53
    0.705253036f,  // i=54
    0.735067240f,  // i=55
    0.765604000f,  // i=56
    0.796868236f,  // i=57
    0.828864815f,  // i=58
    0.861598560f,  // i=59
    0.895074245f,  // i=60
    0.929296597f,  // i=61
    0.964270302f,  // i=62
    1.000000000f   // i=63
};

/* ---- RGB565 -> LAB conversion ------------------------------------------ */

// CIE f() function for XYZ -> LAB
static inline float cie_f(float t) {
    if (t > 0.008856f)
        return f_cbrt(t);
    return 7.787f * t + (16.0f / 116.0f);
}

// Convert one RGB565 pixel to LAB using precomputed const LUTs -- no f_powf
static inline void rgb565_to_lab(unsigned short pixel,
                                  float *L, float *A, float *B) {
    float r = lut_r5[(pixel >> 11) & 0x1Fu];
    float g = lut_g6[(pixel >>  5) & 0x3Fu];
    float b = lut_r5[ pixel        & 0x1Fu];

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

// Module entry point
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    MP_DYNRUNTIME_INIT_ENTRY
    mp_store_global(MP_QSTR_fast_np_count, MP_OBJ_FROM_PTR(&fast_np_count_obj));
    MP_DYNRUNTIME_INIT_EXIT
}