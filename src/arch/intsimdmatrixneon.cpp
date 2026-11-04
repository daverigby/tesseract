///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrixneon.cpp
// Description: matrix-vector product for 8-bit data on neon.
// Author:      Robin Watts (from the AVX2 original by Ray Smith)
//
// (C) Copyright 2017, Google Inc.
// (C) Copyright 2020, Artifex Software Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////

#if defined(__ARM_NEON)

#  include "intsimdmatrix.h"
#  include "tesstypes.h"

#  include <algorithm>
#  include <cstdint>
#  include <vector>
#  include "arm_neon.h"

namespace tesseract {

// Number of outputs held in each register. (Actually, we use a
// pair of 4x32 registers, so 8 x 32 bit ints).
constexpr int kNumOutputsPerRegister = 8;
// Maximum number of registers that we will use.
constexpr int kMaxOutputRegisters = 1;
// Number of inputs in the inputs register.
constexpr int kNumInputsPerRegister = 8;
// Number of inputs in each weight group.
constexpr int kNumInputsPerGroup = 8;

// Function to compute part of a matrix.vector multiplication. The weights
// are in a very specific order (see above) in w, which is multiplied by
// u of length num_in, to produce output v after scaling the integer results
// by the corresponding member of scales.
// The amount of w and scales consumed is fixed and not available to the
// caller.

// Reduce four 32-bit partial sums once per output row after the input loop.
static inline int32_t HorizontalSum32x4(int32x4_t values) {
  const int32x2_t pair = vadd_s32(vget_low_s32(values), vget_high_s32(values));
  return vget_lane_s32(vpadd_s32(pair, pair), 0);
}

// Computes part of matrix.vector v = Wu. Computes N=8 results.
// The weights *must* be arranged so that consecutive reads from wi
// provides (num_in/kNumInputsPerGroup groups of (N output dim groups of
// (kNumInputsPerGroup inputs))). After that there must be N consecutive
// bias weights, before continuing with any more weights.
// u must be padded out with zeros to
// kNumInputsPerGroup*ceil(num_in/kNumInputsPerGroup) elements.
static inline void PartialMatrixDotVector8(const int8_t *__restrict wi,
                                           const TFloat *__restrict scales,
                                           const int8_t *__restrict u, int num_in,
                                           TFloat *__restrict v, int num_out) {
  // Initialize all the results to 0.
  int32x4_t sum0 = vdupq_n_s32(0);
  int32x4_t sum1 = vdupq_n_s32(0);
  int32x4_t sum2 = vdupq_n_s32(0);
  int32x4_t sum3 = vdupq_n_s32(0);
  int32x4_t sum4 = vdupq_n_s32(0);
  int32x4_t sum5 = vdupq_n_s32(0);
  int32x4_t sum6 = vdupq_n_s32(0);
  int32x4_t sum7 = vdupq_n_s32(0);
  int8x8_t bias_scale = {127, 127, 127, 127, 127, 127, 127, 127};
  // Iterate over the input (u), one registerful at a time.
  for (int j = 0; j < num_in; j += 8) {
    int8x8_t vu = vld1_s8(u);              // vu     = u0  u1  u2  u3  u4  u5  u6  u7
    int8x16_t vw01 = vld1q_s8(wi);         // vw0    = w00 w01 w02 w03 w04 w05 w06 w07
                                           // w10 w11 w12 w13 w14 w15 w16 w17
    int8x16_t vw23 = vld1q_s8(wi + 8 * 2); // vw2    = w20 w21 w22 w23 w24 w25 w26 w27 w30
                                           // w31 w32 w33 w34 w35 w36 w37
    int8x16_t vw45 = vld1q_s8(wi + 8 * 4); // vw4    = w40 w41 w42 w43 w44 w45 w46 w47 w50
                                           // w51 w52 w53 w54 w55 w56 w57
    int8x16_t vw67 = vld1q_s8(wi + 8 * 6); // vw6    = w60 w61 w62 w63 w64 w65 w66 w67 w70
                                           // w71 w72 w73 w74 w75 w76 w77

    int16x8_t vrow0q = vmull_s8(vget_low_s8(vw01), vu); // vrow0q = vw00.u0 w01.u1 w02.u2
                                                        // w03.u3 vw04.u4 w05.u5 w06.u6 w07.u7
    int16x8_t vrow1q = vmull_s8(vget_high_s8(vw01),
                                vu);                    // vrow1q = vw10.u0 w11.u1 w12.u2 w13.u3
                                                        // vw14.u4 w15.u5 w16.u6 w17.u7
    int16x8_t vrow2q = vmull_s8(vget_low_s8(vw23), vu); // vrow2q = vw20.u0 w21.u1 w22.u2
                                                        // w23.u3 vw24.u4 w25.u5 w26.u6 w27.u7
    int16x8_t vrow3q = vmull_s8(vget_high_s8(vw23),
                                vu);                    // vrow3q = vw30.u0 w31.u1 w32.u2 w33.u3
                                                        // vw34.u4 w35.u5 w36.u6 w37.u7
    int16x8_t vrow4q = vmull_s8(vget_low_s8(vw45), vu); // vrow4q = vw40.u0 w41.u1 w42.u2
                                                        // w43.u3 vw44.u4 w45.u5 w46.u6 w47.u7
    int16x8_t vrow5q = vmull_s8(vget_high_s8(vw45),
                                vu);                    // vrow5q = vw50.u0 w51.u1 w52.u2 w53.u3
                                                        // vw54.u4 w55.u5 w56.u6 w57.u7
    int16x8_t vrow6q = vmull_s8(vget_low_s8(vw67), vu); // vrow6q = vw60.u0 w61.u1 w62.u2
                                                        // w63.u3 vw64.u4 w65.u5 w66.u6 w67.u7
    int16x8_t vrow7q = vmull_s8(vget_high_s8(vw67),
                                vu); // vrow7q = vw70.u0 w71.u1 w72.u2 w73.u3
                                     // vw74.u4 w75.u5 w76.u6 w77.u7

    sum0 = vpadalq_s16(sum0, vrow0q);
    sum1 = vpadalq_s16(sum1, vrow1q);
    sum2 = vpadalq_s16(sum2, vrow2q);
    sum3 = vpadalq_s16(sum3, vrow3q);
    sum4 = vpadalq_s16(sum4, vrow4q);
    sum5 = vpadalq_s16(sum5, vrow5q);
    sum6 = vpadalq_s16(sum6, vrow6q);
    sum7 = vpadalq_s16(sum7, vrow7q);
    u += 8;
    wi += 64;
  }
  int32x4_t result0123 = {HorizontalSum32x4(sum0), HorizontalSum32x4(sum1),
                           HorizontalSum32x4(sum2), HorizontalSum32x4(sum3)};
  int32x4_t result4567 = {HorizontalSum32x4(sum4), HorizontalSum32x4(sum5),
                           HorizontalSum32x4(sum6), HorizontalSum32x4(sum7)};
  {
    int8x8_t bias = vld1_s8(wi); // vw0    = b0  b1  b2  b3  b4  b5  b6  b7
    int16x8_t scaled_bias = vmull_s8(bias, bias_scale);
    result0123 = vaddw_s16(result0123, vget_low_s16(scaled_bias));
    result4567 = vaddw_s16(result4567, vget_high_s16(scaled_bias));
    *v++ = vget_lane_s32(vget_low_s32(result0123), 0) * *scales++;
    if (num_out > 1)
      *v++ = vget_lane_s32(vget_low_s32(result0123), 1) * *scales++;
    if (num_out > 2)
      *v++ = vget_lane_s32(vget_high_s32(result0123), 0) * *scales++;
    if (num_out > 3)
      *v++ = vget_lane_s32(vget_high_s32(result0123), 1) * *scales++;
    if (num_out > 4)
      *v++ = vget_lane_s32(vget_low_s32(result4567), 0) * *scales++;
    if (num_out > 5)
      *v++ = vget_lane_s32(vget_low_s32(result4567), 1) * *scales++;
    if (num_out > 6)
      *v++ = vget_lane_s32(vget_high_s32(result4567), 0) * *scales++;
    if (num_out > 7)
      *v = vget_lane_s32(vget_high_s32(result4567), 1) * *scales;
  }
}

static void matrixDotVector(int dim1, int dim2, const int8_t *wi, const TFloat *scales,
                            const int8_t *u, TFloat *v) {
  const int num_out = dim1;
  const int num_in = dim2 - 1;
  // Each call to a partial_func_ produces group_size outputs, except the
  // last one, which can produce less.
  const int rounded_num_in = IntSimdMatrix::Roundup(num_in, kNumInputsPerGroup);
  int group_size = kNumOutputsPerRegister * kMaxOutputRegisters;
  int output = 0;

  int w_step = (rounded_num_in + 1) * group_size;

  for (; output + group_size <= num_out; output += group_size) {
    PartialMatrixDotVector8(wi, scales, u, rounded_num_in, v, kNumOutputsPerRegister);
    wi += w_step;
    scales += group_size;
    v += group_size;
  }
  if (output < num_out)
    PartialMatrixDotVector8(wi, scales, u, rounded_num_in, v,
                            num_out & (kNumOutputsPerRegister - 1));
}

const IntSimdMatrix IntSimdMatrix::intSimdMatrixNEON = {
    // Function.
    matrixDotVector,
    // Number of 32 bit outputs held in each register.
    kNumOutputsPerRegister,
    // Maximum number of registers that we will use to hold outputs.
    kMaxOutputRegisters,
    // Number of 8 bit inputs in the inputs register.
    kNumInputsPerRegister,
    // Number of inputs in each weight group.
    kNumInputsPerGroup
};

} // namespace tesseract.

#endif /* __ARM_NEON */
