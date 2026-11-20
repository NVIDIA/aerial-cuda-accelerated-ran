/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensor_desc.hpp"
#include <assert.h>
#include <algorithm>

namespace
{
////////////////////////////////////////////////////////////////////////
// compute_size_in_bytes()
size_t compute_size_in_bytes(cuphyDataType_t          t,
                             const tensor_layout_any& layout)
{
    int    components_per_element = 1;
    size_t element_count          = layout.get_num_elements_with_strides();
    if(CUPHY_BIT == t)
    {
        components_per_element = (sizeof(data_type_traits<CUPHY_BIT>::type) * CHAR_BIT);
        // For rank >= 2, strides will already have been calculated to
        // add padding for the word size. For the 1-D case, add it here.
        if(1 == layout.rank())
        {
            element_count = round_up_to_next(element_count,
                                             static_cast<size_t>(components_per_element));
        }
    }
    return element_count * get_cuphy_type_storage_element_size(t) /
           components_per_element;
}

////////////////////////////////////////////////////////////////////////
// euclid_gcd()
// Euclid's algorithm for determining the greatest common divisor
// https://www.youtube.com/watch?v=7HCd074v8g8
int euclid_gcd(int a, int b)
{
    int dividend = (a >= b) ? a : b;
    int divisor  = (a <= b) ? a : b;
    while(divisor != 0)
    {
        int remainder = dividend % divisor;
        dividend      = divisor;
        divisor       = remainder;
    }
    return dividend;
}

////////////////////////////////////////////////////////////////////////
// least_common_multiple()
int least_common_multiple(int a, int b) { return ((a * b) / euclid_gcd(a, b)); }

} // anonymous namespace

////////////////////////////////////////////////////////////////////////
// get_cuphy_type_storage_element_size()
int get_cuphy_type_storage_element_size(cuphyDataType_t type)
{
    // clang-format off
    switch (type)
    {
    default:
    case CUPHY_VOID:      return 0;                                               // uninitialized type
    case CUPHY_BIT:       return sizeof(data_type_traits<CUPHY_BIT>::type);       // 1-bit value - special handling for sub-byte types
    case CUPHY_R_8I:      return sizeof(data_type_traits<CUPHY_R_8I>::type);      // 8-bit signed integer real values
    case CUPHY_C_8I:      return sizeof(data_type_traits<CUPHY_C_8I>::type);      // 8-bit signed integer complex values
    case CUPHY_R_8U:      return sizeof(data_type_traits<CUPHY_R_8U>::type);      // 8-bit unsigned integer real values
    case CUPHY_C_8U:      return sizeof(data_type_traits<CUPHY_C_8U>::type);      // 8-bit unsigned integer real values
    case CUPHY_R_16I:     return sizeof(data_type_traits<CUPHY_R_16I>::type);     // 16-bit signed integer real values
    case CUPHY_C_16I:     return sizeof(data_type_traits<CUPHY_C_16I>::type);     // 16-bit signed integer real values
    case CUPHY_R_16U:     return sizeof(data_type_traits<CUPHY_R_16U>::type);     // 16-bit unsigned integer real values
    case CUPHY_C_16U:     return sizeof(data_type_traits<CUPHY_C_16U>::type);     // 16-bit unsigned integer real values
    case CUPHY_R_32I:     return sizeof(data_type_traits<CUPHY_R_32I>::type);     // 32-bit signed integer real values
    case CUPHY_C_32I:     return sizeof(data_type_traits<CUPHY_C_32I>::type);     // 32-bit signed integer real values
    case CUPHY_R_32U:     return sizeof(data_type_traits<CUPHY_R_32U>::type);     // 32-bit unsigned integer real values
    case CUPHY_C_32U:     return sizeof(data_type_traits<CUPHY_C_32U>::type);     // 32-bit unsigned integer real values
    case CUPHY_R_16F:     return sizeof(data_type_traits<CUPHY_R_16F>::type);     // half precision (16-bit) real values
    case CUPHY_C_16F:     return sizeof(data_type_traits<CUPHY_C_16F>::type);     // half precision (16-bit) complex values
    case CUPHY_R_32F:     return sizeof(data_type_traits<CUPHY_R_32F>::type);     // single precision (32-bit) real values
    case CUPHY_C_32F:     return sizeof(data_type_traits<CUPHY_C_32F>::type);     // single precision (32-bit) complex values
    case CUPHY_R_64F:     return sizeof(data_type_traits<CUPHY_R_64F>::type);     // double precision (64-bit) real values
    case CUPHY_C_64F:     return sizeof(data_type_traits<CUPHY_C_64F>::type);     // double precision (64-bit) complex values
    case CUPHY_R_8F_E4M3: return sizeof(data_type_traits<CUPHY_R_8F_E4M3>::type); // fp8 (1 sign, 4 exponent, 3 mantissa bits)
    case CUPHY_R_8F_E5M2: return sizeof(data_type_traits<CUPHY_R_8F_E5M2>::type); // fp8 (1 sign, 5 exponent, 2 mantissa bits)
    }
    // clang-format on
}

/////////////////////////////////////////////////////////////////////////
// get_element_multiple_for_alignment()
int get_element_multiple_for_alignment(int byte_align, cuphyDataType_t type)
{
    int type_size  = get_cuphy_type_storage_element_size(type);
    int elem_count = 0;
    switch(type)
    {
    case CUPHY_BIT:
        elem_count = least_common_multiple(byte_align, type_size) * CHAR_BIT;
        break;
    case CUPHY_VOID:
        // Invalid condition
        break;
    default:
        elem_count = least_common_multiple(byte_align, type_size) / type_size;
        break;
    }
    return elem_count;
}

////////////////////////////////////////////////////////////////////////
// word_layout_from_bit_layout()
// Creates a tensor layout that assumes 32-bit words, from an input
// bit layout.
// CUPHY_BIT tensors must be contiguous (i.e. the stride in the first
// dimension must be 1). Since bits are stored in 32-bit words, the
// stride in the second dimension (and thus all higher dimensions) must
// be a multiple of 32.
// The dimensions of the returned layout will be:
// - div_round_up(dim_input[0], 32) for dimension 0
// - the same for dimensions 1:CUPHY_DIM_MAX
// The strides of the returned layout will be strides in uint32_t words,
// and as such will be equal to 1 for dimension 0 and (dim_input[i] / 32)
// for other dimensions.
tensor_layout_any word_layout_from_bit_layout(const tensor_layout_any& kLayout)
{
    cuphy::vec<int, CUPHY_DIM_MAX> newDims    = kLayout.dimensions;
    newDims[0]                         = (newDims[0] + 31) / 32;
    cuphy::vec<int, CUPHY_DIM_MAX> newStrides = kLayout.strides;
    for(int i = 1; i < kLayout.rank(); ++i) // Skip first dim
    {
        newStrides[i] /= 32;
    }
    return tensor_layout_any(kLayout.rank(), newDims.begin(), newStrides.begin());
}

////////////////////////////////////////////////////////////////////////
// tensor_layout_any::validate()
// 1.) The number of dimensions must be greater than zero and less than
//     the maximum number supported by the API.
// 2.) The pointer to the array of dimensions must be non-NULL.
// 3.) All dimensions must be greather than or equal to 1.
// 4.) If strides are provided, the tensor must not be overlapping (i.e.
//     no two distinct index values must produce the same address).
//     Indices will overlap if stride[i] < (stride[i-1] * dim[i-1]) when
//     the strides are in ascending order.
//     See cuDNN documentation:
//     https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor-descriptor
//     Note that we may want to lift the non-overlapping restriction
//     in the future.
bool tensor_layout_any::validate(int             numDim,
                                 const int*      dim,
                                 const int*      str,
                                 cuphyDataType_t type) noexcept
{
    if((numDim <= 0) || (numDim > CUPHY_DIM_MAX)) return false;
    if(!dim) return false;
    if(std::any_of(dim, dim + numDim, [](int d) { return d < 1; })) return false;
    if(str)
    {
        // Make sure that strides describe a non-overlapping tensor
        std::array<int, CUPHY_DIM_MAX> strides{}, strides_sorted{};
        std::array<int, CUPHY_DIM_MAX> dims{}, dims_sorted{};
        std::array<std::size_t, CUPHY_DIM_MAX> indices{};
        const auto indicesEnd = indices.begin() + numDim;

        // Sorting first based on dims to get rid of the case where one dimension is 1 and some strides thus the same.
        std::iota(indices.begin(), indicesEnd, 0);
        std::sort(indices.begin(), indicesEnd, [&](std::size_t i, std::size_t j){ return dim[i] < dim[j]; });
        std::transform(indices.begin(), indicesEnd, strides.begin(), [&](std::size_t i){ return str[i]; });
        std::transform(indices.begin(), indicesEnd, dims.begin(), [&](std::size_t i){ return dim[i]; });

        std::iota(indices.begin(), indicesEnd, 0);
        std::sort(indices.begin(), indicesEnd, [&](std::size_t i, std::size_t j){ return strides[i] < strides[j]; });

        // Apply the sort.
        std::transform(indices.begin(), indicesEnd, strides_sorted.begin(), [&](std::size_t i){ return strides[i]; });
        std::transform(indices.begin(), indicesEnd, dims_sorted.begin(), [&](std::size_t i){ return dims[i]; });

        // Check.
        for(size_t i = 1; i < numDim; ++i)
        {
            if(strides_sorted[i] < (strides_sorted[i - 1] * dims_sorted[i - 1]))
            {
                return false;
            }
        }

        // For sub-byte types, make sure that strides (if provided)
        // have storage word (32-bit) alignment, and that the first
        // dimension is contiguous (stride = 1).
        if(type_is_sub_byte(type))
        {
            if(1 != str[0])
            {
                return false;
            }
            int storage_size = get_cuphy_type_storage_element_size(type);
            assert(type == CUPHY_BIT);
            if((numDim > 1) && (0 != (str[1] % (storage_size * CHAR_BIT))))
            {
                return false;
            }
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////
// tensor_desc::set()
bool tensor_desc::set(cuphyDataType_t t,
                      int             numDim,
                      const int*      dim,
                      const int*      strides) noexcept
{
    // If the input is not valid, the descriptor will remain unchanged
    if((t != CUPHY_VOID) &&
       tensor_layout_any::validate(numDim, dim, strides, t))
    {
        const int*                     layout_strides = strides;
        std::array<int, CUPHY_DIM_MAX> sub_byte_strides;
        // For 2-D (and higher) tensors with a sub-byte element type
        // and no user-provided strides, we need to manually set up
        // strides before creating the tensor layout.
        if(type_is_sub_byte(t) && !strides && (numDim > 1))
        {
            // Get the number of bytes in the storage word
            int storage_size = get_cuphy_type_storage_element_size(t);
            // Sub-byte tensors must have stride[0] = 1
            sub_byte_strides[0] = 1;
            // The first dimension must fill up a storage word
            sub_byte_strides[1] = round_up_to_next(dim[0],
                                                   get_element_multiple_for_alignment(storage_size, t));
            //printf("Calculated strides: (%i, %i)\n", sub_byte_strides[0], sub_byte_strides[1]);
            // Populate remaining stride values
            for(size_t i = 2; i < CUPHY_DIM_MAX; ++i)
            {
                sub_byte_strides[i] = (i < numDim) ? (sub_byte_strides[i - 1] * dim[i - 1]) : 0;
            }
            // Pass these locally computed strides to the tensor layout c'tor
            layout_strides = sub_byte_strides.data();
        }
        type_       = t;
        layout_     = tensor_layout_any(numDim, dim, layout_strides);
        size_bytes_ = compute_size_in_bytes(type_, layout_);
        return true;
    }
    return false;
}
