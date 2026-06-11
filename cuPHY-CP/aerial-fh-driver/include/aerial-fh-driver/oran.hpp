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

#ifndef AERIAL_FH_DRIVER_ORAN__
#define AERIAL_FH_DRIVER_ORAN__

#include <inttypes.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <array>
#include <unordered_map>
#include <stdexcept>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Bits manipulation
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Bitfield template for CUDA device/host code
 *
 * Overcomes NVCC error: "Bitfields and field types containing bitfields are not
 * supported in packed structures and unions for device compilation!"
 *
 * Inspired by https://github.com/preshing/cpp11-on-multicore/blob/master/common/bitfield.h
 * Only supports 8, 16, 32 bit sized bitfields
 *
 * \tparam T Underlying integer type (uint8_t, uint16_t, or uint32_t)
 * \tparam Offset Bit offset within the field
 * \tparam Bits Number of bits for this field
 */
template <typename T, int Offset, int Bits>
class __attribute__((__packed__)) Bitfield {
    static_assert(Offset + Bits <= (int)sizeof(T) * 8, "Member exceeds bitfield boundaries");
    static_assert(Bits < (int)sizeof(T) * 8, "Can't fill entire bitfield with one member");
    static_assert(sizeof(T) == sizeof(uint8_t) ||
                      sizeof(T) == sizeof(uint16_t) ||
                      sizeof(T) == sizeof(uint32_t),
                  "Size not supported by bitfield");
    static const T Maximum = (T(1) << Bits) - 1;
    static const T Mask    = Maximum << Offset;

    T field;
    // T maximum() const { return Maximum; }
    // T one() const { return T(1) << Offset; }
    __host__ __device__ T be_to_le(T value)
    {
        T tmp = value;
        if constexpr(sizeof(T) == sizeof(uint16_t))
        {
            tmp = 0;
            tmp |= (value & 0xFF00) >> 8;
            tmp |= (value & 0x00FF) << 8;
        }
        else if constexpr(sizeof(T) == sizeof(uint32_t))
        {
            tmp = 0;
            tmp |= value >> 24;
            tmp |= (value & 0x00FF0000) >> 8;
            tmp |= (value & 0x0000FF00) << 8;
            tmp |= (value & 0x000000FF) << 24;
        }
        return tmp;
    }
    __host__ __device__ T le_to_be(T value)
    {
        T tmp = value;
        if constexpr(sizeof(T) == sizeof(uint16_t))
        {
            tmp = 0;
            tmp |= (value & 0xFF00) >> 8;
            tmp |= (value & 0x00FF) << 8;
        }
        else if constexpr(sizeof(T) == sizeof(uint32_t))
        {
            tmp = 0;
            tmp |= value >> 24;
            tmp |= (value & 0x00FF0000) >> 8;
            tmp |= (value & 0x0000FF00) << 8;
            tmp |= (value & 0x000000FF) << 24;
        }
        return tmp;
    }

public:
    __host__ __device__ void operator=(T value)
    {
        // value must fit inside the bitfield member. This line guarantees value <= Maximum.
        value &= Maximum;

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        field = be_to_le(field);
#endif
        field = (field & ~Mask) | (value << Offset);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        field = le_to_be(field);
#endif
    }

    __host__ __device__ operator T()
    {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return (T)(be_to_le(field) >> Offset) & Maximum;
#else
        return (T)(field >> Offset) & Maximum;
#endif
    }

    __host__ __device__ T get() const
    {
        #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            if constexpr(sizeof(T) == sizeof(uint16_t))
                return (T)((((field & 0xFF00) >> 8 | (field & 0x00FF) << 8) >> Offset) & Maximum);
            else if constexpr(sizeof(T) == sizeof(uint32_t))
                return (T)(
                    ((
                        field >> 24                |
                        (field & 0x00FF0000) >> 8  |
                        (field & 0x0000FF00) << 8  |
                        (field & 0x000000FF) << 24
                    )>> Offset) & Maximum
                );
            else
                return ((T)(field >> Offset) & Maximum);
        #else
            return ((T)(field >> Offset) & Maximum);
        #endif
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Ethernet generic
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define ORAN_ETHER_ADDR_LEN 6  //!< Ethernet address length (6 bytes)

/**
 * Ethernet MAC address
 */
struct oran_ether_addr
{
    uint8_t addr_bytes[ORAN_ETHER_ADDR_LEN]; //!< Address bytes in transmission order
} __attribute__((__aligned__(2)));

/**
 * Ethernet header
 */
struct oran_ether_hdr
{
    struct oran_ether_addr dst_addr;   //!< Destination MAC address
    struct oran_ether_addr src_addr;   //!< Source MAC address
    uint16_t               ether_type; //!< Ethertype (0x8100 for VLAN, 0xAEFE for eCPRI)
} __attribute__((__aligned__(2)));

/**
 * VLAN header (802.1Q)
 */
struct oran_vlan_hdr
{
    uint16_t vlan_tci;  //!< VLAN TCI: PCP(3) + DEI(1) + VID(12)
    uint16_t eth_proto; //!< Ethertype of encapsulated frame
} __attribute__((__packed__));

/**
 * Complete Ethernet header with VLAN
 */
struct oran_eth_hdr
{
    struct oran_ether_hdr eth_hdr;   //!< Ethernet header
    struct oran_vlan_hdr  vlan_hdr;  //!< VLAN header
};

#define ORAN_ETH_HDR_SIZE (         \
    sizeof(struct oran_ether_hdr) + \
    sizeof(struct oran_vlan_hdr))    //!< Total Ethernet+VLAN header size

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// eCPRI generic (O-RAN specs v01.00)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define ETHER_TYPE_ECPRI 0xAEFE           //!< eCPRI Ethertype
#define ORAN_DEF_ECPRI_VERSION 1          //!< Default eCPRI protocol version
#define ORAN_DEF_ECPRI_RESERVED 0         //!< Default reserved field value
#define ORAN_ECPRI_CONCATENATION_NO 0     //!< No eCPRI message concatenation
#define ORAN_ECPRI_CONCATENATION_YES 1    //!< eCPRI message concatenation enabled

#define ORAN_ECPRI_HDR_OFFSET ORAN_ETH_HDR_SIZE  //!< eCPRI header offset in packet

// ORAN timing constants
#define ORAN_MAX_FRAME_ID 256                                      //!< Maximum frame ID (0-255)
#define ORAN_MAX_SUBFRAME_ID 10                                    //!< Maximum subframe ID (0-9)
#define ORAN_MAX_SLOT_ID 2                                         //!< Maximum slot ID (assuming 500us TTI)
#define ORAN_MAX_SLOT_X_SUBFRAME_ID (ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID)  //!< Slots per frame

// eCPRI message types (Section 3.1.3.1.4)
#define ECPRI_MSG_TYPE_IQ 0x0   //!< IQ data message type
#define ECPRI_MSG_TYPE_RTC 0x2  //!< Real-time control message type
#define ECPRI_MSG_TYPE_ND 0x5   //!< Node discovery message type

#define ORAN_ECPRI_MAX_PAYLOAD_LEN  ((0x1<<16)-1)  //!< Maximum eCPRI payload length (Section 5.1.3.1.5)

/**
 * eCPRI transport header (ORAN-WG4.CUS.0-v01.00 Section 3.1.3.1)
 *
 * Common header for eCPRI messages carrying ORAN fronthaul traffic.
 * Contains version, message type, payload length, and sequence info.
 */
struct oran_ecpri_hdr
{
    /*
    LITTLE ENDIAN FORMAT (8 bits):
    -----------------------------------------------------
    | ecpriVersion | ecpriReserved | ecpriConcatenation |
    -----------------------------------------------------
    |       4      |       3       |        1           |
    -----------------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> ecpriVersion;
        Bitfield<uint8_t, 4, 3> ecpriReserved;
        Bitfield<uint8_t, 7, 1> ecpriConcatenation;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 1> ecpriConcatenation;
        Bitfield<uint8_t, 1, 3> ecpriReserved;
        Bitfield<uint8_t, 4, 4> ecpriVersion;
    };
#endif

    uint8_t  ecpriMessage;
    uint16_t ecpriPayload;
    union
    {
        uint16_t ecpriRtcid;
        uint16_t ecpriPcid;
    };
    uint8_t ecpriSeqid;

    /*
    BIG ENDIAN FORMAT (8 bits):
    -----------------------------
    | ecpriEbit | ecpriSubSeqid |
    -----------------------------
    |     1     |       7       |
    -----------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 1> ecpriEbit;
        Bitfield<uint8_t, 1, 7> ecpriSubSeqid;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 7> ecpriSubSeqid;
        Bitfield<uint8_t, 7, 1> ecpriEbit;
    };
#endif

} __attribute__((__packed__));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Message specific O-RAN header
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * ORAN packet direction
 */
enum oran_pkt_dir : std::uint8_t
{
    DIRECTION_UPLINK = 0,    //!< Uplink (UE to gNB)
    DIRECTION_DOWNLINK       //!< Downlink (gNB to UE)
};

// ORAN protocol constants (from ORAN spec sections)
#define ORAN_DEF_PAYLOAD_VERSION 1  //!< Default payload version (Section 5.4.4.2)
#define ORAN_DEF_FILTER_INDEX 0     //!< Default filter index (Section 5.4.4.3)
#define ORAN_RB_ALL 0               //!< All resource blocks (Section 5.4.5.2)
#define ORAN_RB_OTHER_ALL 1         //!< All other resource blocks (Section 5.4.5.2)
#define ORAN_SYMCINC_NO 0           //!< No symbol number increment (Section 5.4.5.3)
#define ORAN_SYMCINC_YES 1          //!< Symbol number increment (Section 5.4.5.3)
#define ORAN_REMASK_ALL 0x0FFFU     //!< All resource elements mask (Section 5.4.5.5)
#define ORAN_ALL_SYMBOLS 14U        //!< Number of symbols in a slot (Section 5.4.5.7)
#define ORAN_EF_NO 0                //!< No extension flag (Section 5.4.5.8)
#define ORAN_EF_YES 1               //!< Extension flag present (Section 5.4.5.8)
#define ORAN_BEAMFORMING_NO 0x0000  //!< No beamforming (Section 5.4.5.9)

#define ORAN_MAX_PRB_X_SECTION 255  //!< Maximum PRBs per section
#define ORAN_MAX_PRB_X_SLOT 273     //!< Maximum PRBs per slot

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// U-plane
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define SLOT_NUM_SYMS 14U                                      //!< Number of OFDM symbols in a slot
#define PRB_NUM_RE 12U                                         //!< Number of resource elements (REs) in a PRB
#define UD_IQ_WIDH_MAX 16                                      //!< Maximum IQ bit width
#define PRB_SIZE(iq_width) ((iq_width * 2U * PRB_NUM_RE) / 8U) //!< PRB size in bytes for given IQ width

///////////////////////////////////
//// 16bit I/Q
//////////////////////////////////
/* an O-RAN 16-bit I 16-bit Q Resource Element */
struct oran_re_16b
{
    uint16_t I; /* Note: big endian. */
    uint16_t Q; /* Note: big endian. */
} __attribute__((__packed__));

struct oran_prb_16b_uncompressed
{
    struct oran_re_16b re_array[PRB_NUM_RE];
} __attribute__((__packed__));

#define PRB_SIZE_16F PRB_SIZE(16)

///////////////////////////////////
//// 14bit I/Q
//////////////////////////////////
/* an O-RAN 14-bit I 14-bit Q Resource Element will have to use bytes*/
struct oran_prb_14b_compressed
{
    uint8_t re_array[43];
} __attribute__((__packed__));

#define PRB_SIZE_14F sizeof(struct oran_prb_14b_compressed) /* in bytes */

///////////////////////////////////
//// 9bit I/Q
//////////////////////////////////
/* an O-RAN 9-bit I 9-bit Q Resource Element will have to use bytes*/
struct oran_prb_9b_compressed
{
    uint8_t re_array[28];
} __attribute__((__packed__));

#define PRB_SIZE_9F sizeof(struct oran_prb_9b_compressed) /* in bytes */

/* Not considering section id for data placement yet */
#define ORAN_DEF_SECTION_ID 0

#define ORAN_DEF_BFP_NO_COMPRESSION 0
/* header of the IQ data frame U-Plane message in O-RAN FH, all the way up to
* and including symbolid (the fuchsia part of Table 6-2 in the spec) */
struct oran_umsg_iq_hdr
{
    /*
    BIG ENDIAN FORMAT (8 bits):
    ---------------------------------------------------
    | Data Direction | Payload Version | Filter Index |
    ---------------------------------------------------
    |     1          |          3      |      4       |
    ---------------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 1> dataDirection;
        Bitfield<uint8_t, 1, 3> payloadVersion;
        Bitfield<uint8_t, 4, 4> filterIndex;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> filterIndex;
        Bitfield<uint8_t, 4, 3> payloadVersion;
        Bitfield<uint8_t, 7, 1> dataDirection;
    };
#endif

    uint8_t frameId;

/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------------------
    | subframeId | slotId | symbolId |
    ----------------------------------
    |    4       |    6   |    6     |
    ----------------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 4>  subframeId;
        Bitfield<uint16_t, 4, 6>  slotId;
        Bitfield<uint16_t, 10, 6> symbolId;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 6> symbolId;
        Bitfield<uint16_t, 6, 6> slotId;
        Bitfield<uint16_t, 12, 4> subframeId;
    };
#endif

} __attribute__((__packed__));

/* A struct for the section header of uncompressed IQ U-Plane message.
* No compression is used, so the compression header is omitted.
*/
struct oran_u_section_uncompressed
{
/*
    BIG ENDIAN FORMAT (16 bits):
    ---------------------------------------------
    | sectionId | rb | symInc | unused_startPrbu |
    ---------------------------------------------
    |    12     | 1  |    1   |         2        |
    ---------------------------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 12>  sectionId;
        Bitfield<uint32_t, 12, 1>  rb;
        Bitfield<uint32_t, 13, 1>  symInc;
        Bitfield<uint32_t, 14, 10> startPrbu;
        Bitfield<uint32_t, 24, 8>  numPrbu;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 8> numPrbu;
        Bitfield<uint32_t, 8, 10> startPrbu;
        Bitfield<uint32_t, 18, 1> symInc;
        Bitfield<uint32_t, 19, 1> rb;
        Bitfield<uint32_t, 20, 12> sectionId;
    };
#endif
    /* NOTE: no compression header */
} __attribute__((__packed__));

struct oran_u_section_compression_hdr
{
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> udIqWidth;
        Bitfield<uint8_t, 4, 4> udCompMeth;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> udCompMeth;
        Bitfield<uint8_t, 4, 4> udIqWidth;
    };
#endif
    uint8_t reserved;
} __attribute__((__packed__));


/* per-eth frame overhead. NOTE: one eCPRI message per eth frame assumed */
#define ORAN_IQ_HDR_OFFSET ( \
    ORAN_ECPRI_HDR_OFFSET +  \
    sizeof(struct oran_ecpri_hdr))

#define ORAN_IQ_STATIC_OVERHEAD ( \
    ORAN_IQ_HDR_OFFSET +          \
    sizeof(struct oran_umsg_iq_hdr))

/* per-section overhead */
#define ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD ( \
    sizeof(struct oran_u_section_uncompressed))

#define ORAN_IQ_SECTION_COMPRESSION_HDR_OVERHEAD ( \
    sizeof(struct oran_u_section_compression_hdr))

#define ORAN_IQ_COMPRESSED_SECTION_OVERHEAD ( \
    ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD + ORAN_IQ_SECTION_COMPRESSION_HDR_OVERHEAD)


struct oran_umsg_hdrs
{
    struct oran_eth_hdr                ethvlan;
    struct oran_ecpri_hdr              ecpri;
    struct oran_umsg_iq_hdr            iq_hdr;
    struct oran_u_section_uncompressed sec_hdr;
    struct oran_u_section_compression_hdr comp_hdr[];
};

#define ORAN_UMSG_IQ_HDR_SIZE sizeof(struct oran_umsg_hdrs)
#define ORAN_IQ_HDR_SZ (ORAN_IQ_STATIC_OVERHEAD + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// C-plane
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum oran_cmsg_section_type
{
    ORAN_CMSG_SECTION_TYPE_0 = 0,
    ORAN_CMSG_SECTION_TYPE_1 = 1,
    ORAN_CMSG_SECTION_TYPE_3 = 3,
    ORAN_CMSG_SECTION_TYPE_5 = 5,
    ORAN_CMSG_SECTION_TYPE_6 = 6,
    ORAN_CMSG_SECTION_TYPE_7 = 7
};

enum oran_cmsg_section_ext_type
{
    ORAN_CMSG_SECTION_EXT_TYPE_0 = 0,
    ORAN_CMSG_SECTION_EXT_TYPE_1 = 1,
    ORAN_CMSG_SECTION_EXT_TYPE_2 = 2,
    ORAN_CMSG_SECTION_EXT_TYPE_3 = 3,
    ORAN_CMSG_SECTION_EXT_TYPE_4 = 4,
    ORAN_CMSG_SECTION_EXT_TYPE_5 = 5,
    ORAN_CMSG_SECTION_EXT_TYPE_6 = 6,
    ORAN_CMSG_SECTION_EXT_TYPE_7 = 7,
    ORAN_CMSG_SECTION_EXT_TYPE_8 = 8,
    ORAN_CMSG_SECTION_EXT_TYPE_9 = 9,
    ORAN_CMSG_SECTION_EXT_TYPE_10 = 10,
    ORAN_CMSG_SECTION_EXT_TYPE_11 = 11,
    ORAN_CMSG_SECTION_EXT_TYPE_12 = 12,
    ORAN_CMSG_SECTION_EXT_TYPE_13 = 13,
    ORAN_CMSG_SECTION_EXT_TYPE_14 = 14,
    ORAN_CMSG_SECTION_EXT_TYPE_15 = 15,
    ORAN_CMSG_SECTION_EXT_TYPE_16 = 16,
    ORAN_CMSG_SECTION_EXT_TYPE_17 = 17,
    ORAN_CMSG_SECTION_EXT_TYPE_18 = 18,
    ORAN_CMSG_SECTION_EXT_TYPE_19 = 19,
    ORAN_CMSG_SECTION_EXT_TYPE_20 = 20,
    ORAN_CMSG_SECTION_EXT_TYPE_21 = 21,
    ORAN_CMSG_SECTION_EXT_TYPE_22 = 22,
};


/******************************************************************/ /**
 * \brief User data compression methods for BFW
 */
enum class UserDataBFWCompressionMethod
{
    NO_COMPRESSION                 = 0b0000,
    BLOCK_FLOATING_POINT           = 0b0001,
    BLOCK_SCALING                  = 0b0010,
    U_LAW                          = 0b0011,
    BEAMSPACE_1                    = 0b0100,
    BEAMSPACE_2                    = 0b0101,
    RESERVED                       = 0b0110,
};

#define ORAN_CMESG_ALL_PRBC 0x0

struct oran_cmsg_radio_app_hdr
{
    /*
    BIG ENDIAN FORMAT (8 bits):
    ---------------------------------------------------
    | Data Direction | Payload Version | Filter Index |
    ---------------------------------------------------
    |     1          |          3      |      4       |
    ---------------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 1> dataDirection;
        Bitfield<uint8_t, 1, 3> payloadVersion;
        Bitfield<uint8_t, 4, 4> filterIndex;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> filterIndex;
        Bitfield<uint8_t, 4, 3> payloadVersion;
        Bitfield<uint8_t, 7, 1> dataDirection;
    };
#endif

    uint8_t frameId;

    /*
    BIG ENDIAN FORMAT (16 bits):
    ------------------------------------------
    | Subframe ID | Slot ID | startSymbol ID |
    ------------------------------------------
    |     4       |    6    |       6        |
    ------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 4>  subframeId;
        Bitfield<uint16_t, 4, 6>  slotId;
        Bitfield<uint16_t, 10, 6> startSymbolId;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 6> startSymbolId;
        Bitfield<uint16_t, 6, 6> slotId;
        Bitfield<uint16_t, 12, 4> subframeId;
    };
#endif

    uint8_t numberOfSections;
    uint8_t sectionType;
} __attribute__((__packed__));

/*
 * C-message Section Type 0 Common Header Fields
 */
struct oran_cmsg_sect0_common_hdr
{
    struct oran_cmsg_radio_app_hdr radioAppHdr;
    uint16_t                       timeOffset;
    uint8_t                        frameStructure;
    uint16_t                       cpLength;
    uint8_t                        reserved;
} __attribute__((__packed__));

/*
 * C-message Section Type 0 Section fields
 */
struct oran_cmsg_sect0
{
/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------------------------
    | Section ID | RB | SymInc | startPrbu |
    ----------------------------------------
    |   12       | 1  |   1    |     2     |
    ----------------------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 12>  sectionId;
        Bitfield<uint32_t, 12, 1>  rb;
        Bitfield<uint32_t, 13, 1>  symInc;
        Bitfield<uint32_t, 14, 10> startPrbc;
        Bitfield<uint32_t, 24, 8>  numPrbc;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 8> numPrbc;
        Bitfield<uint32_t, 8, 10> startPrbc;
        Bitfield<uint32_t, 18, 1> symInc;
        Bitfield<uint32_t, 19, 1> rb;
        Bitfield<uint32_t, 20, 12> sectionId;
    };
#endif
/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------
    | reMask | numSymbol |
    ----------------------
    |   12   |      4    |
    ----------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 12> reMask;
        Bitfield<uint16_t, 12, 4> numSymbol;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 4> numSymbol;
        Bitfield<uint16_t, 4, 12> reMask;
    };
#endif
/*
    BIG ENDIAN FORMAT (16 bits):
    -----------------
    | ef | reserved |
    -----------------
    | 1  |    15    |
    -----------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 1>  ef;
        Bitfield<uint16_t, 1, 15> reserved;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 15> reserved;
        Bitfield<uint16_t, 15, 1> ef;
    };
#endif
} __attribute__((__packed__));

/*
 * C-message Section Type 1 Common Header Fields
 */
struct oran_cmsg_sect1_common_hdr
{
    struct oran_cmsg_radio_app_hdr radioAppHdr;
    uint8_t                        udCompHdr;
    uint8_t                        reserved;
} __attribute__((__packed__));

/*
 * C-message Section Type 1 Section fields
 */
struct oran_cmsg_sect1
{
/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------------------------
    | Section ID | RB | SymInc | startPrbu |
    ----------------------------------------
    |   12       | 1  |   1    |     2     |
    ----------------------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 12>  sectionId;
        Bitfield<uint32_t, 12, 1>  rb;
        Bitfield<uint32_t, 13, 1>  symInc;
        Bitfield<uint32_t, 14, 10> startPrbc;
        Bitfield<uint32_t, 24, 8>  numPrbc;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 8> numPrbc;
        Bitfield<uint32_t, 8, 10> startPrbc;
        Bitfield<uint32_t, 18, 1> symInc;
        Bitfield<uint32_t, 19, 1> rb;
        Bitfield<uint32_t, 20, 12> sectionId;
    };
#endif
/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------
    | reMask | numSymbol |
    ----------------------
    |   12   |      4    |
    ----------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 12> reMask;
        Bitfield<uint16_t, 12, 4> numSymbol;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 4> numSymbol;
        Bitfield<uint16_t, 4, 12> reMask;
    };
#endif
/*
    BIG ENDIAN FORMAT (16 bits):
    ---------------
    | ef | beamId |
    ---------------
    | 1  |   15   |
    ---------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 1>  ef;
        Bitfield<uint16_t, 1, 15> beamId;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 15> beamId;
        Bitfield<uint16_t, 15, 1> ef;
    };
#endif
} __attribute__((__packed__));

/*
 * C-message Section Type 3 Common Header Fields
 */
struct oran_cmsg_sect3_common_hdr
{
    struct oran_cmsg_radio_app_hdr radioAppHdr;
    uint16_t                       timeOffset;
    uint8_t                        frameStructure;
    uint16_t                       cpLength;
    uint8_t                        udCompHdr;
} __attribute__((__packed__));

/*
 * C-message Section Type 3 Section fields
 */
struct oran_cmsg_sect3
{
/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------------------------
    | Section ID | RB | SymInc | startPrbu |
    ----------------------------------------
    |   12       | 1  |   1    |     2     |
    ----------------------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 12>  sectionId;
        Bitfield<uint32_t, 12, 1>  rb;
        Bitfield<uint32_t, 13, 1>  symInc;
        Bitfield<uint32_t, 14, 10> startPrbc;
        Bitfield<uint32_t, 24, 8>  numPrbc;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 8> numPrbc;
        Bitfield<uint32_t, 8, 10> startPrbc;
        Bitfield<uint32_t, 18, 1> symInc;
        Bitfield<uint32_t, 19, 1> rb;
        Bitfield<uint32_t, 20, 12> sectionId;
    };
#endif
/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------
    | reMask | numSymbol |
    ----------------------
    |   12   |      4    |
    ----------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 12> reMask;
        Bitfield<uint16_t, 12, 4> numSymbol;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 4> numSymbol;
        Bitfield<uint16_t, 4, 12> reMask;
    };
#endif
/*
    BIG ENDIAN FORMAT (16 bits):
    ---------------
    | ef | beamId |
    ---------------
    | 1  |   15   |
    ---------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 1>  ef;
        Bitfield<uint16_t, 1, 15> beamId;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 15> beamId;
        Bitfield<uint16_t, 15, 1> ef;
    };
#endif
/*
    BIG ENDIAN FORMAT (32 bits):
    -------------------------
    | freqOffset | reserved |
    -------------------------
    |     24     |    8     |
    -------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 24> freqOffset;
        Bitfield<uint32_t, 24, 8> reserved;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 8> reserved;
        Bitfield<uint32_t, 8, 24> freqOffset;
    };
#endif
} __attribute__((__packed__));

/*
 * C-message Section Type 5 Common Header Fields
 */
struct oran_cmsg_sect5_common_hdr
{
    struct oran_cmsg_radio_app_hdr radioAppHdr;
    uint8_t                        udCompHdr;
    uint8_t                        reserved;
} __attribute__((__packed__));

/*
 * C-message Section Type 5 Section fields
 */
struct oran_cmsg_sect5
{
/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------------------------
    | Section ID | RB | SymInc | startPrbu |
    ----------------------------------------
    |   12       | 1  |   1    |     2     |
    ----------------------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 12>  sectionId;
        Bitfield<uint32_t, 12, 1>  rb;
        Bitfield<uint32_t, 13, 1>  symInc;
        Bitfield<uint32_t, 14, 10> startPrbc;
        Bitfield<uint32_t, 24, 8>  numPrbc;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 8> numPrbc;
        Bitfield<uint32_t, 8, 10> startPrbc;
        Bitfield<uint32_t, 18, 1> symInc;
        Bitfield<uint32_t, 19, 1> rb;
        Bitfield<uint32_t, 20, 12> sectionId;
    };
#endif
/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------
    | reMask | numSymbol |
    ----------------------
    |   12   |      4    |
    ----------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 12> reMask;
        Bitfield<uint16_t, 12, 4> numSymbol;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 4> numSymbol;
        Bitfield<uint16_t, 4, 12> reMask;
    };
#endif
/*
    BIG ENDIAN FORMAT (16 bits):
    ---------------
    | ef | ueId |
    ---------------
    | 1  |   15   |
    ---------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 1>  ef;
        Bitfield<uint16_t, 1, 15> ueId;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 15> ueId;
        Bitfield<uint16_t, 15, 1> ef;
    };
#endif
} __attribute__((__packed__));

/*
 * C-message Section Type 6 Common Header Fields
 */
struct oran_cmsg_sect6_common_hdr
{
    struct oran_cmsg_radio_app_hdr radioAppHdr;
    uint8_t                        numberOfUEs;
    uint8_t                        reserved;
} __attribute__((__packed__));

/*
 * C-message Section Type 6 Section fields
 */
struct oran_cmsg_sect6
{
/*
    BIG ENDIAN FORMAT (16 bits):
    ---------------
    | ef | ueId |
    ---------------
    | 1  |   15   |
    ---------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 1>  ef;
        Bitfield<uint16_t, 1, 15> ueId;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 15> ueId;
        Bitfield<uint16_t, 15, 1> ef;
    };
#endif

    uint16_t regularizationFactor;

/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------------------------
    | reserved | RB | SymInc | startPrbc |
    ----------------------------------------
    |    4     | 1  |   1    |     10     |
    ----------------------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 4>  reserved;
        Bitfield<uint16_t, 4, 1>  rb;
        Bitfield<uint16_t, 5, 1>  symInc;
        Bitfield<uint16_t, 6, 10> startPrbc;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 10> startPrbc;
        Bitfield<uint16_t, 10, 1> symInc;
        Bitfield<uint16_t, 11, 1> rb;
        Bitfield<uint16_t, 12, 4> reserved;
    };
#endif

    uint8_t  numPrbc;
    uint16_t ciIsample;
    uint16_t ciQsample;
} __attribute__((__packed__));

#define ORAN_CMSG_HDR_OFFSET ( \
    ORAN_ECPRI_HDR_OFFSET +    \
    sizeof(struct oran_ecpri_hdr))

#define ORAN_CMSG_SECT1_FIELDS_OFFSET ( \
    ORAN_CMSG_HDR_OFFSET +              \
    sizeof(struct oran_cmsg_sect1_common_hdr))

#define ORAN_CMSG_SECT1_OVERHEAD (  \
    ORAN_CMSG_SECT1_FIELDS_OFFSET + \
    sizeof(struct oran_cmsg_sect1))

#define ORAN_CMSG_SECT3_FIELDS_OFFSET ( \
    ORAN_CMSG_HDR_OFFSET +              \
    sizeof(struct oran_cmsg_sect3_common_hdr))

#define ORAN_CMSG_SECT3_OVERHEAD (  \
    ORAN_CMSG_SECT3_FIELDS_OFFSET + \
    sizeof(struct oran_cmsg_sect3))

#define ORAN_CMSG_SECT5_FIELDS_OFFSET ( \
    ORAN_CMSG_HDR_OFFSET +              \
    sizeof(struct oran_cmsg_sect5_common_hdr))

#define ORAN_CMSG_SECT5_OVERHEAD (  \
    ORAN_CMSG_SECT5_FIELDS_OFFSET + \
    sizeof(struct oran_cmsg_sect5))

#define ORAN_CMSG_SECT6_FIELDS_OFFSET ( \
    ORAN_CMSG_HDR_OFFSET +              \
    sizeof(struct oran_cmsg_sect6_common_hdr))

#define ORAN_CMSG_SECT6_OVERHEAD (  \
    ORAN_CMSG_SECT6_FIELDS_OFFSET + \
    sizeof(struct oran_cmsg_sect6))

struct oran_cmsg_uldl_hdrs
{
    struct oran_eth_hdr               ethvlan;
    struct oran_ecpri_hdr             ecpri;
    struct oran_cmsg_sect1_common_hdr sect1_hdr;
    struct oran_cmsg_sect1            sect1_fields;
};

struct oran_cmsg_prach_hdrs
{
    struct oran_eth_hdr               ethvlan;
    struct oran_ecpri_hdr             ecpri;
    struct oran_cmsg_sect3_common_hdr sect3_hdr;
    struct oran_cmsg_sect3            sect3_fields;
};

/*
 * C-plane section extension header
 */
struct oran_cmsg_ext_hdr
{
    /*
    BIG ENDIAN FORMAT (8 bits):
    ----------------
    | ef | extType |
    ----------------
    | 1  |    7    |
    ----------------
 */
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 1> ef;
        Bitfield<uint8_t, 1, 7> extType;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 7> extType;
        Bitfield<uint8_t, 7, 1> ef;
    };
#endif


} __attribute__((__packed__));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Section Extensions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct  oran_cmsg_sect_ext_type_4
{
    uint8_t extLen;
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 1> csf;
        Bitfield<uint16_t, 1, 15> modCompScalor;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 15> modCompScalor;
        Bitfield<uint16_t, 15, 1> csf;
    };
#endif
} __attribute__((__packed__));

struct oran_cmsg_sect_ext_type_5
{
    uint8_t extLen;
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    uint64_t mcScaleReMask_1 : 12;
    uint64_t csf_1 : 1;
    uint64_t mcScaleOffset_1 : 15;
    uint64_t mcScaleReMask_2 : 12;
    uint64_t csf_2 : 1;
    uint64_t mcScaleOffset_2 : 15;
    uint64_t zero_padding : 8;
#else
    uint64_t zero_padding : 8;
    uint64_t mcScaleOffset_2 : 15;
    uint64_t csf_2 : 1;
    uint64_t mcScaleReMask_2 : 12;
    uint64_t mcScaleOffset_1 : 15;
    uint64_t csf_1 : 1;
    uint64_t mcScaleReMask_1 : 12;
#endif
    uint16_t extra_zero_padding;
} __attribute__((__packed__));

#define ORAN_SECT_EXT_11_ALIGNMENT 4
static constexpr int ORAN_SECT_EXT_11_L_TRX = 64;

struct oran_cmsg_sect_ext_type_11
{
    uint16_t extLen;
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 1> disableBFWs;
        Bitfield<uint8_t, 1, 1> RAD;
        Bitfield<uint8_t, 2, 6> reserved;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 6> reserved;
        Bitfield<uint8_t, 6, 1> RAD;
        Bitfield<uint8_t, 7, 1> disableBFWs;
    };
#endif
    uint8_t numBundPrb;
    uint8_t body[0];
} __attribute__((__packed__));



struct oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr
{
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> bfwIqWidth;
        Bitfield<uint8_t, 4, 4> bfwCompMeth;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> bfwCompMeth;
        Bitfield<uint8_t, 4, 4> bfwIqWidth;
    };
#endif
    uint8_t next[0];
} __attribute__((__packed__));

struct oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompParam
{
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> reserved;
        Bitfield<uint8_t, 4, 4> exponent;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> exponent;
        Bitfield<uint8_t, 4, 4> reserved;
    };
#endif
} __attribute__((__packed__));

struct oran_cmsg_sect_ext_type_11_disableBFWs_0_beamId
{
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 1> reserved;
        Bitfield<uint16_t, 1, 15> beamId;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 15> beamId;
        Bitfield<uint16_t, 15, 1> reserved;
    };
#endif
} __attribute__((__packed__));

struct oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed
{
    union __attribute__((__packed__))
    {
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
        union __attribute__((__packed__))
        {
            Bitfield<uint16_t, 0, 1> reserved;
            Bitfield<uint16_t, 1, 15> beamId;
        };
#else
        union __attribute__((__packed__))
        {
            Bitfield<uint16_t, 0, 15> beamId;
            Bitfield<uint16_t, 15, 1> reserved;
        };
#endif
        struct oran_cmsg_sect_ext_type_11_disableBFWs_0_beamId beam_id_struct; //refactor

    };

    uint8_t bfw[0];
} __attribute__((__packed__));

struct oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr
{
    oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompParam bfwCompParam;
    union __attribute__((__packed__))
    {
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
        union __attribute__((__packed__))
        {
            Bitfield<uint16_t, 0, 1> reserved;
            Bitfield<uint16_t, 1, 15> beamId;
        };
#else
        union __attribute__((__packed__))
        {
            Bitfield<uint16_t, 0, 15> beamId;
            Bitfield<uint16_t, 15, 1> reserved;
        };
#endif
        struct oran_cmsg_sect_ext_type_11_disableBFWs_0_beamId beam_id_struct; //refactor
    };
    uint8_t bfw[0];
} __attribute__((__packed__));


struct oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle
{
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 1> reserved;
        Bitfield<uint16_t, 1, 15> beamId;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 15> beamId;
        Bitfield<uint16_t, 15, 1> reserved;
    };
#endif
} __attribute__((__packed__));

struct oran_cmsg_sect_ext_type_11_disableBFWs_1
{
    struct oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle bundles[0];
} __attribute__((__packed__));

#define ORAN_CMSG_SECT_EXT11_HDR ( \
   sizeof(struct oran_cmsg_ext_hdr) +    \
    sizeof(struct oran_cmsg_sect_ext_type_11))

#define ORAN_CMSG_SECT_EXT11_DISABLEBFW0_HDR ( \
    ORAN_CMSG_SECT_EXT11_HDR +    \
    sizeof(struct oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompParam))

#define ORAN_CMSG_SECT_EXT11_DISABLEBFW1_HDR ORAN_CMSG_SECT_EXT11_HDR

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Utils functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline const char* ecpri_msgtype_to_string(int x)
{
    if(x == ECPRI_MSG_TYPE_IQ)
        return "Type #0: IQ Data";
    if(x == ECPRI_MSG_TYPE_RTC)
        return "Type #2: Real-Time Control";
    if(x == ECPRI_MSG_TYPE_ND)
        return "Type #5: Network Delay";

    return "Unknown";
}

inline const char* oran_direction_to_string(enum oran_pkt_dir x)
{
    if(x == DIRECTION_UPLINK)
        return "Uplink";
    if(x == DIRECTION_DOWNLINK)
        return "Downlink";

    return "Unknown";
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define F_TYPE __inline__ __device__ __host__

F_TYPE uint8_t oran_umsg_get_frame_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload + ORAN_IQ_HDR_OFFSET))->frameId;
}
F_TYPE uint8_t oran_umsg_get_subframe_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload + ORAN_IQ_HDR_OFFSET))->subframeId;
}
F_TYPE uint8_t oran_umsg_get_slot_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload + ORAN_IQ_HDR_OFFSET))->slotId;
}
F_TYPE uint8_t oran_umsg_get_symbol_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload + ORAN_IQ_HDR_OFFSET))->symbolId;
}
//oran_u_section_uncompressed

F_TYPE uint8_t oran_umsg_get_rb(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_u_section_uncompressed*)(mbuf_payload + ORAN_IQ_STATIC_OVERHEAD))->rb;
}
F_TYPE uint16_t oran_umsg_get_start_prb(uint8_t* mbuf_payload)
{
    return (uint16_t)((struct oran_u_section_uncompressed*)(mbuf_payload + ORAN_IQ_STATIC_OVERHEAD))->startPrbu;
}
F_TYPE uint8_t oran_umsg_get_num_prb(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_u_section_uncompressed*)(mbuf_payload + ORAN_IQ_STATIC_OVERHEAD))->numPrbu;
}

F_TYPE uint8_t* oran_umsg_get_first_section_buf(uint8_t* mbuf_payload)
{
    return (mbuf_payload + ORAN_IQ_STATIC_OVERHEAD);
}

F_TYPE uint16_t oran_umsg_get_section_id(uint8_t* mbuf_payload)
{
    return (uint16_t)((struct oran_u_section_uncompressed*)(mbuf_payload + ORAN_IQ_STATIC_OVERHEAD))->sectionId.get();
}

F_TYPE uint16_t oran_umsg_get_start_prb_from_section_buf(uint8_t* section_buf)
{
    return (uint16_t)((struct oran_u_section_uncompressed*)(section_buf))->startPrbu;
}
F_TYPE uint8_t oran_umsg_get_num_prb_from_section_buf(uint8_t* section_buf)
{
    return (uint8_t)((struct oran_u_section_uncompressed*)(section_buf))->numPrbu;
}
F_TYPE uint16_t oran_umsg_get_section_id_from_section_buf(uint8_t* section_buf)
{
    return (uint16_t)((struct oran_u_section_uncompressed*)(section_buf))->sectionId;
}

F_TYPE uint8_t oran_umsg_get_com_meth_from_section_buf(uint8_t* section_buf)
{
    return (uint8_t)((struct oran_u_section_compression_hdr*)(section_buf + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD))->udCompMeth;
}

F_TYPE uint8_t oran_umsg_get_iq_width_from_section_buf(uint8_t* section_buf)
{
    return (uint8_t)((struct oran_u_section_compression_hdr*)(section_buf + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD))->udIqWidth;
}

F_TYPE uint8_t oran_umsg_get_comp_hdr_reserved_bits_from_section_buf(uint8_t* section_buf)
{
    return (uint8_t)((struct oran_u_section_compression_hdr*)(section_buf + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD))->reserved;
}

// oran_cmsg_radio_app_hdr
F_TYPE uint8_t oran_cmsg_get_frame_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_cmsg_radio_app_hdr*)(mbuf_payload + ORAN_CMSG_HDR_OFFSET))->frameId;
}
F_TYPE uint8_t oran_cmsg_get_subframe_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_cmsg_radio_app_hdr*)(mbuf_payload + ORAN_CMSG_HDR_OFFSET))->subframeId;
}
F_TYPE uint8_t oran_cmsg_get_slot_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_cmsg_radio_app_hdr*)(mbuf_payload + ORAN_CMSG_HDR_OFFSET))->slotId;
}
F_TYPE uint8_t oran_cmsg_get_startsymbol_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_cmsg_radio_app_hdr*)(mbuf_payload + ORAN_CMSG_HDR_OFFSET))->startSymbolId;
}
F_TYPE uint32_t oran_cmsg_get_number_of_sections(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_cmsg_radio_app_hdr*)(mbuf_payload + ORAN_CMSG_HDR_OFFSET))->numberOfSections;
}

F_TYPE uint8_t oran_cmsg_get_section_type(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_cmsg_radio_app_hdr*)(mbuf_payload + ORAN_CMSG_HDR_OFFSET))->sectionType;
}

F_TYPE uint8_t oran_msg_get_data_direction(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_cmsg_radio_app_hdr*)(mbuf_payload + ORAN_CMSG_HDR_OFFSET))->dataDirection;
}

F_TYPE uint8_t oran_msg_get_sect1_common_hdr_reserved_field(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_cmsg_sect1_common_hdr*)(mbuf_payload + ORAN_CMSG_HDR_OFFSET))->reserved;
}

F_TYPE bool oran_cmsg_get_section_1_ef(oran_cmsg_sect1* sect_hdr)
{
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    return !(!(sect_hdr->ef & 0b1000000000000000));
#else
    return !(!(sect_hdr->ef & 0b0000000000000001));
#endif
}

F_TYPE bool oran_cmsg_get_ext_ef(oran_cmsg_ext_hdr* ext_hdr)
{
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    return !(!(ext_hdr->ef & 0b10000000));
#else
    return !(!(ext_hdr->ef & 0b00000001));
#endif
}

F_TYPE bool oran_cmsg_get_ext_11_disableBFWs(oran_cmsg_sect_ext_type_11* ext_11_hdr)
{
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    return !(!(ext_11_hdr->disableBFWs & 0b10000000));
#else
    return !(!(ext_11_hdr->disableBFWs & 0b00000001));
#endif
}

F_TYPE bool oran_cmsg_is_ext_11(oran_cmsg_ext_hdr* ext_hdr)
{
    return ext_hdr->extType.get() == ORAN_CMSG_SECTION_EXT_TYPE_11;
}

/******************************************************************/ /**
 * \brief Get the size of the BFW compression parameter
 * \param bfwCompMeth The BFW compression method
 * \return The size of the BFW compression parameter in bytes, -1 if the method is not supported
 */
F_TYPE int oran_cmsg_get_bfwCompParam_size(UserDataBFWCompressionMethod bfwCompMeth)
{
    switch(bfwCompMeth)
    {
        case UserDataBFWCompressionMethod::NO_COMPRESSION:
            return 0;
        case UserDataBFWCompressionMethod::BLOCK_FLOATING_POINT:
            [[fallthrough]];
        case UserDataBFWCompressionMethod::BLOCK_SCALING:
            [[fallthrough]];
        case UserDataBFWCompressionMethod::U_LAW:
            return 1;
        case UserDataBFWCompressionMethod::BEAMSPACE_1:
            [[fallthrough]];
        case UserDataBFWCompressionMethod::BEAMSPACE_2:
            [[fallthrough]];
        default:
            return -1;
    }
}

/******************************************************************/ /**
 * \brief Get the size of the BFW compression parameter
 * \param bfwCompMeth The BFW compression method
 * \return The size of the BFW compression parameter in bytes, -1 if the method is not supported
 */
F_TYPE int oran_cmsg_get_bfw_bundle_hdr_size(UserDataBFWCompressionMethod bfwCompMeth)
{
    switch(bfwCompMeth)
    {
        case UserDataBFWCompressionMethod::NO_COMPRESSION:
            return sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed);
        case UserDataBFWCompressionMethod::BLOCK_FLOATING_POINT:
            [[fallthrough]];
        case UserDataBFWCompressionMethod::BLOCK_SCALING:
            [[fallthrough]];
        case UserDataBFWCompressionMethod::U_LAW:
            return sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr);
        case UserDataBFWCompressionMethod::BEAMSPACE_1:
            [[fallthrough]];
        case UserDataBFWCompressionMethod::BEAMSPACE_2:
            [[fallthrough]];
        default:
            return -1;
    }
}

/******************************************************************/ /**
 * \brief Calculate padding bytes needed for alignment
 * \param current_len Current length of data
 * \return Number of padding bytes needed
 */
F_TYPE uint32_t oran_cmsg_se11_disableBFWs_0_padding_bytes(uint32_t current_len)
{
    return ORAN_SECT_EXT_11_ALIGNMENT * ((current_len + ORAN_SECT_EXT_11_ALIGNMENT - 1) / ORAN_SECT_EXT_11_ALIGNMENT) - current_len;
}

F_TYPE bool oran_cmsg_is_ext_4(oran_cmsg_ext_hdr* ext_hdr)
{
    return ext_hdr->extType.get() == ORAN_CMSG_SECTION_EXT_TYPE_4;
}

F_TYPE bool oran_cmsg_is_ext_5(oran_cmsg_ext_hdr* ext_hdr)
{
    return ext_hdr->extType.get() == ORAN_CMSG_SECTION_EXT_TYPE_5;
}

// oran_ecpri_hdr
F_TYPE uint16_t oran_cmsg_get_ecpri_payload(const uint8_t* mbuf_payload)
{
    return (uint16_t)(
        ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriPayload << 8 |
        ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriPayload >> 8);
}

// oran_ecpri_hdr
F_TYPE uint16_t oran_umsg_get_ecpri_payload(uint8_t* mbuf_payload)
{
    return (uint16_t)(
        ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriPayload << 8 |
        ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriPayload >> 8);
}

F_TYPE uint8_t oran_get_sequence_id(uint8_t* mbuf_payload)
{
    return ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriSeqid;
}

F_TYPE uint16_t oran_msg_get_flowid(uint8_t* mbuf_payload)
{
    return (uint16_t)(
        ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriRtcid << 8 |
        ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriRtcid >> 8);
}

F_TYPE uint16_t oran_cmsg_get_flowid(uint8_t* mbuf_payload)
{
    return (uint16_t)(
        ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriRtcid << 8 |
        ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriRtcid >> 8);
}

F_TYPE uint16_t oran_umsg_get_flowid(uint8_t* mbuf_payload)
{
    return (uint16_t)(
        ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriPcid << 8 |
        ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriPcid >> 8);
}

F_TYPE uint32_t oran_get_offset_from_hdr(uint8_t* pkt, int flow_index, int symbols_x_slot, int prbs_per_symbol, int prb_size)
{
    return (flow_index * symbols_x_slot * prbs_per_symbol * prb_size) +
           (oran_umsg_get_symbol_id(pkt) * prbs_per_symbol * prb_size) +
           (oran_umsg_get_start_prb(pkt) * prb_size);
}

F_TYPE uint32_t oran_get_offset_from_hdr(uint8_t* pkt, int flow_index, int symbols_x_slot, int prbs_per_symbol, int prb_size, int startPrb)
{
    return (flow_index * symbols_x_slot * prbs_per_symbol * prb_size) +
           (oran_umsg_get_symbol_id(pkt) * prbs_per_symbol * prb_size) +
           (startPrb * prb_size);
}

F_TYPE uint32_t oran_srs_get_offset_from_hdr(uint8_t* pkt, int flow_index, int symbols_x_slot, int prbs_per_symbol, int prb_size, uint8_t start_symbol_x_slot)
{
    return (flow_index * symbols_x_slot * prbs_per_symbol * prb_size) +
           ((oran_umsg_get_symbol_id(pkt)-start_symbol_x_slot) * prbs_per_symbol * prb_size) +
           (oran_umsg_get_start_prb(pkt) * prb_size);
}

F_TYPE uint32_t oran_srs_get_offset_from_hdr(uint8_t* pkt, int flow_index, int symbols_x_slot, int prbs_per_symbol, int prb_size, int startPrb, uint8_t start_symbol_x_slot)
{
    return (flow_index * symbols_x_slot * prbs_per_symbol * prb_size) +
           ((oran_umsg_get_symbol_id(pkt)-start_symbol_x_slot) * prbs_per_symbol * prb_size) +
           (startPrb * prb_size);
}

F_TYPE uint16_t oran_cmsg_get_startprbc(uint8_t* mbuf_payload, uint8_t sect_type)
{
    switch(sect_type)
    {
    case ORAN_CMSG_SECTION_TYPE_1:
        return (uint16_t)((struct oran_cmsg_sect1*)(mbuf_payload + ORAN_CMSG_SECT1_FIELDS_OFFSET))->startPrbc;
        break;
    case ORAN_CMSG_SECTION_TYPE_3:
        return (uint16_t)((struct oran_cmsg_sect3*)(mbuf_payload + ORAN_CMSG_SECT3_FIELDS_OFFSET))->startPrbc;
        break;
    default:
        return 0;
    }
}
F_TYPE uint16_t oran_cmsg_get_numprbc(uint8_t* mbuf_payload, uint8_t sect_type)
{
    switch(sect_type)
    {
    case ORAN_CMSG_SECTION_TYPE_1:
        return (uint16_t)((struct oran_cmsg_sect1*)(mbuf_payload + ORAN_CMSG_SECT1_FIELDS_OFFSET))->numPrbc;
        break;
    case ORAN_CMSG_SECTION_TYPE_3:
        return (uint16_t)((struct oran_cmsg_sect3*)(mbuf_payload + ORAN_CMSG_SECT3_FIELDS_OFFSET))->numPrbc;
        break;
    default:
        return 0;
    }
}

F_TYPE uint16_t oran_cmsg_get_numsymbol(uint8_t* mbuf_payload, uint8_t sect_type)
{
    switch(sect_type)
    {
    case ORAN_CMSG_SECTION_TYPE_1:
        return (uint8_t)((struct oran_cmsg_sect1*)(mbuf_payload + ORAN_CMSG_SECT1_FIELDS_OFFSET))->numSymbol;
        break;
    case ORAN_CMSG_SECTION_TYPE_3:
        return (uint8_t)((struct oran_cmsg_sect3*)(mbuf_payload + ORAN_CMSG_SECT3_FIELDS_OFFSET))->numSymbol;
        break;
    default:
        return 0;
    }
}

F_TYPE uint32_t oran_cmsg_get_section_id(uint8_t* mbuf_payload, uint8_t sect_type)
{
    switch(sect_type)
    {
    case ORAN_CMSG_SECTION_TYPE_1:
        return (uint8_t)((struct oran_cmsg_sect1*)(mbuf_payload + ORAN_CMSG_SECT1_FIELDS_OFFSET))->sectionId.get();
        break;
    case ORAN_CMSG_SECTION_TYPE_3:
        return (uint8_t)((struct oran_cmsg_sect3*)(mbuf_payload + ORAN_CMSG_SECT3_FIELDS_OFFSET))->sectionId.get();
        break;
    default:
        return 0;
    }
}

F_TYPE uint8_t oran_msg_get_message_type(uint8_t* mbuf_payload)
{
    return ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriMessage;
}

inline struct oran_ether_addr* oran_cmsg_get_src_eth_addr(struct oran_eth_hdr* ethvlan_hdr)
{
    return &(ethvlan_hdr->eth_hdr.src_addr);
}

inline struct oran_ether_addr* oran_cmsg_get_dst_eth_addr(struct oran_eth_hdr* ethvlan_hdr)
{
    return &(ethvlan_hdr->eth_hdr.dst_addr);
}

constexpr int ORAN_BFP_COMPRESSION_9_BITS  = 9;
constexpr int ORAN_BFP_COMPRESSION_14_BITS = 14;
constexpr int ORAN_BFP_NO_COMPRESSION      = 16;

constexpr int ORAN_COMPRESSION_METH = 4;

#define ECPRI_REV_UP_TO_20        1

F_TYPE bool ecpri_hdr_sanity_check(uint8_t* mbuf_payload)
{
    bool ret = true;
    auto* ecpri_hdr = (struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET);
    if(ecpri_hdr->ecpriVersion.get() != ECPRI_REV_UP_TO_20)
    {
        printf("Wrong ecpriVersion: %d, we currently support: %d\n", ecpri_hdr->ecpriVersion.get(), ECPRI_REV_UP_TO_20);
        ret = false;
    }

    if(ecpri_hdr->ecpriReserved.get() != 0)
    {
        printf("Wrong ecpriReserved: %d, we currently support: %d\n", ecpri_hdr->ecpriReserved.get(), 0);
        ret = false;
    }

    if(ecpri_hdr->ecpriConcatenation.get() != 0)
    {
        printf("Wrong ecpriConcatenation: %d, we currently support: %d\n", ecpri_hdr->ecpriConcatenation.get(), 0);
        ret = false;
    }

#if 0 //If it's not correct, it will be dropped by NIC due to flow rules
    if(ecpri_hdr->ecpriMessage != ECPRI_MSG_TYPE_IQ)
    {
        printf("Wrong ecpriMessage: %d, should be: %d\n", ecpri_hdr->ecpriMessage, ECPRI_MSG_TYPE_IQ);
        ret = false;
    }
#endif

    if(ecpri_hdr->ecpriSubSeqid.get() != 0)
    {
        printf("Wrong ecpriSubSeqid: %d, we currently support: %d\n", ecpri_hdr->ecpriSubSeqid.get(), 0);
        ret = false;
    }
#if 0
    if(ecpri_hdr->ecpriEbit.get() != 1)
    {
        printf("Wrong ecpriEbit: %d, we currently support: %d\n", ecpri_hdr->ecpriEbit.get(), 1);
        ret = false;
    }
#endif 
    return ret;
}

F_TYPE bool uplane_pkt_sanity_check(uint8_t* pkt_buffer, int comp_bits_cell, int dl_comp_meth)
{
    uint8_t* section_buf          = oran_umsg_get_first_section_buf(pkt_buffer);
    uint16_t ecpri_payload_length = oran_umsg_get_ecpri_payload(pkt_buffer);
    uint16_t current_length       = 4 + sizeof(oran_umsg_iq_hdr);
    uint16_t num_prb              = 0;
    uint16_t compressed_prb_size  = 0;
    uint16_t prb_buffer_size      = 0;
    uint16_t section_overhead     = 0;
    if(dl_comp_meth == ORAN_COMPRESSION_METH)
    {
        auto iq_width       = oran_umsg_get_iq_width_from_section_buf(section_buf);
        compressed_prb_size = iq_width * 3;
        section_overhead    = ORAN_IQ_COMPRESSED_SECTION_OVERHEAD;
    }
    else
    {
        compressed_prb_size = (comp_bits_cell == ORAN_BFP_NO_COMPRESSION) ? PRB_SIZE_16F : (comp_bits_cell == ORAN_BFP_COMPRESSION_14_BITS) ? PRB_SIZE_14F :
                                                                                                                                              PRB_SIZE_9F;
        section_overhead    = ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
    }

    while(current_length < ecpri_payload_length)
    {
        if(ecpri_payload_length - current_length <= section_overhead)
        {
            //printf("Wrong BFP or num_prb\n");
            return false;
        }
        num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
        if(num_prb == 0)
            num_prb = ORAN_MAX_PRB_X_SLOT;
        prb_buffer_size = compressed_prb_size * num_prb;
        if(current_length + prb_buffer_size + section_overhead > ecpri_payload_length)
        {
            //printf("Wrong BFP or num_prb\n");
            return false;
        }
        current_length += prb_buffer_size + section_overhead;
        section_buf += section_overhead + prb_buffer_size;
    }
    return true;
}

/******************************************************************/ /**
 * \brief Type of static compression supported today
 *
 */
enum iq_data_fmt
{
    BFP_COMPRESSION_9_BITS  = 9,
    BFP_COMPRESSION_10_BITS = 10,
    BFP_COMPRESSION_11_BITS = 11,
    BFP_COMPRESSION_12_BITS = 12,
    BFP_COMPRESSION_13_BITS = 13,
    BFP_COMPRESSION_14_BITS = 14,
    BFP_COMPRESSION_15_BITS = 15,
    BFP_NO_COMPRESSION      = 16,
    FIXED_POINT_16_BITS     = 17,
    MOD_COMPRESSION_4_BITS  = 18,
    IQ_DATA_FMT_MAX
};

/******************************************************************/ /**
 * \brief Type of O-RU supported today
 *
 */
enum ru_type
{
    SINGLE_SECT_MODE     = 1,
    MULTI_SECT_MODE      = 2,
    OTHER_MODE           = 3
};


const int MAX_BANDWIDTH = 100; // Maximum channel bandwidth in MHz
const int NUM_SCS = 3; // Number of Subcarrier Spacing options

// Lookup table for maximum transmission bandwidth (in Resource Blocks)
// Table 5.3.2-1: Maximum transmission bandwidth configuration NRB
const static std::array<std::array<int, MAX_BANDWIDTH + 1>, NUM_SCS> max_tx_bandwidth = {{
    // 15 kHz SCS
    {0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 52, 0, 0, 0, 0, 79, 0, 0, 0, 0, 106, 0, 0, 0, 0, 133, 0, 0, 0, 0, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 270},
    // 30 kHz SCS
    {0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 24, 0, 0, 0, 0, 38, 0, 0, 0, 0, 51, 0, 0, 0, 0, 65, 0, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 245, 0, 0, 0, 0, 0, 0, 0, 0, 0, 273},
    // 60 kHz SCS
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 18, 0, 0, 0, 0, 24, 0, 0, 0, 0, 31, 0, 0, 0, 0, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135}
}};

inline int getMaxTransmissionBandwidth(int scs, int channelBandwidth) {
    if (scs < 0 || scs >= NUM_SCS || channelBandwidth <= 0 || channelBandwidth > MAX_BANDWIDTH) {
        return -1; // Invalid input
    }
    return max_tx_bandwidth[scs][channelBandwidth];
}

// Lookup table for channel bandwidth (MHz) based on max transmission bandwidth (RBs) and SCS (kHz)
// Table 5.3.2-1: Maximum transmission bandwidth configuration NRB

static std::unordered_map<int, std::unordered_map<int, int>> bandwidthTable = {
    {15, {{5, 25}, {52, 10}, {79, 15}, {106, 20}, {133, 25}, {160, 30}, {216, 40}, {270, 50}}},
    {30, {{5, 11}, {10, 24}, {15, 38}, {20, 51}, {25, 65}, {30, 78}, {40, 106}, {50, 133}, {60, 162}, {80, 217}, {100, 273}}},
    {60, {{10, 11}, {15, 18}, {20, 24}, {25, 31}, {30, 38}, {40, 51}, {50, 65}, {60, 79}, {80, 107}, {90, 121}, {100, 135}}}
};

inline int getMaxTransmissionBWNRB(int channelBandwidth, int scs) {
    if (bandwidthTable.find(scs) == bandwidthTable.end()) {
        return -1;
    }

    auto& scsBandwidths = bandwidthTable[scs];
    auto it = scsBandwidths.find(channelBandwidth);
    if (it == scsBandwidths.end()) {
        return -1;
    }

    return it->second;
}

// Lookup table for minimum guardband and transmission bandwidth configuration based on max transmission bandwidth (RBs) and SCS (kHz)
// Table 5.3.3-1: Minimum guardband for each UE channel bandwidth and SCS (kHz)
const static std::unordered_map<int, std::unordered_map<int, int>> fr1_guardband = {
    {15, {{5, 242}, {10, 312}, {15, 382}, {20, 452}, {25, 522}, {30, 592}, {40, 552}, {50, 692}}},
    {30, {{5, 505}, {10, 665}, {15, 645}, {20, 805}, {25, 785}, {30, 945}, {40, 905}, {50, 1045}, {60, 825}, {80, 925}, {90, 885}, {100, 845}}},
    {60, {{10, 1010}, {15, 990}, {20, 1330}, {25, 1310}, {30, 1290}, {40, 1610}, {50, 1570}, {60, 1530}, {80, 1450}, {90, 1410}, {100, 1370}}}
};

inline int getGuardband(int scs, int channelBandwidth) {
    if (fr1_guardband.find(scs) != fr1_guardband.end()) {
        const auto& bandwidths = fr1_guardband.at(scs);
        if (bandwidths.find(channelBandwidth) != bandwidths.end()) {
            return bandwidths.at(channelBandwidth);
        }
    }
    return -1; // Invalid input or guardband not found
}

inline int getPRACHStartPRB(int frequencyOffset, int subcarrierSpacing, int ulBandwidth) {
    int guardbw = getGuardband(subcarrierSpacing, ulBandwidth);
    int halfbw = (-(ulBandwidth * 1000)/2);
    int startPrb = ((frequencyOffset * subcarrierSpacing / 2 - halfbw  - guardbw - subcarrierSpacing/2)) / (subcarrierSpacing * PRB_NUM_RE);
    return startPrb;
}

#endif //ifndef AERIAL_FH_DRIVER_ORAN__
