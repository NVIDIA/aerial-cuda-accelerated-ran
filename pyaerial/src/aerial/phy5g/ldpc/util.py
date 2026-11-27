# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pyAerial library - Utility functions for LDPC coding chain."""
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from aerial.phy5g.config import PdschConfig
from aerial import pycuphy


__all__ = [
    "get_mcs",
    "get_tb_size",
    "get_base_graph",
    "max_code_block_size",
    "find_lifting_size",
    "get_num_info_nodes",
    "get_code_block_num_info_bits",
    "get_code_block_size",
    "get_num_code_blocks",
    "code_block_segment",
    "code_block_desegment",
    "add_crc_len",
    "random_tb",
    "get_crc_len",
    "get_pdsch_config_attrs",
    "get_pdsch_tb_sizes"
]


def get_mcs(mcs: int, table_idx: int = 2) -> Tuple[int, float]:
    """Get modulation order and code rate based on MCS index.

    Args:
        mcs (int): MCS index pointing to the table indicated by `table_idx`.
        table_idx (int): Index of the MCS table in TS 38.214 section 5.1.3.1. Values:
            - 1: TS38.214, table 5.1.3.1-1.
            - 2: TS38.214, table 5.1.3.1-2.
            - 3: TS38.214, table 5.1.3.1-3.

    Returns:
        int, float: A tuple containing:

        - *int*:
          Modulation order.
        - *float*:
          Code rate * 1024.
    """
    if table_idx == 1:
        mcs_table = {
            0: [2, 120.],
            1: [2, 157.],
            2: [2, 193.],
            3: [2, 251.],
            4: [2, 308.],
            5: [2, 379.],
            6: [2, 449.],
            7: [2, 526.],
            8: [2, 602.],
            9: [2, 679.],
            10: [4, 340.],
            11: [4, 378.],
            12: [4, 434.],
            13: [4, 490.],
            14: [4, 553.],
            15: [4, 616.],
            16: [4, 658.],
            17: [6, 438.],
            18: [6, 466.],
            19: [6, 517.],
            20: [6, 567.],
            21: [6, 616.],
            22: [6, 666.],
            23: [6, 719.],
            24: [6, 772.],
            25: [6, 822.],
            26: [6, 873.],
            27: [6, 910.],
            28: [6, 948.]
        }

    elif table_idx == 2:
        mcs_table = {
            0: [2, 120.],
            1: [2, 193.],
            2: [2, 308.],
            3: [2, 449.],
            4: [2, 602.],
            5: [4, 378.],
            6: [4, 434.],
            7: [4, 490.],
            8: [4, 553.],
            9: [4, 616.],
            10: [4, 658.],
            11: [6, 466.],
            12: [6, 517.],
            13: [6, 567.],
            14: [6, 616.],
            15: [6, 666.],
            16: [6, 719.],
            17: [6, 772.],
            18: [6, 822.],
            19: [6, 873.],
            20: [8, 682.5],
            21: [8, 711.],
            22: [8, 754.],
            23: [8, 797.],
            24: [8, 841.],
            25: [8, 885.],
            26: [8, 916.5],
            27: [8, 948.],
        }

    elif table_idx == 3:
        mcs_table = {
            0: [2, 30.],
            1: [2, 40.],
            2: [2, 50.],
            3: [2, 64.],
            4: [2, 78.],
            5: [2, 99.],
            6: [2, 120.],
            7: [2, 157.],
            8: [2, 193.],
            9: [2, 251.],
            10: [2, 308.],
            11: [2, 379.],
            12: [2, 449.],
            13: [2, 526.],
            14: [2, 602.],
            15: [4, 340.],
            16: [4, 378.],
            17: [4, 434.],
            18: [4, 490.],
            19: [4, 553.],
            20: [4, 616.],
            21: [6, 438.],
            22: [6, 466.],
            23: [6, 517.],
            24: [6, 567.],
            25: [6, 616.],
            26: [6, 666.],
            27: [6, 719.],
            28: [6, 772.],
        }
    else:
        raise NotImplementedError(f"MCS table {table_idx} not supported!")

    mod_order, code_rate = mcs_table[mcs]
    return int(mod_order), code_rate


def get_tb_size(
        *,
        mod_order: int,
        code_rate: float,
        dmrs_syms: List[int],
        num_prbs: int,
        start_sym: int,
        num_symbols: int,
        num_layers: int) -> int:
    """Get transport block size based on given parameters.

    Determine transport block size as per TS 38.214 section 5.1.3.2.

    Args:
        mod_order (int): Modulation order.
        code_rate (float): Code rate * 1024 as in section 5.1.3.1 of TS 38.214.
        dmrs_syms (List[int]): List of binary numbers indicating which symbols contain DMRS.
        num_prbs (int): Number of PRBs.
        start_sym (int): Starting symbol.
        num_symbols (int): Number of symbols.
        num_layers (int): Number of layers.

    Returns:
        int: Transport block size in bits.
    """
    code_rate = code_rate / 1024
    n_sc = 12
    # Overhead parameter N_oh = 0.
    n_re = num_prbs * (np.array(dmrs_syms[start_sym:start_sym + num_symbols]) == 0).sum() * n_sc
    n_info = n_re * code_rate * mod_order * num_layers
    if n_info <= 3824:

        n = np.max([3, np.floor(np.log2(n_info)) - 6])

        n_info_prime = np.max([24, np.power(2, n) * np.floor(n_info / np.power(2, n))])

        tbs_select = [
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
            136,
            144,
            152,
            160,
            168,
            176,
            184,
            192,
            208,
            224,
            240,
            256,
            272,
            288,
            304,
            320,
            336,
            352,
            368,
            384,
            408,
            432,
            456,
            480,
            504,
            528,
            552,
            576,
            608,
            640,
            672,
            704,
            736,
            768,
            808,
            848,
            888,
            928,
            984,
            1032,
            1064,
            1128,
            1160,
            1192,
            1224,
            1256,
            1288,
            1320,
            1352,
            1416,
            1480,
            1544,
            1608,
            1672,
            1736,
            1800,
            1864,
            1928,
            2024,
            2088,
            2152,
            2216,
            2280,
            2408,
            2472,
            2536,
            2600,
            2664,
            2728,
            2792,
            2856,
            2976,
            3104,
            3240,
            3368,
            3496,
            3624,
            3752,
            3824,
        ]

        for tbs_item in tbs_select:
            if tbs_item >= n_info_prime:
                tbs = tbs_item
                break

    else:

        n = np.floor(np.log2(n_info - 24)) - 5

        n_info_prime = np.max(
            [3840, np.power(2, n) * np.round((n_info - 24) / np.power(2, n))]
        )

        if code_rate < 0.25:
            C = np.ceil((n_info + 24) / 3816)

            tbs = int(8 * C * np.ceil((n_info_prime + 24) / 8 / C))

        else:
            if n_info_prime > 8424:

                C = np.ceil((n_info_prime + 24) / 8424)

                tbs = int(8 * C * np.ceil((n_info_prime + 24) / 8 / C))

            else:
                tbs = 8 * np.ceil((n_info_prime + 24) / 8)

        tbs -= 24

    return int(tbs)


def get_base_graph(tb_size: int, code_rate: float) -> int:  # pylint: disable=invalid-name
    """Get LDPC base graph.

    Args:
        tb_size (int): Transport block size in bits, without CRC.
        code_rate (float): Code rate.

    Returns:
        int: Base graph, 1 or 2.
    """
    if tb_size <= 292 or (tb_size <= 3824 and code_rate <= (2 / 3)) or code_rate <= 0.25:
        return 2
    return 1


def max_code_block_size(base_graph: int) -> int:
    """Get maximum LDPC code block size based on base graph.

    Args:
        base_graph (int): Base graph, 1 or 2.

    Returns:
        int: Maximum code block size.
    """
    if base_graph == 1:
        max_size = 8448
    elif base_graph == 2:
        max_size = 3840
    else:
        raise ValueError(f"Invalid value {base_graph} given for base graph!")
    return max_size


def get_num_info_nodes(
    base_graph: int, tb_size: int
) -> int:  # pylint: disable=invalid-name,too-many-return-statements
    """Get number of information nodes.

    Note: This is the value `K_b` in TS 38.212.

    Args:
        base_graph (int): Base graph, 1 or 2.
        tb_size (int): Transport block size without any CRCs.

    Returns:
        int: The number of information nodes (`K_b`).
    """
    tb_size = add_crc_len(tb_size)

    if base_graph < 1 or base_graph > 2:
        raise ValueError(f"Invalid value {base_graph} given for base graph!")

    if base_graph == 1:
        return 22
    if tb_size > 640:
        return 10
    if tb_size > 560:
        return 9
    if tb_size > 192:
        return 8
    return 6


def find_lifting_size(base_graph: int, tb_size: int) -> int:
    """Find lifting size for base graph.

    Args:
        base_graph (int): Base graph, 1 or 2.
        tb_size (int): Transport block size in bits without CRC.

    Returns:
        int: Lifting size.
    """
    lifting_sizes = [
        2,   3,   4,   5,   6,   7,    # noqa: E241
        8,   9,   10,  11,  12,  13,   # noqa: E241
        14,  15,  16,  18,  20,  22,   # noqa: E241
        24,  26,  28,  30,  32,  36,   # noqa: E241
        40,  44,  48,  52,  56,  60,   # noqa: E241
        64,  72,  80,  88,  96,  104,  # noqa: E241
        112, 120, 128, 144, 160, 176,  # noqa: E241
        192, 208, 224, 240, 256, 288,  # noqa: E241
        320, 352, 384                  # noqa: E241
    ]

    if base_graph < 1 or base_graph > 2:
        raise ValueError(f"Invalid value {base_graph} given for base graph!")

    # Disable pylint and follow TS 38.212 notation here.
    Kprime = get_code_block_num_info_bits(base_graph, tb_size)  # pylint: disable=invalid-name
    Kb = get_num_info_nodes(base_graph, tb_size)  # pylint: disable=invalid-name

    for Zc in lifting_sizes:  # pylint: disable=invalid-name
        if Zc * Kb >= Kprime:
            return Zc

    raise ValueError(
        f"Unable to find lifting size for base graph {base_graph} and TB size {tb_size}!"
    )


def get_code_block_num_info_bits(base_graph: int, tb_size: int) -> int:
    """Get number of information bits in a code block.

    This is the number K' in TS 38.212, i.e. the number of information
    bits without the filler bits.

    Args:
        base_graph (int): Base graph, 1 or 2.
        tb_size (int): Transport block size in bits, without CRC.

    Returns:
        int: Number of information bits in a code block.
    """
    tb_size = add_crc_len(tb_size)
    max_cb_size = max_code_block_size(base_graph)

    crc_len = 24  # Code block CRC.
    if tb_size > max_cb_size:
        num_code_blocks = math.ceil(tb_size / (max_cb_size - crc_len))
        tb_size_with_cb_crc = tb_size + num_code_blocks * crc_len  # This is B' in TS 38.212.
    else:
        num_code_blocks = 1
        tb_size_with_cb_crc = tb_size
    Kprime = tb_size_with_cb_crc / num_code_blocks  # pylint: disable=invalid-name
    return int(Kprime)


def get_code_block_size(tb_size: int, code_rate: float) -> int:
    """Get code block size.

    This is the number K in TS 38.212, i.e. the number of information bits
    including filler bits.

    Args:
        tb_size (int): Transport block size in bits, without CRC.
        code_rate (float): Code rate.

    Returns:
        int: Code block size.
    """
    base_graph = get_base_graph(tb_size, code_rate)
    lifting_size = find_lifting_size(base_graph, tb_size)
    if base_graph == 1:
        code_block_size = lifting_size * 22
    else:
        code_block_size = lifting_size * 10
    return code_block_size


def get_num_code_blocks(tb_size: int, code_rate: float) -> int:
    """Return the number of code blocks for a transport block.

    Args:
        tb_size (int): Transport block size in bits, without CRC.
        code_rate (float): Code rate.

    Returns:
        int: The number of code blocks (C).
    """
    base_graph = get_base_graph(tb_size, code_rate)

    # Check if this TB size requires code block segmentation.
    crc_len = 24
    max_cb_size = max_code_block_size(base_graph)
    tb_size_with_crc = add_crc_len(tb_size)
    if tb_size_with_crc > max_cb_size:
        num_code_blocks = math.ceil(tb_size_with_crc / (max_cb_size - crc_len))
    else:
        num_code_blocks = 1

    return num_code_blocks


def code_block_segment(tb_size: int, transport_block: np.ndarray, code_rate: float) -> np.ndarray:
    """Do code block segmentation.

    This function does code block segmentation as per TS 38.212 section 5.2.2.
    Randomly generated 24-bit string is attached to each code block to emulate code
    block CRC if there is more than one code block.

    Args:
        tb_size (int): Transport block size in bits, without CRC.
        transport_block (np.ndarray): Transport block in bits, CRC included.
        code_rate (float): Code rate.

    Returns:
        np.ndarray: The code blocks.
    """
    code_block_size = get_code_block_size(tb_size, code_rate)
    num_code_blocks = get_num_code_blocks(tb_size, code_rate)

    base_graph = get_base_graph(tb_size, code_rate)
    Kprime = get_code_block_num_info_bits(base_graph, tb_size)  # pylint: disable=invalid-name

    code_blocks = np.zeros((code_block_size, num_code_blocks), dtype=np.uint8)
    if num_code_blocks > 1:
        code_blocks_no_crc = transport_block.reshape(Kprime - 24, num_code_blocks, order="F")
        crc = np.random.randint(0, 1, size=(24, num_code_blocks), dtype=np.uint8)
        code_blocks[:Kprime, :] = np.concatenate((code_blocks_no_crc, crc), axis=0)
    else:
        code_blocks[:Kprime, :] = transport_block[:, None]

    return code_blocks


def code_block_desegment(
        code_blocks: np.ndarray,
        tb_size: int,
        code_rate: float,
        return_bits: bool = True) -> np.ndarray:
    """Concatenate code blocks coming from LDPC decoding into a transport block.

    This function desegments code blocks into a transport block as per TS 38.212, and
    removes the CRCs, i.e. does the opposite of :func:`~aerial.phy5g.ldpc.util.code_block_segment`.

    Args:
        code_blocks (np.ndarray): The code blocks coming out of the LDPC decoder as a N x C array.
        tb_size (int): Transport block size in bits, without CRC.
        code_rate (float): Code rate.
        return_bits (bool): If True (default), give the return value in bits. Otherwise convert
            to bytes.

    Returns:
        np.ndarray: The transport block with CRC, in bits or bytes depending on
        the value of `return_bits`.
    """
    base_graph = get_base_graph(tb_size, code_rate)
    Kprime = get_code_block_num_info_bits(base_graph, tb_size)  # pylint: disable=invalid-name
    num_code_blocks = get_num_code_blocks(tb_size, code_rate)

    # Remove padding and code block CRCs.
    if num_code_blocks > 1:
        code_blocks = code_blocks[:Kprime - 24, :]
    else:
        code_blocks = code_blocks[:Kprime, :]

    # Concatenate.
    transport_block = code_blocks.reshape(-1, order="F")
    transport_block = transport_block.astype(np.uint8)

    if not return_bits:
        transport_block = np.packbits(transport_block)  # type: ignore[assignment]

    return transport_block


def add_crc_len(tb_size: int) -> int:
    """Append CRC length to transport block size.

    Args:
        tb_size (int): Transport block size in bits without CRC.

    Returns:
        int: Transport block size in bits with CRC.
    """
    if tb_size > 3824:
        length = tb_size + 24
    else:
        length = tb_size + 16

    return length


def get_crc_len(tb_size: int) -> int:
    """Return CRC length based on transport block size.

    Args:
        tb_size (int): Transport block size in bits without CRC.

    Returns:
        int: CRC length (either 16 or 24 bits).
    """
    if tb_size > 3824:
        length = 24
    else:
        length = 16

    return length


def random_tb(
        *,
        mod_order: int,
        code_rate: float,
        dmrs_syms: List[int],
        num_prbs: int,
        start_sym: int,
        num_symbols: int,
        num_layers: int,
        return_bits: bool = False) -> np.ndarray:
    """Generate a random transport block.

    Generates random transport block according to given parameters. The transport
    block size is first determined as per TS 38.214 section 5.1.3.2.

    Args:
        mod_order (int): Modulation order.
        code_rate (float): Code rate * 1024 as in section 5.1.3.1 of TS 38.214.
        dmrs_syms (List[int]): List of binary numbers indicating which symbols contain DMRS.
        num_prbs (int): Number of PRBs.
        start_sym (int): Starting symbol.
        num_symbols (int): Number of symbols.
        num_layers (int): Number of layers.
        return_bits (bool): Whether to return the transport block in bits (True) or bytes (False).

    Returns:
        np.ndarray: Random transport block payload.
    """
    tbs = get_tb_size(
        mod_order=mod_order,
        code_rate=code_rate,
        dmrs_syms=dmrs_syms,
        num_prbs=num_prbs,
        start_sym=start_sym,
        num_symbols=num_symbols,
        num_layers=num_layers
    )
    if return_bits:
        payload = np.random.randint(0, 2, size=tbs, dtype=np.uint8)
    else:
        payload = np.random.randint(0, 255, size=tbs // 8, dtype=np.uint8)

    return payload


def get_pdsch_config_attrs(pdsch_configs: List[PdschConfig],
                           attrs: List[str]) -> Dict[str, List[Any]]:
    """Get attributes from PDSCH configurations.

    This function is used to get attributes from PDSCH configurations. It returns a dictionary
    where each key corresponds to a list of attribute values, one entry per transport block.

    Args:
        pdsch_configs (List[PdschConfig]): List of PDSCH configurations.
        attrs (List[str]): List of attributes to get.

    Returns:
        Dict[str, List[Any]]: Dictionary of lists of attributes.
    """
    attrs_dict = {attr: [] for attr in attrs}  # type: Dict[str, List[Any]]

    for pdsch_config in pdsch_configs:
        for ue_config in pdsch_config.ue_configs:
            for cw_config in ue_config.cw_configs:
                for attr in attrs:
                    try:
                        value = getattr(cw_config, attr)
                    except AttributeError:
                        pass
                    else:
                        attrs_dict[attr].append(value)
                        continue

                    try:
                        value = getattr(ue_config, attr)
                    except AttributeError:
                        pass
                    else:
                        attrs_dict[attr].append(value)
                        continue

                    try:
                        value = getattr(pdsch_config, attr)
                    except AttributeError as exc:
                        raise AttributeError(
                            f"Attribute {attr} not found in the given PDSCH configurations!"
                        ) from exc
                    attrs_dict[attr].append(value)

    return attrs_dict


def get_pdsch_tb_sizes(pdsch_configs: List[PdschConfig]) -> List[int]:
    """Get transport block sizes from PDSCH configurations.

    Args:
        pdsch_configs (List[PdschConfig]): List of PDSCH configurations.

    Returns:
        List[int]: List of transport block sizes.
    """
    tb_sizes = []
    attrs = ["code_rate",
             "dmrs_syms",
             "start_sym",
             "num_symbols",
             "num_dmrs_cdm_grps_no_data",
             "num_prbs",
             "layers",
             "mod_order"]
    pdsch_config_attrs = get_pdsch_config_attrs(pdsch_configs, attrs)
    num_tb = len(pdsch_config_attrs["code_rate"])
    for tb_idx in range(num_tb):
        dmrs_syms = pdsch_config_attrs["dmrs_syms"][tb_idx]
        start_sym = pdsch_config_attrs["start_sym"][tb_idx]
        num_symbols = pdsch_config_attrs["num_symbols"][tb_idx]
        num_dmrs_symbols = sum(dmrs_syms[start_sym:start_sym + num_symbols])
        num_data_symbols = num_symbols - num_dmrs_symbols
        tb_size = pycuphy.get_tb_size(
            num_data_symbols,
            num_dmrs_symbols,
            pdsch_config_attrs["num_prbs"][tb_idx],
            pdsch_config_attrs["layers"][tb_idx],
            pdsch_config_attrs["code_rate"][tb_idx] / 10240.,
            pdsch_config_attrs["mod_order"][tb_idx],
            pdsch_config_attrs["num_dmrs_cdm_grps_no_data"][tb_idx]
        )
        tb_sizes.append(tb_size)
    return tb_sizes
