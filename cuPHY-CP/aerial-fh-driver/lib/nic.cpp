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

#include "nic.hpp"

#include "flow.hpp"
#include "gpu_mempool.hpp"
#include "peer.hpp"
#include "utils.hpp"
#include "gpu_comm.hpp"
#include <time.h>
#include <stdio.h>
#include <net/if.h>
#include <cstdlib>
#include <cstring>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <doca_dpdk.h>
#include <doca_rdma_bridge.h>
#include <doca_error.h>
#pragma GCC diagnostic pop

// TODO FIXME remove when DOCA warnings are fixed
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#define TAG (NVLOG_TAG_BASE_FH_DRIVER + 19) // "FH.NIC"
#define TAG_STATS (NVLOG_TAG_BASE_FH_DRIVER + 20) //"FH.STATS"


namespace aerial_fh
{
Nic::Nic(Fronthaul* fhi, NicInfo const* info) :
    fhi_{fhi},
    info_{*info},
    metrics_{this},
    queue_manager_{this}
{
    NVLOGI_FMT(TAG, "Adding NIC {}" , info_.name);

    validate_input();
    doca_probe_device();
    validate_driver();
    configure();

    if(driver_name_ == kMlxPciDriverName)
    {
        set_pcie_max_read_request_size();
        disable_ethernet_flow_control();
    }

    set_mtu();
    restrict_ingress_traffic();
    create_cpu_mbuf_pool();
    if (fhi->get_info().cuda_device_ids.empty() && !fhi->get_info().cuda_device_ids_for_compute.empty()) //CPU Init Comms mode
    {
        create_cpu_pinned_mbuf_pool();
        //create_gpu_mbuf_pool();
    }
    create_tx_request_uplane_pool();
    create_tx_request_cplane_pool();
    setup_tx_queues();

    setup_rx_queues();
    start();
    setup_tx_queues_gpu();
    check_physical_link_status();
    metrics_.cache_metric_ids();
    warm_up_txqs();

    fhi->add_nic(this);
    if( fhi->rmax_init_nic(this) )
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Failed to initialize Rivermax for NIC {}", info_.name);
    }

    if (!(fhi->get_info().cuda_device_ids.empty())) {
        gcomm_ = std::make_unique<GpuComm>(this);
        gpu_comm_init_tx_queues();
    }
    if(is_cx6())
    {
        set_qp_clock_id();
    }
    if (!(fhi->get_info().cuda_device_ids.empty()))
        set_flow_comm_buf();
    fh_extended_stats_init();
}

Nic::~Nic()
{
    fhi_->remove_nic(this);

    if(port_id_ != (uint8_t)-1)
    {
        gpu_mbuf_pool_.reset();
        remove_device();
    }
}

static doca_error_t
get_dpdk_port_id_doca_dev(struct doca_dev *dev_input, uint16_t *port_id)
{
	struct doca_dev *dev_local = NULL;
	char pci_addr_input[DOCA_DEVINFO_PCI_ADDR_SIZE];
	doca_error_t result;
	uint16_t dpdk_port_id;
	uint8_t is_addr_equal = 0;

	if (dev_input == NULL || port_id == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	*port_id = RTE_MAX_ETHPORTS;

	result = doca_devinfo_get_pci_addr_str(doca_dev_as_devinfo(dev_input), pci_addr_input);
	if (result != DOCA_SUCCESS) {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Failed to get device PCI address {} {}",pci_addr_input, doca_error_get_descr(result));
		return result;
	}

	for (dpdk_port_id = 0; dpdk_port_id < RTE_MAX_ETHPORTS; dpdk_port_id++) {
		/* search for the probed devices */
		if (!rte_eth_dev_is_valid_port(dpdk_port_id))
			continue;

		result = doca_dpdk_port_as_dev(dpdk_port_id, &dev_local);
		if (result != DOCA_SUCCESS) {
            NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Failed to find DOCA device associated with port ID {} {}",dpdk_port_id, doca_error_get_descr(result));
			return result;
		}

		result = doca_devinfo_is_equal_pci_addr(doca_dev_as_devinfo(dev_local), pci_addr_input, &is_addr_equal);
		if (result != DOCA_SUCCESS) {
            NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Failed to get device PCI address {} {}",pci_addr_input, doca_error_get_descr(result));
			return result;
		}

		if (is_addr_equal) {
			*port_id = dpdk_port_id;
			break;
		}
	}

	if (*port_id == RTE_MAX_ETHPORTS) {
		NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "No DPDK port matches the DOCA device.");
		return DOCA_ERROR_NOT_FOUND;
	}

	return DOCA_SUCCESS;
}

static doca_error_t
open_doca_device_with_pci(const char *pcie_value, struct doca_dev **retval)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	doca_error_t result;
	size_t i;
	uint8_t is_addr_equal = 0;

	/* Set default return value */
	*retval = NULL;

	result = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (result != DOCA_SUCCESS) {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Failed to load doca devices list {}", doca_error_get_descr(result));
		return result;
	}

	/* Search */
	for (i = 0; i < nb_devs; i++) {
		result = doca_devinfo_is_equal_pci_addr(dev_list[i], pcie_value, &is_addr_equal);
		if (result == DOCA_SUCCESS && is_addr_equal) {
			/* if device can be opened */
			result = doca_dev_open(dev_list[i], retval);
			if (result == DOCA_SUCCESS) {
				doca_devinfo_destroy_list(dev_list);
				return result;
			}
		}
	}

	NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Matching device not found.");
	result = DOCA_ERROR_NOT_FOUND;

	doca_devinfo_destroy_list(dev_list);
	return result;
}

void Nic::doca_probe_device()
{
    doca_error_t ret_doca;
	int ret;
	auto name                  = info_.name.c_str();
    auto accu_tx_sched_disable = fhi_->get_info().accu_tx_sched_disable;
    auto accu_tx_sched_res_ns  = fhi_->get_info().accu_tx_sched_res_ns;
    enum doca_eth_wait_on_time_type wait_on_time_mode;

    StringBuilder devargs_builder;


	ret_doca = open_doca_device_with_pci(name, &ddev_);
	if (ret_doca != DOCA_SUCCESS)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"open_doca_device_with_pci returned {}", doca_error_get_descr(ret_doca));
    }

	ret_doca = doca_eth_txq_cap_get_wait_on_time_offload_supported(doca_dev_as_devinfo(ddev_), &wait_on_time_mode);
	if (ret_doca != DOCA_SUCCESS)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"doca_eth_txq_get_wait_on_time_offload_supported returned {}", doca_error_get_descr(ret_doca));
    }

    if(wait_on_time_mode == DOCA_ETH_WAIT_ON_TIME_TYPE_DPDK)
    {
        cx6 = true;
        NVLOGI_FMT(TAG, "cx6 device, wait_on_time_mode ={}", +wait_on_time_mode);
    }
    else
    {
        cx6 = false;
        NVLOGI_FMT(TAG, "Non cx6 device, wait_on_time_mode={}", +wait_on_time_mode);
    }

    if(cx6) //tx_pp application only for CX-6 device
    {
        if(!accu_tx_sched_disable)
            devargs_builder << "tx_pp=" << accu_tx_sched_res_ns << ",";
    }

    devargs_builder << "txq_inline_max=0,dv_flow_en=2"; //HWS
    auto devargs = std::string(devargs_builder);

    NVLOGI_FMT(TAG, "DOCA hotplug-adding NIC {} with following devargs: {}", name, devargs.c_str());


    //Explicitly enable DPDK steering here (should be done before port probe)
    if(rte_pmd_mlx5_driver_enable_steering()<0)
    {
        THROW_FH(rte_errno, StringBuilder() << "Failed to enable DPDK steering");
    }

    ret_doca = doca_dpdk_port_probe(ddev_, devargs.c_str());
    if (ret_doca != DOCA_SUCCESS)
    {
        THROW_FH(ret_doca, StringBuilder() << "doca_dpdk_port_probe returned " << doca_error_get_descr(ret_doca));
    }

    ret_doca = get_dpdk_port_id_doca_dev(ddev_, &port_id_);
    if (ret_doca != DOCA_SUCCESS)
    {
        THROW_FH(ret_doca, StringBuilder() << "get_dpdk_port_id_doca_dev returned " << doca_error_get_descr(ret_doca));
    }
}

struct doca_dev * Nic::get_doca_dev()
{
    return ddev_;
}

void Nic::validate_input()
{
    auto cpu_mbuf_num = info_.cpu_mbuf_num;
    if(!(fhi_->get_info().cuda_device_ids.empty()) && cpu_mbuf_num == 0)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Invalid cpu_mbuf_num value: {}", cpu_mbuf_num);
    }

    auto tx_request_num = info_.tx_request_num;
    if(tx_request_num == 0)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Invalid tx_request_num value: {}", tx_request_num);
    }

    auto txq_size = info_.txq_size;
    if(txq_size == 0)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Invalid txq_size value: {}", txq_size);
    }

    auto rxq_size = info_.rxq_size;
    if(rxq_size == 0)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Invalid rxq_size value: {}", rxq_size);
    }
}

void Nic::print_rx_offloads(uint64_t offloads)
{
    uint64_t      single_offload;
    int           begin;
    int           end;
    int           bit;
    StringBuilder rx_offloads;

    if(offloads == 0)
    {
        return;
    }

    rx_offloads << "NIC " << info_.name << " RX offloads:";

    begin = __builtin_ctzll(offloads);
    end   = sizeof(offloads) * CHAR_BIT - __builtin_clzll(offloads);

    single_offload = 1ULL << begin;
    for(bit = begin; bit < end; bit++)
    {
        if(offloads & single_offload)
        {
            rx_offloads << " " << rte_eth_dev_rx_offload_name(single_offload);
        }

        single_offload <<= 1;
    }

    NVLOGD_FMT(TAG,"{}", std::string(rx_offloads).c_str());
}

void Nic::print_tx_offloads(uint64_t offloads)
{
    uint64_t      single_offload;
    int           begin;
    int           end;
    int           bit;
    StringBuilder tx_offloads;

    if(offloads == 0)
    {
        return;
    }

    tx_offloads << "NIC " << info_.name << " TX offloads:";

    begin = __builtin_ctzll(offloads);
    end   = sizeof(offloads) * CHAR_BIT - __builtin_clzll(offloads);

    single_offload = 1ULL << begin;
    for(bit = begin; bit < end; bit++)
    {
        if(offloads & single_offload)
        {
            tx_offloads << " " << rte_eth_dev_tx_offload_name(single_offload);
        }

        single_offload <<= 1;
    }

    NVLOGD_FMT(TAG,"{}", std::string(tx_offloads).c_str());
}

void Nic::configure()
{
    rte_eth_conf eth_conf{};
    auto& rxq_count = info_.rxq_count;
    auto  txq_count = info_.txq_count + info_.txq_count_gpu;

    eth_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_TIMESTAMP;
    eth_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MULTI_SEGS;

    if(is_cx6())
    {
        if(!fhi_->get_info().accu_tx_sched_disable)
            eth_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_SEND_ON_TIMESTAMP;
    }
    else
    {
        eth_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_SEND_ON_TIMESTAMP;
    }

    if(fhi_->pdump_enabled())
        rxq_count++;

    NVLOGI_FMT(TAG, "Initializing NIC {} with {} RX queues and {} TX queues", info_.name.c_str(), rxq_count, txq_count);

    auto ret = rte_eth_dev_configure(port_id_, rxq_count, txq_count, &eth_conf);
    if(ret)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Failed to initialize NIC {}: {}", info_.name, rte_strerror(-ret));
    }

    print_rx_offloads(eth_conf.rxmode.offloads);
    print_tx_offloads(eth_conf.txmode.offloads);
}

bool Nic::is_cx6() {
    return cx6;
}

void Nic::set_mtu()
{
    auto mtu = info_.mtu;

    NVLOGI_FMT(TAG, "Setting MTU size to {} for NIC {}", mtu, info_.name.c_str());

    auto ret = rte_eth_dev_set_mtu(port_id_, mtu);
    if(ret)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to set MTU to {} for NIC {}: {}" , mtu, info_.name, rte_strerror(-ret));
    }
}

void Nic::setup_tx_queues()
{
    auto txq_count = info_.txq_count;
    auto txq_count_gpu = info_.txq_count_gpu;
    auto txq_size  = info_.txq_size;
    uint16_t txq_idx = 0;

    for(txq_idx = 0; txq_idx < txq_count; txq_idx++)
    {
        queue_manager_.add(new Txq(this, txq_idx, false));
    }
}

void Nic::setup_tx_queues_gpu()
{
    auto txq_count = info_.txq_count;
    auto txq_count_gpu = info_.txq_count_gpu;
    auto txq_size  = info_.txq_size;
    uint16_t txq_idx = 0;

    for(txq_idx = txq_count; txq_idx < txq_count + txq_count_gpu; txq_idx++)
    {
        queue_manager_.add(new Txq(this, txq_idx, true));
    }
}

void Nic::setup_rx_queues()
{
    uint16_t rxq_idx = 0;
    if(fhi_->pdump_enabled())
    {
        queue_manager_.add(new RxqPcap(this, rxq_idx++));
    }

    auto rxq_count = info_.rxq_count;
    while(rxq_idx < rxq_count)
    {
        queue_manager_.add(new Rxq(this, rxq_idx++));
    }
}

void Nic::create_cpu_mbuf_pool()
{
    uint16_t droom_sz  = RTE_ALIGN_MUL_CEIL(info_.mtu + RTE_PKTMBUF_HEADROOM + RTE_ETHER_HDR_LEN + RTE_ETHER_CRC_LEN, kMbufPoolDroomSzAlign);
    uint16_t priv_size = RTE_ALIGN(sizeof(rte_mbuf_ext_shared_info), RTE_MBUF_PRIV_ALIGN);
    uint32_t mbuf_num;

    if(!(fhi_->get_info().cuda_device_ids.empty()))
    {
        mbuf_num  = rte_align32pow2(info_.cpu_mbuf_num) - 1;
    }
    else
    {
        mbuf_num  = rte_align32pow2(info_.cpu_mbuf_tx_num) - 1;
    }


    std::string name = StringBuilder() << "cpu_mbuf_" << info_.name;

    NVLOGI_FMT(TAG, "Initializing CPU mbuf mempool for NIC {}. It will have {} mbuf entries. Mbuf droom sz: {}", info_.name.c_str(), mbuf_num, droom_sz);

    auto mp = rte_pktmbuf_pool_create(name.c_str(), mbuf_num, 0, priv_size, droom_sz, rte_eth_dev_socket_id(port_id_));

    if(mp == nullptr)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Could not create {} mempool: {}", name , rte_strerror(rte_errno));
    }

    auto mbuf_initializer = [](struct rte_mempool* mp, void* opaque, void* obj, __rte_unused unsigned obj_idx) -> void {
        auto m      = static_cast<struct rte_mbuf*>(obj);
        auto shinfo = static_cast<rte_mbuf_ext_shared_info*>(rte_mbuf_to_priv(m));

        m->buf_iova = RTE_BAD_IOVA;
        m->shinfo   = shinfo;
    };

    auto objs_iterated = rte_mempool_obj_iter(mp, mbuf_initializer, this);
    if(objs_iterated != mbuf_num)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to pre-initialize CPU mbufs");
    }
    cpu_mbuf_pool_.reset(mp);

    if(info_.split_cpu_mp && !info_.per_rxq_mempool)
    {
        uint16_t droom_sz  = RTE_ALIGN_MUL_CEIL(info_.mtu + RTE_PKTMBUF_HEADROOM + RTE_ETHER_HDR_LEN + RTE_ETHER_CRC_LEN, kMbufPoolDroomSzAlign);
        uint16_t priv_size = RTE_ALIGN(sizeof(rte_mbuf_ext_shared_info), RTE_MBUF_PRIV_ALIGN);
        uint32_t mbuf_num;
        mbuf_num  = rte_align32pow2(info_.cpu_mbuf_rx_num) - 1;

        std::string name = StringBuilder() << "cpu_tx_mbuf_" << info_.name;

        NVLOGI_FMT(TAG, "Initializing CPU mbuf mempool for NIC {}. It will have {} mbuf entries. Mbuf droom sz: {}", info_.name.c_str(), mbuf_num, droom_sz);

        auto mp = rte_pktmbuf_pool_create(name.c_str(), mbuf_num, 0, priv_size, droom_sz, rte_eth_dev_socket_id(port_id_));

        if(mp == nullptr)
        {
            NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Could not create {} mempool: {}", name , rte_strerror(rte_errno));
        }

        auto mbuf_initializer = [](struct rte_mempool* mp, void* opaque, void* obj, __rte_unused unsigned obj_idx) -> void {
            auto m      = static_cast<struct rte_mbuf*>(obj);
            auto shinfo = static_cast<rte_mbuf_ext_shared_info*>(rte_mbuf_to_priv(m));

            m->buf_iova = RTE_BAD_IOVA;
            m->shinfo   = shinfo;
        };

        auto objs_iterated = rte_mempool_obj_iter(mp, mbuf_initializer, this);
        if(objs_iterated != mbuf_num)
        {
            NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to pre-initialize CPU mbufs");
        }
        cpu_tx_mbuf_pool_.reset(mp);
    }
}

void Nic::create_gpu_mbuf_pool()
{
    auto cuda_device = info_.cuda_device;
    if(cuda_device < 0)
    {
        return;
    }

    gpu_mbuf_pool_.reset(new GpuMempool(fhi_->gpus()[cuda_device].get(), this,false));
}

void Nic::create_cpu_pinned_mbuf_pool()
{
    auto cuda_device = info_.cuda_device;
    if(cuda_device < 0)
    {
        return;
    }

    cpu_pinned_mbuf_pool_.reset(new GpuMempool(fhi_->gpus()[cuda_device].get(), this,true));
}


void Nic::create_tx_request_uplane_pool()
{
    size_t      num_entries = rte_align32pow2(info_.tx_request_num) - 1;
    size_t      object_size = sizeof(TxRequestUplane);
    std::string name        = StringBuilder() << "tx_request_" << info_.name;

    NVLOGD_FMT(TAG, "Initializing TX request mempool with {} entries; object size: {}.", num_entries, object_size );

    auto mp = rte_mempool_create(name.c_str(), num_entries, object_size, 0, 0, nullptr, nullptr, nullptr, nullptr, rte_eth_dev_socket_id(port_id_), 0);

    if(mp == nullptr)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Could not create {} mempool: {}", name, rte_strerror(rte_errno));
    }

    tx_request_pool_.reset(mp);
}

void Nic::create_tx_request_cplane_pool()
{
    size_t      num_entries = rte_align32pow2(info_.tx_request_num) - 1;
    size_t      object_size = sizeof(TxRequestCplane);
    std::string name        = StringBuilder() << "tx_request_c_" << info_.name;

    NVLOGD_FMT(TAG, "Initializing TX request C-plane mempool with {} entries; object size: {}.",num_entries, object_size);

    auto mp = rte_mempool_create(name.c_str(), num_entries, object_size, 0, 0, nullptr, nullptr, nullptr, nullptr, rte_eth_dev_socket_id(port_id_), 0);

    if(mp == nullptr)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Could not create {} mempool: {}", name, rte_strerror(rte_errno));
    }

    tx_request_cplane_pool_.reset(mp);
}

void Nic::check_physical_link_status() const
{
    rte_eth_link link{};
    char         link_status_text[RTE_ETH_LINK_MAX_STR_LEN];

    auto ret = rte_eth_link_get(port_id_, &link);
    if(ret)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to get link info for NIC {} : {}", info_.name, rte_strerror(-ret));
    }

    std::string link_status = link.link_status == RTE_ETH_LINK_DOWN ? "DOWN" : "UP";
    std::string link_duplex = link.link_duplex == RTE_ETH_LINK_FULL_DUPLEX ? "full-duplex" : "half-duplex";

    if(link.link_status == RTE_ETH_LINK_DOWN)
    {
        // A missing physical link is fatal by default. Cable-less bring-up
        // (development workstations, CI without a wired RU, pre-RU lab setups)
        // can set AERIAL_FH_LINK_REQUIRED=0 to downgrade this to an error and
        // let initialization continue. The comparison is an exact string match
        // rather than an integer parse, so that anything unrecognized -- an
        // empty value, "false", a typo -- leaves the fatal default in place.
        const char* link_required_env = std::getenv("AERIAL_FH_LINK_REQUIRED");
        if((link_required_env != nullptr) && (std::strcmp(link_required_env, "0") == 0))
        {
            NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "NIC {} has no physical Ethernet link (cable unplugged or peer port down); continuing because AERIAL_FH_LINK_REQUIRED=0", info_.name);
        }
        else
        {
            NVLOGF_FMT(TAG, AERIAL_DPDK_API_EVENT, "NIC {} has no physical Ethernet link (cable unplugged or peer port down)", info_.name);
        }
    }

    NVLOGI_FMT(TAG, "NIC {} link status: {}, {}, {}",
        info_.name.c_str(), link_status.c_str(), rte_eth_link_speed_to_str(link.link_speed), link_duplex);
}

void Nic::print_stats() const
{
    rte_eth_stats stats{};

    auto ret = rte_eth_stats_get(port_id_, &stats);
    if(ret)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to get NIC {} stats: {}", info_.name,rte_strerror(-ret));
    }

    NVLOGC_FMT(TAG, "NIC {} stats:", info_.name);
    NVLOGC_FMT(TAG, "  tx_packets: {}", stats.opackets);
    NVLOGC_FMT(TAG, "  rx_packets: {}", stats.ipackets);
    NVLOGC_FMT(TAG, "  tx_bytes: {}", stats.obytes);
    NVLOGC_FMT(TAG, "  rx_bytes: {}", stats.ibytes);
    NVLOGC_FMT(TAG, "  tx_errors: {}", stats.oerrors);
    NVLOGC_FMT(TAG, "  rx_errors: {}", stats.ierrors);
    NVLOGC_FMT(TAG, "  rx_missed: {}", stats.imissed);
    NVLOGC_FMT(TAG, "  rx_nombuf: {}", stats.rx_nombuf);
}

void Nic::fh_extended_stats_init()
{
    // Get count
    auto cnt_xstats = rte_eth_xstats_get_names(port_id_, nullptr, 0);
    if(cnt_xstats < 0)
    {
        THROW_FH(EINVAL, StringBuilder() << "Failed to get NIC " << info_.name << " xstats count");
    }

    // Get id-name lookup table
    auto xstats_names = static_cast<rte_eth_xstat_name*>(malloc(sizeof(rte_eth_xstat_name) * cnt_xstats));
    if(xstats_names == nullptr)
    {
        THROW_FH(ENOMEM, "Cannot allocate memory for xstats lookup");
    }

    UniquePtr xstats_names_unique{static_cast<void*>(xstats_names), free};

    if(cnt_xstats != rte_eth_xstats_get_names(port_id_, xstats_names, cnt_xstats))
    {
        THROW_FH(EINVAL, StringBuilder() << "Failed to look up NIC " << info_.name << " xstats");
    }

    // Get stats themselves
    auto xstats = static_cast<rte_eth_xstat*>(malloc(sizeof(rte_eth_xstat) * cnt_xstats));
    if(xstats == nullptr)
    {
        THROW_FH(ENOMEM, "Cannot allocate memory for xstats");
    }

    UniquePtr xstats_unique{static_cast<void*>(xstats), free};

    if(cnt_xstats != rte_eth_xstats_get(port_id_, xstats, cnt_xstats))
    {
        THROW_FH(EINVAL, StringBuilder() << "Failed to get NIC " << info_.name << " xstats");
    }

    fh_xstats_.num_xstats = 0;

    // Display xstats
    NVLOGD_FMT(TAG, "NIC {} extended stats:", info_.name.c_str());
    for(int idx_xstat = 0; idx_xstat < cnt_xstats; idx_xstat++)
    {
        NVLOGI_FMT(TAG, "  name: {} -> id: {}", xstats_names[idx_xstat].name, xstats[idx_xstat].id);

#ifdef TX_PP_ONLY
        if(strncmp(xstats_names[idx_xstat].name, "tx_pp", strlen("tx_pp")) != 0)
        {
            continue;
        }
#endif

        strncpy(fh_xstats_.names[fh_xstats_.num_xstats], xstats_names[idx_xstat].name, sizeof(xstats_names[idx_xstat].name));
        fh_xstats_.ids[fh_xstats_.num_xstats] = xstats[idx_xstat].id;
        fh_xstats_.values[fh_xstats_.num_xstats] = xstats[idx_xstat].value;
        fh_xstats_.num_xstats++;
        fh_xstats_.prev_values[xstats[idx_xstat].id] = xstats[idx_xstat].value;
    }
}

int Nic::fh_extended_stats_retrieval()
{
    try
    {
        auto cnt_xstats = rte_eth_xstats_get_by_id(port_id_, fh_xstats_.ids, fh_xstats_.values, fh_xstats_.num_xstats);
        if(cnt_xstats < 0)
        {
            THROW_FH(EINVAL, StringBuilder() << "Failed to get NIC " << info_.name << " xstats");
        }

        bool any_diff = false;
        for(int idx_xstat = 0; idx_xstat < cnt_xstats; idx_xstat++)
        {
            if(fh_xstats_.prev_values[fh_xstats_.ids[idx_xstat]] != fh_xstats_.values[idx_xstat])
            {
                any_diff = true;
                fh_xstats_.prev_values[fh_xstats_.ids[idx_xstat]] = fh_xstats_.values[idx_xstat];
            }
        }
        if(any_diff == false)
        {
            return 0;
        }

        bool error_log = false;
        for(int idx_xstat = 0; idx_xstat < cnt_xstats; idx_xstat++)
        {
            if(strncmp(fh_xstats_.names[idx_xstat], "tx_pp_sync_lost", strlen("tx_pp_sync_lost")) == 0 && fh_xstats_.values[idx_xstat])
            {
                error_log = true;
                break;
            }
        }

        // Display xstats
        if(error_log)
        {
            NVLOGE_FMT(TAG_STATS, AERIAL_ORAN_FH_EVENT, "NIC {} extended stats:", info_.name.c_str());
            for(int idx_xstat = 0; idx_xstat < cnt_xstats; idx_xstat++)
            {
                NVLOGE_FMT(TAG_STATS, AERIAL_ORAN_FH_EVENT, "  {}: {}", fh_xstats_.names[idx_xstat], fh_xstats_.values[idx_xstat]);
            }
        }
        else
        {
            NVLOGD_FMT(TAG_STATS, "NIC {} extended stats:", info_.name.c_str());
            for(int idx_xstat = 0; idx_xstat < cnt_xstats; idx_xstat++)
            {
                NVLOGD_FMT(TAG_STATS, "  {}: {}", fh_xstats_.names[idx_xstat], fh_xstats_.values[idx_xstat]);
            }
        }
    }
    catch(aerial_fh::FronthaulException const& e)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Exception! {}", e.what());
        return e.err_code();
    }
    return 0;
}

void Nic::print_extended_stats() const
{
    // Get count
    auto cnt_xstats = rte_eth_xstats_get_names(port_id_, nullptr, 0);
    if(cnt_xstats < 0)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to get NIC {} xstats count",info_.name);
    }

    // Get id-name lookup table
    auto xstats_names = static_cast<rte_eth_xstat_name*>(malloc(sizeof(rte_eth_xstat_name) * cnt_xstats));
    if(xstats_names == nullptr)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Cannot allocate memory for xstats lookup");
    }

    UniquePtr xstats_names_unique{static_cast<void*>(xstats_names), free};

    if(cnt_xstats != rte_eth_xstats_get_names(port_id_, xstats_names, cnt_xstats))
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to look up NIC {} xstats", info_.name);
    }

    // Get stats themselves
    auto xstats = static_cast<rte_eth_xstat*>(malloc(sizeof(rte_eth_xstat) * cnt_xstats));
    if(xstats == nullptr)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Cannot allocate memory for xstats");
    }

    UniquePtr xstats_unique{static_cast<void*>(xstats), free};

    if(cnt_xstats != rte_eth_xstats_get(port_id_, xstats, cnt_xstats))
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to get NIC {} xstats", info_.name);
    }

    // Display xstats
    NVLOGD_FMT(TAG, "NIC {} extended stats:", info_.name.c_str());
    for(int idx_xstat = 0; idx_xstat < cnt_xstats; idx_xstat++)
    {
        NVLOGD_FMT(TAG, "  {}: {}", xstats_names[idx_xstat].name, xstats[idx_xstat].value);
    }
}

void Nic::reset_stats() const
{
    NVLOGD_FMT(TAG, "Resetting NIC {} stats", info_.name.c_str());
    auto ret = rte_eth_stats_reset(port_id_);
    if(ret)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to reset NIC {} xstats", info_.name);
    }

    ret = rte_eth_xstats_reset(port_id_);
    if(ret)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to reset NIC {} xstats", info_.name);
    }
}

void Nic::set_pcie_max_read_request_size()
{
    NVLOGD_FMT(TAG, "Setting NIC {} PCIe MRRS to 4096 bytes for NIC", info_.name.c_str());

    StringBuilder read_mrrs_command;
    read_mrrs_command << "setpci -s " << info_.name << " 68.w";

    auto pipe_deleter = [](FILE* f) { if (f) pclose(f); };
    std::unique_ptr<FILE, decltype(pipe_deleter)> pipe(popen(std::string(read_mrrs_command).c_str(), "r"), pipe_deleter);
    if(pipe == nullptr)
    {
        NVLOGW_FMT(TAG, "Failed to read NIC {} PCIe MRRS", info_.name.c_str());
        return;
    }

    std::array<char, 128> buffer;
    std::string           result;

    while(fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }

    result[0] = '5';

    StringBuilder set_mrrs_command;
    set_mrrs_command << "setpci -s " << info_.name << " 68.w=" << result << " > /dev/null";

    auto ret = std::system(std::string(set_mrrs_command).c_str());
    if(ret)
    {
        NVLOGW_FMT(TAG, "Failed to set NIC {} PCIe MRRS", info_.name.c_str());
    }
}

void Nic::restrict_ingress_traffic()
{
    rte_flow_error flowerr{};

    NVLOGD_FMT(TAG, "Restricting ingress traffic for NIC {}", info_.name.c_str());

    auto ret = rte_flow_isolate(port_id_, 1, &flowerr);
    if(ret)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to enable flow isolation on NIC {}: {}", info_.name, flowerr.message);
    }
}

uint64_t Nic::get_tx_bytes_phy()
{
    std::string query = "ethtool -S " + if_name_;
    FILE*       fp    = popen(query.c_str(), "r");
    if(fp == NULL)
    {
        return 1;
    }

    std::stringstream ss;
    char              buffer[128];
    while(fgets(buffer, sizeof(buffer), fp) != NULL)
    {
        ss << buffer;
    }

    pclose(fp);

    uint64_t res = 0;

    std::string        line;
    std::istringstream iss(ss.str());
    while(std::getline(iss, line))
    {
        if(line.find("tx_bytes_phy") != std::string::npos)
        {
            std::string str_tx_bytes_phy = line.substr(line.find(":") + 2);
            NVLOGD_FMT(TAG, "tx_bytes_phy of NIC port {}:  {}", port_id_, str_tx_bytes_phy.c_str());
            for(auto& c : str_tx_bytes_phy)
            {
                res = res * 10 + (c - '0');
            }
            break;
        }
    }
    return res;
}

void Nic::add_device()
{
    auto name                  = info_.name.c_str();
    auto accu_tx_sched_disable = fhi_->get_info().accu_tx_sched_disable;
    auto accu_tx_sched_res_ns  = fhi_->get_info().accu_tx_sched_res_ns;

    StringBuilder devargs_builder;

    if(!accu_tx_sched_disable && fhi_->get_info().cuda_device_ids.empty())
    {
        devargs_builder << "tx_pp=" << accu_tx_sched_res_ns << ",";
    }

    devargs_builder << "txq_inline_max=0";
    auto devargs = std::string(devargs_builder);

    NVLOGD_FMT(TAG, "Hotplug-adding NIC {} with following devargs: {}", name, devargs.c_str());

    std::array<std::string, 2> supported_buses{kPciBusName, kAuxBusName};
    for(auto bus_name : supported_buses)
    {
        auto ret = rte_eal_hotplug_add(bus_name.c_str(), name, devargs.c_str());
        if(ret)
        {
            NVLOGD_FMT(TAG, "NIC {} is NOT {} bus device", name, bus_name.c_str());
        }
        else
        {
            NVLOGD_FMT(TAG, "NIC {} is {} bus device", name, bus_name.c_str());
            return;
        }
    }

    NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "No suitable bus found to attach NIC {}", name);
}

void Nic::remove_device()
{
    NVLOGI_FMT(TAG, "Removing NIC {}", info_.name.c_str());

    auto ret = rte_eth_dev_stop(port_id_);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Failed to stop NIC {}: {}", info_.name.c_str(), rte_strerror(-ret));
    }

    ret = rte_eth_dev_close(port_id_);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Failed to close NIC {}: {}", info_.name, rte_strerror(-ret));
    }
}

void Nic::set_port_id()
{
    auto name = info_.name.c_str();
    auto ret  = rte_eth_dev_get_port_by_name(name, &port_id_);
    if(ret)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to get port ID for NIC {}: {}", name, rte_strerror(-ret));
    }
}

void Nic::set_qp_clock_id()
{
    int qp_clock_id = rte_pmd_mlx5_txpp_idx(get_port_id());
    if(qp_clock_id < 0)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "rte_pmd_mlx5_txpp_idx failed");
    }
    else
    {
        qp_clock_id_ = (uint32_t)qp_clock_id;
    }

    qp_clock_id_be_ = rte_cpu_to_be_32(qp_clock_id_);
}

void Nic::set_flow_comm_buf()
{
    num_packets                  = kGpuCommSendPeers * kMaxFlows * kMaxPktsFlow;
    packet_size_rnd              = ((get_mtu() + pageSizeAlign - 1) / pageSizeAlign) * pageSizeAlign;
    packet_size_rnd_local_ = packet_size_rnd;
    if(rte_is_power_of_2(packet_size_rnd_local_) == 0)
        packet_size_rnd_local_ = rte_align32pow2(packet_size_rnd_local_);

    for(int i = 0; i < kGpuCommSendPeers * kMaxFlows; i++)
    {
        flow_idx_q_.push(i);
    }

    doca_error_t ret = doca_create_tx_buf(&flow_tx_buf, get_fronthaul()->get_docaGpuParams()->gpu, get_doca_dev(), DOCA_GPU_MEM_TYPE_GPU, num_packets, packet_size_rnd_local_,fhi_->get_info().enable_gpu_comm_via_cpu);
    if(ret != DOCA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Could not alloc flow DOCA tx buffer");
        THROW_FH(EINVAL, StringBuilder() << "Could not alloc flow DOCA tx buffer");
    }

}

struct doca_tx_buf* Nic::get_flow_comm_buf()
{
    return &flow_tx_buf;
}

void Nic::validate_driver()
{
    rte_eth_dev_info dev_info{};

    NVLOGD_FMT(TAG, "Validating device driver for NIC {}", info_.name.c_str());

    auto ret = rte_eth_dev_info_get(port_id_, &dev_info);
    if(ret)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to get device info for NIC {}: {}", info_.name, rte_strerror(-ret));
    }

    auto driver_name = std::string(dev_info.driver_name);

    NVLOGD_FMT(TAG, "NIC {} is using {} driver", info_.name.c_str(), driver_name.c_str());

    if((driver_name != kMlxPciDriverName) & (driver_name != kMlxAuxDriverName))
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Non-Mellanox NICs are not supported");
    }

    char if_name[IFNAMSIZ];
    if(if_indextoname(dev_info.if_index, if_name) != nullptr)
    {
        if_name_ = std::string(if_name);
        NVLOGD_FMT(TAG, "NIC {} if name is: {}", if_name_, if_name);
    }
    else
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to get NIC {} if_name", info_.name);
    }

    driver_name_ = driver_name;
}

void Nic::disable_ethernet_flow_control()
{
    // Disable flow control
    rte_eth_fc_conf flow_control{};

    NVLOGD_FMT(TAG, "Disabling Ethernet flow control for NIC {}", info_.name.c_str());

    auto ret = rte_eth_dev_flow_ctrl_get(port_id_, &flow_control);
    if(ret)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to get NIC {} current Ethernet link flow control status: {}",info_.name, rte_strerror(-ret));
    }

    flow_control.mode = RTE_ETH_FC_NONE;

    ret = rte_eth_dev_flow_ctrl_set(port_id_, &flow_control);
    if(ret)
    {
        NVLOGW_FMT(TAG, "Failed to get NIC {} current Ethernet link flow control status: {}", info_.name.c_str(), rte_strerror(-ret));
    }
}

void Nic::start()
{
#if 0
    #define DPDK_LAYER_UT_DEFAULT_QUEUE_DEPTH (256)
    #define DPDK_LAYER_UT_DEFAULT_QUEUE (16)

    struct rte_flow_port_attr port_attr = {0};
    struct rte_flow_queue_attr queue_attr;
    const struct rte_flow_queue_attr *queue_attrs[DPDK_LAYER_UT_DEFAULT_QUEUE];
    struct rte_flow_error err = {0};

    port_attr.nb_counters = 1;
    port_attr.nb_aging_objects = 1;
    port_attr.nb_meters = 1;
    queue_attr.size = DPDK_LAYER_UT_DEFAULT_QUEUE_DEPTH;
    for (i = 0; i < 1; ++i)
        queue_attrs[i] = &queue_attr;

    ret = rte_flow_configure(port_id_, &port_attr, 1, queue_attrs, &err);
#endif

    auto ret = rte_eth_dev_start(port_id_);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to start NIC {} with error {}", info_.name,rte_strerror(ret));
    }
}

void Nic::warm_up_txqs()
{
    queue_manager_.warm_up_txqs();
    reset_stats();
}

Fronthaul* Nic::get_fronthaul() const
{
    return fhi_;
}

std::string Nic::get_name() const
{
    return info_.name;
}

std::string Nic::get_if_name() const
{
    return if_name_;
}

uint16_t Nic::get_port_id() const
{
    return port_id_;
}

GpuId Nic::get_cuda_device() const
{
    return info_.cuda_device;
}

uint32_t Nic::get_qp_clock_id() const
{
    return qp_clock_id_;
}

uint32_t Nic::get_qp_clock_id_be() const
{
    return qp_clock_id_be_;
}

uint16_t Nic::get_mtu() const
{
    return info_.mtu;
}

uint16_t Nic::get_nxt_flow_idx()
{
    const std::lock_guard<aerial_fh::FHMutex> lock(flow_idx_q_lock_);
    if(flow_idx_q_.empty())
    {
        NVLOGE_FMT(TAG, AERIAL_MEMORY_EVENT, "No more flow id available, please increase buffer size");
    }
    auto flow_idx = flow_idx_q_.front();
    flow_idx_q_.pop();
    return flow_idx;
}

void Nic::free_flow_idx(uint16_t flow_idx)
{
    const std::lock_guard<aerial_fh::FHMutex> lock(flow_idx_q_lock_);
    flow_idx_q_.push(flow_idx);
}

rte_mempool* Nic::get_cpu_mbuf_pool() const
{
    return cpu_mbuf_pool_.get();
}

rte_mempool* Nic::get_cpu_tx_mbuf_pool() const
{
    return info_.split_cpu_mp ? cpu_tx_mbuf_pool_.get() : get_cpu_mbuf_pool();
}

rte_mempool* Nic::get_rx_mbuf_pool(bool hostPinned) const
{
    if(gpu_mbuf_pool_==nullptr)
    {
        NVLOGI_FMT(TAG,"{}:hostPinned {}",__func__,hostPinned);
    }
    return gpu_mbuf_pool_ == nullptr ? ((hostPinned==true)? cpu_pinned_mbuf_pool_->get_pool():cpu_mbuf_pool_.get()) : gpu_mbuf_pool_->get_pool();
}

rte_mempool* Nic::get_tx_request_pool() const
{
    return tx_request_pool_.get();
}

rte_mempool* Nic::get_tx_request_cplane_pool() const
{
    return tx_request_cplane_pool_.get();
}

void Nic::update_metrics()
{
    metrics_.update();
}

bool Nic::pdump_enabled() const
{
    return fhi_->pdump_enabled() && (driver_name_ == kMlxPciDriverName);
}

NicInfo const& Nic::get_info() const
{
    return info_;
}

QueueManager& Nic::get_queue_manager()
{
    return queue_manager_;
}

RxqPcap* Nic::get_pcap_rxq() const
{
    return queue_manager_.get_pcap_rxq();
}

void Nic::create_cpu_rx_mbuf_per_queue_pool(MempoolUnique& cpu_mbuf_pool_queue_,uint16_t id)
{
    uint16_t droom_sz  = RTE_ALIGN_MUL_CEIL(info_.mtu + RTE_PKTMBUF_HEADROOM + RTE_ETHER_HDR_LEN + RTE_ETHER_CRC_LEN, kMbufPoolDroomSzAlign);
    uint16_t priv_size = RTE_ALIGN(sizeof(rte_mbuf_ext_shared_info), RTE_MBUF_PRIV_ALIGN);
    uint32_t mbuf_num  = rte_align32pow2(info_.cpu_mbuf_rx_num_per_rxq) - 1;

    std::string name = StringBuilder() << "rx_mbuf_" << port_id_<< "_" <<id;

    NVLOGI_FMT(TAG, "Initializing CPU mbuf mempool for NIC {} queue ID {}. It will have {} mbuf entries. Mbuf droom sz: {}", info_.name.c_str(),id ,mbuf_num, droom_sz);

    auto mp = rte_pktmbuf_pool_create(name.c_str(), mbuf_num, 0, priv_size, droom_sz, rte_eth_dev_socket_id(port_id_));

    if(mp == nullptr)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Could not create {} mempool: {}", name , rte_strerror(rte_errno));
    }

    auto mbuf_initializer = [](struct rte_mempool* mp, void* opaque, void* obj, __rte_unused unsigned obj_idx) -> void {
        auto m      = static_cast<struct rte_mbuf*>(obj);
        auto shinfo = static_cast<rte_mbuf_ext_shared_info*>(rte_mbuf_to_priv(m));

        m->buf_iova = RTE_BAD_IOVA;
        m->shinfo   = shinfo;
    };

    auto objs_iterated = rte_mempool_obj_iter(mp, mbuf_initializer, this);
    if(objs_iterated != mbuf_num)
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Failed to pre-initialize CPU mbufs");
    }
    cpu_mbuf_pool_queue_.reset(mp);
}


std::string Nic::get_mac_address()
{
    rte_ether_addr src_addr;
    std::string mac_address;
    char eth_addr[RTE_ETHER_ADDR_LEN*2];
    int i=0, j=0;

    if (rte_eth_macaddr_get(port_id_, &src_addr))
    {
        NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT, "Could not get NIC ({}) MAC address", get_name());
    }

    for(i=0, j=0; i<RTE_ETHER_ADDR_LEN * 2 && j < RTE_ETHER_ADDR_LEN; i+=2, j++)
        sprintf(&(eth_addr[i]), "%02x", src_addr.addr_bytes[j]);
    mac_address = eth_addr;

    return mac_address;
}

void Nic::set_socket(socket_handle * sockh) {
    sockh_ = (*sockh);
}

socket_handle& Nic::get_socket() {
    return sockh_;
}

GpuComm* Nic::get_gpu_comm() {
    return gcomm_.get();
}

void Nic::gpu_comm_init_tx_queues()
{
    queue_manager_.init_gpu_txqs();
}


int Nic::ring_cpu_doorbell(TxRequestGpuPercell* pTxRequestGpuPercell, PreparePRBInfo &prb_info, PacketTimingInfo &packet_timing_info) {
    int ret = 0;

    if (pTxRequestGpuPercell->size == 0)
        return -EINVAL;

    // TODO add error code for GPU Comm
    if(gcomm_->getErrorFlag() != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "FATAL ERROR: {} more packets expected than MAX_PACKETS_PER_SYM {}", __FUNCTION__, MAX_PACKETS_PER_SYM);
        return -1;
    }

    ret = gcomm_->cpu_send(
        pTxRequestGpuPercell,
        prb_info,
        packet_timing_info
    );

    return ret;
}

int Nic::gpu_comm_send_uplane(TxRequestGpuPercell *pTxRequestGpuPercell, PreparePRBInfo &prb_info)
{
    int ret = 0;

    if (pTxRequestGpuPercell->size == 0)
        return -EINVAL;

    // TODO add error code for GPU Comm
    if(gcomm_->getErrorFlag() != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "FATAL ERROR: {} more packets expected than MAX_PACKETS_PER_SYM {}", __FUNCTION__, MAX_PACKETS_PER_SYM);
        return -1;
    }

    ret = gcomm_->send(
        pTxRequestGpuPercell,
        4096, // Update this if we're using jumboframes
        4 + sizeof(oran_umsg_iq_hdr) + sizeof(oran_u_section_uncompressed),
        prb_info
    );

    return ret;
}

int Nic::gpu_comm_set_trigger_ts(uint32_t slot_idx,uint64_t trigger_ts)
{
    int ret = 0;

    gcomm_->setDlSlotTriggerTs(slot_idx,trigger_ts);

    return ret;
}

int Nic::gpu_comm_trigger_cqe_tracer_cb(TxRequestGpuPercell *pTxRequestGpuPercell)
{
    int ret = 0;
    if (pTxRequestGpuPercell->size == 0)
        return -EINVAL;

    gcomm_->traceCqe(pTxRequestGpuPercell);

    return ret;
}

//#define LOG_MAX_DELAY_ENABLED

#ifdef LOG_MAX_DELAY_ENABLED
struct mlx5_fcqs fcqs[48*MLX5_FCQS_N];
#endif

void Nic::log_max_delays()
{
#ifdef LOG_MAX_DELAY_ENABLED // TODO comment out until DPDK issue is resolved
    uint16_t port_id = get_port_id();
    int log_full_queue = 0;

    for (int queue_id = 42; queue_id < 48; queue_id++)
    {
        if (rte_pmd_mlx5_query_fcqs(port_id, queue_id, &fcqs[queue_id*MLX5_FCQS_N]))
        {
            int max_delay = 0;
            for (int i = 0; i < MLX5_FCQS_N; i++)
            {
                struct mlx5_fcqs *cqs = &fcqs[queue_id*MLX5_FCQS_N+i];
                int64_t delay = cqs->done_ts - cqs->fire_ts;
                if (delay > max_delay)
                {
                    max_delay = delay;
                }
            }
            if (max_delay > 30000)
            {
                log_full_queue = 1;
                NVLOGC_FMT(NVLOG_TAG_BASE_FH_DRIVER,"Max delay port {} queue {} : {} ns", port_id, queue_id, max_delay);
            }
        }
    }
    if (log_full_queue)
    {
        for (int queue_id = 42; queue_id < 48; queue_id++)
        {
            for (int i = 0; i < MLX5_FCQS_N; i++)
            {
                struct mlx5_fcqs *cqs = &fcqs[queue_id*MLX5_FCQS_N+i];
                NVLOGI_FMT(NVLOG_TAG_BASE_FH_DRIVER,
                       "queue log ==> port_id={}, queue_id={} :  {:02X} {:04X} {:6d} {:12d} {:12d} {}\n",
                       port_id,
                       queue_id,
                       cqs->wqe_t,
                       rte_be_to_cpu_16(cqs->wqe_i),
                       cqs->fire_ts - cqs->wait_ts,
                       cqs->fire_ts,
                       cqs->done_ts,
                       cqs->done_ts - cqs->fire_ts);
            }
        }
    }
#endif
}


} // namespace aerial_fh
