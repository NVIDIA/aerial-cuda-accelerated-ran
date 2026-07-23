#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

script_name=run_l2sa.sh
function usage {
    echo "Usage: $script_name <duration> <test_group> <test_case> [--channels <channels>]"
    echo "    duration:    The duration to checking status before exit. Unit: second"
    echo "    test_group:  F08, F13, nrSim"
    echo "    test_case:   For F08/F13:  1C, 2C, 3C, 4C"
    echo "                 For nrSim:    Case number, like 1001, 2001, ..."
    echo "Example: ./$script_name 20 nrSim 6001 --channels PUCCH"
}

if [ "$3" = "" ]; then
    usage
    exit 1
fi

if [ "$cuBB_SDK" = "" ]; then
    echo "Please set cuBB_SDK first"
    exit 1
fi

LOCAL_DIR=$(dirname $(readlink -f "$0"))

echo "cuBB_SDK=$cuBB_SDK"

if [ "$BUILD" = "" ]; then
    BUILD="build"
fi

if [[ -d $cuBB_SDK/build-$(arch) ]]; then
    BUILD=build-$(arch)
fi

duration=$1
test_group=$2
test_case=$3

if [ "$test_group" == "nrSim" ]; then
    test_case=$(echo $test_case | sed 's/^0*//')
    printf -v test_case "%04d" $test_case
    if [[ "${test_case}" == "1"* ]]; then
        channels="PBCH";
    elif [[ "${test_case}" == "2"* ]]; then
        channels="PDCCH_DL";
    elif [[ "${test_case}" == "3"* ]]; then
        channels="PDSCH";
    elif [[ "${test_case}" == "4"* ]]; then
        channels="CSI_RS";
    elif [[ "${test_case}" == "5"* ]]; then
        channels="PRACH";
    elif [[ "${test_case}" == "6"* ]]; then
        channels="PUCCH";
    elif [[ "${test_case}" == "7"* ]]; then
        channels="PUSCH";
    fi
fi

if [ "$5" != "" ]; then
    channels=$5
fi

if [ "$LOG_PREFIX" = "" ]; then
    log_prefix=$(date -u "+%Y%m%d_%H%M%S")
else
    log_prefix=${LOG_PREFIX}
fi

log_folder=${log_prefix}_${test_group}_${test_case}

if [ ! -z "${channels}" ]; then
    channel_option=" --channels $channels"
    log_folder=${log_folder}_${channels}
else
    channel_option=""
fi

if [ "$BFP" != "" ]; then
    log_folder=${log_folder}_BFP${BFP}
fi

log_folder=${log_folder}_${duration}s

echo "Test case: $duration seconds ${test_group} ${test_case} $channels BFP=$BFP"

if [ `whoami` = "root" ];then
    USE_SUDO=""
else
    USE_SUDO="sudo -E"
fi

if [ "$LOG_PATH" = "" ]; then
    export LOG_PATH=$cuBB_SDK/logs/latest
fi
echo "LOG_PATH=$LOG_PATH"
if [ ! -d $LOG_PATH ]; then
    mkdir -p $LOG_PATH
fi

# chmod to fix NFS storage permission issue
chmod 777 $LOG_PATH
cd $LOG_PATH

function kill_all {
    signal=$1
    if [ "$signal" = "" ]; then
        signal=SIGKILL
    fi

    echo "$(date +%T) start killing ... signal=$signal"
    start_time=$(date +%s)
    sudo killall -9 -q phy_main l2_adapter_cuphycontroller_scf
    sudo killall -9 -q mac_main test_mac
    interval=$(($(date +%s) - $start_time))
    echo "$(date +%T) killed all in ${interval} seconds"
}

# Kill previous running processes if exist
kill_all SIGKILL

if [ "$DOCA" = "" ]; then
    export DOCA=1
fi

# Debian multiarch directory name, e.g. x86_64-linux-gnu or aarch64-linux-gnu
LIB_ARCH=$(arch)-linux-gnu

if [ "$DOCA" = "1" ]; then
    DPDK_LINK_PATH=/opt/mellanox/doca/lib/$LIB_ARCH:/opt/mellanox/dpdk/lib/$LIB_ARCH
else
    DPDK_LINK_PATH=$cuBB_SDK/gpu-dpdk/$BUILD/install/lib/$LIB_ARCH
fi


LOCAL_PHY_CMD="$cuBB_SDK/$BUILD/cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/l2_adapter_cuphycontroller_scf"

if [ "$TIMEOUT_BASE" = "" ]; then
    TIMEOUT_BASE=300
fi
N_SEC=$(($TIMEOUT_BASE + $duration))
LOCAL_PHY_CMD="cd $LOG_PATH && $USE_SUDO LD_LIBRARY_PATH=${DPDK_LINK_PATH} timeout -s 9 $N_SEC $LOCAL_PHY_CMD"

if [ "$NVIPC_DEBUG_EN" != "" ]; then
    LOCAL_PHY_CMD="export NVIPC_DEBUG_EN=$NVIPC_DEBUG_EN && $LOCAL_PHY_CMD"
fi

LOCAL_RU_CMD="export cuBB_SDK=${cuBB_SDK} && cd $LOG_PATH && $USE_SUDO LD_LIBRARY_PATH=${DPDK_LINK_PATH} $cuBB_SDK/$BUILD/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator ${test_group} ${test_case}${channel_option}"
LOCAL_MAC_CMD="export cuBB_SDK=${cuBB_SDK} && cd $LOG_PATH && $USE_SUDO LD_LIBRARY_PATH=${DPDK_LINK_PATH} $cuBB_SDK/$BUILD/cuPHY-CP/testMAC/testMAC/test_mac ${test_group} ${test_case}${channel_option}"
LOCAL_CMD_PHY_LOG="cp /tmp/l2sa.log /tmp/testmac.log $LOG_PATH"
LOCAL_CMD_PCAP_LOG="export cuBB_SDK=${cuBB_SDK} && cd $LOG_PATH && $USE_SUDO $cuBB_SDK/$BUILD/cuPHY-CP/gt_common_libs/nvIPC/tests/pcap/pcap_collect nvipc ."

# Disable test_mac FAPI validation for L2SA test
LOCAL_MAC_CMD="${LOCAL_MAC_CMD} --no-validation"


# Clean old logs
rm -rf $LOG_PATH/*

is_screen_running() {
    title=$1
    check=$(screen -ls | grep "${title}")
    if [ "$check" != "" ]; then
        echo 1
    else
        echo 0
    fi
}

moniter_all_running() {
    titles=$1
    pids=$2
    ret=0
    echo "monitor: watching screens [$titles] and process pids [$pids]"
    while :
    do
        sleep 1

        for title in $titles; do
            if [ $(is_screen_running $title) -eq 0 ]; then
	            echo "monitor: $title had exited"
	            ret=1
	            break
	        fi
        done

        for pid in $pids; do
            running=$(ps -o pid= -p $pid)
            if [ "$running" = "" ]; then
	            # echo "monitor: process pid=$pid had exited"
	            ret=2
	            break
	        fi
        done

        if [ $ret -ne 0 ]; then
            break
        fi
    done
    return $ret
}

run_in_screen() {
    title=$1
    cmd=$2
    debug=$3
    if [ "$debug" != "" ]; then
        echo "screen[$title]: $cmd"
    fi
    screen -L -t $title -dmS $title bash -c "$cmd";
}

run_in_screen l2sa "$LOCAL_PHY_CMD" 1
sleep 2
run_in_screen mac "$LOCAL_MAC_CMD" 1

# Check running screens
screen -ls

if [ "$CHECK_RESULT_SCRIPT_PATH" = "" ]; then
    CHECK_RESULT_SCRIPT_PATH=$LOCAL_DIR
fi
export L2SA_TEST=1
$CHECK_RESULT_SCRIPT_PATH/check_result.py ${duration} & checker_pid=$!
echo "check_result.py running in background: pid=$checker_pid"

# Handler Ctrl + C to kill the background check_result.py
clean_up() {
    echo "Single received, kill $checker_pid and exit"
    kill -9 $checker_pid
    exit 1
}
trap clean_up SIGINT

moniter_all_running "l2sa mac" "$checker_pid"
ret=$?
if [ $ret -eq 1 ]; then
    echo "monitor: unexpected exit - kill check_result.py pid=$checker_pid"
    sleep 3
    kill $checker_pid
    echo "Test FAILED"
fi

wait $checker_pid
test_result=$?

# Kill after test
kill_all SIGINT

# Collect logs
bash -c "$LOCAL_CMD_PHY_LOG" > $LOG_PATH/nvlog_collect.log 2>&1
if [[ "$NVIPC_DEBUG_EN" != "" && "$NVIPC_DEBUG_EN" != "0" ]]; then
    echo "$LOCAL_CMD_PCAP_LOG"
    bash -c "$LOCAL_CMD_PCAP_LOG" | tee -a $LOG_PATH/nvlog_collect.log
fi

# $USE_SUDO chown $(whoami):$(whoami) ./*

if [ "${QA_TEST}" != "" ]; then
    exit $test_result
fi

echo "copy logs ..."
cd ${LOG_PATH}/..
cp -r ${LOG_PATH} ${log_folder}
error=$(grep " E " ${LOG_PATH}/*.log)
if [ $test_result -ne 0 ]; then
    mv ${log_folder} ${log_folder}_FAIL
elif [[ -n $error ]]; then
    mv ${log_folder} ${log_folder}_ERR
    test_result=1
else
    mv ${log_folder} ${log_folder}_PASS
fi

exit $test_result
