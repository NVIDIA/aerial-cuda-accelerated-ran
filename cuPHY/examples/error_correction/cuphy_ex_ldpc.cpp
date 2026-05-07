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

#include "cuphy.h"
#include <cstdio>
#include <string>
#include <nvtx3/nvToolsExt.h>
#include "CLI/CLI.hpp"  // CLI11 header
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "ldpc_decode_test_vec_file.hpp"
#include "ldpc_decode_test_vec_gen.hpp"
#include "ldpc/ldpc_api.hpp"

using namespace cuphy;

////////////////////////////////////////////////////////////////////////
// LDPC_decode_error_stats
class LDPC_decode_error_stats
{
public:
    typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> err_count_tensor_t;

    LDPC_decode_error_stats() :
        bit_error_count_(0),
        bit_count_(0),
        block_error_count_(0),
        block_count_(0)
    {
    }
    //------------------------------------------------------------------
    // update()
    template <class TSrc, class TDecoded>
    void update(TSrc& src, TDecoded& decoded)
    {
        typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tensor_uint32_p_t;
        
        const int             B      = src.dimensions()[0];
        const int             NUM_CW = decoded.dimensions()[1];
        cuphy::tensor_device  xor_results(CUPHY_BIT, B, NUM_CW);
        tensor_uint32_p_t     err_count(1, NUM_CW);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Generate a reference to ONLY the information bits of the
        // decoder output, without any possible filler bits.
        // tDecodeB = tDecode(0:B-1, :)
        cuphy::tensor_ref tDecodeB = decoded.subset(cuphy::index_group(cuphy::index_range(0, B),
                                                                       cuphy::dim_all()));
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // XOR decoder output with source bits. Each set bit in the
        // xor_results output indicates a bit error.
        // xor_results = tDecodeB ^ src_bits
        cuphy::tensor_xor(xor_results,
                          tDecodeB,
                          src);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Count the number of set bits in each column (codeword)
        cuphy::tensor_reduction_sum(err_count, xor_results, 0);
        cudaStreamSynchronize(0);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update error statistics
        update_statistics(err_count, B);
    }
    void update_statistics(err_count_tensor_t& tErrorCount,
                           int                 bitsPerCodeword)
    {
        // Input is uint32_t tensor with dimensions (1, NUM_CW)
        int NUM_CW = tErrorCount.dimensions()[1];
        
        for(int i = 0; i < NUM_CW; ++i)
        {
            uint32_t cwBitErrors = tErrorCount(0, i);
            //printf("%i: %u\n", i, err_count(0, i));
            bit_error_count_ += cwBitErrors;
            if(cwBitErrors > 0)
            {
                ++block_error_count_;
            }
        }
        bit_count_   += (bitsPerCodeword * NUM_CW);
        block_count_ += NUM_CW;
    }
    uint64_t bit_error_count()   const { return bit_error_count_;   }
    uint64_t bit_count()         const { return bit_count_;         }
    uint32_t block_error_count() const { return block_error_count_; }
    uint32_t block_count()       const { return block_count_;       }
    float    BER()               const { return static_cast<float>(bit_error_count_)   / bit_count_;   }
    float    BLER()              const { return static_cast<float>(block_error_count_) / block_count_; }
private:
    uint64_t bit_error_count_;
    uint64_t bit_count_;
    uint32_t block_error_count_;
    uint32_t block_count_;
};

////////////////////////////////////////////////////////////////////////
// LDPC_decode_timing_stats
class LDPC_decode_timing_stats
{
public:
    LDPC_decode_timing_stats() :
        run_count_(0),
        total_time_milliseconds_(0.0),
        total_bits_(0.0)
    {
    }
    //------------------------------------------------------------------
    // update()
    void update(float t_milliseconds, int nruns, int64_t bits_per_run)
    {
        run_count_               += nruns;
        total_time_milliseconds_ += t_milliseconds;
        total_bits_              += (bits_per_run * nruns);
        //printf("Average (%u runs) elapsed time in usec = %.1f, throughput = %.2f Gbps\n",
        //       nruns,
        //       t_milliseconds * 1000 / nruns,
        //       (bits_per_run * nruns) / (t_milliseconds / 1000.0f) / 1.0e9);
    }
    int64_t num_runs()          const { return run_count_; }
    float   average_time_usec() const { return (total_time_milliseconds_ * 1000) / run_count_; }
    float   throughput()        const { return (total_bits_ * 1.0e-9) / (total_time_milliseconds_ / 1000.0); }
private:
    int64_t run_count_;
    double  total_time_milliseconds_;
    double  total_bits_;
};

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    bool quiet = false;
    
    cuphyNvlogFmtHelper nvlog_fmt("ldpc_decoder.log");
    
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments using CLI11
        CLI::App app{"LDPC Decoder Example"};
        
        // Command line options
        std::string  inputFilename;
        int          numIterations        = 1;
        float        clampValue           = 32.0f;
        bool         useHalf              = false;
        int          parityNodes          = 8;
        int          algoIndex            = 0;
        bool         compareDecodeOutput  = true;
        unsigned int numRuns              = 10;
        int          numCBLimit           = -1;
        bool         doWarmup             = true;
        float        minSumNorm           = 0.0f;
        int          BG                   = 1;
        bool         puncture             = false;
        int          Zi                   = 384;
        float        SNR                  = 10.0f;
        bool         useTBInterface       = false;
        int          blockSize            = -1;
        float        codeRate             = 0.0f;
        int          modulatedBits        = -1;
        int          log2QAM              = CUPHY_QAM_4;  // Default to QPSK
        int          min_block_err_cnt    = 0;
        int          max_block_cnt        = 1000000;
        bool         chooseHighThroughput = false;
        bool         spreadTB             = false;
        bool         writeSoftOutputs     = false;
        std::string  outputFilename;

        // Create option groups to match usage() organization
        auto execution_common = app.add_option_group("Execution (Common) Options", 
            "---------------------------");
        auto input_data = app.add_option_group("Input Data Options", 
            "-------------------");
        auto file_based = input_data->add_option_group("File Based Input", 
            "When using file based input, no additional puncturing or shortening is performed. BER/BLER\n"
            "will reflect the puncturing/shortening conditions used to generate the input data.\n"
            "The number of input information bits is determined from the 'sourceData' data set in\n"
            "the input file, and the lifting size Z is derived appropriately.");
        auto generating = input_data->add_option_group("Generating Input Data",
            "Ways to specify randomly generated input data:\n"
            "B, N   (input block size and number of modulated bits)\n"
            "B, R   (input block size and code rate)\n"
            "p, Z   (num parity nodes and lifting size)");

        // Execution (Common) Options
        app.add_option("-a", algoIndex, 
            "Use specific implementation [26..40] (default: 0 - let library decide)")
            ->check([](const std::string &str) -> std::string {
                if (const auto val = std::stoi(str); val == 0 || (val >= 26 && val <= 40)) { return {}; }
                return "Value must be either 0 or between 26 and 40, inclusive";
            })
            ->group("Execution (Common) Options");
            
        app.add_flag("-b", useTBInterface, 
            "Use the transport block LDPC interface (instead of the tensor interface)")
            ->group("Execution (Common) Options");
        
        app.add_option("-c", max_block_cnt, 
            "Terminate the data loop when the given number of blocks has been decoded.\n"
            "When the '-e' error count option is provided, this option can be used to avoid\n"
            "an infinite loop, as the provided SNR may not generate any bit or block errors.")
            ->check(CLI::PositiveNumber)
            ->group("Execution (Common) Options");
            
        app.add_flag("-d", spreadTB, 
            "When using the transport block interface, 'spread' the data over multiple\n"
            "transport blocks (" + std::to_string(CUPHY_LDPC_DECODE_DESC_MAX_TB) + "), instead of one large transport block")
            ->group("Execution (Common) Options");
        
        app.add_option("-e", min_block_err_cnt, 
            "Generate data and accumulate error statistics until specified blocks containing\n"
            "an error have occurred. (Not used for file-based input.)")
            ->check(CLI::NonNegativeNumber)
            ->group("Execution (Common) Options");
            
        app.add_flag("-f", useHalf, 
            "Use half precision instead of single precision (Volta and later only)")
            ->group("Execution (Common) Options");
            
        app.add_flag("-k", doWarmup, 
            "Skip 'warmup' run before timing loop")
            ->default_val(true)
            ->group("Execution (Common) Options");
        
        app.add_option("-m", minSumNorm, 
            "Normalization factor for min-sum. If no value is provided, the library\n"
            "will choose an appropriate value, based on the LDPC configuration.")
            ->group("Execution (Common) Options");
        
        app.add_option("-n", numIterations, 
            "Maximum number of LDPC iterations (default: 1)")
            ->check(CLI::PositiveNumber)
            ->group("Execution (Common) Options");

        app.add_option("-C", clampValue, "Clamp value for LLR values (default: 32.0)")
           ->check([](const std::string& str) -> std::string {
               if(const auto val = std::stof(str); val <= 0.0f || val >= 65504.0f)
               {
                   NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid clamp value: {}", str);
                   return "Clamp value must be greater than 0 and less than 65504";
               };
               return std::string();
           })
           ->group("Execution (Common) Options");

        app.add_option("-o", outputFilename, 
            "Write output data to an HDF5 file with the given name.\n"
            "If an output file name is provided and the -u option is used,\n"
            "the soft output LLR values will be placed in the file.")
            ->group("Execution (Common) Options");
            
        app.add_option("-r", numRuns, 
            "Number of times to perform batch decoding (default: 10)")
            ->check(CLI::PositiveNumber)
            ->group("Execution (Common) Options");
            
        app.add_flag("-s", compareDecodeOutput, 
            "Skip comparison of decoder output to input data")
            ->default_val(true)
            ->group("Execution (Common) Options");
            
        app.add_flag("-t", chooseHighThroughput, 
            "Instruct the library algorithm chooser to choose a kernel optimized for\n"
            "throughput (instead of latency) when a high throughput kernel is available.\n"
            "(Only valid when algo_index is 0 or is not specified on the command line.)")
            ->group("Execution (Common) Options");
        
        app.add_flag("-u", writeSoftOutputs, 
            "Write soft output data into a buffer. If an output file name is provided\n"
            "via the -o option, the soft output LLR values will be placed in the file.")
            ->group("Execution (Common) Options");

        app.add_flag("-q,--quiet", quiet,
            "Quiet mode: suppress all output")
            ->group("Execution (Common) Options");

        // File Based Input options
        app.add_option("-i", inputFilename, 
            "Input HDF5 file name, which must contain the following datasets:\n"
            "    sourceData:    uint8 data set with source information bits\n"
            "    inputLLR:      Log-likelihood ratios for coded, modulated symbols\n"
            "    inputCodeWord: uint8 data set with encoded bits (optional)\n"
            "                  (Initial bits are sourceData. No puncturing assumed.)")
            ->group("File Based Input");
            
        app.add_option("-g", BG, 
            "Base graph used to generate input data (default: 1)")
            ->check(CLI::Range(1, 2))
            ->group("File Based Input");
            
        app.add_option("-p", parityNodes, 
            "Number of parity nodes mb (must be between 4 and 46 for BG1, and between\n"
            "4 and 42 for BG2). This value is not used if the code rate 'R' is specified,\n"
            "or if the number of modulated bits 'N' is specified. (default: 8)")
            ->check(CLI::Range(4, 46))
            ->group("File Based Input");
            
        app.add_option("-w", numCBLimit, 
            "For file input: Decode numCBLimit code blocks (instead of the total number contained\n"
            "in the input file). Must be less than or equal to the number of codewords in the file.\n"
            "For generated input: Number of codewords to generate (default: 80)")
            ->check(CLI::PositiveNumber)
            ->group("File Based Input");

        // Generating Input Data options
        app.add_option("-B", blockSize, 
            "Input data block size (before LDPC encoding). Uses the base graph selection\n"
            "to determine the lifting size Z.")
            ->check(CLI::PositiveNumber)
            ->group("Generating Input Data");
            
        app.add_option("-M", "Modulation used before adding noise. Valid values are 'BPSK', 'QPSK',\n"
            "'QAM16', 'QAM64', or 'QAM256'. (default: 'QPSK')")
            ->transform([&log2QAM](const std::string& mod) {
                if(mod == "QAM256") { log2QAM = CUPHY_QAM_256; return std::string("QAM256"); }
                if(mod == "QAM64")  { log2QAM = CUPHY_QAM_64;  return std::string("QAM64"); }
                if(mod == "QAM16")  { log2QAM = CUPHY_QAM_16;  return std::string("QAM16"); }
                if(mod == "QPSK")   { log2QAM = CUPHY_QAM_4;   return std::string("QPSK"); }
                if(mod == "BPSK")   { log2QAM = CUPHY_QAM_2;   return std::string("BPSK"); }
                throw CLI::ValidationError("Invalid modulation");
            })
            ->group("Generating Input Data");
            
        app.add_option("-N", modulatedBits, 
            "Number of modulated bits (info + parity) in each codeword.")
            ->check(CLI::PositiveNumber)
            ->group("Generating Input Data");
            
        app.add_option("-R", codeRate, 
            "Code rate. Used with block size parameter 'B' to determine the number of parity\n"
            "nodes and punctured parity bits. Ignored if '-N' option is used, and instead\n"
            "derived from that value.")
            ->check(CLI::Range(0.0f, 1.0f))
            ->group("Generating Input Data");
            
        app.add_flag("-P", puncture, 
            "Puncture the generated test vector data (default: false)")
            ->group("Generating Input Data");
            
        app.add_option("-S", SNR, 
            "SNR (in dB) for generated noise. The (complex) noise variance is given by\n"
            "10^(-SNR_dB/10). The variance of the real and imaginary components are assumed\n"
            "to be equal, and in this case each is equal to half of the complex variance.\n"
            "(default SNR: 10)")
            ->group("Generating Input Data");
            
        app.add_option("-Z", Zi, 
            "Lifting size for generated data. This option is only used if the data block\n"
            "size is NOT specified. If this option is specified, the number of filler bits\n"
            "is zero, and no parity bits are punctured.")
            ->check(CLI::PositiveNumber)
            ->group("Generating Input Data");

        // Parse command line arguments
        CLI11_PARSE(app, argc, argv);

        //--------------------------------------------------------------
        // Display device (GPU) info
        cuphy::device gpuDevice;
        if(!quiet)
        {
            printf("*********************************************************************\n");
            printf("%s\n", gpuDevice.desc().c_str());
        }
        //--------------------------------------------------------------
        // Create a cuPHY context
        cuphy::context ctx;
        //--------------------------------------------------------------
        // Create a random number generator, in case we need it to
        // generate source input data
        cuphy::rng rng_gen;
        //--------------------------------------------------------------
        // Initialize test data and the LDPC configuration using command
        // line arguments.
        cuphyDataType_t                       LLR_type = useHalf ? CUPHY_R_16F : CUPHY_R_32F;
        std::unique_ptr<ldpc_decode_test_vec> ptv;
        if(!inputFilename.empty())
        {
            // Load a test vector from an input file
            ptv.reset(new ldpc_decode_test_vec_file(test_vec_file_params(inputFilename.c_str(), // input file name
                                                                         LLR_type,              // LLR data type
                                                                         BG,                    // base graph
                                                                         parityNodes,           // num parity nodes
                                                                         numCBLimit)));         // limit num CWs
        }
        else
        {
            // Generate test vector data randomly
            ptv.reset(new ldpc_decode_test_vec_gen(ctx,                                  // cuPHY context
                                                   rng_gen,                              // random number generator
                                                   test_vec_gen_params(LLR_type,         // LLR data type
                                                                       BG,               // base graph
                                                                       Zi,               // lifting size
                                                                       parityNodes,      // num parity nodes
                                                                       numCBLimit,       // number of codewords
                                                                       blockSize,        // number of input bits
                                                                       codeRate,         // code rate
                                                                       modulatedBits,    // number of modulated bits
                                                                       log2QAM,          // modulation
                                                                       SNR,              // signal-to-noise ratio
                                                                       puncture)));      // puncture first 2Z LLRs
        }
        //------------------------------------------------------------------
        // Display LDPC test vector configuration info
        ldpc_decode_test_vec&              tv  = *ptv;
        const ldpc_decode_test_vec_config& tv_cfg = tv.config();
        if(!quiet)
        {
            tv.print_config();
        }

        //------------------------------------------------------------------
        // Allocate an output buffer for decoded bits
        tensor_device tDecode(CUPHY_BIT,
                              tv.config().K, //MAX_DECODED_CODE_BLOCK_BIT_SIZE,
                              tv.config().num_cw,
                              cuphy::tensor_flags::align_coalesce);
        //------------------------------------------------------------------
        // Create a tensor descriptor for soft outputs
        tensor_device tSoftOutputs;
        if(writeSoftOutputs)
        {
            tSoftOutputs = tensor_device(CUPHY_R_16F,
                                         tv.config().K,
                                         tv.config().num_cw);
        }

        //printf("Decode: addr: %p, %s, size: %.1f kB\n\n",
        //       tDecode.addr(),
        //       tDecode.desc().get_info().to_string().c_str(),
        //       tDecode.desc().get_size_in_bytes() / 1024.0);
        //--------------------------------------------------------------
        // Create an LDPC decoder instance
        cuphy::LDPC_decoder dec(ctx);
        //--------------------------------------------------------------
        // Initialize an LDPC decode configuration. This is used for
        // both the tensor and transport block interfaces.
        uint32_t decode_flags = chooseHighThroughput ? CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT : 0;
        cuphy::LDPC_decode_config dec_cfg(LLR_type,      // LLR type (fp16 or fp32)
                                          tv_cfg.mb,     // num parity nodes
                                          tv_cfg.Z,      // lifting size
                                          numIterations, // max num iterations
                                          clampValue,    // clamp value
                                          tv_cfg.Kb,     // info nodes
                                          minSumNorm,    // normalization value
                                          decode_flags,  // flags
                                          tv_cfg.BG,     // base graph
                                          algoIndex,     // algorithm index
                                          nullptr);      // workspace address
        //--------------------------------------------------------------
        // If no normalization value was provided, query the library for
        // an appropriate value.
        if(minSumNorm <= 0.0f)
        {
            dec.set_normalization(dec_cfg);
        }
        if(!quiet)
        {
            printf("Normalization                    = %f\n", dec_cfg.get_norm());
            printf("Number of iterations             = %i\n", numIterations);
            printf("\n");
        }
        //--------------------------------------------------------------
        // Initialize an LDPC decode descriptor structure. (This is only
        // used when the transport block interface is selected.)
        LDPC_decode_desc dec_desc(dec_cfg,CUPHY_LDPC_DECODE_DESC_MAX_TB);
        if(useTBInterface)
        {
            if(spreadTB)
            {
                // Spread the codewords out into multiple transport blocks,
                // with addresses that point back to the original input
                // tensor.
                const int         CW_PER_TB = (tv_cfg.num_cw + (CUPHY_LDPC_DECODE_DESC_MAX_TB - 1)) /
                                              CUPHY_LDPC_DECODE_DESC_MAX_TB;
                cuphy::tensor_ref tLLR(tv.LLR_desc(), tv.LLR_addr());
                for(int iCW = 0; iCW < tv_cfg.num_cw; iCW += CW_PER_TB)
                {
                    cuphy::index_group slice(cuphy::dim_all(),
                                             cuphy::index_range(iCW, std::min(iCW + CW_PER_TB, tv_cfg.num_cw)));
                    cuphy::tensor_ref  sLLR    = tLLR.subset(slice);
                    cuphy::tensor_ref  sDecode = tDecode.subset(slice);
                    //printf("start = %i, end = %i\n", slice.ranges()[1].start(), slice.ranges()[1].end());
                    if(writeSoftOutputs)
                    {
                        cuphy::tensor_ref  sSoftOutputs = tSoftOutputs.subset(slice);
                        dec_desc.add_tensor_as_tb(sLLR.desc(),         sLLR.addr(),
                                                  sDecode.desc(),      sDecode.addr(),
                                                  sSoftOutputs.desc(), sSoftOutputs.addr());
                    }
                    else
                    {
                        dec_desc.add_tensor_as_tb(sLLR.desc(),    sLLR.addr(),
                                                  sDecode.desc(), sDecode.addr());
                    }
                }
            }
            else
            {
                if(writeSoftOutputs)
                {
                    dec_desc.add_tensor_as_tb(tv.LLR_desc(),
                                              tv.LLR_addr(),
                                              tDecode.desc(),
                                              tDecode.addr(),
                                              tSoftOutputs.desc(),
                                              tSoftOutputs.addr());
                }
                else
                {
                    dec_desc.add_tensor_as_tb(tv.LLR_desc(),
                                              tv.LLR_addr(),
                                              tDecode.desc(),
                                              tDecode.addr());
                }
            }
        }
        //--------------------------------------------------------------
        // Initialize an LDPC decode tensor params structure. (This is
        // only used when the tensor-based decoder interface is selected.)
        LDPC_decode_tensor_params dec_tensor(dec_cfg,                 // LDPC configuration
                                             tDecode.desc().handle(), // output descriptor
                                             tDecode.addr(),          // output address
                                             tv.LLR_desc().handle(),  // LLR descriptor
                                             tv.LLR_addr(),           // LLR address
                                             writeSoftOutputs ? tSoftOutputs.desc().handle() : nullptr, // Soft output descriptor (optional)
                                             writeSoftOutputs ? tSoftOutputs.addr() : nullptr);         // Soft output address (optional)
        //--------------------------------------------------------------
        // Decoder execution loop
        LDPC_decode_error_stats  error_stats;
        LDPC_decode_timing_stats timing_stats;
        do
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Generate test vector data
            tv.generate();
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Warmup run
            if(doWarmup)
            {
                if(useTBInterface)
                {
                    dec.decode(dec_desc);
                }
                else
                {
                    dec.decode(dec_tensor);
                }
            }
            cudaDeviceSynchronize();
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Timed run
            nvtxRangePush("LDPC_decode_timed");
            cuphy::event_timer tmr;

            tmr.record_begin();
            for(unsigned int uRun = 0; uRun < numRuns; ++uRun)
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Decode
                if(useTBInterface)
                {
                    dec.decode(dec_desc);
                }
                else
                {
                    dec.decode(dec_tensor);
                }
            }
            tmr.record_end();
            tmr.synchronize();
            nvtxRangePop();
            timing_stats.update(tmr.elapsed_time_ms(), numRuns, tv_cfg.B * tv_cfg.num_cw);
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Compare decoder output to source bits
            if(compareDecodeOutput)
            {
                error_stats.update(tv.src_bits(), tDecode);
            }
        } while((error_stats.block_count()       < max_block_cnt)     &&
                (error_stats.block_error_count() < min_block_err_cnt));
        //--------------------------------------------------------------
        // Optional: export to HDF5
        if(!outputFilename.empty())
        {
            if((outputFilename.length() < 3) ||
               (0 != strcmp(outputFilename.c_str() +  outputFilename.length() - 3, ".h5")))
            {
                outputFilename.append(".h5");
            }
            hdf5hpp::hdf5_file f = hdf5hpp::hdf5_file::create(outputFilename.c_str());
            tv.export_hdf5(f);
            // Also write out the soft output data. We will convert to FP32
            // for convenience in reading.
            if(writeSoftOutputs)
            {
                cuphy::tensor_device tSoftOutput_f32(CUPHY_R_32F, tSoftOutputs.layout());
                cuphy::tensor_convert(tSoftOutput_f32, tSoftOutputs);
                cuphy::write_HDF5_dataset(f, tSoftOutput_f32, "outputLLR");
            }
        }
        //--------------------------------------------------------------
        // Display aggregated timing and error statistics
        if(!quiet)
        {
            printf("Average (%li runs) elapsed time in usec = %.1f, throughput = %.2f Gbps\n",
                   timing_stats.num_runs(),
                   timing_stats.average_time_usec(),
                   timing_stats.throughput());

            if(compareDecodeOutput)
            {
                printf("bit error count = %lu, bit error rate (BER) = (%lu / %lu) = %.5e, block error rate (BLER) = (%u / %u) = %.5e\n",
                       error_stats.bit_error_count(),
                       error_stats.bit_error_count(),
                       error_stats.bit_count(),
                       error_stats.BER(),
                       error_stats.block_error_count(),
                       error_stats.block_count(),
                       error_stats.BLER());
            }
        }
    }

    catch(std::exception& e)
    {
        if(!quiet)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        }
        returnValue = 1;
    }
    catch(...)
    {
        if(!quiet)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        }
        returnValue = 2;
    }
    return returnValue;
}
