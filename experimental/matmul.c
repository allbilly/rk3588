       alu_case_matmul: { // matmul
          MatmulParams params = matmul_params;
          if (params.align_in <= 0 || params.align_out <= 0 || params.out_width <= 0 ||
              params.out_width_stride <= 0 || params.align_out_atomic <= 0 ||
              params.M <= 0 || params.N <= 0 || params.K <= 0) {
             params = make_matmul_params(params.M, params.N, params.K);
          }
          int dataout_width = params.out_width > 0 ? params.out_width : 1;
          int dataout_height = params.M > 0 ? params.M : 1;
          int data_in_width = dataout_width;
          int data_in_height = dataout_height;
          int align_in = params.align_in > 0 ? params.align_in : 32;
          int align_out = params.align_out > 0 ? params.align_out : 32;
          int out_width_stride = params.out_width_stride > 0 ? params.out_width_stride : dataout_width;
          const bool is_KN_64 = ( params.K == 64 && params.N == 64);
          const bool is_matmul_64 = (params.M == 64 && params.K == 64 && params.N == 64);
          const bool is_KN_256 = (params.K == 256 && params.N == 256);
          const bool is_KN_512 = (params.K == 512 && params.N == 512);
          const bool is_KN_lg_512 = (params.K > 512 && params.N > 512);
          const bool is_matmul_256 = (params.M == 256 && params.K == 256 && params.N == 256);
          const bool is_matmul_768 = (params.M == 1 && params.K == 768 && params.N == 768) ;
          const bool is_matmul_768_2048 = (params.M == 1 && params.K == 768 && params.N == 2048 ) ;
          const bool is_matmul_2048 = (params.M == 1 && params.K == 2048 && params.N == 2048 ) ;

          EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
          uint32_t conv_con1 = CNA_CONV_CON1_PROC_PRECISION(2) | CNA_CONV_CON1_IN_PRECISION(2);
          if (!is_KN_64 && !is_KN_256 && !is_KN_512 && !is_KN_lg_512 && !is_matmul_768 && !is_matmul_768_2048 && !is_matmul_2048) 
             conv_con1 |= CNA_CONV_CON1_GROUP_LINE_OFF(1);
          EMIT(REG_CNA_CONV_CON1, conv_con1);
          // int feature_grains = data_in_height + 1;
          // if (params.M > 128 && params.M <= 192) feature_grains = data_in_height;
          // if (params.M > 192 && params.M <= 224) feature_grains = 148;
          // if (params.M > 224 && params.M < 256) feature_grains = 128;
          // if (params.M > 256 && params.M <= 288) feature_grains = 114;
          // if (params.M > 288 && params.M <= 320) feature_grains = 104;
          // if (params.M > 320 && params.M <= 352) feature_grains = 94;
          // if (params.M > 352 && params.M <= 384) feature_grains = 86;
          // if (params.M > 384 && params.M < 512) feature_grains = 80;
          int feature_grains = data_in_height + 1;
          if (params.K > 7872) {
             feature_grains = 2 ;
          } else if (params.K > 128 && params.K <= 192) {
             feature_grains = data_in_height;
          } else if (params.K > 192 && params.K != 256) {
             uint32_t denom = (uint32_t)align_in * (uint32_t)sizeof(__fp16);
             uint32_t grains = (2u * NPU_CBUF_BANK_SIZE + denom - 1) / denom; // ~2 banks
             grains = (grains + 1u) & ~1u; // round up to even
             if (grains < 80u) grains = 80u;
             feature_grains = (int)grains;
          }
          EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(feature_grains));
          EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_Y_STRIDE(1) | CNA_CONV_CON3_CONV_X_STRIDE(1));
          EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH((uint32_t)data_in_width) | CNA_DATA_SIZE0_DATAIN_HEIGHT((uint32_t)data_in_height));
          EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL((uint32_t)align_in - 1) | CNA_DATA_SIZE1_DATAIN_CHANNEL((uint32_t)align_in));
          EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH((uint32_t)dataout_width));
          EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS((uint32_t)dataout_width * dataout_height));

          uint32_t weight_bytes_per_kernel = (uint32_t)align_in * (uint32_t)sizeof(__fp16);
          EMIT(REG_CNA_WEIGHT_SIZE0, weight_bytes_per_kernel * align_out);
          EMIT(REG_CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(weight_bytes_per_kernel));
          EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(1) | CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(1) | CNA_WEIGHT_SIZE2_WEIGHT_KERNELS((uint32_t)align_out));
          
          // uint32_t fd_bytes = data_in_width * data_in_height * align_in * sizeof(__fp16);
          // uint32_t data_bank = (fd_bytes / NPU_CBUF_BANK_SIZE);
          // data_bank += (uint32_t)(data_bank == 0) ;
          // if (params.M > 128 && params.M <= 170) data_bank = 2;
          // if (params.M > 170 && params.M <= 219) data_bank = 3;
          // if (params.M > 219 && params.M < 256) data_bank = 4;
          // if (params.M > 256 && params.M < 284) data_bank = 5;
          // if (params.M > 284 && params.M < 307) data_bank = 6;
          // if (params.M > 307 && params.M < 325) data_bank = 7;
          // if (params.M > 325 && params.M < 512) data_bank = 8;
          uint64_t fd_bytes = (uint64_t)data_in_width * data_in_height * align_in * sizeof(__fp16);
          uint32_t data_bank = (uint32_t)((fd_bytes + NPU_CBUF_BANK_SIZE - 1) / NPU_CBUF_BANK_SIZE);
          if (data_bank == 0) data_bank = 1;
          if (data_bank > NPU_CBUF_BANKS - 1) data_bank = NPU_CBUF_BANKS - 1;

          EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(NPU_CBUF_BANKS - data_bank) | CNA_CBUF_CON0_DATA_BANK(data_bank));
          EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES( (uint32_t)((data_in_width * align_in + 31)/32) ));
          EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) | CNA_CVT_CON0_CVT_BYPASS(1));
          EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
          EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
          EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
          EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
          EMIT(REG_CNA_FEATURE_DATA_ADDR, CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR(input_dma));
          EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));

          // uint32_t line_stride = (uint32_t)data_in_width * 4u;
          // if (params.M > 32 && params.M < 64) line_stride = 8;
          // else if (params.M > 64 && params.M <= 96) line_stride = 12;
          // else if (params.M > 96 && params.M <= 128) line_stride = 16;
          // else if (params.M > 128 && params.M <= 160) line_stride = 20;
          // else if (params.M > 160 && params.M <= 192) line_stride = 24;
          // else if (params.M > 192 && params.M <= 224) line_stride = 28;
          // else if (params.M > 224 && params.M < 256) line_stride = 32;
          // else if (params.M > 256 && params.M <= 288) line_stride = 36;
          // else if (params.M > 288 && params.M <= 320) line_stride = 40;
          // else if (params.M > 320 && params.M <= 352) line_stride = 44;
          // else if (params.M > 352 && params.M < 512) line_stride = 48;
          uint32_t line_stride = (uint32_t)data_in_width * 4u;
          if (params.K > 32 && params.K < 512 && params.K != 64 && params.K != 256) {
             uint32_t stride_steps = ((uint32_t)params.K + 31u) / 32u;
             if (stride_steps > 13u) stride_steps = 13u;
             line_stride = stride_steps * 4u;
          }

          int32_t surf_groups = data_in_height / 4;
          int32_t surf_stride_signed = (int32_t)line_stride * (surf_groups - 1) + (surf_groups == 0);
          uint32_t surf_stride = (uint32_t)(surf_stride_signed * (int32_t)(align_in >= 64));
          if (params.K > 32 && params.K < 64) surf_stride = 0 ;
          else if (params.K > 64 && params.K <= 128) surf_stride = 0 ;
          else if (params.K > 128 && params.K < 256) surf_stride = 0 ;
          else if (params.K > 256 && params.K < 512) surf_stride = 0 ;
          EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(line_stride));
          EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(surf_stride));

          EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH((uint32_t)data_in_width) | CNA_FC_DATA_SIZE0_DMA_HEIGHT((uint32_t)data_in_height));
          EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL((uint32_t)align_in));
          // We place regcmds at the start of the weights buffer; actual weights start after REGCMD_RESERVED.
          EMIT(REG_CNA_DCOMP_ADDR0, CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0(weights_dma + REGCMD_RESERVED));
          EMIT(REG_CORE_MISC_CFG, CORE_MISC_CFG_PROC_PRECISION(2) | CORE_MISC_CFG_QD_EN(1));
          EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT((uint32_t)(dataout_height - 1)) | CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH((uint32_t)(dataout_width - 1)));
          EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL((uint32_t)align_out - 1));
          emit_raw(&regs, CORE | 0x1, 0x3030, 0);

          EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2));
          EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(5) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
          EMIT(REG_DPU_DST_BASE_ADDR, DPU_DST_BASE_ADDR_DST_BASE_ADDR(output_dma));

          uint32_t dst_surf_stride = is_matmul_64 ? 64u : (is_matmul_256 ? 256u : (uint32_t)out_width_stride);
          EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(dst_surf_stride));
          EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH((uint32_t)(dataout_width - 1)));
          EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT((uint32_t)(dataout_height - 1)));

          // uint32_t notch_val = (is_matmul_64 || is_matmul_256) ? 0u : 7u;
          // if (params.M > 32 && params.M < 64) notch_val = 15 ;
          // else if (params.M > 64 && params.M <= 96) notch_val = 23 ;
          // else if (params.M > 96 && params.M <= 128) notch_val = 31;
          // else if (params.M > 128 && params.M <= 160) notch_val = 39;
          // else if (params.M > 160 && params.M <= 192) notch_val = 47;
          // else if (params.M > 192 && params.M <= 224) notch_val = 55;
          // else if (params.M > 224 && params.M < 256) notch_val = 63;
          // else if (params.M > 256 && params.M <= 288) notch_val = 71;
          // else if (params.M > 288 && params.M <= 320) notch_val = 79;
          // else if (params.M > 320 && params.M <= 352) notch_val = 87;
          // else if (params.M > 352 && params.M < 512) notch_val = 95;
          // uint32_t notch_val = (is_KN_64 || is_KN_256 || is_KN_512 || params.K > 7872) ? 0u : 7u;
          // if (params.K > 32 && params.K < 512 && params.K != 64 && params.K != 256) {
          //    uint32_t notch_steps = ((uint32_t)params.K - 1u) / 32u;
          //    if (notch_steps > 12u) notch_steps = 12u;
          //    notch_val = 7u + 8u * notch_steps;
          // }
          // if (params.M == 33 && params.K == 1 && params.N == 33 ) notch_val = 15;
          
          uint32_t notch_blocks = (uint32_t)align_out / 32u; /* align_out is already 32-aligned */
          if (notch_blocks > 13u) notch_blocks = 13u;
          uint32_t notch_val = 8u * notch_blocks - 1u;
          if (is_KN_64 || is_KN_256 || is_KN_512 || params.K > 7872) notch_val = 0u;
          EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1(notch_val) |DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0(notch_val));
          
          EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL((uint32_t)align_out - 1) | DPU_DATA_CUBE_CHANNEL_CHANNEL((uint32_t)align_out - 1));
          EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
          EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(3) | DPU_BS_OW_CFG_SIZE_E_1(3) | DPU_BS_OW_CFG_SIZE_E_0(3) | DPU_BS_OW_CFG_OD_BYPASS(1));
          EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA((uint32_t)align_out - 1));
          EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA((uint32_t)(dataout_height - 1)) | DPU_WDMA_SIZE_1_WIDTH_WDMA((uint32_t)(dataout_width - 1)));
          EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
          EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
          // EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
          // EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
          EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(dst_surf_stride * 4u));
          // emit_raw(&regs, 0x0 | 0x1, 0x40c4, 0);
          emit_raw(&regs, 0x81, REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
          goto alu_case_done;
       }
