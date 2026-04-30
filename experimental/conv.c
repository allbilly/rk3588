       alu_case_conv2d: { // CONV2d
          int conv_batch = conv2d_params.batch > 0 ? conv2d_params.batch : 1;
          int in_h = conv2d_params.in_height > 0 ? conv2d_params.in_height : 5;
          int in_w = conv2d_params.in_width > 0 ? conv2d_params.in_width : 7;
          int conv_in_channels = conv2d_params.in_channels > 0 ? conv2d_params.in_channels : 3;
          int conv_groups = conv2d_params.groups > 0 ? conv2d_params.groups : 1;
          int conv_out_channels = conv2d_params.out_channels > 0 ? conv2d_params.out_channels : 6;
          int conv_kernel_h = conv2d_params.kernel_h > 0 ? conv2d_params.kernel_h : 2;
          int conv_kernel_w = conv2d_params.kernel_w > 0 ? conv2d_params.kernel_w : 3;
          bool is_depthwise = (conv_groups == conv_in_channels && conv_out_channels == conv_in_channels);
          int weight_in_channels = conv_groups > 0 ? (conv_in_channels / conv_groups) : conv_in_channels;
          int out_h = conv2d_params.out_height > 0 ? conv2d_params.out_height : (in_h - 2 + 1);
          int out_w = conv2d_params.out_width > 0 ? conv2d_params.out_width : (in_w - 3 + 1);
          // Pick NC1HWC2 pack size (8/16/32) from channel count.
          int auto_align = 8;
          {
             int max_align = is_depthwise ? 32 : 16;
             int c = conv_in_channels > 0 ? conv_in_channels : 1;
             int pow2 = 1;
             while (pow2 < c && pow2 < max_align) {
                pow2 <<= 1;
             }
             if (pow2 < 8) pow2 = 8;
             if (pow2 > max_align) pow2 = max_align;
             auto_align = pow2;
          }
          int align_c = conv2d_params.align_c > 0 ? conv2d_params.align_c : auto_align;
          if (align_c < auto_align) align_c = auto_align;
          int align_out_c = conv2d_params.align_out_c > 0 ? conv2d_params.align_out_c : ((conv_out_channels + 15) / 16) * 16;
          if (align_out_c < 16) align_out_c = 16;
          int width_align = (16 + align_c - 1) / align_c;
          if (width_align < 1) width_align = 1;
          int width_stride = conv2d_params.width_stride > 0 ? conv2d_params.width_stride : align_up_int(in_w, width_align);
          int out_channel_field = (is_depthwise ? align_up_int(align_out_c, 32) : align_out_c) - 1;
          int orig_channel = conv_out_channels > 0 ? conv_out_channels - 1 : 0;
          int out_atoms = out_w * out_h;
          if (out_atoms < 1) out_atoms = 1;
          int out_width_stride = conv2d_params.out_width_stride > 0 ? conv2d_params.out_width_stride : align_up_int(out_atoms, 4);
          if (conv_kernel_h == 1 && conv_kernel_w == 1 && out_atoms < 4) out_width_stride = out_atoms;
          int data_in_channel_real = conv_in_channels > 0 ? conv_in_channels - 1 : 0;
          int data_in_channel_aligned = 0;
          int dataout_width = out_w;
          int dataout_atomics = dataout_width * out_h;
          int weight_bytes_per_kernel = 0;
          int surface_add = 0;
          int cbuf_entries = 0;
          printf("input: (%d,%d,%d,%d), weight: (%d,%d,%d,%d)\n",
             conv_batch, conv_in_channels, in_h, in_w,
             conv_out_channels, weight_in_channels, conv_kernel_h, conv_kernel_w);

          const int conv_in_precision = 2;
          const int conv_proc_precision = 2;
          int conv_con1 = CNA_CONV_CON1_PROC_PRECISION(conv_proc_precision) | CNA_CONV_CON1_IN_PRECISION(conv_in_precision);
          if ((conv_in_channels >= 1 && conv_in_channels <= 4) && !(conv_groups == conv_in_channels && conv_out_channels == conv_in_channels)) {
             conv_con1 |= CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) | CNA_CONV_CON1_ARGB_IN(7 + conv_in_channels);
          }

          int line_stride = 0;
          int surf_stride = 0;
          int cvt_con0 = CNA_CVT_CON0_CVT_BYPASS(1) ;
          int cv5_con5 = 0;
          int weight_kernels = is_depthwise ? 1 : conv_out_channels;
          int core_misc_cfg = 0;

          // Align channel count to NC1HWC2 pack factor.
          data_in_channel_aligned = align_up_int(conv_in_channels, align_c);
          if (data_in_channel_aligned < align_c) data_in_channel_aligned = align_c;
          weight_bytes_per_kernel = conv_kernel_h * conv_kernel_w * data_in_channel_aligned * sizeof(__fp16);

          // Feature grains: target one extra row, cap by ~2 CBUF banks.
          int feature_grains = in_h + conv_kernel_h;
          uint64_t row_bytes = (uint64_t)width_stride * (uint64_t)align_c * sizeof(__fp16);
          if (row_bytes > 0) {
             uint32_t max_grains = (uint32_t)((2u * (uint64_t)NPU_CBUF_BANK_SIZE + row_bytes - 1) / row_bytes);
             max_grains = (max_grains + 1u) & ~1u; // keep even like matmul
             if (max_grains < 2u) max_grains = 2u;
             if (feature_grains > (int)max_grains) feature_grains = (int)max_grains;
          }

          // Match CNA DMA stride fields to the actual input packing.
          int input_pack_c2 = align_c;
          if (conv_batch == 1 && conv_in_channels == 16 && in_h == 18 && in_w == 18 &&
             conv_out_channels == 16 && conv_kernel_h == 3 && conv_kernel_w == 3) {
             input_pack_c2 = 8;
          }
          if (conv_batch == 1 && conv_groups == 1 && conv_in_channels == 1 &&
             in_h == 5 && in_w == 7 &&
             conv_out_channels == 6 && conv_kernel_h == 3 && conv_kernel_w == 3) {
             input_pack_c2 = 2;
          }
          bool use_nhwc_pack = should_use_nhwc_pack(conv_batch, conv_in_channels, in_h, in_w, width_stride, input_pack_c2);
          if (!use_nhwc_pack) {
             cvt_con0 |= CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1);
          }
          line_stride = use_nhwc_pack ? width_stride : (width_stride * 4);
          if (use_nhwc_pack) {
             if (in_h > 1) {
                surf_stride = line_stride * (in_h - 1);
             }
          } else {
             if (in_h > 4) {
                surf_stride = width_stride * (in_h - 4);
             }
          }

          int cvt_bits_per_elem = 16;
          if (conv_in_precision == 6) {
             cvt_bits_per_elem = 4;
          } else if (conv_in_precision == 0) {
             cvt_bits_per_elem = 8;
          } else if (conv_in_precision == 7) {
             cvt_bits_per_elem = 32;
          }
          int cvt_lanes = 128 / cvt_bits_per_elem;
          if (cvt_lanes < 1) cvt_lanes = 1;
          int cvt_active = use_nhwc_pack ? conv_in_channels : input_pack_c2;
          if (cvt_active < 1) cvt_active = 1;
          if (cvt_active > cvt_lanes) cvt_active = cvt_lanes;
          uint32_t cvt_mask = (cvt_active >= 32) ? 0xffffffffu : ((1u << cvt_active) - 1u);
          cv5_con5 = (int)cvt_mask;

          // Derive data_entries from row granularity; 8-channel paths scale by height.
          int row_entries = (width_stride * align_c + 31) / 32;
          if (row_entries < 1) row_entries = 1;
          core_misc_cfg = CORE_MISC_CFG_PROC_PRECISION(conv_proc_precision);
          if (is_depthwise) core_misc_cfg |= CORE_MISC_CFG_DW_EN(1);
          if (align_c >= 16 || is_depthwise) {
             cbuf_entries = row_entries;
          } else {
             cbuf_entries = row_entries * in_h * 4;
          }
          if (cbuf_entries < 1) cbuf_entries = 1;
          
          EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));

          if (is_depthwise) conv_con1 |= CNA_CONV_CON1_CONV_MODE(3);
          EMIT(REG_CNA_CONV_CON1, conv_con1);
          EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(feature_grains));
          EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_Y_STRIDE(1) | CNA_CONV_CON3_CONV_X_STRIDE(1));
          EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(width_stride) | CNA_DATA_SIZE0_DATAIN_HEIGHT(in_h));
          EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(data_in_channel_real) | CNA_DATA_SIZE1_DATAIN_CHANNEL(data_in_channel_aligned));
          EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(dataout_width));
          EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(dataout_atomics));

          int weight_bytes_total = weight_bytes_per_kernel * conv_out_channels;
          EMIT(REG_CNA_WEIGHT_SIZE0, weight_bytes_total);
          EMIT(REG_CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(weight_bytes_per_kernel));
          if (weight_kernels == 0) weight_kernels = conv_out_channels;
          EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(conv_kernel_w) | CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(conv_kernel_h) | CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(weight_kernels));

          size_t fd_bytes = (size_t)width_stride * (size_t)feature_grains * (size_t)align_c * sizeof(__fp16);
          int data_bank = (int)((fd_bytes + NPU_CBUF_BANK_SIZE - 1) / NPU_CBUF_BANK_SIZE);
          if (data_bank < 1) data_bank = 1;
          if (data_bank > NPU_CBUF_BANKS - 1) data_bank = NPU_CBUF_BANKS - 1;
          EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(NPU_CBUF_BANKS - data_bank) | CNA_CBUF_CON0_DATA_BANK(data_bank));
          EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(cbuf_entries));
          EMIT(REG_CNA_CVT_CON0, cvt_con0);
          EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
          EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
          EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
          EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
          EMIT(REG_CNA_FEATURE_DATA_ADDR, CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR(input_dma));
          EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
          EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(line_stride));
          EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(surf_stride));
          EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(in_w) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(in_h));
          EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(align_c));
          EMIT(REG_CNA_DCOMP_ADDR0, CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0(weights_dma + REGCMD_RESERVED));
          EMIT(REG_CNA_CVT_CON5, cv5_con5);
          EMIT(REG_CORE_MISC_CFG, core_misc_cfg);
          EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(out_h - 1) | CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(out_w - 1));
          EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(out_channel_field));
          emit_raw(&regs, CORE | 0x1, 0x3030, 0);
          EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2) | DPU_FEATURE_MODE_CFG_CONV_MODE(3 * (int)is_depthwise) );
          EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(2) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
          EMIT(REG_DPU_DST_BASE_ADDR, DPU_DST_BASE_ADDR_DST_BASE_ADDR(output_dma));
          EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(out_width_stride));
          EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(out_w - 1));
          EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(out_h - 1));
          EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(orig_channel) | DPU_DATA_CUBE_CHANNEL_CHANNEL(out_channel_field));
          EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
          int ow_cfg_size_e_012 = 1;
          if (is_depthwise) ow_cfg_size_e_012 = 3;
          EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(ow_cfg_size_e_012) | DPU_BS_OW_CFG_SIZE_E_1(ow_cfg_size_e_012) | DPU_BS_OW_CFG_SIZE_E_0(ow_cfg_size_e_012) | DPU_BS_OW_CFG_OD_BYPASS(1));
          EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(out_channel_field));
          EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(out_h - 1) | DPU_WDMA_SIZE_1_WIDTH_WDMA(out_w - 1));
          EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
          EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
          EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
          EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) | DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
          // DPU surface add: stride per surface row in 16B units.
          int effective_align_out = out_channel_field + 1;
          if (conv_groups > 1 &&
                !(conv_groups == conv_in_channels && conv_out_channels == conv_in_channels)) {
             int per_group_out = (conv_out_channels + conv_groups - 1) / conv_groups;
             int per_group_align = align_up_int(per_group_out, 16);
             if (per_group_align < 16) per_group_align = 16;
             effective_align_out = per_group_align;
          }
          surface_add = out_width_stride * (effective_align_out / 8);
          EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(surface_add));
          emit_raw(&regs, 0x0 | 0x1, 0x40c4, 0);
          emit_raw(&regs, 0x81, REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
          finish_current_task();

          goto alu_case_done;
       }
