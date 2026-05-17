# DeepWiki Q&A with Code Context for Repository: nvdla/hw
## Q1
how does it pack input
### Answer
Based on the NVDLA codebase context, input packing occurs in several different contexts:

## Test Vector Packing (Verification)

The `input_txn_to_verilator.pl` script converts text-based test vectors to binary format using Perl's `pack` function for Verilator simulation hw:82-107 . Commands are packed as:
- `wait`: 1 byte (code 1)
- `write_reg`: 1 byte code + 4-byte address + 4-byte data (code 2)
- `read_reg`: 1 byte code + 4-byte address + 4-byte mask + 4-byte expected data (code 3)
- `load_mem`/`dump_mem`: 1 byte code + 4-byte address + 4-byte size + memory contents (codes 5/4)

## Hardware Data Packing

### CSC Module (Convolution Spatial Computing)
The CSC module packs input data when sending to the MAC sequencer. For INT8 precision with packed storage:
- When `last_super_channel_element_num <= 32`, it packs 4 atoms into one CBUF entry hw:781-786 
- When `last_super_channel_element_num > 32`, it packs 2 atoms into one CBUF entry hw:783-786 
- The packed channel iterator is calculated based on `input_atom_coor_width % 4` or `% 2` depending on packing mode hw:835-839 

### CDMA Module (Convolution DMA)
CDMA packs data when storing to CBUF. For the last super surface with odd surface count:
- If `last_super_surface_size <= ATOM_CUBE_SIZE` (32B), it packs 4 atoms with 4 different width coordinates into one CBUF entry hw:1499-1511 
- Otherwise, it packs 4 atoms with 2 different width coordinates into one CBUF entry hw:1513-1522 

### SDP Core Packing
The SDP core has dedicated packing modules:
- `NV_NVDLA_SDP_CORE_Y_dmapack`: Packs 256-bit data into four 64-bit segments using a 2-bit counter hw:143-165 
- `NV_NVDLA_SDP_CORE_Y_dppack`: Similar structure but with 128-bit segments hw:143-165 
- `NV_NVDLA_SDP_CORE_Y_dpunpack`: Unpacks data by concatenating four segments hw:172-176 

## Notes
The packing mechanisms serve different purposes: test vector packing is for verification, while hardware packing optimizes memory bandwidth and CBUF utilization. The CSC and CDMA packing strategies depend on data format (INT8/INT16/FP16) and channel dimensions to maximize CBUF entry usage.

Wiki pages you might want to explore:
- [Verilator Integration (nvdla/hw)](/wiki/nvdla/hw#5.3)
### Citations
**File:** verif/verilator/input_txn_to_verilator.pl (L82-107)
```text
    if ($values[0] =~ /wait/) {
      print $ouf pack("C", 1);
    }
    elsif($values[0] =~ /write_reg/) {
      if($size != 3) { die "\nERROR: in write_reg command: $input_line\n" };

      # For CSB, top 16 bits are misc, lower 16 are addr
      my $address = $values[1]; 
      my $data = $values[2];

      print $ouf pack("CLL", 2, hex($address), hex($data));

    } elsif ($values[0] =~ /read_reg/) {
      if($size != 4) { die "\n ERROR: in read_reg command: $input_line\n" };

      my $address;
      my $bitmask;
      my $cmp_mode;
      my $exp_data;
      my $poll_attempts = $read_reg_poll_retries;

      $address = $values[1];
      $bitmask = $values[2];
      $exp_data = $values[3];
      
      print $ouf pack("CLLL", 3, hex($address), hex($bitmask), hex($exp_data));
```
**File:** cmod/csc/NV_NVDLA_csc.cpp (L781-786)
```cpp
                                            if (last_super_channel_element_num <= 32) { // Pack 4 atoms into one entry
                                                cbuf_entry_addr = (data_entry_idx_free_ + (input_atom_coor_height * batch_num + batch_iter) * cbuf_entry_per_slice + cube_in_width*(super_channel_num/2) + input_atom_coor_width/4)%cbuf_entry_for_data;
                                            } else {    // Pack 2 atoms into one entry
                                                // (last_super_channel_element_num > 32) && (last_super_channel_element_num < 64)
                                                cbuf_entry_addr = (data_entry_idx_free_+ (input_atom_coor_height * batch_num + batch_iter) * cbuf_entry_per_slice + cube_in_width*(super_channel_num/2) + input_atom_coor_width/2)%cbuf_entry_for_data;
                                            }
```
**File:** cmod/csc/NV_NVDLA_csc.cpp (L835-839)
```cpp
                                                    if (last_super_channel_element_num <= 32) { // Pack 4 atoms of 4 rows into one cbuf entry
                                                        packed_channel_iter = (input_atom_coor_width%4) * 32 + channel_iter;
                                                    } else { // Pack 2 atoms of 2 rows into one cbuf entry
                                                        packed_channel_iter = (input_atom_coor_width%2) * 64 + channel_iter;
                                                    }
```
**File:** cmod/cdma/NV_NVDLA_cdma.cpp (L1499-1511)
```cpp
                                if (last_super_surface_size <= ATOM_CUBE_SIZE) { // At most 32B in last super surface. pack 4 atoms with 4 different width coordinates into one cbuf entry
                                    memcpy(cbuf_payload_data_ptr, &fetched_data_ptr[height_iter*cube_width+atom_stored][0], ATOM_CUBE_SIZE);
                                    if (atom_stored+1 < cube_width)
                                        memcpy(&cbuf_payload_data_ptr[ATOM_CUBE_SIZE], &fetched_data_ptr[height_iter*cube_width+atom_stored+1][0], ATOM_CUBE_SIZE);
                                    else
                                        memset(&cbuf_payload_data_ptr[ATOM_CUBE_SIZE], 0, ATOM_CUBE_SIZE);
                                    cbuf_entry_addr = (data_entry_idx_working_ + ((super_slice_iter * fetch_slice_grain + height_iter) * batch_num + batch_iter ) * cbuf_entry_per_slice + (super_surface_iter/2) * cube_width + (atom_stored/4)) % data_entry_num;
                                    if (atom_stored%4 == 0) {
                                        cdma2cbuf_data_payload_->hsel = 0;
                                    } else { 
                                        cdma2cbuf_data_payload_->hsel = 1;
                                    }
                                    atom_stored+=2;
```
**File:** cmod/cdma/NV_NVDLA_cdma.cpp (L1513-1522)
```cpp
                                else {  // pack 4 atoms with 2 different width coordinates into one cbuf entry
                                    memcpy(cbuf_payload_data_ptr, fetched_data_ptr[height_iter*cube_width+atom_stored], ATOM_CUBE_SIZE*2);
                                    cbuf_entry_addr = (data_entry_idx_working_ + ((super_slice_iter * fetch_slice_grain + height_iter) * batch_num + batch_iter ) * cbuf_entry_per_slice + (super_surface_iter/2) * cube_width + (atom_stored/2)) % data_entry_num;
                                    if (atom_stored%2 == 0) {
                                        cdma2cbuf_data_payload_->hsel = 0;
                                    } else { 
                                        cdma2cbuf_data_payload_->hsel = 1;
                                    }
                                    atom_stored++;
                                }
```
**File:** vmod/nvdla/sdp/NV_NVDLA_SDP_CORE_Y_dmapack.v (L143-165)
```text
 assign pack_seg0 = pack_data_ext[((64*0) + 64 - 1):64*0];
 assign pack_seg1 = pack_data_ext[((64*1) + 64 - 1):64*1];
 assign pack_seg2 = pack_data_ext[((64*2) + 64 - 1):64*2];
 assign pack_seg3 = pack_data_ext[((64*3) + 64 - 1):64*3];
always @(
  pack_cnt
  or pack_seg0
  or pack_seg1
  or pack_seg2
  or pack_seg3
  ) begin
    case (pack_cnt)
     0: mux_data = pack_seg0;
     1: mux_data = pack_seg1;
     2: mux_data = pack_seg2;
     3: mux_data = pack_seg3;
    //VCS coverage off
    default : begin 
                mux_data[63:0] = {64{`x_or_0}};
              end  
    //VCS coverage on
    endcase
end
```
**File:** vmod/nvdla/sdp/NV_NVDLA_SDP_CORE_Y_dppack.v (L143-165)
```text
 assign pack_seg0 = pack_data_ext[((128*0) + 128 - 1):128*0];
 assign pack_seg1 = pack_data_ext[((128*1) + 128 - 1):128*1];
 assign pack_seg2 = pack_data_ext[((128*2) + 128 - 1):128*2];
 assign pack_seg3 = pack_data_ext[((128*3) + 128 - 1):128*3];
always @(
  pack_cnt
  or pack_seg0
  or pack_seg1
  or pack_seg2
  or pack_seg3
  ) begin
    case (pack_cnt)
     0: mux_data = pack_seg0;
     1: mux_data = pack_seg1;
     2: mux_data = pack_seg2;
     3: mux_data = pack_seg3;
    //VCS coverage off
    default : begin 
                mux_data[127:0] = {128{`x_or_0}};
              end  
    //VCS coverage on
    endcase
end
```
**File:** vmod/nvdla/sdp/NV_NVDLA_SDP_CORE_Y_dpunpack.v (L172-176)
```text
//====================================
assign pack_prdy = out_prdy;
assign out_pvld  = pack_pvld;
assign out_data  = {pack_seg3 , pack_seg2 , pack_seg1 , pack_seg0};

```
## Q2
where is CDMA_DC source code and test script
### Answer
The **CDMA_DC** (Convolution DMA - Direct Convolution) source code is primarily located in the `cdma` directory of the hardware repository, and its testing is integrated into the NVDLA trace-based verification system.

| Component | File Path |
| :--- | :--- |
| **RTL Source (Verilog)** | `vmod/nvdla/cdma/NV_NVDLA_CDMA_dc.v` hw:9-11  |
| **C-Model (SystemC)** | `cmod/cdma/NV_NVDLA_cdma.cpp` hw:1039-1045  |
| **Test Infrastructure** | `verif/sim/Makefile` and `verif/traces/traceplayer/` hw:0-0  |

---

## Source Code Details

### RTL Implementation
The core logic for the Direct Convolution (DC) DMA engine is implemented in `NV_NVDLA_CDMA_dc.v`. This module handles data requests to memory (MCIF/CVIF) and manages the flow of data into the convolution buffers hw:11-81 . 
- **Supporting FIFO**: The module uses a dedicated FIFO defined in `NV_NVDLA_CDMA_DC_fifo.v` hw:9-13 .
- **Integration**: It is instantiated within the top-level CDMA module and included in the DUT (Design Under Test) file list for simulation hw:40-40 .

### C-Model Implementation
The SystemC model for CDMA simulates the DC behavior in `NV_NVDLA_cdma.cpp`. Specifically, the `DirectConvDataRequestSequencerCommon` function handles the calculation of fetch parameters like `super_surface_num` and `fetch_slice_grain` hw:1039-1047 . Constants and definitions for DC mode are found in the corresponding header file hw:91-91 .

## Test Scripts and Verification

NVDLA uses a trace-based verification system where hardware transactions are "played" back against the RTL.

1.  **Simulation Makefile**: The main entry point for running tests is `verif/sim/Makefile` hw:0-0 .
2.  **Test Vectors**: Direct Convolution tests are typically found as transaction traces in `verif/traces/traceplayer/`. While there isn't a single "DC test script," the regression suite includes sanity and functional tests that exercise the CDMA DC path hw:0-0 .
3.  **Verification Flow**:
    *   **Build**: `make build` compiles the RTL including the CDMA_DC module hw:0-0 .
    *   **Run**: `make run TESTDIR=...` executes a specific trace hw:0-0 .
    *   **Check**: Results are verified using `checktest.pl` against expected logs hw:0-0 .

## Notes
- The CDMA module also contains sub-units for Image (IMG) and Winograd (WG) modes, which share some infrastructure with the DC unit hw:70-90 .
- For FPGA-based verification using Vivado, the file list is maintained in `verif/dut/dut.vivado.f` hw:40-40 .

Wiki pages you might want to explore:
- [Verification System (nvdla/hw)](/wiki/nvdla/hw#5)
### Citations
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_dc.v (L9-81)
```text
// File Name: NV_NVDLA_CDMA_dc.v

module NV_NVDLA_CDMA_dc (
   nvdla_core_clk            //|< i
  ,nvdla_core_ng_clk         //|< i
  ,nvdla_core_rstn           //|< i
  ,cvif2dc_dat_rd_rsp_pd     //|< i
  ,cvif2dc_dat_rd_rsp_valid  //|< i
  ,dc2sbuf_p0_rd_data        //|< i
  ,dc2sbuf_p1_rd_data        //|< i
  ,dc_dat2cvif_rd_req_ready  //|< i
  ,dc_dat2mcif_rd_req_ready  //|< i
  ,mcif2dc_dat_rd_rsp_pd     //|< i
  ,mcif2dc_dat_rd_rsp_valid  //|< i
  ,pwrbus_ram_pd             //|< i
  ,reg2dp_batch_stride       //|< i
  ,reg2dp_batches            //|< i
  ,reg2dp_conv_mode          //|< i
  ,reg2dp_data_bank          //|< i
  ,reg2dp_data_reuse         //|< i
  ,reg2dp_datain_addr_high_0 //|< i
  ,reg2dp_datain_addr_low_0  //|< i
  ,reg2dp_datain_channel     //|< i
  ,reg2dp_datain_format      //|< i
  ,reg2dp_datain_height      //|< i
  ,reg2dp_datain_ram_type    //|< i
  ,reg2dp_datain_width       //|< i
  ,reg2dp_dma_en             //|< i
  ,reg2dp_entries            //|< i
  ,reg2dp_grains             //|< i
  ,reg2dp_in_precision       //|< i
  ,reg2dp_line_packed        //|< i
  ,reg2dp_line_stride        //|< i
  ,reg2dp_op_en              //|< i
  ,reg2dp_proc_precision     //|< i
  ,reg2dp_skip_data_rls      //|< i
  ,reg2dp_surf_packed        //|< i
  ,reg2dp_surf_stride        //|< i
  ,sc2cdma_dat_pending_req   //|< i
  ,status2dma_free_entries   //|< i
  ,status2dma_fsm_switch     //|< i
  ,status2dma_valid_slices   //|< i *
  ,status2dma_wr_idx         //|< i
  ,cvif2dc_dat_rd_rsp_ready  //|> o
  ,dc2cvt_dat_wr_addr        //|> o
  ,dc2cvt_dat_wr_data        //|> o
  ,dc2cvt_dat_wr_en          //|> o
  ,dc2cvt_dat_wr_hsel        //|> o
  ,dc2cvt_dat_wr_info_pd     //|> o
  ,dc2sbuf_p0_rd_addr        //|> o
  ,dc2sbuf_p0_rd_en          //|> o
  ,dc2sbuf_p0_wr_addr        //|> o
  ,dc2sbuf_p0_wr_data        //|> o
  ,dc2sbuf_p0_wr_en          //|> o
  ,dc2sbuf_p1_rd_addr        //|> o
  ,dc2sbuf_p1_rd_en          //|> o
  ,dc2sbuf_p1_wr_addr        //|> o
  ,dc2sbuf_p1_wr_data        //|> o
  ,dc2sbuf_p1_wr_en          //|> o
  ,dc2status_dat_entries     //|> o
  ,dc2status_dat_slices      //|> o
  ,dc2status_dat_updt        //|> o
  ,dc2status_state           //|> o
  ,dc_dat2cvif_rd_req_pd     //|> o
  ,dc_dat2cvif_rd_req_valid  //|> o
  ,dc_dat2mcif_rd_req_pd     //|> o
  ,dc_dat2mcif_rd_req_valid  //|> o
  ,dp2reg_dc_rd_latency      //|> o
  ,dp2reg_dc_rd_stall        //|> o
  ,mcif2dc_dat_rd_rsp_ready  //|> o
  ,slcg_dc_gate_img          //|> o
  ,slcg_dc_gate_wg           //|> o
  );
```
**File:** cmod/cdma/NV_NVDLA_cdma.cpp (L1039-1047)
```cpp
    // super_normal_ratio: original number of atoms fetched for half cbuf entry(64B)
    super_normal_ratio      = 2;
    element_per_super_atom  = element_per_atom * super_normal_ratio;
    fetch_slice_grain       = cdma_grains_ + 1; // For DC, it's the number of lines because one slice contains one line in DC mode.
    super_slice_num         = (cube_height+fetch_slice_grain-1) / fetch_slice_grain;
    super_surface_num       = (cube_channel + element_per_super_atom - 1)/element_per_super_atom;
    fetch_slice_grain_last  = cube_height - fetch_slice_grain*(super_slice_num-1);
    is_line_packed          = cdma_line_packed_;
    is_surf_packed          = cdma_surf_packed_;
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_DC_fifo.v (L9-13)
```text
// File Name: NV_NVDLA_CDMA_DC_fifo.v

`define FORCE_CONTENTION_ASSERTION_RESET_ACTIVE 1'b1
`include "simulate_x_tick.vh"
module NV_NVDLA_CDMA_DC_fifo (
```
**File:** verif/dut/dut.f (L40-40)
```text
../../outdir/nv_full/vmod/nvdla/cdma/NV_NVDLA_CDMA_dc.v
```
**File:** cmod/cdma/NV_NVDLA_cdma.h (L91-91)
```text
#define INPUT_DATA_FORMAT_DC                0
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_dma_mux.v (L70-90)
```text
input         dc_dat2mcif_rd_req_valid;  /* data valid */
output        dc_dat2mcif_rd_req_ready;  /* data return handshake */
input  [78:0] dc_dat2mcif_rd_req_pd;

input         dc_dat2cvif_rd_req_valid;  /* data valid */
output        dc_dat2cvif_rd_req_ready;  /* data return handshake */
input  [78:0] dc_dat2cvif_rd_req_pd;

input         wg_dat2mcif_rd_req_valid;  /* data valid */
output        wg_dat2mcif_rd_req_ready;  /* data return handshake */
input  [78:0] wg_dat2mcif_rd_req_pd;

input         wg_dat2cvif_rd_req_valid;  /* data valid */
output        wg_dat2cvif_rd_req_ready;  /* data return handshake */
input  [78:0] wg_dat2cvif_rd_req_pd;

input         img_dat2mcif_rd_req_valid;  /* data valid */
output        img_dat2mcif_rd_req_ready;  /* data return handshake */
input  [78:0] img_dat2mcif_rd_req_pd;

input         img_dat2cvif_rd_req_valid;  /* data valid */
```
**File:** verif/dut/dut.vivado.f (L40-40)
```text
../../outdir/nv_full/vmod/nvdla/cdma/NV_NVDLA_CDMA_dc.v
```
## Q3
CDMA_WG vs CDMA_DC vs CDMA_IMG
### Answer
CDMA_WG, CDMA_DC, and CDMA_IMG are three distinct operation modes within the NVDLA Convolution DMA (CDMA) module, each optimized for different convolution algorithms and input data formats hw:91-93 .

---

## Comparison Overview

| Aspect | CDMA_DC | CDMA_WG | CDMA_IMG |
|--------|---------|---------|----------|
| **Full Name** | Direct Convolution | Winograd Convolution | Image Convolution |
| **Data Format** | `INPUT_DATA_FORMAT_DC` (0) | `INPUT_DATA_FORMAT_WINO` (2) | `INPUT_DATA_FORMAT_IMAGE` (1) |
| **Convolution Mode** | Direct | Winograd | Direct |
| **Input Type** | Feature data | Feature data | Pixel data |
| **RTL Module** | `NV_NVDLA_CDMA_dc.v` | `NV_NVDLA_CDMA_wg.v` | `NV_NVDLA_CDMA_img.v` |
| **Key Parameters** | `grains`, `line_packed`, `surf_packed` | `conv_x_stride`, `conv_y_stride`, padding | `pixel_format`, `pixel_mapping`, `pixel_x_offset` |

---

## CDMA_DC (Direct Convolution)

CDMA_DC handles standard direct convolution with feature data input. It is activated when `conv_mode` is DIRECT and `datain_format` is FEATURE hw:229-235 .

**Key Characteristics:**
- Uses `DirectConvDataRequestSequencerCommon()` and `DirectConvDataResponseSequencerCommon()` for data sequencing hw:316-332 
- Supports batch processing with `batches` parameter
- Uses `fetch_slice_grain` (calculated as `cdma_grains_ + 1`) to control data fetching granularity hw:1039-1047 
- Supports line and surface packing via `line_packed` and `surf_packed` flags
- Calculates `super_surface_num` and `super_slice_num` for efficient CBUF utilization hw:1247-1254 

---

## CDMA_WG (Winograd Convolution)

CDMA_WG implements the Winograd algorithm for convolution, which can be more efficient for certain kernel sizes. It is activated when `conv_mode` is WINOGRAD hw:288-290 .

**Key Characteristics:**
- Uses `WinoConvDataRequestSequencerCommon()` and `WinoConvDataResponseSequencerCommon()` for data sequencing hw:323-327 
- Supports extended width and height parameters (`datain_width_ext`, `datain_height_ext`) for Winograd transformations hw:30-33 
- Uses convolution strides (`conv_x_stride`, `conv_y_stride`) specific to Winograd algorithm hw:21-22 
- Includes padding parameters (`pad_left`, `pad_right`, `pad_top`, `pad_bottom`) for Winograd input preparation hw:39-43 
- Has a dedicated FSM with states: IDLE, PEND, BUSY, DONE hw:683-686 

---

## CDMA_IMG (Image Convolution)

CDMA_IMG handles direct convolution with pixel/image data input (e.g., from camera or image files). It is activated when `conv_mode` is DIRECT and `datain_format` is PIXEL hw:296-301 .

**Key Characteristics:**
- Uses `ImageConvDataRequestSequencerCommon()` and `ImageConvDataResponseSequencerCommon()` for data sequencing hw:328-332 
- Handles various pixel formats (YUV, RGB, planar/interleaved) via `pixel_format` parameter hw:25-26 
- Supports pixel format conversion and reordering (e.g., RGBA order) hw:2761-2794 
- Includes pixel-specific parameters: `pixel_mapping`, `pixel_sign_override`, `pixel_x_offset` hw:27-29 
- Has dedicated packing logic in `NV_NVDLA_CDMA_IMG_pack.v` for pixel data organization hw:11-60 
- Supports planar pixel formats with separate bundle limits for planar0 and planar1 hw:44-58 

---

## Mode Selection Logic

The C-model determines the operation mode based on register configuration hw:221-313 :

```cpp
// DC mode: conv_mode=DIRECT, datain_format=FEATURE
if (cdma_conv_mode_ == NVDLA_CDMA_D_MISC_CFG_0_CONV_MODE_DIRECT) {
    if (cdma_datain_format_ == NVDLA_CDMA_D_DATAIN_FORMAT_0_DATAIN_FORMAT_FEATURE)
        cdma_input_data_format_ = INPUT_DATA_FORMAT_DC;
    else
        cdma_input_data_format_ = INPUT_DATA_FORMAT_IMAGE;
} else
    cdma_input_data_format_ = INPUT_DATA_FORMAT_WINO;  // WG mode
```

## Notes
- All three modes share common DMA infrastructure through the DMA mux (`NV_NVDLA_CDMA_dma_mux.v`) which arbitrates memory requests hw:11-60 
- Each mode has its own clock gating control (SLCG) for power management hw:636-738 
- The CSC module also has corresponding mode handling for weight loading based on these convolution modes hw:428-449 .
### Citations
**File:** cmod/cdma/NV_NVDLA_cdma.h (L91-93)
```text
#define INPUT_DATA_FORMAT_DC                0
#define INPUT_DATA_FORMAT_IMAGE             1
#define INPUT_DATA_FORMAT_WINO              2
```
**File:** cmod/cdma/NV_NVDLA_cdma.cpp (L221-313)
```cpp
void NV_NVDLA_cdma::ActDataReadRequestSequenceThread () {
    uint32_t cdma_act_data_operation_mode;
    uint8_t  cdma_input_data_format_;
    while (true) {
        cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, before cdma_kickoff_\n"));
        wait(cdma_kickoff_);
        cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, after cdma_kickoff_\n"));

        if (cdma_conv_mode_ == NVDLA_CDMA_D_MISC_CFG_0_CONV_MODE_DIRECT) {
            if (cdma_datain_format_ == NVDLA_CDMA_D_DATAIN_FORMAT_0_DATAIN_FORMAT_FEATURE)
                cdma_input_data_format_ = INPUT_DATA_FORMAT_DC;
            else
                cdma_input_data_format_ = INPUT_DATA_FORMAT_IMAGE;
        } else
            cdma_input_data_format_ = INPUT_DATA_FORMAT_WINO;

        if (dat_first_layer) {  // The first layer
            dat_first_layer         = false;
            data_entry_idx_free_    = (cdma_data_bank_ + 1) * CBUF_ENTRY_PER_BANK - 1;
            data_entry_idx_working_ = 0;
            data_entry_idx_planed_  = -1;
            // if (cdma_data_reuse_)
            //    FAIL(("Can't reuse data when first layer\n"));

            cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread dat_first_layer. data_entry_idx_free_=0x%x data_entry_idx_working_=0x%x data_entry_idx_planed_=0x%x\n", data_entry_idx_free_, data_entry_idx_working_, data_entry_idx_planed_));
        }
        else if ((cdma_req_prev_conv_mode_ != cdma_conv_mode_) || (cdma_req_prev_data_bank_ != cdma_data_bank_) || (cdma_req_prev_weight_bank_ != cdma_weight_bank_) || (cdma_req_prev_input_data_format_ != cdma_input_data_format_)) {
            // Not reuse data of previous layer. wait until cbuf is empty
            if (cdma_req_prev_data_bank_ != cdma_data_bank_) {
                cslInfo(("Before WaitUntilDataEntryPlanedIndexEqualEntryFreeIndex\n"));
                WaitUntilDataEntryPlanedIndexEqualEntryFreeIndex();
                cslInfo(("After WaitUntilDataEntryPlanedIndexEqualEntryFreeIndex\n"));
                data_entry_idx_free_    = (cdma_data_bank_ + 1) * CBUF_ENTRY_PER_BANK - 1;
                data_entry_idx_working_ = 0;
                data_entry_idx_planed_  = -1;
                cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread not dat_first_layer. Bank changed, not reuse data. data_entry_idx_free_=0x%x data_entry_idx_working_=0x%x data_entry_idx_planed_=0x%x\n", data_entry_idx_free_, data_entry_idx_working_, data_entry_idx_planed_));
            } else {
                // Continue to use cbuf following the end of previous layer
                // data_entry_idx_free_ will be updated by sc2cdma
                data_entry_idx_working_ = data_entry_idx_planed_ + 1;
                cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread not dat_first_layer. Not reuse data. data_entry_idx_free_=0x%x data_entry_idx_working_=0x%x data_entry_idx_planed_=0x%x\n", data_entry_idx_free_, data_entry_idx_working_, data_entry_idx_planed_));
            }
        }
        else if (cdma_req_prev_skip_data_rls_ && cdma_data_reuse_ && (cdma_req_prev_conv_mode_ == cdma_conv_mode_) && (cdma_req_prev_data_bank_ == cdma_data_bank_) && 
            (cdma_req_prev_weight_bank_ == cdma_weight_bank_) && (cdma_req_prev_input_data_format_ == cdma_input_data_format_)) {
            // Reuse all data of previous layer. Skip fetching new data.
            cdma_req_prev_skip_data_rls_= cdma_skip_data_rls_;
            cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread not dat_first_layer. Reuse all data of previous layer. data_entry_idx_free_=0x%x data_entry_idx_working_=0x%x data_entry_idx_planed_=0x%x\n", data_entry_idx_free_, data_entry_idx_working_, data_entry_idx_planed_));
            continue;
        }
        else {
            // reuse partial data of previous layer OR not reuse data of previous layer
            // Continue to use cbuf following the end of previous layer
            // data_entry_idx_free_ will be updated by sc2cdma
            data_entry_idx_working_ = data_entry_idx_planed_ + 1;
            cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread not dat_first_layer. Partial resue or not reuse data. data_entry_idx_free_=0x%x data_entry_idx_working_=0x%x data_entry_idx_planed_=0x%x\n", data_entry_idx_free_, data_entry_idx_working_, data_entry_idx_planed_));
        }

        if ( NVDLA_CDMA_D_DATAIN_FORMAT_0_DATAIN_FORMAT_FEATURE == cdma_datain_format_) {
            // Direct convolution or Winograd convolution
            if (NVDLA_CDMA_D_MISC_CFG_0_CONV_MODE_DIRECT == cdma_conv_mode_) {
                // Direct convolution
                if (0 == cdma_batches_) {
                    cdma_act_data_operation_mode = ACT_MODE_DIRECT_CONV_NONE_BATCH;
                } else {
                    cdma_act_data_operation_mode = ACT_MODE_DIRECT_CONV_BATCH;
                }
            } else if (NVDLA_CDMA_D_MISC_CFG_0_CONV_MODE_WINOGRAD == cdma_conv_mode_) {
                cdma_act_data_operation_mode = ACT_MODE_WINOGRAD_CONV;
#pragma CTC SKIP
            } else {
                FAIL(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, unsupport mode, cdma_conv_mode_ is 0x%X", cdma_conv_mode_));
            }
#pragma CTC ENDSKIP
        }
        else if (NVDLA_CDMA_D_DATAIN_FORMAT_0_DATAIN_FORMAT_PIXEL == cdma_datain_format_) {
            // Image load, only Direct convolution is supported
            if (NVDLA_CDMA_D_MISC_CFG_0_CONV_MODE_DIRECT == cdma_conv_mode_) {
                if (0 == cdma_batches_) {
                    cdma_act_data_operation_mode = ACT_MODE_DIRECT_PIXEL;
#pragma CTC SKIP
                } else {
                    FAIL(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, unsupport mode, cdma_conv_mode_ is 0x%X", cdma_conv_mode_));
                }
            } else {
                FAIL(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, unsupport mode, cdma_conv_mode_ is 0x%X", cdma_conv_mode_));
            }
        }
        else {
            // HOG
            FAIL(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, image load sequences has not been implemented."));
        }
#pragma CTC ENDSKIP
```
**File:** cmod/cdma/NV_NVDLA_cdma.cpp (L316-332)
```cpp
        switch(cdma_act_data_operation_mode) {
            case ACT_MODE_DIRECT_CONV_NONE_BATCH:
            case ACT_MODE_DIRECT_CONV_BATCH:
                cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, before DirectConvDataRequestSequencerCommon\n"));
                DirectConvDataRequestSequencerCommon();
                cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, after DirectConvDataRequestSequencerCommon\n"));
                break;
            case ACT_MODE_WINOGRAD_CONV:
                cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, before WinotConvDataRequestSequencerCommon\n"));
                WinoConvDataRequestSequencerCommon();
                cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, after WinoConvDataRequestSequencerCommon\n"));
                break;
            case ACT_MODE_DIRECT_PIXEL:
                cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, before ImageConvDataRequestSequencerCommon\n"));
                ImageConvDataRequestSequencerCommon();
                cslInfo(("NV_NVDLA_cdma::ActDataReadRequestSequenceThread, after ImageConvDataRequestSequencerCommon\n"));
                break;
```
**File:** cmod/cdma/NV_NVDLA_cdma.cpp (L1039-1047)
```cpp
    // super_normal_ratio: original number of atoms fetched for half cbuf entry(64B)
    super_normal_ratio      = 2;
    element_per_super_atom  = element_per_atom * super_normal_ratio;
    fetch_slice_grain       = cdma_grains_ + 1; // For DC, it's the number of lines because one slice contains one line in DC mode.
    super_slice_num         = (cube_height+fetch_slice_grain-1) / fetch_slice_grain;
    super_surface_num       = (cube_channel + element_per_super_atom - 1)/element_per_super_atom;
    fetch_slice_grain_last  = cube_height - fetch_slice_grain*(super_slice_num-1);
    is_line_packed          = cdma_line_packed_;
    is_surf_packed          = cdma_surf_packed_;
```
**File:** cmod/cdma/NV_NVDLA_cdma.cpp (L1247-1254)
```cpp
    super_normal_ratio      = 2;

    element_per_super_atom   = element_per_atom * super_normal_ratio;
    fetch_slice_grain       = cdma_grains_ + 1;
    super_slice_num         = (cube_height+fetch_slice_grain-1)/fetch_slice_grain;
    super_surface_num       = (cube_channel + element_per_super_atom - 1)/element_per_super_atom;
    fetch_slice_grain_last  = cube_height - fetch_slice_grain*(super_slice_num-1);
    data_entry_num          = (cdma_data_bank_+1) * CBUF_ENTRY_PER_BANK;
```
**File:** cmod/cdma/NV_NVDLA_cdma.cpp (L2761-2794)
```cpp
                case NVDLA_CDMA_D_DATAIN_FORMAT_0_PIXEL_FORMAT_T_Y10___U10V10_N444:
                case NVDLA_CDMA_D_DATAIN_FORMAT_0_PIXEL_FORMAT_T_Y12___U12V12_N444:
                case NVDLA_CDMA_D_DATAIN_FORMAT_0_PIXEL_FORMAT_T_Y16___U16V16_N444:
                    for (i=0;i<3;i++) {
                        rgba_idx = (i==0)? 0: (i==1)?2: 1;
                        pixel_16  = fetch_buffer_16[ori_pixel_idx*element_num+rgba_idx]; // Perfrom reorder. Store to CBUF in the order of RGBA
                        if (NVDLA_CDMA_D_MEAN_FORMAT_0_MEAN_FORMAT_ENABLE == mean_format)
                            cvt_mean = (i==0)? cdma_mean_ry_:
                                (i==1)? cdma_mean_gu_:
                                (i==2)? cdma_mean_bv_: cdma_mean_ax_;
                        else
                            cvt_mean = cvt_offset;
                        process_one_element_16(pixel_16, pixel_idx, element_num, i, PIXEL_UNSIGNED_INT16, cvt_en, convert_16to8, pad_buffer_8, pad_buffer_16, cvt_mean, cvt_scale, cvt_truncate, precision, proc_precision);
                    }
                    break;
                case NVDLA_CDMA_D_DATAIN_FORMAT_0_PIXEL_FORMAT_T_Y10___V10U10_N444:
                case NVDLA_CDMA_D_DATAIN_FORMAT_0_PIXEL_FORMAT_T_Y12___V12U12_N444:
                case NVDLA_CDMA_D_DATAIN_FORMAT_0_PIXEL_FORMAT_T_Y16___V16U16_N444:
                    for (i=0;i<3;i++) {
                        rgba_idx = i;
                        pixel_16  = fetch_buffer_16[ori_pixel_idx*element_num+rgba_idx]; // Perfrom reorder. Store to CBUF in the order of RGBA
                        if (NVDLA_CDMA_D_MEAN_FORMAT_0_MEAN_FORMAT_ENABLE == mean_format)
                            cvt_mean = (i==0)? cdma_mean_ry_:
                                (i==1)? cdma_mean_gu_:
                                (i==2)? cdma_mean_bv_: cdma_mean_ax_;
                        else
                            cvt_mean = cvt_offset;
                        process_one_element_16(pixel_16, pixel_idx, element_num, i, PIXEL_UNSIGNED_INT16, cvt_en, convert_16to8, pad_buffer_8, pad_buffer_16, cvt_mean, cvt_scale, cvt_truncate, precision, proc_precision);
                    }
                    break;
                default:
                    FAIL(("Unexpected pixel format %d", pixel_format));
            }
            pixel_idx++;
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_wg.v (L21-22)
```text
  ,reg2dp_conv_x_stride      //|< i
  ,reg2dp_conv_y_stride      //|< i
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_wg.v (L30-33)
```text
  ,reg2dp_datain_height_ext  //|< i
  ,reg2dp_datain_ram_type    //|< i
  ,reg2dp_datain_width       //|< i
  ,reg2dp_datain_width_ext   //|< i
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_wg.v (L39-43)
```text
  ,reg2dp_pad_bottom         //|< i *
  ,reg2dp_pad_left           //|< i
  ,reg2dp_pad_right          //|< i
  ,reg2dp_pad_top            //|< i
  ,reg2dp_pad_value          //|< i
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_wg.v (L683-686)
```text
localparam WG_STATE_IDLE = 2'b00;
localparam WG_STATE_PEND = 2'b01;
localparam WG_STATE_BUSY = 2'b10;
localparam WG_STATE_DONE = 2'b11;
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_IMG_ctrl.v (L25-26)
```text
  ,reg2dp_pixel_format
  ,reg2dp_pixel_mapping
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_IMG_ctrl.v (L27-29)
```text
  ,reg2dp_pixel_sign_override
  ,reg2dp_pixel_x_offset
  ,reg2dp_proc_precision
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_IMG_ctrl.v (L44-58)
```text
  ,pixel_planar0_bundle_limit
  ,pixel_planar0_bundle_limit_1st
  ,pixel_planar0_byte_sft
  ,pixel_planar0_lp_burst
  ,pixel_planar0_lp_vld
  ,pixel_planar0_rp_burst
  ,pixel_planar0_rp_vld
  ,pixel_planar0_sft
  ,pixel_planar0_width_burst
  ,pixel_planar1_bundle_limit
  ,pixel_planar1_bundle_limit_1st
  ,pixel_planar1_byte_sft
  ,pixel_planar1_lp_burst
  ,pixel_planar1_lp_vld
  ,pixel_planar1_rp_burst
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_IMG_pack.v (L11-60)
```text
module NV_NVDLA_CDMA_IMG_pack (
   nvdla_core_clk
  ,nvdla_core_rstn
  ,img2sbuf_p0_rd_data
  ,img2sbuf_p1_rd_data
  ,is_running
  ,layer_st
  ,pixel_bank
  ,pixel_data_expand
  ,pixel_data_shrink
  ,pixel_early_end
  ,pixel_packed_10b
  ,pixel_planar
  ,pixel_planar0_sft
  ,pixel_planar1_sft
  ,pixel_precision
  ,pixel_uint
  ,reg2dp_datain_channel
  ,reg2dp_datain_width
  ,reg2dp_mean_ax
  ,reg2dp_mean_bv
  ,reg2dp_mean_gu
  ,reg2dp_mean_ry
  ,reg2dp_pad_left
  ,reg2dp_pad_right
  ,sg2pack_data_entries
  ,sg2pack_entry_end
  ,sg2pack_entry_mid
  ,sg2pack_entry_st
  ,sg2pack_height_total
  ,sg2pack_img_pd
  ,sg2pack_img_pvld
  ,sg2pack_mn_enable
  ,sg2pack_sub_h_end
  ,sg2pack_sub_h_mid
  ,sg2pack_sub_h_st
  ,status2dma_wr_idx
  ,img2cvt_dat_wr_addr
  ,img2cvt_dat_wr_data
  ,img2cvt_dat_wr_en
  ,img2cvt_dat_wr_hsel
  ,img2cvt_dat_wr_info_pd
  ,img2cvt_dat_wr_pad_mask
  ,img2cvt_mn_wr_data
  ,img2sbuf_p0_rd_addr
  ,img2sbuf_p0_rd_en
  ,img2sbuf_p1_rd_addr
  ,img2sbuf_p1_rd_en
  ,img2status_dat_entries
  ,img2status_dat_slices
```
**File:** vmod/nvdla/cdma/NV_NVDLA_CDMA_dma_mux.v (L11-60)
```text
module NV_NVDLA_CDMA_dma_mux (
   nvdla_core_clk
  ,nvdla_core_rstn
  ,dc_dat2mcif_rd_req_valid
  ,dc_dat2mcif_rd_req_ready
  ,dc_dat2mcif_rd_req_pd
  ,dc_dat2cvif_rd_req_valid
  ,dc_dat2cvif_rd_req_ready
  ,dc_dat2cvif_rd_req_pd
  ,wg_dat2mcif_rd_req_valid
  ,wg_dat2mcif_rd_req_ready
  ,wg_dat2mcif_rd_req_pd
  ,wg_dat2cvif_rd_req_valid
  ,wg_dat2cvif_rd_req_ready
  ,wg_dat2cvif_rd_req_pd
  ,img_dat2mcif_rd_req_valid
  ,img_dat2mcif_rd_req_ready
  ,img_dat2mcif_rd_req_pd
  ,img_dat2cvif_rd_req_valid
  ,img_dat2cvif_rd_req_ready
  ,img_dat2cvif_rd_req_pd
  ,cdma_dat2mcif_rd_req_valid
  ,cdma_dat2mcif_rd_req_ready
  ,cdma_dat2mcif_rd_req_pd
  ,cdma_dat2cvif_rd_req_valid
  ,cdma_dat2cvif_rd_req_ready
  ,cdma_dat2cvif_rd_req_pd
  ,mcif2cdma_dat_rd_rsp_valid
  ,mcif2cdma_dat_rd_rsp_ready
  ,mcif2cdma_dat_rd_rsp_pd
  ,cvif2cdma_dat_rd_rsp_valid
  ,cvif2cdma_dat_rd_rsp_ready
  ,cvif2cdma_dat_rd_rsp_pd
  ,mcif2dc_dat_rd_rsp_valid
  ,mcif2dc_dat_rd_rsp_ready
  ,mcif2dc_dat_rd_rsp_pd
  ,cvif2dc_dat_rd_rsp_valid
  ,cvif2dc_dat_rd_rsp_ready
  ,cvif2dc_dat_rd_rsp_pd
  ,mcif2wg_dat_rd_rsp_valid
  ,mcif2wg_dat_rd_rsp_ready
  ,mcif2wg_dat_rd_rsp_pd
  ,cvif2wg_dat_rd_rsp_valid
  ,cvif2wg_dat_rd_rsp_ready
  ,cvif2wg_dat_rd_rsp_pd
  ,mcif2img_dat_rd_rsp_valid
  ,mcif2img_dat_rd_rsp_ready
  ,mcif2img_dat_rd_rsp_pd
  ,cvif2img_dat_rd_rsp_valid
  ,cvif2img_dat_rd_rsp_ready
```
**File:** vmod/nvdla/cdma/NV_NVDLA_cdma.v (L636-738)
```text
NV_NVDLA_CDMA_slcg u_slcg_dc (
   .dla_clk_ovr_on_sync           (dla_clk_ovr_on_sync)             //|< i
  ,.global_clk_ovr_on_sync        (global_clk_ovr_on_sync)          //|< i
  ,.nvdla_core_clk                (nvdla_core_clk)                  //|< i
  ,.nvdla_core_rstn               (nvdla_core_rstn)                 //|< i
  ,.slcg_en_src_0                 (slcg_op_en[1])                   //|< w
  ,.slcg_en_src_1                 (slcg_wg_gate_dc)                 //|< w
  ,.slcg_en_src_2                 (slcg_img_gate_dc)                //|< w
  ,.tmc2slcg_disable_clock_gating (tmc2slcg_disable_clock_gating)   //|< i
  ,.nvdla_core_gated_clk          (nvdla_op_gated_clk_dc)           //|> w
  );

//==========================================================
// Winograd convolution DMA
//==========================================================
NV_NVDLA_CDMA_wg u_wg (
   .nvdla_core_clk                (nvdla_op_gated_clk_wg)           //|< w
  ,.nvdla_core_rstn               (nvdla_core_rstn)                 //|< i
  ,.pwrbus_ram_pd                 (pwrbus_ram_pd[31:0])             //|< i
  ,.wg_dat2mcif_rd_req_valid      (wg_dat2mcif_rd_req_valid)        //|> w
  ,.wg_dat2mcif_rd_req_ready      (wg_dat2mcif_rd_req_ready)        //|< w
  ,.wg_dat2mcif_rd_req_pd         (wg_dat2mcif_rd_req_pd[78:0])     //|> w
  ,.wg_dat2cvif_rd_req_valid      (wg_dat2cvif_rd_req_valid)        //|> w
  ,.wg_dat2cvif_rd_req_ready      (wg_dat2cvif_rd_req_ready)        //|< w
  ,.wg_dat2cvif_rd_req_pd         (wg_dat2cvif_rd_req_pd[78:0])     //|> w
  ,.mcif2wg_dat_rd_rsp_valid      (mcif2wg_dat_rd_rsp_valid)        //|< w
  ,.mcif2wg_dat_rd_rsp_ready      (mcif2wg_dat_rd_rsp_ready)        //|> w
  ,.mcif2wg_dat_rd_rsp_pd         (mcif2wg_dat_rd_rsp_pd[513:0])    //|< w
  ,.cvif2wg_dat_rd_rsp_valid      (cvif2wg_dat_rd_rsp_valid)        //|< w
  ,.cvif2wg_dat_rd_rsp_ready      (cvif2wg_dat_rd_rsp_ready)        //|> w
  ,.cvif2wg_dat_rd_rsp_pd         (cvif2wg_dat_rd_rsp_pd[513:0])    //|< w
  ,.wg2cvt_dat_wr_en              (wg2cvt_dat_wr_en)                //|> w
  ,.wg2cvt_dat_wr_addr            (wg2cvt_dat_wr_addr[11:0])        //|> w
  ,.wg2cvt_dat_wr_hsel            (wg2cvt_dat_wr_hsel)              //|> w
  ,.wg2cvt_dat_wr_data            (wg2cvt_dat_wr_data[511:0])       //|> w
  ,.wg2cvt_dat_wr_info_pd         (wg2cvt_dat_wr_info_pd[11:0])     //|> w
  ,.status2dma_fsm_switch         (status2dma_fsm_switch)           //|< w
  ,.wg2status_state               (wg2status_state[1:0])            //|> w
  ,.wg2status_dat_updt            (wg2status_dat_updt)              //|> w
  ,.wg2status_dat_entries         (wg2status_dat_entries[11:0])     //|> w
  ,.wg2status_dat_slices          (wg2status_dat_slices[11:0])      //|> w
  ,.status2dma_valid_slices       (status2dma_valid_slices[11:0])   //|< w
  ,.status2dma_free_entries       (status2dma_free_entries[11:0])   //|< w
  ,.status2dma_wr_idx             (status2dma_wr_idx[11:0])         //|< w
  ,.wg2sbuf_p0_wr_en              (wg2sbuf_p0_wr_en)                //|> w
  ,.wg2sbuf_p0_wr_addr            (wg2sbuf_p0_wr_addr[7:0])         //|> w
  ,.wg2sbuf_p0_wr_data            (wg2sbuf_p0_wr_data[255:0])       //|> w
  ,.wg2sbuf_p1_wr_en              (wg2sbuf_p1_wr_en)                //|> w
  ,.wg2sbuf_p1_wr_addr            (wg2sbuf_p1_wr_addr[7:0])         //|> w
  ,.wg2sbuf_p1_wr_data            (wg2sbuf_p1_wr_data[255:0])       //|> w
  ,.wg2sbuf_p0_rd_en              (wg2sbuf_p0_rd_en)                //|> w
  ,.wg2sbuf_p0_rd_addr            (wg2sbuf_p0_rd_addr[7:0])         //|> w
  ,.wg2sbuf_p0_rd_data            (wg2sbuf_p0_rd_data[255:0])       //|< w
  ,.wg2sbuf_p1_rd_en              (wg2sbuf_p1_rd_en)                //|> w
  ,.wg2sbuf_p1_rd_addr            (wg2sbuf_p1_rd_addr[7:0])         //|> w
  ,.wg2sbuf_p1_rd_data            (wg2sbuf_p1_rd_data[255:0])       //|< w
  ,.sc2cdma_dat_pending_req       (sc2cdma_dat_pending_req)         //|< i
  ,.nvdla_core_ng_clk             (nvdla_core_clk)                  //|< i
  ,.reg2dp_op_en                  (reg2dp_op_en[0])                 //|< w
  ,.reg2dp_conv_mode              (reg2dp_conv_mode[0])             //|< w
  ,.reg2dp_in_precision           (reg2dp_in_precision[1:0])        //|< w
  ,.reg2dp_proc_precision         (reg2dp_proc_precision[1:0])      //|< w
  ,.reg2dp_data_reuse             (reg2dp_data_reuse[0])            //|< w
  ,.reg2dp_skip_data_rls          (reg2dp_skip_data_rls[0])         //|< w
  ,.reg2dp_datain_format          (reg2dp_datain_format[0])         //|< w
  ,.reg2dp_datain_width           (reg2dp_datain_width[12:0])       //|< w
  ,.reg2dp_datain_height          (reg2dp_datain_height[12:0])      //|< w
  ,.reg2dp_datain_width_ext       (reg2dp_datain_width_ext[12:0])   //|< w
  ,.reg2dp_datain_height_ext      (reg2dp_datain_height_ext[12:0])  //|< w
  ,.reg2dp_datain_channel         (reg2dp_datain_channel[12:0])     //|< w
  ,.reg2dp_datain_ram_type        (reg2dp_datain_ram_type[0])       //|< w
  ,.reg2dp_datain_addr_high_0     (reg2dp_datain_addr_high_0[31:0]) //|< w
  ,.reg2dp_datain_addr_low_0      (reg2dp_datain_addr_low_0[26:0])  //|< w
  ,.reg2dp_line_stride            (reg2dp_line_stride[26:0])        //|< w
  ,.reg2dp_surf_stride            (reg2dp_surf_stride[26:0])        //|< w
  ,.reg2dp_entries                (reg2dp_entries[11:0])            //|< w
  ,.reg2dp_conv_x_stride          (reg2dp_conv_x_stride[2:0])       //|< w
  ,.reg2dp_conv_y_stride          (reg2dp_conv_y_stride[2:0])       //|< w
  ,.reg2dp_pad_left               (reg2dp_pad_left[4:0])            //|< w
  ,.reg2dp_pad_right              (reg2dp_pad_right[5:0])           //|< w
  ,.reg2dp_pad_top                (reg2dp_pad_top[4:0])             //|< w
  ,.reg2dp_pad_bottom             (reg2dp_pad_bottom[5:0])          //|< w
  ,.reg2dp_pad_value              (reg2dp_pad_value[15:0])          //|< w
  ,.reg2dp_data_bank              (reg2dp_data_bank[3:0])           //|< w
  ,.reg2dp_dma_en                 (reg2dp_dma_en[0])                //|< w
  ,.slcg_wg_gate_dc               (slcg_wg_gate_dc)                 //|> w
  ,.slcg_wg_gate_img              (slcg_wg_gate_img)                //|> w
  ,.dp2reg_wg_rd_stall            (dp2reg_wg_rd_stall[31:0])        //|> w
  ,.dp2reg_wg_rd_latency          (dp2reg_wg_rd_latency[31:0])      //|> w
  );

//-------------- SLCG for WG DMA --------------//
NV_NVDLA_CDMA_slcg u_slcg_wg (
   .dla_clk_ovr_on_sync           (dla_clk_ovr_on_sync)             //|< i
  ,.global_clk_ovr_on_sync        (global_clk_ovr_on_sync)          //|< i
  ,.nvdla_core_clk                (nvdla_core_clk)                  //|< i
  ,.nvdla_core_rstn               (nvdla_core_rstn)                 //|< i
  ,.slcg_en_src_0                 (slcg_op_en[2])                   //|< w
  ,.slcg_en_src_1                 (slcg_dc_gate_wg)                 //|< w
  ,.slcg_en_src_2                 (slcg_img_gate_wg)                //|< w
  ,.tmc2slcg_disable_clock_gating (tmc2slcg_disable_clock_gating)   //|< i
  ,.nvdla_core_gated_clk          (nvdla_op_gated_clk_wg)           //|> w
  );
```
**File:** cmod/csc/NV_NVDLA_csc.cpp (L428-449)
```cpp
        // Evaluation operation mode
        if (NVDLA_CSC_D_MISC_CFG_0_CONV_MODE_DIRECT == csc_conv_mode_) {
            // Direct convolution
            if ( NVDLA_CSC_D_DATAIN_FORMAT_0_DATAIN_FORMAT_FEATURE == csc_datain_format_) {
                if (0 == csc_batches_) {
                    weight_load_operation_mode = WEIGHT_LOAD_MODE_DIRECT_CONV_NONE_BATCH;
                } else {
                    weight_load_operation_mode = WEIGHT_LOAD_MODE_DIRECT_CONV;
                }
            }
            else if ( NVDLA_CSC_D_DATAIN_FORMAT_0_DATAIN_FORMAT_PIXEL == csc_datain_format_) {
                weight_load_operation_mode = WEIGHT_LOAD_MODE_IMAGE_CONV_NONE_BATCH;
            }
            else {  //HOG
                FAIL(("NV_NVDLA_csc::DataLoadSequenceThread, unsupport datain_format, csc_datain_format_ is 0x%X", csc_datain_format_));
            }
        } else if (NVDLA_CSC_D_MISC_CFG_0_CONV_MODE_WINOGRAD == csc_conv_mode_) {
            // Winograd convolution
            weight_load_operation_mode = WEIGHT_LOAD_MODE_WINOGRAD_CONV;
        } else {
            FAIL(("NV_NVDLA_csc::WeightLoadSequenceThread, unsupport mode, csc_conv_mode_ is 0x%X", csc_conv_mode_));
        }
```# NVDLA CDMA (Convolution DMA) — Architecture Deep Dive

## Table of Contents

1. [Overview](#1-overview)
2. [Key Constants](#2-key-constants)
3. [Constructor — FIFO Allocations](#3-constructor--fifo-allocations)
4. [Reset](#4-reset)
5. [Register Programming Model](#5-register-programming-model)
6. [CdmaConsumerThread — Main Orchestrator](#6-cdmaconsumerthread--main-orchestrator)
7. [The 10 SC_THREADs](#7-the-10-sc_threads)
    - [Thread 1: ActDataReadRequestSequenceThread](#thread-1-actdatareadrequestsequencethread)
    - [Thread 2: ActDataReadResponseSequenceThread](#thread-2-actdatareadresponsesequencethread)
    - [Thread 3: WeightReadRequestSequenceThread](#thread-3-weightreadrequestsequencethread)
    - [Thread 4: WeightReadResponseSequenceThread](#thread-4-weightreadresponsesequencethread)
    - [Thread 5: WGSReadRequestSequenceThread](#thread-5-wgsreadrequestsequencethread)
    - [Thread 6: WGSReadResponseSequenceThread](#thread-6-wgsreadresponsesequencethread)
    - [Thread 7: WMBReadRequestSequenceThread](#thread-7-wmbreadrequestsequencethread)
    - [Thread 8: WMBReadResponseSequenceThread](#thread-8-wmbreadresponsesequencethread)
    - [Thread 9: WtReadRequestThread](#thread-9-wtreadrequestthread)
    - [Thread 10: Cdma2ScUpdateThread](#thread-10-cdma2scupdatethread)
8. [Key Data Processing Functions](#8-key-data-processing-functions)
    - [DirectConvDataResponseSequencerCommon](#directconvdataresponsesequencercommon)
    - [DirectConvWeightResponseSequencerCommon](#directconvweightresponsesequencercommon)
    - [ImageConvDataResponseSequencerCommon](#imageconvdataresponsesequencercommon)
    - [WriteOneEntryToCbuf](#writeoneentrytocbuf)
    - [WinoConvDataResponseSequencerCommon](#winoconvdataresponsesequencercommon)
    - [process_one_element_16 / process_one_element_8](#process_one_element_16--process_one_element_8)
9. [Complete Data Flow](#9-complete-data-flow)
10. [CDMA_DC vs CDMA_WG vs CDMA_IMG](#10-cdma_dc-vs-cdma_wg-vs-cdma_img)
11. [CBUF Three-Way Pointer Management](#11-cbuf-three-way-pointer-management)
12. [Data Reuse Optimization](#12-data-reuse-optimization)

---

## 1. Overview

The **CDMA (Convolution DMA)** is the data feeder for the entire NVDLA convolution pipeline. It fetches **activation data**, **weights**, **WMB** (weight mean/bias), and **WGS** (weight group size for compression) from external DRAM via DMA, processes/rearranges them, and writes them into the **CBUF** (Convolution Buffer, an on-chip SRAM). The CSC (Convolution Stream Controller) then reads from CBUF and feeds the MAC array.

Source: `cmod/cdma/NV_NVDLA_cdma.cpp` in the NVDLA hardware reference model.

### Three Operating Modes

| Mode | Input Source | Typical Use |
|------|-------------|-------------|
| **CDMA_DC** (Direct Convolution) | Feature maps (NC1HWC0 format) | All non-Winograd conv layers |
| **CDMA_WG** (Winograd) | Feature maps, 4×4×4 sub-cubes | 3×3 kernels with F(4,3) transform |
| **CDMA_IMG** (Image/Pixel) | Camera/raw pixels (RGB/YUV/RAW) | First layer only |

---

## 2. Key Constants

From `NV_NVDLA_cdma.h`:

| Constant | Value | Meaning |
|---|---|---|
| `ATOM_CUBE_SIZE` | 32 | Smallest DMA transaction unit (bytes) |
| `CBUF_BANK_NUM` | 16 | Number of CBUF banks |
| `CBUF_ENTRY_PER_BANK` | 256 | Entries per bank |
| `CBUF_ENTRY_SIZE` | 128 | Bytes per CBUF entry |
| `CBUF_HALF_ENTRY_SIZE` | 64 | Half-entry (one write transaction) |
| `MAX_MEM_TRANSACTION_SIZE` | varies | Max bytes per DMA burst |
| `MEM_BUSWIDTH_IN_BIT` | 512 | Bus width (= 64 bytes) |
| `WGS_FIFO_DEPTH` | 32 | Depth of WGS FIFO in RTL (32×4Bytes) |
| `ATOM_PER_DMA_TRANSACTION` | varies | Atoms per single DMA transaction |
| `ELEMENT_PER_ATOM_INT8` | 32 | 8-bit elements per 32B atom |
| `ELEMENT_PER_ATOM_INT16` | 16 | 16-bit elements per 32B atom |
| `ELEMENT_PER_ATOM_FP16` | 16 | FP16 elements per 32B atom |

---

## 3. Constructor — FIFO Allocations

### Data path FIFOs (DRAM → CBUF)

| FIFO | Type / Size | Purpose |
|---|---|---|
| `act_data_read_rsp_fifo_` | `uint8_t*`, `MAX*10/ATOM_CUBE` | Activation data response from DMA |
| `mean_data_read_rsp_fifo_` | `uint8_t*`, `MAX/ATOM_CUBE` | Mean data response (for image load) |
| `weight_read_rsp_fifo_` | `uint8_t*`, `MAX/ATOM_CUBE` | Weight data response from DMA |
| `wmb_read_rsp_fifo_` | `uint8_t*`, `MAX/ATOM_CUBE` | WMB data response |
| `wgs_read_rsp_fifo_` | `uint8_t*`, `MAX/ATOM_CUBE` | WGS data response |

### Weight request arbitration FIFOs

| FIFO | Type/Size | Purpose |
|---|---|---|
| `wt_dma_rtl_source_id_fifo_` | `int`, 256 | Source ID indicating weight/WMB/WGS request type |
| `cdma_wt_req_fifo_` | `cdma_wt_req_t*`, 1024 | Weight DMA request queue |
| `cdma_wmb_req_fifo_` | `cdma_wt_req_t*`, 1024 | WMB DMA request queue |
| `cdma_wgs_req_fifo_` | `cdma_wt_req_t*`, 1024 | WGS DMA request queue |
| `cdma_wt_info_fifo_` | `cdma_wt_info_t*`, 1024 | Request metadata (source + payload size) |

### SC (System Controller) update FIFOs

| FIFO | Type/Size | Purpose |
|---|---|---|
| `wmb2sc_up_fifo_` | `int32_t`, 1024 | WMB entry count to SC |
| `wt2sc_up_kernel_fifo_` | `int32_t`, 1024 | Kernel count to SC |
| `wt2sc_up_entry_fifo_` | `int32_t`, 1024 | Weight entry count to SC |

### Synchronization FIFOs

| FIFO | Size | Purpose |
|---|---|---|
| `cdma_data_fetch_done_fifo_` | 1 | Signal: data fetch complete |
| `cdma_weight_fetch_done_fifo_` | 1 | Signal: weight fetch complete |
| `cdma_wgs_fetch_done_fifo_` | 1 | Signal: WGS fetch complete |
| `cdma_wmb_fetch_done_fifo_` | 1 | Signal: WMB fetch complete |
| `cdma_mean_fetch_done_fifo_` | 1 | Signal: mean data fetch complete |
| `wino_req2resp_sync_fifo_` | 480 | Winograd request/response sync (1920/4) |
| `cdma_wgs2wt_sync_fifo_` | 1 | WGS-to-weight sync for SCVE |
| `wgs2wt_sync_fifo_` | 64 | WGS-to-weight sync for headless NTB |
| `cdma_wmb_fetch_done2wt_fifo_` | 1 | WMB done signal for weight done interrupt |
| `wino_fetch_data_fifo_[4]` | `2*READ_WINO_BUF_WIDTH` | Winograd fetch FIFOs (4 banks, one per row) |

### Payload pointers

| Pointer | Type | Purpose |
|---|---|---|
| `dma_act_rd_req_payload_` | `nvdla_dma_rd_req_t` | DMA read request for activation data |
| `dma_wt_rd_req_payload_` | `nvdla_dma_rd_req_t` | DMA read request for weight data |
| `dma_wgs_rd_req_payload_` | `nvdla_dma_rd_req_t` | DMA read request for WGS |
| `dma_wmb_rd_req_payload_` | `nvdla_dma_rd_req_t` | DMA read request for WMB |
| `cdma2cbuf_data_payload_` | `nvdla_ram_wr_port_WADDR_12_WDATA_512_BE_1_t` | CBUF write payload |
| `data2sc_data_update_payload_` | `nvdla_dat_info_update_t` | SC data info update |
| `data2sc_weight_update_payload_` | `nvdla_wt_info_update_t` | SC weight info update |

---

## 4. Reset

```
Reset() procedure:
  CdmaRegReset()                       // Clear all registers
  CdmaUpdateStatusRegister(CBUF_FLUSH_STATUS, 0, 0x1)  // Mark CBUF as flushed

  // CBUF pointer tracking - reset all
  data_entry_idx_working_     = 0      // CSC read position
  data_entry_idx_free_        = 0      // CDMA write position
  data_entry_idx_planed_      = -1     // Next planned write

  weight_entry_idx_working_   = 0
  weight_entry_idx_free_      = 0
  weight_entry_idx_planed_    = -1
  weight_byte_idx_planed_     = 0

  wmb_entry_idx_working_      = 0
  wmb_entry_idx_free_         = 0
  wmb_entry_idx_planed_       = -1
  wmb_byte_idx_planed_        = 0

  // Bank tracking for bank-aligned access
  cdma_req_prev_data_bank_        = CBUF_BANK_NUM + 1   // Invalid sentinel
  cdma_resp_prev_data_bank_       = CBUF_BANK_NUM + 1
  wt_req_cdma_prev_weight_bank_   = CBUF_BANK_NUM + 1

  // Reuse flags
  dat_first_layer     = true
  wgs_first_layer     = true
  wmb_first_layer     = true
  wt_first_layer      = true
```

---

## 5. Register Programming Model

CDMA behavior is controlled by registers accessed via `csb2cdma_req_b_transport()`. Key registers:

| Register | Fields | Effect |
|---|---|---|
| `D_OP_ENABLE` | `OP_EN` | Enables a register group (0 or 1) |
| `D_MISC_CFG` | `CONV_MODE`, `IN_PRECISION`, `PROC_PRECISION`, `DATA_REUSE`, `WEIGHT_REUSE`, `SKIP_DATA_RLS`, `SKIP_WEIGHT_RLS` | Sets convolution mode, precision, reuse |
| `D_DATAIN_FORMAT` | `DATAIN_FORMAT` (FEATURE=0, PIXEL=1) | Determines CDMA_DC/WG vs CDMA_IMG |
| `D_DATA_FORMAT` | `DATA_FORMAT`, `PIXEL_FORMAT`, `PIXEL_MAPPING` | Activation data layout, pixel type |
| `D_WEIGHT_FORMAT` | `WEIGHT_FORMAT` (compressed/uncompressed) | Weight compression mode |
| `D_DATA_CUBE_WIDTH` | width | Input cube width in elements |
| `D_DATA_CUBE_HEIGHT` | height | Input cube height in elements |
| `D_DATA_CUBE_CHANNEL` | channel | Input cube channel count |
| `D_LINE_STRIDE` | stride | Byte stride between lines |
| `D_SURFACE_STRIDE` | stride | Byte stride between surface planes |
| `D_DATA_BASE_ADDR_LOW/HIGH` | 64-bit address | DRAM base for activation data |
| `D_WEIGHT_BASE_ADDR_LOW/HIGH` | 64-bit address | DRAM base for weights |
| `D_MEAN_BASE_ADDR_LOW/HIGH` | 64-bit address | DRAM base for mean data (IMAGE mode) |
| `D_DATA_BANK` | bank ID | CBUF start bank for activation data |
| `D_WEIGHT_BANK` | bank ID | CBUF start bank for weights |
| `D_ENTRY_PER_SLICE` | entries | CBUF entries consumed per slice |
| `D_CVT_CONFIG` | `cvt_en`, `cvt_offset`, `cvt_scale`, `cvt_truncate` | Pixel→feature conversion params |
| `D_MEAN_FORMAT` | format | Per-channel vs global mean selection |
| `D_MEAN_GLOBAL_0..3` | values | Global mean values for R/G/B/A |

**Dual register group** (`group_0` and `group_1`) enables ping-pong: one group can be programmed while the other is executing, enabling continuous layer processing.

---

## 6. CdmaConsumerThread — Main Orchestrator

```
CdmaConsumerThread():

  while (true):
    ── Register Group 0 ──────────────────────────
    wait(event_cdma_reg_group_0_operation_enable)  // SW set D_OP_ENABLE.OP_EN=1
    CdmaUpdateWorkingStatus(0, 1)                   // Set working=1 (HW busy)
    CdmaUpdateVariables(register_group_0)           // Read registers → internal state
    CdmaHardwareLayerExecutionTrigger()             // Start all fetch threads

    // Wait for all 4 done signals:
    |  cdma_data_fetch_done_fifo_.read()            // Activation fetch complete
    |  cdma_wgs_fetch_done_fifo_.read()             // WGS fetch complete
    |  cdma_wmb_fetch_done_fifo_.read()             // WMB fetch complete
    |  cdma_weight_fetch_done_fifo_.read()          // Weight fetch complete
    ▼  (All 4 must complete before proceeding)

    CdmaUpdateStatRegisters(0, data_nan, weight_nan, data_inf, weight_inf)
    CdmaUpdateWorkingStatus(0, 0)                   // Set working=0 (HW idle)
    CdmaClearOpeartionEnable(register_group_0)      // Clear OP_ENABLE

    ── Register Group 1 ──────────────────────────
    wait(event_cdma_reg_group_1_operation_enable)
    ...same sequence for group 1...
```

---

## 7. The 10 SC_THREADs

### Thread 1: ActDataReadRequestSequenceThread

**Role**: Generate DMA read requests for activation data from DRAM.

```
ActDataReadRequestSequenceThread():

  while (true):
    wait(cdma_kickoff_)                    // Woken by CdmaHardwareLayerExecutionTrigger

    if (cdma_data_reuse_ && !dat_first_layer):
      // Layer identical to previous — no DMA needed
      cdma_data_fetch_done_fifo_.write(true)
      continue

    switch (cdma_datain_format_):
      case INPUT_DATA_FORMAT_DC:           // Feature map → Direct Conv or Winograd
        switch (cdma_conv_mode_):
          case DIRECT:
            DirectConvDataRequestSequencerCommon()
          case WINOGRAD:
            WinoConvDataRequestSequencerCommon()

      case INPUT_DATA_FORMAT_IMAGE:        // Pixel input
        ImageConvDataRequestSequencerCommon()

    cdma_data_fetch_done_fifo_.write(true)

    dat_first_layer = false
```

#### DirectConvDataRequestSequencerCommon

```
DirectConvDataRequestSequencerCommon():

  Calculates:
    atom_per_width   = ceil(cube_width  * bpe / ATOM_CUBE_SIZE)
    atom_per_channel = ceil(cube_channel * bpe / ATOM_CUBE_SIZE)

  For batch = 0..num_batches-1:
    For height = 0..cube_height-1:
      For surface = 0..cube_channel-1 (in atom_per_channel steps):
        addr = base_addr
             + batch * batch_stride
             + height * line_stride
             + surface * ATOM_CUBE_SIZE

        // Send atom-sized DMA read requests
        for atom_x = 0..atom_per_width-1:
          dma_act_rd_req_payload_.addr = addr + atom_x * ATOM_CUBE_SIZE
          SendActDmaReadRequest()    // → MCIF or CVIF
```

#### WinoConvDataRequestSequencerCommon

```
WinoConvDataRequestSequencerCommon():

  // Process 4 rows at a time (super_height)
  super_height_num = ceil(cube_height / (4 * conv_y_stride))

  for super_iter = 0..super_height_num-1:
    // Fetch 8 atoms (256B) per row = 4 rows × 8 atoms
    for row = 0..3:
      for atom_x = 0..atoms_per_row-1 (up to 8):
        addr = base_addr + (row * conv_y_stride + super_iter*4) * line_stride
                         + atom_x * ATOM_CUBE_SIZE
        dma_act_rd_req_payload_.addr = addr
        SendActDmaReadRequest()

    wino_req2resp_sync_fifo_.write(true)  // Sync: data ready for response thread
```

#### ImageConvDataRequestSequencerCommon

```
ImageConvDataRequestSequencerCommon():

  for row = 0..cube_height-1:
    // Interleaved format (RGB, etc.):
    addr = base_addr + row * line_stride

    // Planar format (YUV):
    addr_y  = y_base  + row * y_line_stride
    addr_uv = uv_base + row * uv_line_stride  // may be subsampled for NV12

    // Send DMA requests for entire row (split into ATOM chunks)
    for chunk = 0..bytes_per_row / ATOM_CUBE_SIZE:
      SendActDmaReadRequest(addr + chunk * ATOM_CUBE_SIZE)

    // Fetch separate mean data if needed
```

### Thread 2: ActDataReadResponseSequenceThread

**Role**: Process DMA responses → rearrange → write to CBUF.

```
ActDataReadResponseSequenceThread():

  while (true):
    wait(cdma_kickoff_)

    if (cdma_skip_data_rls_):
      // Data already in CBUF from previous layer (reuse)
      continue

    if (cdma_data_reuse_ && !dat_first_layer):
      // Reuse: just resend SC update
      dat_up_cdma2sc_b_transport(...)
      continue

    switch (cdma_act_data_mode_):       // Internal enum from mode detection
      case ACT_MODE_DIRECT_CONV_NONE_BATCH:
      case ACT_MODE_DIRECT_CONV_BATCH:
        DirectConvDataResponseSequencerCommon()

      case ACT_MODE_WINOGRAD_CONV:
        wait(wino_req2resp_sync_fifo_.read())   // Wait for request thread
        WinoConvDataResponseSequencerCommon()

      case ACT_MODE_DIRECT_PIXEL:
        ImageConvDataResponseSequencerCommon()
```

### Thread 3: WeightReadRequestSequenceThread

**Role**: Generate DMA read requests for weights.

```
WeightReadRequestSequenceThread():

  while (true):
    wait(cdma_kickoff_)

    if (cdma_weight_reuse_ && !wt_first_layer):
      cdma_weight_fetch_done_fifo_.write(true)
      continue

    // Determine weight mode from CONV_MODE and DATAIN_FORMAT
    // All modes (DC, WG, IMG) use DirectConvWeightRequestSequencerCommon()
    DirectConvWeightRequestSequencerCommon()

    wt_first_layer = false
```

### Thread 4: WeightReadResponseSequenceThread

**Role**: Process weight DMA responses → write to CBUF.

```
WeightReadResponseSequenceThread():

  while (true):
    wait(cdma_kickoff_)

    if (cdma_skip_weight_rls_):
      cdma_weight_fetch_done_fifo_.write(true)
      continue

    if (cdma_weight_reuse_ && !wt_first_layer):
      cdma_weight_fetch_done_fifo_.write(true)
      continue

    DirectConvWeightResponseSequencerCommon()

    cdma_weight_fetch_done_fifo_.write(true)
```

### Thread 5: WGSReadRequestSequenceThread

**Role**: Fetch Weight Group Size data (for compressed weights).

```
WGSReadRequestSequenceThread():

  while (true):
    wait(cdma_kickoff_)

    if (cdma_weight_format_ == COMPRESSED):
      if (!wgs_first_layer):  // Check reuse
        cdma_wgs_fetch_done_fifo_.write(true)
        continue
      total_wgs_bytes = calc_total_wgs_bytes()
      wgs_buffer_ = new uint8_t[total_wgs_bytes]
      ConvWGSRequestSequencerCommon()   // → DMA reads

    else if (cdma_wt_dma_arbiter_override_enable):
      cdma_wgs2wt_sync_fifo_.write(true)  // Signal sync even without WGS
```

### Thread 6: WGSReadResponseSequenceThread

**Role**: Process WGS DMA responses → store in `wgs_buffer_`.

```
WGSReadResponseSequenceThread():

  while (true):
    wait(cdma_kickoff_)

    if (cdma_weight_format_ == COMPRESSED):
      ConvWGSResponseSequencerCommon()
        // Read from wgs_read_rsp_fifo_
        // Store into wgs_buffer_[]

      cdma_wgs_fetch_done_fifo_.write(true)
      cdma_wgs2wt_sync_fifo_.write(true)  // Signal WGS ready for weight decoder
```

### Thread 7: WMBReadRequestSequenceThread

**Role**: Fetch Weight Mean Bias data (for compressed weights).

```
WMBReadRequestSequenceThread():

  while (true):
    wait(cdma_kickoff_)

    // Manage CBUF pointer triple for WMB
    wmb_entry_idx_free_    = WMB_BANK_BASE * CBUF_ENTRY_PER_BANK
    wmb_entry_idx_working_ = wmb_entry_idx_free_
    wmb_entry_idx_planed_  = wmb_entry_idx_free_ - 1

    if (cdma_weight_format_ == COMPRESSED):
      if (wmb_config_changed() && !wmb_first_layer):
        // Re-fetch needed due to config change
      ConvWMBRequestSequencerCommon()   // → DMA reads
```

### Thread 8: WMBReadResponseSequenceThread

**Role**: Process WMB DMA responses → write to CBUF.

```
WMBReadResponseSequenceThread():

  while (true):
    wait(cdma_kickoff_)

    if (cdma_weight_format_ == COMPRESSED):
      ConvWMBResponseSequencerCommon()
        // Read from wmb_read_rsp_fifo_
        // Write to CBUF (WMB_BANK)
        // wmb2sc_up_fifo_.write(entries)

      cdma_wmb_fetch_done_fifo_.write(true)
      cdma_wmb_fetch_done2wt_fifo_.write(true)
```

### Thread 9: WtReadRequestThread

**Role**: Arbiter — merges weight/WMB/WGS DMA requests into one stream.

```
WtReadRequestThread():

  while (true):
    source_id = wt_dma_rtl_source_id_fifo_.read()  // From RTL arbiter

    cdma_wt_info_t* info = new cdma_wt_info_t
    info->source = source_id

    switch (source_id):
      case CDMA_WEIGHT_DATA:
        req = cdma_wt_req_fifo_->read()
        info->size = req->size
        break
      case CDMA_WMB_DATA:
        req = cdma_wmb_req_fifo_->read()
        info->size = req->size
        break
      case CDMA_WGS_DATA:
        req = cdma_wgs_req_fifo_->read()
        info->size = req->size
        break

    cdma_wt_info_fifo_->write(info)

    // Route to MCIF or CVIF based on cdma_weight_ram_type_
    if (ram_type == MC):
      dma_wt_rd_req_payload_->addr = req->addr
      mcif2cdma_wt_rd_req_b_transport(...)   // → MCIF (DDR)
    else:
      cvif2cdma_wt_rd_req_b_transport(...)   // → CVIF (SRAM)
```

### Thread 10: Cdma2ScUpdateThread

**Role**: Send weight metadata updates to the System Controller (SC).

```
Cdma2ScUpdateThread():

  while (true):
    wt_kernels = wt2sc_up_kernel_fifo_.read()    // Kernel count
    wt_entries = wt2sc_up_entry_fifo_.read()      // CBUF entry count

    wmb_entries = 0
    if (cdma_weight_format_ == COMPRESSED):
      wmb_entries = wmb2sc_up_fifo_.read()        // WMB entry count

    // Package and send to SC
    wt_up_cdma2sc_payload.kernels = wt_kernels
    wt_up_cdma2sc_payload.entries = wt_entries
    wt_up_cdma2sc_payload.wmb_entries = wmb_entries

    wt_up_cdma2sc_b_transport(wt_up_cdma2sc_payload, delay)
```

---

## 8. Key Data Processing Functions

### DirectConvDataResponseSequencerCommon

Processes feature map data for direct convolution.

```
DirectConvDataResponseSequencerCommon():

  element_per_atom = (IN_PRECISION == INT8)  ? 32 : 16
  atom_size        = (IN_PRECISION == INT8)  ? ATOM_SIZE_INT8  : ATOM_SIZE_INT16
  element_size     = (IN_PRECISION == INT8)  ? 1 : 2

  atom_per_channel = ceil(cube_channel * element_size / ATOM_CUBE_SIZE)
  cbuf_entry_per_slice = ceil(cube_channel * element_size * cube_width / CBUF_ENTRY_SIZE)

  for each height step (slice):
    for each surface (channel group):
      for each element in width × channel:
        data = act_data_read_rsp_fifo_.read()    // 32B atom

        if (IN_PRECISION == FP16):
          for each of 16 elements in atom:
            if is_nan(element): data_nan_num_perlayer_++
            if is_inf(element): data_inf_num_perlayer_++

        if (cvt_en):
          // Apply offset, scale, truncate
          data = hls_convertor(data, cvt_offset, cvt_scale, cvt_truncate)

        // NaN flush: replace NaN with 0
        if (cvt_nan_flush_en && is_nan(data)):
          data = 0

        // Pack into 128B CBUF entry buffer
        cbuf_buf[cbuf_offset++] = data

        // When 128B full, write to CBUF
        if (cbuf_offset == CBUF_ENTRY_SIZE):
          cdma2cbuf_data_payload_.addr = cbuf_entry_addr
          cdma2cbuf_data_payload_.data = cbuf_buf[0..63]    // Left half
          cdma2cbuf_data_payload_.hsel = 0
          cdma2buf_dat_wr_b_transport(payload, delay)

          cdma2cbuf_data_payload_.data = cbuf_buf[64..127]  // Right half
          cdma2cbuf_data_payload_.hsel = 1
          cdma2buf_dat_wr_b_transport(payload, delay)

          cbuf_entry_addr++
          cbuf_offset = 0

  // Notify SC about available data
  dat_up_cdma2sc_b_transport(entries_per_slice, num_slices, ...)
```

### DirectConvWeightResponseSequencerCommon

Processes weight data for all convolution modes.

```
DirectConvWeightResponseSequencerCommon():

  kernel_per_group = (WEIGHT_FORMAT == INT8)  ? KERNEL_PER_GROUP_INT8
                    : (WEIGHT_FORMAT == INT16) ? KERNEL_PER_GROUP_INT16
                    :                            KERNEL_PER_GROUP_FP16

  for each kernel group:
    for each entry:
      // Read weight data from DMA response FIFO
      weight_data = weight_read_rsp_fifo_.read()

      // Copy into CBUF write payload
      cdma2buf_wt_wr_payload.data = weight_data
      cdma2buf_wt_wr_payload.addr = weight_entry_addr

      // Write to CBUF
      cdma2buf_wt_wr_b_transport(payload, delay)

      // For FP16: scan for NaN/Inf
      if (IN_PRECISION == FP16):
        for each of 16 elements:
          if is_nan(element): weight_nan_num_perlayer_++
          if is_inf(element): weight_inf_num_perlayer_++

      // Track CBUF entry address (bank management)
      weight_entry_addr = advance_cbuf_addr(weight_entry_addr)

    // After each kernel group: update SC
    wt2sc_up_kernel_fifo_.write(kernels_in_group)
    wt2sc_up_entry_fifo_.write(entries_for_group)
```

### ImageConvDataResponseSequencerCommon

Most complex sequencer — handles camera input (RGB/YUV/RAW) with pixel format conversion, mean subtraction, scaling, truncation, and padding.

```
ImageConvDataResponseSequencerCommon():

  bytes_per_pixel = lookup[pixel_format].bpp / 8
  element_num     = lookup[pixel_format].elements
  planar_num      = lookup[pixel_format].planes

  for each row in cube_height + pad_top + pad_bottom:
    pad_buf_offset = 0

    // 1. Left padding
    for p = 0..pad_left-1:
      for e = 0..element_num-1:
        pad_buffer[pad_buf_offset++] = pad_value

    // 2. Main pixel row (from DMA)
    for x = 0..cube_width-1:
      pixel_data = act_data_read_rsp_fifo_.read()

      // Parse pixel based on format (e.g., RGB unpacking)
      switch pixel_format:
        case T_R8G8B8A8:
          // RGBA interleaved, 4 bytes per pixel
          elements[0] = pixel_data[0]  // R
          elements[1] = pixel_data[1]  // G
          elements[2] = pixel_data[2]  // B
          elements[3] = pixel_data[3]  // A
          break
        case Y8___U8V8_N444:
          // Planar Y + interleaved UV
          elements[0] = y_plane[row * y_stride + x]
          elements[1] = uv_plane[row * uv_stride + x/2 * 2]
          elements[2] = uv_plane[row * uv_stride + x/2 * 2 + 1]
          break
        // ... other formats ...

      // 3. Per-element processing
      for e = 0..element_num-1:
        if (cvt_en):
          // Mean subtraction, scaling, truncation
          cvt_mean = get_mean(e)   // From MEAN_FORMAT registers
          converted = hls_convertor(elements[e], cvt_mean, cvt_scale, cvt_truncate)

          if (convert_16to8):
            pad_buffer[pad_buf_offset++] = converted & 0xFF
          else:
            pad_buffer[pad_buf_offset++] = converted & 0xFFFF
        else:
          pad_buffer[pad_buf_offset++] = pixel & clip_mask

    // 4. Right padding
    for p = 0..pad_right-1:
      for e = 0..element_num-1:
        pad_buffer[pad_buf_offset++] = pad_value

    // 5. Write filled CBUF entries
    for entry_start = 0; entry_start < pad_buf_offset; entry_start += CBUF_ENTRY_SIZE:
      WriteOneEntryToCbuf(&pad_buffer[entry_start], cbuf_entry_addr++)
      dat_up_cdma2sc_b_transport(entries_per_slice, 1, ...)
```

### WriteOneEntryToCbuf

Writes a 128-byte CBUF entry as two 64-byte halves.

```
WriteOneEntryToCbuf(uint8_t* read_data_ptr, int cbuf_entry_addr):

  // Left half (bytes 0-63, hsel=0)
  cdma2cbuf_data_payload_->addr  = cbuf_entry_addr
  cdma2cbuf_data_payload_->hsel  = 0
  memcpy(cdma2cbuf_data_payload_->data, read_data_ptr, 64)
  cdma2buf_dat_wr_b_transport(cdma2cbuf_data_payload_, b_transport_delay_)

  // Right half (bytes 64-127, hsel=1)
  cdma2cbuf_data_payload_->hsel  = 1
  memcpy(cdma2cbuf_data_payload_->data, read_data_ptr + 64, 64)
  cdma2buf_dat_wr_b_transport(cdma2cbuf_data_payload_, b_transport_delay_)
```

### WinoConvDataResponseSequencerCommon

Processes Winograd data into 4×4×4 sub-cubes.

```
WinoConvDataResponseSequencerCommon():

  // Each super_height processes 4 rows
  for each super_height_iter:
    // Wait for all 4 row FIFOs to have data
    wait(wino_req2resp_sync_fifo_.read())

    // Read 4 rows from 4 bank FIFOs
    for row = 0..3:
      row_data[row] = wino_fetch_data_fifo_[row].read()   // 8 atoms = 256B

    // Apply per-row left/right padding
    for row = 0..3:
      pad_row(row_data[row], pad_left, pad_right, pad_value)

    // Compose 4×4×4 sub-cubes
    // Each sub-cube = 4 rows × 4 col-atoms × 4 ch-atoms
    for x_tile = 0..width_in_subcubes-1:
      for c_tile = 0..channel_in_subcubes-1:
        // Build one 128B CBUF entry
        // Layout:
        //   Left half (64B): rows 0-3, cols 0-1, 4 channels
        //     row0: [col0-ch0..3] [col1-ch0..3]  = 8 × 2B (FP16) = 16B
        //     row1: [col0-ch0..3] [col1-ch0..3]  = 16B
        //     row2: [col0-ch0..3] [col1-ch0..3]  = 16B
        //     row3: [col0-ch0..3] [col1-ch0..3]  = 16B
        //     Total: 64B ✓
        //   Right half (64B): rows 0-3, cols 2-3, 4 channels (same pattern)

        cbuf_entry_addr = calc_cbuf_addr(x_tile, c_tile, super_iter)

        // Write left half
        cdma2cbuf_data_payload_->addr = cbuf_entry_addr
        cdma2cbuf_data_payload_->hsel = 0
        compose_left_half(row_data, x_tile, c_tile, cdma2cbuf_data_payload_->data)
        cdma2buf_dat_wr_b_transport(payload, delay)

        // Write right half
        cdma2cbuf_data_payload_->hsel = 1
        compose_right_half(row_data, x_tile, c_tile, cdma2cbuf_data_payload_->data)
        cdma2buf_dat_wr_b_transport(payload, delay)

    // Notify SC after each super_height iteration
    dat_up_cdma2sc_b_transport(entries_per_slice, 1, ...)
```

### process_one_element_16 / process_one_element_8

Perform pixel-level format conversion.

```
process_one_element_16(int16_t pixel_16):

  if (FP16 input):
    pixel = pixel_16  // Already FP16
  else if (pixel_sign_override_en):
    pixel = sign_extend(pixel_16, 16)   // Treat as signed
  else:
    pixel = pixel_16                     // Unsaturated

  if (cvt_en):
    // Mean subtraction + scaling + truncation
    // cvt_mean, cvt_scale, cvt_truncate from registers
    convert_result = hls_convertor(pixel, cvt_mean, cvt_scale, cvt_truncate)
  else:
    convert_result = pixel & clip_mask

  if (convert_16to8):
    pad_buffer_8[offset++] = convert_result & 0xFF
  else:
    pad_buffer_16[offset++] = convert_result & 0xFFFF


process_one_element_8(int8_t pixel_8):

  if (pixel_sign_override_en):
    pixel = sign_extend(pixel_8, 8)
  else:
    pixel = pixel_8

  if (cvt_en):
    convert_result = hls_convertor(pixel, cvt_mean, cvt_scale, cvt_truncate)
  else:
    convert_result = pixel

  if (convert_8to16):
    pad_buffer_16[offset++] = convert_result & 0xFFFF
  else:
    pad_buffer_8[offset++] = convert_result & 0xFF
```

---

## 9. Complete Data Flow

```
                    DRAM (DDR)
                    /    |    \
                   /     |     \
         DMA Read Req   DMA Read Req   DMA Read Req
         (Activation)   (Weights)      (WMB/WGS)
              |              |              |
    ActDataReadReq    WeightReadReq    WMB/WGS Read Req
    SequenceThread    SequenceThread   SequenceThreads
              |              |              |
              v              v              v
       MCIF / CVIF  ->  DMA Controller  ->  MCIF / CVIF
              |              |              |
              v              v              v
    ActDataReadResp    WeightReadResp    WMB/WGS Read Resp
    SequenceThread    SequenceThread    SequenceThreads
       (act_data_       (weight_         (wmb_/wgs_
        read_rsp_fifo_)  read_rsp_fifo_)  read_rsp_fifo_)
              |              |              |
              v              v              v
    ┌──────────────────────────────────────────────┐
    │          Data Rearrangement & Conversion      │
    │  (DirectConvRespSeq / ImageConvRespSeq /     │
    │   WinoConvRespSeq / process_one_element_*)  │
    └──────────────┬───────────────────────────────┘
                   | cdma2cbuf_data_payload_
                   v
    ┌──────────────────────────────────────────────┐
    │              CBUF (on-chip SRAM)             │
    │  Bank 0..15, 256 entries/bank, 128B/entry   │
    │  Data banks | Weight banks | WMB bank        │
    └──────────────┬───────────────────────────────┘
                   | dat_up_cdma2sc_b_transport()
                   v
    ┌──────────────────────────────────────────────┐
    │          SC (System Controller)              │
    │  Tracks free/used CBUF entries per slice     │
    │  Notifies CSC when data ready                │
    └──────────────┬───────────────────────────────┘
                   | sc2cdma_dat_up_b_transport() (consumed)
                   v
    ┌──────────────────────────────────────────────┐
    │    CSC (Convolution Stream Controller)       │
    │  Reads CBUF → feeds MAC array                │
    └──────────────────────────────────────────────┘
```

---

## 10. CDMA_DC vs CDMA_WG vs CDMA_IMG

### Mode Selection

```
D_MISC_CFG.CONV_MODE    D_DATAIN_FORMAT.DATAIN_FORMAT     →  Mode
─────────────────────    ─────────────────────────────────      ────
DIRECT (0)              FEATURE (0)                            → CDMA_DC
WINOGRAD (1)            FEATURE (0)                            → CDMA_WG
DIRECT (0)              PIXEL (1)                              → CDMA_IMG
```

### Comparison Table

| Aspect | CDMA_DC | CDMA_WG | CDMA_IMG |
|---|---|---|---|
| **Input source** | Feature maps (NC1HWC0) | Feature maps (NC1HWC0) | Camera/raw pixels |
| **Data unit** | ATOM (32B) | 8-atom burst (256B) per row | Pixel rows |
| **CBUF organization** | Channel-major, packed across width | 4×4×4 sub-cubes | Same as DC after pixel→feature conversion |
| **Data transformation** | NaN flush, optional cvt_offset/scale | Padding, 4×4×4 rearrangement | Mean-sub, scale, truncate, padding |
| **Height processing** | slice = 1 row | super_height = 4 rows | row by row |
| **Padding** | N/A (handled by CSC) | left/right padding | all 4 sides |
| **Stride** | line_stride, surface_stride | line_stride, surface_stride | line_stride, uv_line_stride |
| **DMA sync FIFOs** | act_data_read_rsp | wino_fetch_data_[4], wino_req2resp_sync | act_data_read_rsp, mean_data_read_rsp |
| **Request sequencer** | DirectConvDataReq | WinoConvDataReq | ImageConvDataReq |
| **Response sequencer** | DirectConvDataResp | WinoConvDataResp | ImageConvDataResp |
| **Common use case** | Most conv layers (1×1, 3×3, 5×5...) | 3×3 kernels with F(4,3) transform | First layer only |

### CBUF Layout by Mode

```
CDMA_DC:
  ┌──────┬──────┬──────┬──────┬──────┬──────┬─────────┐
  │Entry0│Entry1│Entry2│Entry3│ ...  │EntryN│         │  ← cbuf_entry_per_slice
  ├──────┴──────┴──────┴──────┴──────┴──────┤         │
  │          Slice 0 (one height step)      │         │
  ├──────┬──────┬──────┬──────┬──────┬──────┤         │
  │Entry0│Entry1│Entry2│Entry3│ ...  │EntryN│         │  ← cbuf_entry_per_slice
  ├──────┴──────┴──────┴──────┴──────┴──────┤ entries │
  │          Slice 1                        │         │
  ├─────────────────────────────────────────┤         │
  │         ... slices = cube_height        │         │
  └─────────────────────────────────────────┘         │

CDMA_WG:
  ┌──────────────────────────────────────────────────┐
  │  Entry per 4×4×4 sub-cube                        │
  │  ┌─ subcube 0,0 ──┐  ┌─ subcube 0,1 ──┐        │
  │  │ row0-col0-c0..3 │  │ row0-col2-c0..3 │        │
  │  │ row1-col0-c0..3 │  │ row1-col2-c0..3 │        │
  │  │ row2-col0-c0..3 │  │ row2-col2-c0..3 │        │
  │  │ row3-col0-c0..3 │  │ row3-col2-c0..3 │        │
  │  └─────────────────┘  └─────────────────┘        │
  │  ┌─ subcube 1,0 ──┐  ┌─ subcube 2,0 ──┐        │
  │  │ ...             │  │ ...             │        │
  └──────────────────────────────────────────────────┘

CDMA_IMG: Same layout as CDMA_DC after pixel→feature conversion
```

---

## 11. CBUF Three-Way Pointer Management

CDMA tracks CBUF usage with a **producer-consumer-free** pointer triple:

| Pointer | Who updates | Meaning |
|---|---|---|
| `*_idx_free_` | CDMA (when writing to CBUF) | Next free CBUF entry to write |
| `*_idx_working_` | CDMA (initialized at layer start) | Current read position for CSC |
| `*_idx_planed_` | CDMA (before writing) | Next entry planned for write |

The SC updates `data_entry_idx_free_` via `dat_up_sc2cdma_b_transport()` when CSC has consumed data, implementing a sliding window protocol.

```
  data_entry_idx_free_    ◄── CDMA writes, SC advances via dat_up_sc2cdma
  data_entry_idx_working_ ◄── CDMA sets at layer start (CSC read position)
  data_entry_idx_planed_  ◄── CDMA pre-allocates before writing

  Same pattern for: weight_entry_idx_*, wmb_entry_idx_*
```

---

## 12. Data Reuse Optimization

The CDMA has a critical optimization for multi-layer convolutions where output of one layer feeds the next without writing back to DRAM.

| Register Field | Effect |
|---|---|
| `D_MISC_CFG.DATA_REUSE=1` | Skip DMA for activation data — use CBUF content from previous layer |
| `D_MISC_CFG.WEIGHT_REUSE=1` | Skip DMA for weights — use CBUF content from previous layer |
| `D_MISC_CFG.SKIP_DATA_RLS=1` | Don't signal data availability to SC (data already in CBUF) |
| `D_MISC_CFG.SKIP_WEIGHT_RLS=1` | Don't signal weight availability to SC (weights already in CBUF) |

```
Reuse detection flow:

  dat_first_layer = true    ← Reset at power-on

  On each layer:
    if (DATA_REUSE && !dat_first_layer):
      Skip all DMA reads for activation
      CDMA acts as if data is already in CBUF
      Only SC update may be needed

    if (WEIGHT_REUSE && !wt_first_layer):
      Skip all DMA reads for weights
      Weights from previous layer stay in CBUF

    dat_first_layer = false
    wt_first_layer  = false

  When REUSE=0: force re-fetch from DRAM
    dat_first_layer = true
    wt_first_layer  = true
```

---

## 13. CDMA_WG vs CDMA_DC vs CDMA_IMG — ASCII Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║  CDMA_DC — DIRECT CONVOLUTION (feature map input)                                ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

  DRAM layout (NC1HWC0 format):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  surface_stride                                                          │
  │  ┌─────────────────────────────────────────────────────────────┐        │
  │  │  line_stride                                                 │        │
  │  │  ┌──────────────────────────────────┐                       │        │
  │  │  │  cube_width (pixels)             │                       │        │
  │  │  │  ┌────┬────┬────┬────┬────┐      │                       │        │
  │  │  │  │ A0 │ A1 │ A2 │ A3 │ A4 │ ...  │                       │        │
  │  │  │  ├────┼────┼────┼────┼────┤      │  cube_height          │        │
  │  │  │  │ B0 │ B1 │ B2 │ B3 │ B4 │ ...  │                       │        │
  │  │  │  ├────┼────┼────┼────┼────┤      │                       │        │
  │  │  │  │ C0 │ C1 │ C2 │ C3 │ C4 │ ...  │                       │        │
  │  │  │  └────┴────┴────┴────┴────┘      │                       │        │
  │  │  └──────────────────────────────────┘                       │        │
  │  └─────────────────────────────────────────────────────────────┘        │
  └─────────────────────────────────────────────────────────────────────────┘
                │  cube_channel (surfaces stacked in memory)
                ▼

  DMA fetch: ATOM_CUBE_SIZE = 32 bytes at a time

  CBUF layout:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  cbuf_entry_per_slice = ceil(cube_channel * bytes_per_element       │
  │                              * cube_width / CBUF_ENTRY_SIZE)        │
  │                                                                     │
  │  Entry 0: [ ch0-w0 │ ch0-w1 │ ... │ ch0-wN │ ch1-w0 │ ... ]        │
  │  Entry 1: [ ... cont. channels packed across width ... ]            │
  │  ...                                                                │
  │  Next slice → next set of entries                                   │
  │  CBUF banks: DATA_BASE_BANK .. DATA_HEAD_BANK                       │
  └─────────────────────────────────────────────────────────────────────┘

  Data response sequencer:
    1. Read atoms from act_data_read_rsp_fifo_
    2. For FP16: scan each element for NaN/Inf
    3. Optional: cvt_offset/cvt_scale/cvt_truncate conversion
    4. NaN flush: replace NaN with 0
    5. Pack into 128B CBUF entries (2 × 64B halves)
    6. cdma2buf_dat_wr_b_transport() → CBUF
    7. dat_up_cdma2sc_b_transport(entries_per_slice, slices)


╔═══════════════════════════════════════════════════════════════════════════════════╗
║  CDMA_WG — WINOGRAD CONVOLUTION (4×4×4 sub-cubes)                               ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

  super_height_iter: processes 4 rows at a time
  ┌───────────────────────────────────────────────────────────────┐
  │  super_height_num = ceil(cube_height / (4 * conv_y_stride))  │
  │                                                               │
  │  For each super_height_iter:                                  │
  │    Fetch 4 rows of data (8 atoms = 256B per row)             │
  │    Store each row into wino_fetch_data_fifo_[bank][i]         │
  │    bank 0 = row 0, bank 1 = row 1, bank 2 = row 2, bank 3 = row 3 │
  │                                                               │
  │    ┌── row 0 ──→ wino_fetch_data_fifo_[0] ──┐                │
  │    │  row 1 ──→ wino_fetch_data_fifo_[1]      │               │
  │    │  row 2 ──→ wino_fetch_data_fifo_[2]      │               │
  │    │  row 3 ──→ wino_fetch_data_fifo_[3]      │               │
  │    └───────────────────────────────────────────┘               │
  └───────────────────────────────────────────────────────────────┘

  4×4×4 sub-cube assembly:
  ┌───────────────────────────────────────────────────────────────┐
  │  Each sub-cube = 4 rows × 4 atoms(columns) × 4 channels      │
  │                                                               │
  │  CBUF entry layout for one sub-cube:                          │
  │  ┌─────────────────────────────────────────────────────┐      │
  │  │  Left half (64B)     │  Right half (64B)            │      │
  │  │  [row0-col0-4ch]     │  [row0-col2-4ch]             │      │
  │  │  [row1-col0-4ch]     │  [row1-col2-4ch]             │      │
  │  │  [row2-col0-4ch]     │  [row2-col2-4ch]             │      │
  │  │  [row3-col0-4ch]     │  [row3-col2-4ch]             │      │
  │  └─────────────────────────────────────────────────────┘      │
  │                                                               │
  │  hsel=0 (left): rows 0-3, columns 0-1, 4 channels            │
  │  hsel=1 (right): rows 0-3, columns 2-3, 4 channels           │
  └───────────────────────────────────────────────────────────────┘

  Key constants:
    Max height = 1920 pixels
    Each super_height = 4 rows
    wino_req2resp_sync_fifo_ depth = 480 = 1920/4

  Data response sequencer:
    1. Read atoms from wino_fetch_data_fifo_[bank]
    2. Apply left/right padding per row
    3. Compose 4×4×4 sub-cube from 4 rows × 4 col atoms × 4 ch
    4. Write to CBUF: left half (hsel=0) then right half (hsel=1)
    5. After each super_height_iter: dat_up_cdma2sc_b_transport()


╔═══════════════════════════════════════════════════════════════════════════════════╗
║  CDMA_IMG — IMAGE/PIXEL INPUT (camera/raw input)                                ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

  Supported pixel formats:
  ┌──────────────────────────────────────────────────────┐
  │  Format                   │ bpp │ elements │ planar?  │
  │  ─────────────────────────┼─────┼──────────┼───────── │
  │  T_R8G8B8A8              │ 32  │ RGBA     │ no       │
  │  T_A2B10G10R10           │ 32  │ ABGR     │ no       │
  │  T_A2Y10U10V10           │ 32  │ AYUV     │ no       │
  │  Y8___U8V8_N444          │ 16  │ Y+UV     │ planar   │
  │  Y8___V8U8_N444          │ 16  │ Y+VU     │ planar   │
  │  Y8_U8_V8_N420           │ 12  │ YUV420   │ planar   │
  │  T_R8                    │ 8   │ R        │ no       │
  │  T_R16_G16               │ 32  │ RG(16b)  │ no       │
  └──────────────────────────────────────────────────────┘

  Data flow:
  ┌──────────────────────────────────────────────────────────────────┐
  │  DRAM pixel data (interleaved or planar)                        │
  │       │                                                         │
  │       ▼                                                         │
  │  Fetch rows from DRAM via DMA                                   │
  │  (line_stride for interleaved, uv_line_stride for planar YUV)  │
  │       │                                                         │
  │       ▼                                                         │
  │  process_one_element_16() or process_one_element_8():           │
  │    for each pixel component:                                    │
  │      if cvt_en:                                                 │
  │        converted = (pixel - cvt_mean) * cvt_scale >> cvt_trunc  │
  │      else:                                                      │
  │        converted = pixel (lower bits clipped)                   │
  │      store in pad_buffer                                        │
  │       │                                                         │
  │       ▼                                                         │
  │  Apply padding (pad_left, pad_right, pad_top, pad_bottom):      │
  │    pad_value repeated for padded regions before/after data      │
  │       │                                                         │
  │       ▼                                                         │
  │  WriteOneEntryToCbuf(pad_buffer, cbuf_entry_addr):              │
  │    hsel=0 → 64B left half                                      │
  │    hsel=1 → 64B right half                                     │
  │       │                                                         │
  │       ▼                                                         │
  │  dat_up_cdma2sc_b_transport(entries, slices)                    │
  └──────────────────────────────────────────────────────────────────┘
```
