#!/usr/bin/env python3
import numpy as np
FP16_BYTES=2; FP16_ATOM_ELEMENTS=16; WEIGHT_ATOMIC_ELEMENTS=32
CBUF_ENTRY_BYTES=128; CBUF_ENTRIES_PER_BANK=256; RK_CBUF_BANKS=12; CBUF_BANK_SIZE=CBUF_ENTRIES_PER_BANK*CBUF_ENTRY_BYTES
def _ceil_div(x,y): return (x+y-1)//y
def _align_up(x,a): return _ceil_div(x,a)*a
def E(t,r,v): return (t<<48)|((v&0xFFFFFFFF)<<16)|r

class reg:
    CNA=0x0201;CORE=0x0801;DPU=0x1001;RDMA=0x2001;PC=0x0081;PC_REG=0x0101;VERSION=0x0041
    S_POINTER=0x4004;FEATURE_MODE_CFG=0x400c;DATA_FORMAT=0x4010;DST_BASE_ADDR=0x4020
    DST_SURF_STRIDE=0x4024;DATA_CUBE_WIDTH=0x4030;DATA_CUBE_HEIGHT=0x4034
    DATA_CUBE_NOTCH=0x4038;DATA_CUBE_CHANNEL=0x403c;BS_CFG=0x4040;BS_OW_CFG=0x4050
    WDMA_SIZE_0=0x4058;WDMA_SIZE_1=0x405c;BN_CFG=0x4060;EW_CFG=0x4070
    EW_CVT_SCALE_VALUE=0x4078;OUT_CVT_SCALE=0x4084;SURFACE_ADD=0x40c0
    RDMA_S_POINTER=0x5004;RDMA_ERDMA_CFG=0x5034;RDMA_FEATURE_MODE_CFG=0x5044
    CNA_CONV_CON1=0x100c;CNA_CONV_CON2=0x1010;CNA_CONV_CON3=0x1014
    CNA_DATA_SIZE0=0x1020;CNA_DATA_SIZE1=0x1024;CNA_DATA_SIZE2=0x1028
    CNA_DATA_SIZE3=0x102c;CNA_WEIGHT_SIZE0=0x1030;CNA_WEIGHT_SIZE1=0x1034
    CNA_WEIGHT_SIZE2=0x1038;CNA_CBUF_CON0=0x1040;CNA_CBUF_CON1=0x1044
    CNA_CVT_CON0=0x104c;CNA_CVT_CON1=0x1050;CNA_CVT_CON2=0x1054
    CNA_CVT_CON3=0x1058;CNA_CVT_CON4=0x105c;CNA_CVT_CON5=0x1180
    CNA_FEATURE_DATA_ADDR=0x1070;CNA_DMA_CON0=0x1078;CNA_DMA_CON1=0x107c
    CNA_DMA_CON2=0x1080;CNA_FC_DATA_SIZE0=0x1084;CNA_FC_DATA_SIZE1=0x1088
    CNA_DCOMP_ADDR0=0x1110;CORE_MISC_CFG=0x3010;CORE_DATAOUT_SIZE_0=0x3014
    CORE_DATAOUT_SIZE_1=0x3018;CORE_RESERVED_3030=0x3030

# --- conv.py helpers ---
def _is_dw_c(in_c,out_c,g): return g==in_c==out_c
def _sh_nhwc_c(c,c2): return c>0 and (c2//c==2 or (c==2 and c2//c==4))
def _ac_c(in_c,g,out_c):
    if not _is_dw_c(in_c,out_c,g) and (g>1 or in_c>4): return 16
    return max(8,min(1<<(max(1,in_c)-1).bit_length(),32 if _is_dw_c(in_c,out_c,g) else 16))
def _ic2_c(in_c,g,out_c,ac):
    if in_c==1: return 2
    if _is_dw_c(in_c,out_c,g) or g>1 or in_c>4: return 8
    return ac
def _ds_c(ih,ws,un):
    if un: return ws, ws*(ih-1) if ih>1 else 0
    return ws*4, ws*(ih-4) if ih>4 else 0
def _ce_c(ws,ac,ih,is_dw):
    re=max(1,_ceil_div(ws*ac,2*FP16_ATOM_ELEMENTS))
    return re if ac>=16 or is_dw else re*ih*4
def _fg_c(rb,fg,un=False,isp=False,is_dw=False):
    if un and isp: return fg
    if is_dw and isp: return min(13,fg)
    return min(fg,(_ceil_div(2*CBUF_BANK_SIZE,rb)+1)&~1)
def _db_c(ws,fg,ac,un=False,isp=False,is_dw=False):
    if isp and (un or is_dw): return RK_CBUF_BANKS-1
    return int(np.clip(_ceil_div(ws*fg*ac*FP16_BYTES,CBUF_BANK_SIZE),1,RK_CBUF_BANKS-1))
def _pc_c(b,ic,ih,iw,oc,kh,kw,g):
    dw=_is_dw_c(ic,oc,g); sp=(kh!=1 or kw!=1); oh=ih-kh+1; ow=iw-kw+1
    ac=_ac_c(ic,g,oc); aoc=max(16,_align_up(oc,16))
    ws=_align_up(iw,max(1,_ceil_div(16,ac)))
    oa=max(1,oh*ow); ows=oa if (not sp and oa<4) else _align_up(oa,4)
    ic2=_ic2_c(ic,g,oc,ac)
    un=(not dw and not (g>1 and sp) and _sh_nhwc_c(ic,ic2))
    return dict(ic=ic,ih=ih,iw=iw,oc=oc,kh=kh,kw=kw,g=g,dw=dw,oh=oh,ow=ow,ac=ac,aoc=aoc,ws=ws,ows=ows,ic2=ic2,un=un)

def mkr_c(b,ic,ih,iw,oc,kh,kw,idm,wdm,odm,g=1,ows_o=None,wr=False,fdb=False,of16=False):
    p=_pc_c(b,ic,ih,iw,oc,kh,kw,g); dw=p['dw']; sp=(kh!=1 or kw!=1); oh=p['oh']; ow=p['ow']
    ac=p['ac']; aoc=p['aoc']; ws=p['ws']; ows=ows_o or p['ows']; un=p['un']; ic2=p['ic2']
    cm=(1<<ic if un else ic2)-1; dica=_align_up(ic,ac)
    wpk=kh*kw*dica*FP16_BYTES; wt=wpk if dw else wpk*oc
    fg=_fg_c(ws*dica*FP16_BYTES,ih+kh,un,sp,dw)
    db=_db_c(ws,fg,dica,un,sp,dw)
    if fdb: db=RK_CBUF_BANKS-1
    ocf=aoc-1 if not dw else _align_up(aoc,32)-1
    eao=max(16,_align_up(_ceil_div(oc,g),16)) if (g>1 and not dw) else ocf+1
    op=2 if of16 else 5; se=1 if of16 else 3; bse=3 if dw else se
    return [E(reg.DPU,reg.S_POINTER,0xe),
        E(reg.CNA,reg.CNA_CONV_CON1,((2<<4)|(2<<7)|(((1<<30)|(1<<29)|((7+ic)<<12)) if (un and ic<=4 and not dw) else 0)|(3 if dw else 0))),
        E(reg.CNA,reg.CNA_CONV_CON2,fg<<4),
        E(reg.CNA,reg.CNA_CONV_CON3,0x9),
        E(reg.CNA,reg.CNA_DATA_SIZE0,(ws<<16)|ih),
        E(reg.CNA,reg.CNA_DATA_SIZE1,((ic-1)<<16)|dica),
        E(reg.CNA,reg.CNA_DATA_SIZE2,ow),E(reg.CNA,reg.CNA_DATA_SIZE3,ow*oh),
        E(reg.CNA,reg.CNA_WEIGHT_SIZE0,wt),E(reg.CNA,reg.CNA_WEIGHT_SIZE1,wpk),
        E(reg.CNA,reg.CNA_WEIGHT_SIZE2,(kw<<24)|(kh<<16)|(1 if dw else oc)),
        E(reg.CNA,reg.CNA_CBUF_CON0,(wr<<13)|((RK_CBUF_BANKS-db)<<4)|db),
        E(reg.CNA,reg.CNA_CBUF_CON1,_ce_c(ws,dica,ih,dw)),
        E(reg.CNA,reg.CNA_CVT_CON0,(un<<3)|(un<<1)|1),
        E(reg.CNA,reg.CNA_CVT_CON1,1<<16),E(reg.CNA,reg.CNA_CVT_CON2,1<<16),
        E(reg.CNA,reg.CNA_CVT_CON3,1<<16),E(reg.CNA,reg.CNA_CVT_CON4,1<<16),
        E(reg.CNA,reg.CNA_FEATURE_DATA_ADDR,idm),
        E(reg.CNA,reg.CNA_DMA_CON0,0xf000f),
        E(reg.CNA,reg.CNA_DMA_CON1,_ds_c(ih,ws,un)[0]),
        E(reg.CNA,reg.CNA_DMA_CON2,_ds_c(ih,ws,un)[1]),
        E(reg.CNA,reg.CNA_FC_DATA_SIZE0,(iw<<16)|ih),
        E(reg.CNA,reg.CNA_FC_DATA_SIZE1,dica),E(reg.CNA,reg.CNA_DCOMP_ADDR0,wdm),
        E(reg.CNA,reg.CNA_CVT_CON5,cm),
        E(reg.CORE,reg.CORE_MISC_CFG,(2<<8)|(dw<<1)|sp),
        E(reg.CORE,reg.CORE_DATAOUT_SIZE_0,((oh-1)<<16)|(ow-1)),
        E(reg.CORE,reg.CORE_DATAOUT_SIZE_1,ocf),E(reg.CORE,reg.CORE_RESERVED_3030,0),
        E(reg.DPU,reg.FEATURE_MODE_CFG,(15<<5)|((3*dw)<<3)|(2<<1)),
        E(reg.DPU,reg.DATA_FORMAT,(op<<29)|(2<<26)|2),
        E(reg.DPU,reg.DST_BASE_ADDR,odm),E(reg.DPU,reg.DST_SURF_STRIDE,ows<<4),
        E(reg.DPU,reg.DATA_CUBE_WIDTH,ow-1),E(reg.DPU,reg.DATA_CUBE_HEIGHT,oh-1),
        E(reg.DPU,reg.DATA_CUBE_NOTCH,0),
        E(reg.DPU,reg.DATA_CUBE_CHANNEL,((oc-1)<<16)|ocf),
        E(reg.DPU,reg.BS_CFG,0x53),
        E(reg.DPU,reg.BS_OW_CFG,(bse<<8)|(bse<<5)|(bse<<2)|(1<<1)),
        E(reg.DPU,reg.WDMA_SIZE_0,ocf),
        E(reg.DPU,reg.WDMA_SIZE_1,((oh-1)<<16)|(ow-1)),
        E(reg.DPU,reg.BN_CFG,0x53),E(reg.DPU,reg.EW_CFG,0x3a3),
        E(reg.DPU,reg.EW_CVT_SCALE_VALUE,1),
        E(reg.DPU,reg.OUT_CVT_SCALE,(0x10001 if of16 else 0)),
        E(reg.DPU,reg.SURFACE_ADD,(ows*max(2,eao//16))<<4)]

# --- conv_new.py helpers ---
def _is_dw_n(ic,oc,g): return g==ic==oc
def _upwal(ic,kh,kw,g): return g==1 and kh==1 and kw==1 and _ceil_div(max(ic,FP16_ATOM_ELEMENTS),WEIGHT_ATOMIC_ELEMENTS)>1
def _sh_nhwc_n(c,c2): return c>0 and c2//c==2
def _ac_n(ic,g,oc):
    if not _is_dw_n(ic,oc,g) and g==1 and ic>1: return 16
    if not _is_dw_n(ic,oc,g) and ic>4: return 16
    return np.clip(1<<(max(1,ic)-1).bit_length(),8,32)
def _ic2_n(ic,g,oc,ac):
    if ic==1: return 2
    if not _is_dw_n(ic,oc,g) and g==1 and 1<ic<=4: return 8
    if _is_dw_n(ic,oc,g) or g>1 or ic>4: return 8
    return ac
def _ds_n(ih,ws,un):
    if un: return ws, ws*(ih-1) if ih>1 else 0
    return ws*4, ws*(ih-4) if ih>4 else 0
def _ce_n(ws,ac,ih,dw):
    re=max(1,_ceil_div(ws*ac,2*FP16_ATOM_ELEMENTS))
    return re if ac>=16 or dw else re*ih*4
def _fg_n(rb,fg,un=False,isp=False,dw=False):
    if un and isp: return fg
    if dw and isp: return min(13,fg)
    return min(fg,(_ceil_div(2*CBUF_BANK_SIZE,rb)+1)&~1)
def _db_n(ws,fg,ac,un=False,isp=False,dw=False):
    if isp and (un or dw): return RK_CBUF_BANKS-1
    return int(np.clip(_ceil_div(ws*fg*ac*FP16_BYTES,CBUF_BANK_SIZE),1,RK_CBUF_BANKS-1))
def _pc_n(b,ic,ih,iw,oc,kh,kw,g,s=1):
    dw=_is_dw_n(ic,oc,g); sp=(kh!=1 or kw!=1)
    oh=(ih-kh)//s+1; ow=(iw-kw)//s+1
    ac=WEIGHT_ATOMIC_ELEMENTS if _upwal(ic,kh,kw,g) else _ac_n(ic,g,oc)
    aoc=max(32,_align_up(oc,16))
    ws=_align_up(iw,max(1,_ceil_div(16,ac)))
    oa=max(1,oh*ow); ows=oa if not sp else _align_up(oa,4)
    mas=(not dw) and g==1 and 1<ic<=4
    ic2=_ic2_n(ic,g,oc,ac); un=(not mas) and (not dw) and (ic>0 and ic2//ic==2)
    return dict(ic=ic,ih=ih,iw=iw,oc=oc,kh=kh,kw=kw,g=g,s=s,dw=dw,oh=oh,ow=ow,ac=ac,aoc=aoc,ws=ws,ows=ows,ic2=ic2,un=un,mas=mas)

def mkr_n(b,ic,ih,iw,oc,kh,kw,idm,wdm,odm,g=1,s=1,ows_o=None,wr=False,fdb=False):
    p=_pc_n(b,ic,ih,iw,oc,kh,kw,g,s); dw=p['dw']; sp=(kh!=1 or kw!=1); oh=p['oh']; ow=p['ow']
    ac=p['ac']; aoc=p['aoc']; ws=p['ws']; ows=ows_o or p['ows']; un=p['un']; ic2=p['ic2']
    c5=65535 if (ic==1 and not dw) else (0 if (not dw and g==1) else ((1<<ic if un else ic2)-1))
    dica=_align_up(ic,ac); wpk=kh*kw*dica*FP16_BYTES; wt=wpk if dw else wpk*oc
    rb=ws*dica*FP16_BYTES; fg=_fg_n(rb,ih+kh,un,sp,dw)
    if not sp: fg=max(fg,52)
    ls,ss=_ds_n(ih,ws,un); ce=_ce_n(ws,dica,ih,dw)
    db=_db_n(ws,fg,dica,un,sp,dw)
    if fdb: db=RK_CBUF_BANKS-1
    ocf=aoc-1 if not dw else _align_up(aoc,32)-1
    eao=max(16,_align_up(_ceil_div(oc,g),16)) if (g>1 and not dw) else ocf+1
    return [E(reg.DPU,reg.S_POINTER,0xe),
        E(reg.RDMA,reg.RDMA_S_POINTER,0),E(reg.RDMA,reg.RDMA_ERDMA_CFG,0),
        E(reg.RDMA,reg.RDMA_FEATURE_MODE_CFG,0),
        E(reg.CNA,reg.CNA_CONV_CON1,((2<<4)|(2<<7)|(((1<<30)|(1<<29)|((7+ic)<<12)) if (ic==1 and not dw) else 0)|(3 if dw else 0))),
        E(reg.CNA,reg.CNA_CONV_CON2,fg<<4),
        E(reg.CNA,reg.CNA_CONV_CON3,(s<<3)|(s<<0)),
        E(reg.CNA,reg.CNA_DATA_SIZE0,(ws<<16)|ih),
        E(reg.CNA,reg.CNA_DATA_SIZE1,((ic-1)<<16)|dica),
        E(reg.CNA,reg.CNA_DATA_SIZE2,ow),E(reg.CNA,reg.CNA_DATA_SIZE3,ow*oh),
        E(reg.CNA,reg.CNA_WEIGHT_SIZE0,wt),E(reg.CNA,reg.CNA_WEIGHT_SIZE1,wpk),
        E(reg.CNA,reg.CNA_WEIGHT_SIZE2,(kw<<24)|(kh<<16)|(1 if dw else oc)),
        E(reg.CNA,reg.CNA_CBUF_CON0,(wr<<13)|((RK_CBUF_BANKS-db)<<4)|db),
        E(reg.CNA,reg.CNA_CBUF_CON1,ce),
        E(reg.CNA,reg.CNA_CVT_CON0,(un<<3)|(un<<1)|1),
        E(reg.CNA,reg.CNA_CVT_CON1,1<<16),E(reg.CNA,reg.CNA_CVT_CON2,1<<16),
        E(reg.CNA,reg.CNA_CVT_CON3,1<<16),E(reg.CNA,reg.CNA_CVT_CON4,1<<16),
        E(reg.CNA,reg.CNA_FEATURE_DATA_ADDR,idm),
        E(reg.CNA,reg.CNA_DMA_CON0,0xf000f),
        E(reg.CNA,reg.CNA_DMA_CON1,ls),E(reg.CNA,reg.CNA_DMA_CON2,ss),
        E(reg.CNA,reg.CNA_FC_DATA_SIZE0,(iw<<16)|ih),
        E(reg.CNA,reg.CNA_FC_DATA_SIZE1,dica),E(reg.CNA,reg.CNA_DCOMP_ADDR0,wdm),
        E(reg.CNA,reg.CNA_CVT_CON5,c5),
        E(reg.CORE,reg.CORE_MISC_CFG,(2<<8)|(dw<<1)|sp),
        E(reg.CORE,reg.CORE_DATAOUT_SIZE_0,((oh-1)<<16)|(ow-1)),
        E(reg.CORE,reg.CORE_DATAOUT_SIZE_1,ocf),E(reg.CORE,reg.CORE_RESERVED_3030,0),
        E(reg.DPU,reg.FEATURE_MODE_CFG,(15<<5)|((3*dw)<<3)|(2<<1)),
        E(reg.DPU,reg.DATA_FORMAT,(2<<29)|(2<<26)|2),
        E(reg.DPU,reg.DST_BASE_ADDR,odm),E(reg.DPU,reg.DST_SURF_STRIDE,ows<<4),
        E(reg.DPU,reg.DATA_CUBE_WIDTH,ow-1),E(reg.DPU,reg.DATA_CUBE_HEIGHT,oh-1),
        E(reg.DPU,reg.DATA_CUBE_NOTCH,0),
        E(reg.DPU,reg.DATA_CUBE_CHANNEL,((oc-1)<<16)|ocf),
        E(reg.DPU,reg.BS_CFG,0x53),
        E(reg.DPU,reg.BS_OW_CFG,((3 if dw else 1)<<8)|((3 if dw else 1)<<5)|((3 if dw else 1)<<2)|(1<<1)),
        E(reg.DPU,reg.WDMA_SIZE_0,ocf),
        E(reg.DPU,reg.WDMA_SIZE_1,((oh-1)<<16)|(ow-1)),
        E(reg.DPU,reg.BN_CFG,0x53),E(reg.DPU,reg.EW_CFG,0x3a3),
        E(reg.DPU,reg.EW_CVT_SCALE_VALUE,1),
        E(reg.DPU,reg.OUT_CVT_SCALE,0x10001),
        E(reg.DPU,reg.SURFACE_ADD,(ows*max(2,eao//16))<<4)]

# --- Print comparison ---
NAMES = {(reg.DPU,reg.S_POINTER):'DPU.S_POINTER',(reg.RDMA,reg.RDMA_S_POINTER):'RDMA.RDMA_S_POINTER',
(reg.RDMA,reg.RDMA_ERDMA_CFG):'RDMA.RDMA_ERDMA_CFG',(reg.RDMA,reg.RDMA_FEATURE_MODE_CFG):'RDMA.RDMA_FEATURE_MODE_CFG',
(reg.CNA,reg.CNA_CONV_CON1):'CNA.CNA_CONV_CON1',(reg.CNA,reg.CNA_CONV_CON2):'CNA.CNA_CONV_CON2',
(reg.CNA,reg.CNA_CONV_CON3):'CNA.CNA_CONV_CON3',
(reg.CNA,reg.CNA_DATA_SIZE0):'CNA.CNA_DATA_SIZE0',(reg.CNA,reg.CNA_DATA_SIZE1):'CNA.CNA_DATA_SIZE1',
(reg.CNA,reg.CNA_DATA_SIZE2):'CNA.CNA_DATA_SIZE2',(reg.CNA,reg.CNA_DATA_SIZE3):'CNA.CNA_DATA_SIZE3',
(reg.CNA,reg.CNA_WEIGHT_SIZE0):'CNA.CNA_WEIGHT_SIZE0',(reg.CNA,reg.CNA_WEIGHT_SIZE1):'CNA.CNA_WEIGHT_SIZE1',
(reg.CNA,reg.CNA_WEIGHT_SIZE2):'CNA.CNA_WEIGHT_SIZE2',(reg.CNA,reg.CNA_CBUF_CON0):'CNA.CNA_CBUF_CON0',
(reg.CNA,reg.CNA_CBUF_CON1):'CNA.CNA_CBUF_CON1',(reg.CNA,reg.CNA_CVT_CON0):'CNA.CNA_CVT_CON0',
(reg.CNA,reg.CNA_CVT_CON1):'CNA.CNA_CVT_CON1',(reg.CNA,reg.CNA_CVT_CON2):'CNA.CNA_CVT_CON2',
(reg.CNA,reg.CNA_CVT_CON3):'CNA.CNA_CVT_CON3',(reg.CNA,reg.CNA_CVT_CON4):'CNA.CNA_CVT_CON4',
(reg.CNA,reg.CNA_CVT_CON5):'CNA.CNA_CVT_CON5',(reg.CNA,reg.CNA_FEATURE_DATA_ADDR):'CNA.CNA_FEATURE_DATA_ADDR',
(reg.CNA,reg.CNA_DMA_CON0):'CNA.CNA_DMA_CON0',(reg.CNA,reg.CNA_DMA_CON1):'CNA.CNA_DMA_CON1',
(reg.CNA,reg.CNA_DMA_CON2):'CNA.CNA_DMA_CON2',(reg.CNA,reg.CNA_FC_DATA_SIZE0):'CNA.CNA_FC_DATA_SIZE0',
(reg.CNA,reg.CNA_FC_DATA_SIZE1):'CNA.CNA_FC_DATA_SIZE1',(reg.CNA,reg.CNA_DCOMP_ADDR0):'CNA.CNA_DCOMP_ADDR0',
(reg.CORE,reg.CORE_MISC_CFG):'CORE.CORE_MISC_CFG',(reg.CORE,reg.CORE_DATAOUT_SIZE_0):'CORE.CORE_DATAOUT_SIZE_0',
(reg.CORE,reg.CORE_DATAOUT_SIZE_1):'CORE.CORE_DATAOUT_SIZE_1',(reg.CORE,reg.CORE_RESERVED_3030):'CORE.CORE_RESERVED_3030',
(reg.DPU,reg.FEATURE_MODE_CFG):'DPU.FEATURE_MODE_CFG',(reg.DPU,reg.DATA_FORMAT):'DPU.DATA_FORMAT',
(reg.DPU,reg.DST_BASE_ADDR):'DPU.DST_BASE_ADDR',(reg.DPU,reg.DST_SURF_STRIDE):'DPU.DST_SURF_STRIDE',
(reg.DPU,reg.DATA_CUBE_WIDTH):'DPU.DATA_CUBE_WIDTH',(reg.DPU,reg.DATA_CUBE_HEIGHT):'DPU.DATA_CUBE_HEIGHT',
(reg.DPU,reg.DATA_CUBE_NOTCH):'DPU.DATA_CUBE_NOTCH',(reg.DPU,reg.DATA_CUBE_CHANNEL):'DPU.DATA_CUBE_CHANNEL',
(reg.DPU,reg.BS_CFG):'DPU.BS_CFG',(reg.DPU,reg.BS_OW_CFG):'DPU.BS_OW_CFG',
(reg.DPU,reg.WDMA_SIZE_0):'DPU.WDMA_SIZE_0',(reg.DPU,reg.WDMA_SIZE_1):'DPU.WDMA_SIZE_1',
(reg.DPU,reg.BN_CFG):'DPU.BN_CFG',(reg.DPU,reg.EW_CFG):'DPU.EW_CFG',
(reg.DPU,reg.EW_CVT_SCALE_VALUE):'DPU.EW_CVT_SCALE_VALUE',(reg.DPU,reg.OUT_CVT_SCALE):'DPU.OUT_CVT_SCALE',
(reg.DPU,reg.SURFACE_ADD):'DPU.SURFACE_ADD'}

pc=_pc_c(1,144,32,56,24,1,1,1); pn=_pc_n(1,144,32,56,24,1,1,1)
print("_conv_params diff:")
for k in sorted(pc):
    vc=pc[k];vn=pn.get(k,'N/A');m=" DIFF" if (vn!='N/A' and vc!=vn) else ""
    print(f"  {k:25s}: conv.py={str(vc):22s}  conv_new.py={str(vn):22s}{m}")
for k in sorted(pn):
    if k not in pc: print(f"  {k:25s}: conv.py={'N/A':22s}  conv_new.py={str(pn[k]):22s}  NEW")
print()

rc=mkr_c(1,144,32,56,24,1,1,0,1,2); rn=mkr_n(1,144,32,56,24,1,1,0,1,2)
def dd(r): return r>>48, r&0xFFFF, (r>>16)&0xFFFFFFFF
lc={}; ln={}
for r in rc: t,a,v=dd(r); lc[(t,a)]=v
for r in rn: t,a,v=dd(r); ln[(t,a)]=v
ak=sorted(set(list(lc.keys())+list(ln.keys())),key=lambda x:(x[0],x[1]))

print(f"{'Register':<42s} {'Value conv.py':<24s} {'Value conv_new.py':<24s} Status")
print("-"*120)
dc=0; sc=0; oc=0; on=0; diffs=[]
for k in ak:
    vc=lc.get(k);vn=ln.get(k);nm=NAMES.get(k,f'REG_{k[0]:04x}_{k[1]:04x}')
    if vc is not None and vn is not None:
        if vc==vn: print(f"{nm:<42s} 0x{vc:08x} ({vc:6d})      0x{vn:08x} ({vn:6d})      SAME"); sc+=1
        else: print(f"{nm:<42s} 0x{vc:08x} ({vc:6d})      0x{vn:08x} ({vn:6d})      DIFF"); dc+=1; diffs.append((nm,vc,vn,k))
    elif vc is not None: print(f"{nm:<42s} 0x{vc:08x} ({vc:6d})      {'---':>24s}      ONLY conv.py"); oc+=1; diffs.append((nm,vc,None,k))
    else: print(f"{nm:<42s} {'---':>24s}      0x{vn:08x} ({vn:6d})      ONLY conv_new.py"); on+=1; diffs.append((nm,None,vn,k))

print(f"\nSummary: {sc} same, {dc} diff, {oc} only conv.py, {on} only conv_new.py")

if diffs:
    print("\n"+"="*120+"\nDETAILED DIFF ANALYSIS\n"+"="*120)
    for nm,vc,vn,k in diffs:
        print(f"\n--- {nm} ---")
        if vc is not None and vn is not None:
            x=vc^vn; print(f"  conv.py=0x{vc:08x}  conv_new.py=0x{vn:08x}  XOR=0x{x:08x}")
            for b in range(32):
                if x&(1<<b): print(f"    bit{b:2d}: {((vc>>b)&1)}->{((vn>>b)&1)}")
        elif vc is not None: print(f"  Only in conv.py: 0x{vc:08x}")
        else: print(f"  Only in conv_new.py: 0x{vn:08x}")

print("\n"+"="*120+"\nROOT CAUSE SUMMARY\n"+"="*120)

dica_c=_align_up(144,pc['ac']); dica_n=_align_up(144,pn['ac'])
fg_c=_fg_c(pc['ws']*dica_c*FP16_BYTES,32+1,pc['un'],False,False)
fg_n=_fg_n(pn['ws']*dica_n*FP16_BYTES,32+1,pn['un'],False,False)
fg_n2=max(fg_n,52)
db_c=_db_c(pc['ws'],fg_c,dica_c,pc['un'],False,False)
db_n=_db_n(pn['ws'],fg_n2,dica_n,pn['un'],False,False)
ce_c=_ce_c(pc['ws'],dica_c,32,False)
ce_n=_ce_n(pn['ws'],dica_n,32,False)

print(f"\nKey intermediate value differences:")
print(f"  align_c:            conv.py={pc['ac']}, conv_new.py={pn['ac']}")
print(f"  align_out_c:        conv.py={pc['aoc']}, conv_new.py={pn['aoc']}")
print(f"  data_in_ch_aligned: conv.py={dica_c}, conv_new.py={dica_n}")
print(f"  width_stride:       conv.py={pc['ws']}, conv_new.py={pn['ws']}")
print(f"  out_width_stride:   conv.py={pc['ows']}, conv_new.py={pn['ows']}")
print(f"  input_pack_c2:      conv.py={pc['ic2']}, conv_new.py={pn['ic2']}")
print(f"  use_nhwc:           conv.py={pc['un']}, conv_new.py={pn['un']}")
print(f"  feature_grains:     conv.py={fg_c}, conv_new.py(raw)={fg_n}, conv_new.py(clamped)={fg_n2}")
print(f"  data_bank:          conv.py={db_c}, conv_new.py={db_n}")
print(f"  cbuf_entries:       conv.py={ce_c}, conv_new.py={ce_n}")
print(f"  cvt_channel_mask:   conv.py={(1<<144 if pc['un'] else pc['ic2'])-1}, conv_new.py=0")
print(f"  out_channel_field:  conv.py={pc['aoc']-1}, conv_new.py={pn['aoc']-1}")
