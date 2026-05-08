project('ops_reg', 'c')

incdir = include_directories('../include')
ops_reg_exe = executable('ops_reg', 'main.c',
  include_directories: incdir,
  c_args: ['-Wno-unused-variable', '-Wno-unused-function'],
  link_args: ['-ldrm', '-lm'],
)

matmul_cases = [
  ['33', '1', '33'],
  ['34', '1', '34'],
  ['65', '1', '33'],
  ['32', '32', '32'],
  ['64', '64', '64'],
  ['128', '128', '128'],
  ['256', '256', '256'],
  ['394', '394', '394'], 
  ['1', '8192', '8192'], 
  ['1', '8192', '8193'], 
  ['1', '8193', '8192'], 
  ['1', '8193', '8193'], 
  ['1', '32', '16'],
  ['1', '394', '394'],
  ['1', '768', '768'],
  ['1', '768', '2048'],
  ['1', '2048', '2048'],
  ['1', '4096', '4096'],
]

foreach dims : matmul_cases
 test('matmul @0@x@1@x@2@'.format(dims[0], dims[1], dims[2]),
   ops_reg_exe,
   is_parallel: false,
   suite: 'matmul',
   args: ['matmul', dims[0], dims[1], dims[2]])
endforeach

# foreach n : range(1, 8192, 10)
#   test('matmul @0@x1x@0@'.format(n),
#     ops_reg_exe,
#     is_parallel: false,
#     suite: 'matmul',
#     timeout: 180,
#     args: ['matmul', n.to_string(), '1', n.to_string()])
# endforeach

# llm_linear_cases = [
#   ['1', '1024', '2048', 'attn_q_proj'],
#   ['1', '1024', '1024', 'attn_k_proj'],
#   ['1', '1024', '1024', 'attn_v_proj'],
#   ['1', '2048', '1024', 'attn_o_proj'],
#   ['1', '1024', '3072', 'ffn_gate_proj'],
#   ['1', '1024', '3072', 'ffn_up_proj'],
#   ['1', '3072', '1024', 'ffn_down_proj'],
#   ['1', '1024', '151936', 'lm_head'], # >30s
# ]

# foreach dims : llm_linear_cases
#   test('matmul @0@'.format(dims[3]),
#     ops_reg_exe,
#     is_parallel: false,
#     suite: 'matmul',
#     timeout: 180,
#     args: ['matmul', dims[0], dims[1], dims[2]])
# endforeach


##################

conv1d_cases = [
  ['0', 'conv1d_bs1'],
  ['1', 'conv1d_bs8'],
  ['2', 'conv1d_bs1_612'],
  ['3', 'conv1d_bs1_615'],
  ['4', 'conv1d_bs1_1311_631'],
  ['5', 'conv1d_bs1_1311_632'],
  ['6', 'conv1d_bs1_1311_635'],
  ['7', 'conv1d_bs1_1311_615'],
  ['8', 'conv1d_bs8_8111_611'],
  ['9', 'conv1d_bs8_8111_612'],
  ['10', 'conv1d_bs8_8111_612'],
  ['11', 'conv1d_bs8_8111_615'],
  ['12', 'conv1d_bs8_8311_631'],
  ['13', 'conv1d_bs8_8311_632'],
  ['14', 'conv1d_bs8_8311_635'],
  ['15', 'conv1d_bs8_8311_635'],
]

foreach dims : conv1d_cases
  test('conv1d_@0@_@1@'.format(dims[0], dims[1]),
    ops_reg_exe,
    is_parallel: false,
    suite: 'conv1d',
    args: ['conv1d', dims[0]])
endforeach

##################

conv2d_cases = [
  # ['1', '3', '224', '224', '32', '3', '3', '3', '1', 'conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1'], # need 5 ops
  # ['1', '32', '112', '112', '32', '1', '3', '3', '32', 'conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32'],
  # ['1', '32', '112', '112', '64', '32', '1', '1', '1', 'conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1'],
  # ['1', '64', '112', '112', '64', '1', '3', '3', '64', 'conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64'],
  # ['1', '64', '56', '56', '128', '64', '1', '1', '1', 'conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1'],
  # ['1', '128', '56', '56', '128', '1', '3', '3', '128', 'conv2d_cc_b1_c128_h56_w56_oc128_wic1_k3x3_g128'],
  # ['1', '128', '56', '56', '128', '128', '1', '1', '1', 'conv2d_cc_b1_c128_h56_w56_oc128_wic128_k1x1_g1'],
  # ['1', '128', '28', '28', '256', '128', '1', '1', '1', 'conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1'],
  # ['1', '256', '28', '28', '256', '1', '3', '3', '256', 'conv2d_cc_b1_c256_h28_w28_oc256_wic1_k3x3_g256'],
  # ['1', '256', '28', '28', '256', '256', '1', '1', '1', 'conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1'],
  # ['1', '256', '14', '14', '512', '256', '1', '1', '1', 'conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1'],
  # ['1', '512', '14', '14', '512', '1', '3', '3', '512', 'conv2d_cc_b1_c512_h14_w14_oc512_wic1_k3x3_g512'],
  # ['1', '512', '14', '14', '512', '512', '1', '1', '1', 'conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1'],
  # ['1', '512', '7', '7', '1024', '512', '1', '1', '1', 'conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1'],
  # ['1', '1024', '7', '7', '1024', '1', '3', '3', '1024', 'conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024'],
  # ['1', '1024', '7', '7', '1024', '1024', '1', '1', '1', 'conv2d_cc_b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1'],
  # ['1', '1024', '7', '7', '1024', '1', '7', '7', '1024', 'conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024'],
  # ['1', '1024', '1', '1', '1001', '1024', '1', '1', '1', 'conv2d_cc_b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1'],
  ['1', '16', '18', '18', '16', '16', '3', '3', '1', 'conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1'], 
  ['1', '4', '9', '9', '4', '4', '1', '1', '1', 'conv2d_b1_c4_h9_w9_oc4_wic4_k1x1_g1'], 
  ['1', '16', '32', '32', '16', '16', '1', '1', '1', 'conv2d_b1_c16_h32_w32_oc16_wic16_k1x1_g1'], 
  ['1', '4', '9', '9', '4', '4', '3', '3', '1', 'conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1'], 
  ['2', '4', '9', '9', '4', '4', '3', '3', '1', 'conv2d_b2_c4_h9_w9_oc4_wic4_k3x3_g1'], 
  ['1', '1', '5', '7', '6', '1', '3', '3', '1', 'conv2d_b1_c1_h5_w7_oc6_wic1_k3x3_g1'], 
  ['1', '4', '1', '1', '2', '2', '1', '1', '2', 'conv2d_b1_c4_h1_w1_oc2_wic2_k1x1_g2'], 
  ['1', '32', '32', '32', '32', '1', '1', '1', '32', 'conv2d_b1_c32_h32_w32_oc32_wic1_k1x1_g32'], 
  ['1', '15', '5', '5', '35', '3', '3', '3', '5', 'conv2d_b1_c15_h5_w5_oc35_wic3_k3x3_g5'], 
  ['4', '15', '5', '5', '35', '3', '3', '3', '5', 'conv2d_b4_c15_h5_w5_oc35_wic3_k3x3_g5'], 
  ['1', '3', '11', '28', '3', '1', '3', '3', '3', 'conv2d_b1_c3_h11_w28_oc3_wic1_k3x3_g3'], 
  ['2', '3', '11', '28', '3', '1', '3', '3', '3', 'conv2d_b2_c3_h11_w28_oc3_wic1_k3x3_g3'], 
  ['1', '3', '5', '7', '6', '3', '2', '1', '1', 'conv2d_b1_c3_h5_w7_oc6_wic3_k2x1_g1'],
  ['1', '3', '5', '7', '6', '3', '2', '3', '1', 'conv2d_b1_c3_h5_w7_oc6_wic3_k2x3_g1'],
  ['1', '3', '5', '7', '6', '3', '2', '5', '1', 'conv2d_b1_c3_h5_w7_oc6_wic3_k2x5_g1'],
  ['1', '3', '5', '7', '6', '3', '3', '1', '1', 'conv2d_b1_c3_h5_w7_oc6_wic3_k3x1_g1'],
  ['1', '3', '5', '7', '6', '3', '3', '3', '1', 'conv2d_b1_c3_h5_w7_oc6_wic3_k3x3_g1'],
  ['1', '3', '5', '7', '6', '1', '3', '3', '3', 'conv2d_b1_c3_h5_w7_oc6_wic1_k3x3_g3'],
  ['1', '3', '5', '7', '6', '3', '3', '5', '1', 'conv2d_b1_c3_h5_w7_oc6_wic3_k3x5_g1'],
]

foreach dims : conv2d_cases
  test(dims[9],
    ops_reg_exe,
    is_parallel: false,
    suite: 'conv2d',
    args: ['conv2d', dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7], dims[8]])
endforeach

# foreach n : range(2032, 2034, 1)
#   test('conv2d_1x3_@0@x@0@_k1'.format(n),
#     ops_reg_exe,
#     is_parallel: false,
#     suite: 'conv2d',
#     timeout: 180,
#     args: ['conv2d', '1', '3', n.to_string(), n.to_string(), '6', '3', '1', '1', '1'])
# endforeach

#################

lut_cases = [
  ['sigmoid'],
  ['sin'],
  ['cos'],
  ['tan'],
  ['tanh'],
  ['celu'],
  ['selu'],
  ['swish'],
  ['softsign'],
  ['logsigmoid'],
  ['hardsigmoid'],
  ['softplus'],
  ['gelu'],
  ['quick_gelu'],
  ['elu'],
  ['relu6'],
  ['hardswish'],
  ['mish'],
  ['hardtanh'],
  ['exp'],
  ['exp2'],
  ['asin'],
  ['acos'],
  ['atan'],
  ['asinh'],
  ['acosh'],
  ['atanh'],
  ['sinh'],
  ['cosh'],
  ['silu'],
]

foreach op : lut_cases
  test('@0@_lut'.format(op[0]),
    ops_reg_exe,
    is_parallel: false,
    suite: 'lut',
    args: [op[0]])
endforeach

ops_reg_smoke_cases = [
  'add',
  'minus',
  'mul',
  'div',
  'idiv',
  'abs',
  'neg',
  'max',
  'cmple',
  'cmpgt',
  'cmpge',
  'cmplt',
  'cmpeq',
  'cmpneq',
  'roundoff',
  'rounddown',
  'where',
  'relu',
  'minpool',
  'maxpool',
  'avgpool',
  'globalminpool',
  'globalmaxpool',
  'globalavgpool',
]

sh_prog = find_program('sh')

foreach op : ops_reg_smoke_cases
  test('ops_reg_@0@'.format(op),
    sh_prog,
    is_parallel: false,
    suite: 'ops_reg',
    args: ['-c', '''
set -eu
log="/tmp/test_ops_reg_@0@.log"
echo "=== @0@ ==="
"@1@" "@0@" | tee "$log"
"grep" -q "matches CPU -> YES" "$log"
'''.format(op, ops_reg_exe.full_path())]
  )
endforeach

run_target('test_matmul',
  command: ['meson', 'test', '--maxfail=1', '-C', meson.project_build_root(), '--suite', 'matmul'],
)

run_target('test_conv1d',
  command: ['meson', 'test', '--maxfail=1', '-C', meson.project_build_root(), '--suite', 'conv1d'],
)

run_target('test_conv2d',
  command: ['meson', 'test', '--maxfail=1', '-C', meson.project_build_root(), '--suite', 'conv2d'],
)

run_target('test_lut',
  command: ['meson', 'test', '--maxfail=1', '-C', meson.project_build_root(), '--suite', 'lut'],
)

run_target('test_ops_reg',
  command: ['meson', 'test', '--maxfail=1', '-C', meson.project_build_root(), '--suite', 'ops_reg:ops_reg'],
)
