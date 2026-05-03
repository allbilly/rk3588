set pagination off
set breakpoint pending on

break rknn_matmul_run
commands
  printf "begin rknn_matmul_run gem dump\n"
  shell python3 dump.py 1
  shell python3 dump.py 2
  shell python3 dump.py 3
  shell python3 dump.py 4
  shell python3 dump.py 5
  shell python3 dump.py 6
  printf "end rknn_matmul_run gem dump\n"
  continue
end

run
q
