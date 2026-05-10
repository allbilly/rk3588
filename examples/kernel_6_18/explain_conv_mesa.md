```
EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_X_STRIDE(task->stride_x) |
                         CNA_CONV_CON3_CONV_Y_STRIDE(task->stride_y));
```

where is the stride from                        

```
def fill_task(op, task):
    task.stride_x = op.stride
    task.stride_y = op.stride
```

op.stride
```
    while s < op.input_height:
        prev = op.tasks[-1]
        while s <= prev.bottom_slice:
            s += op.stride
        if s > prev.bottom_slice:
            s -= op.stride
```

op.stride forms top_slice
```
t.top_slice = min(s, prev.bottom_slice) - (op.weights_height - 1) + op.stride
```

and forms input_height, output_width, output_height
```
cur.input_height = cur.bottom_slice - cur.top_slice + 1

cur.output_width = (cur.input_width + cur.pad_left + cur.pad_right -
                    op.weights_width) // op.stride + 1
cur.output_height = (cur.input_height + cur.pad_top + cur.pad_bottom -
                        op.weights_height) // op.stride + 1

emit(REG_CNA_DATA_SIZE0,
        CNA_DATA_SIZE0_DATAIN_WIDTH(task.input_width) |
        CNA_DATA_SIZE0_DATAIN_HEIGHT(task.input_height))
```