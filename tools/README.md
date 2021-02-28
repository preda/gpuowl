Simple ISA instruction-counts diff tool:
if you dumped ISA to two folders A and B, to see the simple diff use:
```sh
./tools/delta.sh A/*.s B/*.s
```

Example output:

```
~/gpuowl$ ./tools/delta.sh tmp4/5M_0_gfx906.s tmp6/5M_0_gfx906.s
tailFused : s_mov_b32 119				      |	tailFused : s_mov_b32 113
tailFused : v_add_f64 443				      |	tailFused : v_add_f64 437
tailFused : v_mul_f64 176				      |	tailFused : v_mul_f64 170
```
