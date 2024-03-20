# GpuOwl code style, C++ and OpenCL

- indent 2 spaces
- no "tab" chars in source code -- configure the editor to convert tabs to spaces
- open bracket on the same line
- always curly-braces {} after if and else
- no space between function name and open parens (e.g. in a function call)
- one space between if/while/for and open parens

Example:

```C++
int example(int value) {
  if (value > 0) {
    return value;
  } else {
    return -value + 1;
  }
}

```
