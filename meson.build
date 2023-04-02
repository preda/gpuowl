# -*- mode:Python -*-

project('gpuowl', 'cpp', default_options:['cpp_std=c++17', 'buildtype=debugoptimized'])

expander = find_program('./tools/expand.py')

subdir('src')

version = vcs_tag(input:'version.inc.in', output:'version.inc')

cpp = meson.get_compiler('cpp')
amdocl = cpp.find_library('amdocl64', dirs:['/opt/rocm/lib'])

executable('gpuowl', sources: srcs + [gpuowl_wrap, version], dependencies:[amdocl, dependency('gmp')])


# Meson experiments below:

#executable('gpuowl', sources:srcs, link_with:[amdocl, dependency('gmp')])
# amdocl = declare_dependency(link_args : ['-L/opt/rocm-3.3.0/opencl/lib/x86_64', '-lamdocl64'])
# dependency('amdocl64')
# thread_dep = dependency('threads')
# gpuowl_wrap = custom_target('gpuowl_wrap', command:[python3, '@INPUT@', '@OUTPUT@'], input:['tools/expand.py', 'gpuowl.cl'], output:'gpuowl-wrap.cpp')
