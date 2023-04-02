# -*- mode:Python -*-

python3 = import('python').find_installation('python3')

wrap = generator(expander, output:'@PLAINNAME@.cpp', arguments:['@INPUT@', '@OUTPUT@'])

gpuowl_wrap = wrap.process('gpuowl.cl')

srcs = files('ProofCache.cpp Proof.cpp Memlock.cpp log.cpp md5.cpp sha3.cpp AllocTrac.cpp GmpUtil.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp Saver.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp'.split())
