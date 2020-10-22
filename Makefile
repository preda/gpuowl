CXXFLAGS = -Wall -O2 -std=c++17

LIBPATH = -L/opt/rocm/opencl/lib -L/opt/rocm-3.3.0/opencl/lib/x86_64 -L/opt/rocm-3.1.0/opencl/lib/x86_64 -L/opt/rocm/opencl/lib/x86_64 -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L.

LDFLAGS = -lstdc++fs -lOpenCL -lgmp -pthread ${LIBPATH}

LINK = $(CXX) -o $@ ${OBJS} ${LDFLAGS}

SRCS = Proof.cpp Pm1Plan.cpp B1Accumulator.cpp Memlock.cpp log.cpp GmpUtil.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp checkpoint.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp FFTConfig.cpp AllocTrac.cpp gpuowl-wrap.cpp sha3.cpp md5.cpp
OBJS = $(SRCS:%.cpp=%.o)
DEPDIR := .d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@


gpuowl: ${OBJS}
	${LINK}

gpuowl-win.exe: ${OBJS}
	${LINK} -static
	strip $@

D:	D.o Pm1Plan.o log.o common.o timeutil.o
	$(CXX) -o $@ $^ ${LDFLAGS}

clean:
	rm -f ${OBJS} gpuowl gpuowl-win.exe

%.o : %.cpp
%.o : %.cpp $(DEPDIR)/%.d gpuowl-wrap.cpp version.inc
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

version.inc: FORCE
	echo \"`git describe --tags --long --dirty --always`\" > version.new
	diff -q -N version.new version.inc >/dev/null || mv version.new version.inc
	echo Version: `cat version.inc`

gpuowl-expanded.cl: gpuowl.cl
	./tools/expand.py < gpuowl.cl > gpuowl-expanded.cl

gpuowl-wrap.cpp: gpuowl-expanded.cl head.txt tail.txt
	cat head.txt gpuowl-expanded.cl tail.txt > gpuowl-wrap.cpp

FORCE:

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS))))
include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename D.cpp)))
