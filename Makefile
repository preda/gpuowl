SRCS = Pm1Plan.cpp GmpUtil.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp checkpoint.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp FFTConfig.cpp clpp.cpp

CXXFLAGS = -Wall -O2 -std=c++17

LIBPATH = -L/opt/rocm/opencl/lib/x86_64 -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32 -L.

LDFLAGS = -lOpenCL -lgmp -lstdc++fs -pthread ${LIBPATH}

LINK = $(CXX) -o $@ ${OBJS} ${LDFLAGS}


OBJS = $(SRCS:%.cpp=%.o)
DEPDIR := .d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@


gpuowl: ${OBJS}
	${LINK}

gpuowl-win: ${OBJS}
	{LINK} -static
	strip $@

%.o : %.cpp
%.o : %.cpp $(DEPDIR)/%.d
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

version.inc: FORCE
	echo \"`git describe --long --dirty --always`\" > version.new
	diff -q -N version.new version.inc >/dev/null || mv version.new version.inc
	echo Version: `cat version.inc`

FORCE:

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS))))
