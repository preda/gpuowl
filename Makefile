BIN=build
CXXFLAGS = -Wall -g -O3 -std=gnu++17
CPPFLAGS = -I$(BIN)

LIBPATH = -L/opt/rocm-5.7.2/opencl/lib -L/opt/rocm-5.1.1/opencl/lib -L/opt/rocm-4.0.0/opencl/lib -L/opt/rocm-3.3.0/opencl/lib/x86_64 -L/opt/rocm/opencl/lib -L/opt/rocm/opencl/lib/x86_64 -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L.

LDFLAGS = -lstdc++fs -lOpenCL -lgmp -pthread ${LIBPATH}

LINK = $(CXX) $(CXXFLAGS) -o $@ ${OBJS} ${LDFLAGS}

SRCS1 = ProofCache.cpp Proof.cpp Memlock.cpp log.cpp GmpUtil.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp Saver.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp FFTConfig.cpp AllocTrac.cpp gpuowl-wrap.cpp sha3.cpp md5.cpp

SRCS=$(addprefix src/, $(SRCS1))

OBJS = $(SRCS1:%.cpp=$(BIN)/%.o)
DEPDIR := $(BIN)/.d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@


$(BIN)/gpuowl: ${OBJS}
	${LINK}

gpuowl-win.exe: ${OBJS}
	${LINK} -static
	strip $@

clean:
	rm -rf $(BIN) gpuowl-win.exe

$(BIN)/%.o : src/%.cpp $(DEPDIR)/%.d $(BIN)/version.inc
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

$(BIN)/gpuowl-wrap.o : $(BIN)/gpuowl-wrap.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(OUTPUT_OPTION) $<

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

$(BIN)/version.inc: FORCE
	echo \"`git describe --tags --long --dirty --always`\" > $(BIN)/version.new
	diff -q -N $(BIN)/version.new $(BIN)/version.inc >/dev/null || mv $(BIN)/version.new $(BIN)/version.inc
	echo Version: `cat $(BIN)/version.inc`

$(BIN)/gpuowl-wrap.cpp: src/gpuowl.cl
	python3 tools/expand.py src/gpuowl.cl $(BIN)/gpuowl-wrap.cpp

FORCE:

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS1))))


# include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename D.cpp)))
