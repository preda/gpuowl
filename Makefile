# Use "make DEBUG=1" for a debug build
# The build will be stored in either build-debug or build-release subfolder
# On Windows invoke with either "make exe" or "make all"

# CXX = g++-12
# DEBUG = 1

ifeq ($(DEBUG), 1)

BIN=build-debug
CXXFLAGS = -Wall -g -std=gnu++17

else

BIN=build-release
CXXFLAGS = -Wall -O2 -std=gnu++17

endif

CPPFLAGS = -I$(BIN)

# Add any path that may be needed to find libraries here with -L<path>
LIBPATH =

LDFLAGS = -lstdc++fs -lOpenCL -lgmp -pthread ${LIBPATH}

LINK = $(CXX) $(CXXFLAGS) -o $@ ${OBJS} ${LDFLAGS}

SRCS1 = gpuid.cpp File.cpp ProofCache.cpp Proof.cpp Memlock.cpp log.cpp GmpUtil.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp Saver.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp FFTConfig.cpp AllocTrac.cpp gpuowl-wrap.cpp sha3.cpp md5.cpp

SRCS=$(addprefix src/, $(SRCS1))

OBJS = $(SRCS1:%.cpp=$(BIN)/%.o)
DEPDIR := $(BIN)/.d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

$(BIN)/gpuowl: ${OBJS}
	${LINK}

$(BIN)/gpuowl-win.exe: ${OBJS}
	${LINK} -static
	strip $@

exe: $(BIN)/gpuowl-win.exe
all: $(BIN)/gpuowl exe

clean:
	rm -rf build-debug build-release

$(BIN)/%.o : src/%.cpp $(DEPDIR)/%.d $(BIN)/version.inc
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

#$(BIN)/gpuowl-wrap.o : $(BIN)/gpuowl-wrap.cpp
#	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(OUTPUT_OPTION) $<

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

$(BIN)/version.inc: FORCE
	echo \"`git describe --tags --long --dirty --always`\" > $(BIN)/version.new
	diff -q -N $(BIN)/version.new $(BIN)/version.inc >/dev/null || mv $(BIN)/version.new $(BIN)/version.inc
	echo Version: `cat $(BIN)/version.inc`

src/gpuowl-wrap.cpp: src/*.cl
	python3 tools/expand.py src/gpuowl.cl src/gpuowl-wrap.cpp

FORCE:

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS1))))
