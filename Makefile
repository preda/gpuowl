# Use "make DEBUG=1" for a debug build

# The build artifacts are put in the "build-release" subfolder (or "build-debug" for a debug build).

# On Windows invoke with "make exe" or "make all"

# Uncomment below as desired to set a particular compiler or force a debug build:
# CXX = g++-12
# DEBUG = 1
# or export those into environment, or pass on the command line e.g.
# make all DEBUG=1 CXX=g++-12

ifeq ($(DEBUG), 1)

BIN=build-debug

CXXFLAGS = -Wall -g -std=c++20 -static-libstdc++ -static-libgcc
STRIP=

else

BIN=build-release

CXXFLAGS = -Wall -O2 -DNDEBUG -std=c++20 -static-libstdc++ -static-libgcc
STRIP=-s

endif

SRCS1 = Primes.cpp tune.cpp CycleFile.cpp TrigBufCache.cpp Event.cpp Queue.cpp TimeInfo.cpp Profile.cpp bundle.cpp Saver.cpp SaveMan.cpp KernelCompiler.cpp Kernel.cpp gpuid.cpp File.cpp Proof.cpp log.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp FFTConfig.cpp AllocTrac.cpp sha3.cpp md5.cpp version.cpp

SRCS=$(addprefix src/, $(SRCS1))

OBJS = $(SRCS1:%.cpp=$(BIN)/%.o)
DEPDIR := $(BIN)/.d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

all: prpll

prpll: $(BIN)/prpll

amd: $(BIN)/prpll-amd

$(BIN)/prpll: ${OBJS}
	$(CXX) $(CXXFLAGS) -o $@ ${OBJS} $(LIBPATH) -lOpenCL ${STRIP}

# Instead of linking with libOpenCL, link with libamdocl64
$(BIN)/prpll-amd: ${OBJS}
	$(CXX) $(CXXFLAGS) -o $@ ${OBJS} $(LIBPATH) -lamdocl64 -L/opt/rocm/lib ${STRIP}

clean:
	rm -rf build-debug build-release

$(BIN)/%.o : src/%.cpp $(DEPDIR)/%.d
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)


# src/bundle.cpp is just a wrapping of the OpenCL sources (*.cl) as a C string.

src/bundle.cpp: genbundle.sh src/cl/*.cl
	./genbundle.sh $^ > src/bundle.cpp

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

src/version.cpp : src/version.inc

src/version.inc: FORCE
	echo \"`basename \`git describe --tags --long --dirty --always --match v/prpll/*\``\" > $(BIN)/version.new
	diff -q -N $(BIN)/version.new $@ >/dev/null || mv $(BIN)/version.new $@
	echo Version: `cat $@`

FORCE:

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS1))))
