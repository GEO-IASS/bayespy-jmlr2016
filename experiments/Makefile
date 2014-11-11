
INFERNET_PATH=$(HOME)/Workspace/Infer.NET/Bin
INFERNET_FILES=$(INFERNET_PATH)/Infer.Runtime.dll,$(INFERNET_PATH)/Infer.Compiler.dll

all: pca_infernet.exe mog_infernet.exe

%.exe: %.cs
	mcs -r:$(INFERNET_FILES) $^

clean:
	rm -f *.exe
