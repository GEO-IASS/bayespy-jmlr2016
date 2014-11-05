
INFERNET_PATH=$(HOME)/Workspace/Infer.NET/Bin
INFERNET_FILES=$(INFERNET_PATH)/Infer.Runtime.dll,$(INFERNET_PATH)/Infer.Compiler.dll

all: pca_infernet.exe

pca_infernet.exe: pca_infernet.cs
	mcs -r:$(INFERNET_FILES) pca_infernet.cs

clean:
	rm -f *.exe
