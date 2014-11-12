
INFERNET_FILES=$(MONO_PATH)/Infer.Runtime.dll,$(MONO_PATH)/Infer.Compiler.dll

all: pca_infernet.exe

pca_infernet.exe: pca_infernet.cs
	mcs -r:$(INFERNET_FILES) pca_infernet.cs

clean:
	rm -f *.exe
