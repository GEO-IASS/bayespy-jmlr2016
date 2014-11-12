
INFERNET_FILES=$(MONO_PATH)/Infer.Runtime.dll,$(MONO_PATH)/Infer.Compiler.dll

all: pca_infernet.exe mog_infernet.exe

%.exe: %.cs
	mcs -r:$(INFERNET_FILES) $^

clean:
	rm -f *.exe
