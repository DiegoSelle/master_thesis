.PHONY: clean dist-clean libgp_prepare libgp_build libgp_install libgp install uninstall all

TMP_OBJS = build tmp *.cpp *.c
TARGET_OBJS = *.so lib/libgp.a include/libgp

PYTHON=python

install: libgp

all: libgp

libgp_prepare:
	mkdir -p build/libgp
	mkdir -p tmp
	@if [ -e tmp/libgp ]; then\
		cd tmp/libgp; git pull;\
	else\
		git clone https://github.com/crey0/libgp.git tmp/libgp;\
	fi
	cd build/libgp && cmake ../../tmp/libgp

libgp_build: libgp_prepare
	$(MAKE) -C build/libgp

libgp_install: libgp_build
	mkdir -p include/libgp
	mkdir -p lib
	cp -r tmp/libgp/include/* include/libgp/
	cp  build/libgp/libgp.a lib/libgp.a

libgp: libgp_install
	$(PYTHON) setup.py build_ext --inplace


clean:
	rm -rfv $(TMP_OBJS)

uninstall: clean
	rm -rfv $(TARGET_OBJS)

