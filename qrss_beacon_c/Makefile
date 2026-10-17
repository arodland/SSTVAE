# QRSSTVAE CE reference beacon (design section 9).
#
#   make                 build $(BUILD)/qrss_ce_test
#   make check           build and run the SHA-256 self-test
#   make tables          regenerate qrss_tables.h from sstvae/qrss
#   make BUILD=/tmp/x    build somewhere else (the tests do)
#
# C99, no allocation; any C compiler will do.

CC      ?= cc
CFLAGS  ?= -O2
CFLAGS  += -std=c99 -Wall -Wextra -pedantic
BUILD   ?= build
PYTHON  ?= python3

SRCS = qrss_ce.c sha256.c test_main.c
HDRS = qrss_ce.h sha256.h qrss_tables.h

all: $(BUILD)/qrss_ce_test

$(BUILD)/qrss_ce_test: $(SRCS) $(HDRS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(SRCS)

check: $(BUILD)/qrss_ce_test
	$(BUILD)/qrss_ce_test selftest

tables:
	$(PYTHON) ../tools/gen_qrss_tables.py

clean:
	rm -rf $(BUILD)

.PHONY: all check tables clean
