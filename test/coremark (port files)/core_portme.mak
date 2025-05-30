# Copyright 2018 Embedded Microprocessor Benchmark Consortium (EEMBC)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Original Author: Shay Gal-on

#File : core_portme.mak

# Flag : OUTFLAG
#	Use this flag to define how to to get an executable (e.g -o)
OUTFLAG= -o
# Flag : CC
#	Use this flag to define compiler to use
CC 		= riscv32-unknown-elf-gcc
# Flag : LD
#	Use this flag to define compiler to use
LD		= riscv32-unknown-elf-ld
# Flag : AS
#	Use this flag to define compiler to use
AS		= riscv32-unknown-elf-as
# Flag : CFLAGS
#	Use this flag to define compiler options. Note, you can add compiler options from the command line using XCFLAGS="other flags"
PORT_CFLAGS = -march=rv32imf_zicsr_zicntr -mabi=ilp32f -O3 -g -ggdb -nostdlib -nostartfiles -ffunction-sections -fdata-sections -ffreestanding -Wl,--gc-sections -W -Wall -Wextra
FLAGS_STR = "$(PORT_CFLAGS) $(XCFLAGS) $(XLFLAGS) $(LFLAGS_END)"
CFLAGS = $(PORT_CFLAGS) -I$(PORT_DIR) -I. -DFLAGS_STR=\"$(FLAGS_STR)\" 
#Flag : LFLAGS_END
#	Define any libraries needed for linking or other flags that should come at the end of the link line (e.g. linker scripts). 
#	Note : On certain platforms, the default clock_gettime implementation is supported but requires linking of librt.
SEPARATE_COMPILE=1
# Flag : SEPARATE_COMPILE
# You must also define below how to create an object file, and how to link.
OBJOUT 	= -o
LFLAGS 	= -m elf32lriscv -T ./riscv/linksc.ld
ASFLAGS = -march=rv32imf_zicsr -mabi=ilp32f
OFLAG 	= -o
COUT 	= -c

LFLAGS_END = -lm -lgcc -L /opt/riscv/lib/gcc/riscv32-unknown-elf/13.2.0/ -L /opt/riscv/riscv32-unknown-elf/lib/
# Flag : PORT_SRCS
# 	Port specific source files can be added here
#	You may also need cvt.c if the fcvt functions are not provided as intrinsics by your compiler!
PORT_SRCS = $(PORT_DIR)/core_portme.c $(PORT_DIR)/ee_printf.c $(PORT_DIR)/crt0.s $(PORT_DIR)/cvt.c
PORT_OBJS = $(PORT_DIR)/core_portme$(OEXT) $(PORT_DIR)/ee_printf$(OEXT) $(PORT_DIR)/crt0$(OEXT) $(PORT_DIR)/cvt$(OEXT)
vpath %.c $(PORT_DIR)
vpath %.s $(PORT_DIR)

# Flag : LOAD
#	For a simple port, we assume self hosted compile and run, no load needed.

# Flag : RUN
#	For a simple port, we assume self hosted compile and run, simple invocation of the executable

LOAD = 
RUN = echo "Please set LOAD to the process of running the executable (e.g. via jtag, or board reset)"

OEXT = .o
EXE = .elf

$(OPATH)$(PORT_DIR)/%$(OEXT) : %.c
	$(CC) $(CFLAGS) $(XCFLAGS) $(COUT) $< $(OBJOUT) $@

$(OPATH)%$(OEXT) : %.c
	$(CC) $(CFLAGS) $(XCFLAGS) $(COUT) $< $(OBJOUT) $@

$(OPATH)$(PORT_DIR)/%$(OEXT) : %.s
	$(AS) $(ASFLAGS) $< $(OBJOUT) $@

# Target : port_pre% and port_post%
# For the purpose of this simple port, no pre or post steps needed.

.PHONY : port_prebuild port_postbuild port_prerun port_postrun port_preload port_postload
port_postbuild:
	riscv32-unknown-elf-objcopy -O binary coremark.elf coremark.bin
	./rom_generator coremark.bin

# FLAG : OPATH
# Path to the output folder. Default - current folder.
OPATH = ./
MKDIR = mkdir -p

