// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// HEADERS
+incdir+/home/deniz/riscv-dv/uvm-1.2/src
+incdir+${RISCV_DV_ROOT}/src
+incdir+${RISCV_DV_ROOT}/src/isa
+incdir+${RISCV_DV_ROOT}/test

// SOURCES
/home/deniz/riscv-dv/uvm-1.2/src/uvm_pkg.sv
${RISCV_DV_ROOT}/src/riscv_signature_pkg.sv
${RISCV_DV_ROOT}/src/riscv_instr_pkg.sv
${RISCV_DV_ROOT}/src/riscv_instr_gen_config.sv
${RISCV_DV_ROOT}/src/isa/riscv_instr.sv
${RISCV_DV_ROOT}/src/isa/rv32i_instr.sv
${RISCV_DV_ROOT}/src/isa/rv32m_instr.sv
${RISCV_DV_ROOT}/src/isa/rv32c_instr.sv
${RISCV_DV_ROOT}/test/riscv_instr_test_pkg.sv
${RISCV_DV_ROOT}/test/riscv_instr_gen_tb_top.sv
