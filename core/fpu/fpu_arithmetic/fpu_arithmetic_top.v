module fpu_arithmetic_top
(
    // inputs
    input        clk,
    input        reset,
    input        start,
    input  [4:0] op,
    input  [2:0] rounding_mode,
    input [31:0] A,
    input [31:0] B,
    input        rs2_lsb,
    // outputs
    output [31:0] fpu_arith_out,
    output        done,
    output        overflow,
    output        underflow,
    output        invalid,
    output        inexact,
    output        div_by_zero
    
);
/*
       OP   |     operation
   5'b00000 |      FADD
   5'b00001 |      FSUB
   5'b00010 |      FMUL
   5'b00011 |      FDIV
   5'b01011 |      FSQRT
   5'b00100 |      FSGNJ, FSGNJN , FSGNJX
   5'b00101 |      FMIN, FMAX
   5'b11000 |      FCVT.W.S, FCVT.WU.S
   5'b11010 |      FCVT.S.W, FCVT.S.WU
   5'b10100 |      FEQ, FLT, FLE

*/


// decoder signals
wire       sign_A, sign_B;
wire [7:0] exp_A, exp_B;
wire[23:0] sig_A, sig_B;
wire       isSubnormalA, isSubnormalB;
wire       isZeroA, isZeroB;
wire       isInfA, isInfB, isSignalingA;
wire       isNaNA, isNaNB, isSignalingB;

fpu_decoder  decA(.in(A), .sign_o(sign_A), .exp_o(exp_A), .sig_o(sig_A), .isSubnormal(isSubnormalA), .isZero(isZeroA), .isInf(isInfA), .isNaN(isNaNA), .isSignaling(isSignalingA));          
fpu_decoder  decB(.in(B), .sign_o(sign_B), .exp_o(exp_B), .sig_o(sig_B), .isSubnormal(isSubnormalB), .isZero(isZeroB), .isInf(isInfB), .isNaN(isNaNB), .isSignaling(isSignalingB)); 

wire       isSignaling;
wire       isBothSubnorm;
assign isSignaling = isSignalingA | isSignalingB;
assign isBothSubnorm = isSubnormalA & isSubnormalB;


// ADD-SUB signals

wire        overflow_add;
wire        underflow_add;
wire        invalid_add;
wire        inexact_add;
wire [31:0] add_sub_out;
wire        sub_op;
assign sub_op = op[0] ? 1'b1 : 1'b0;

fpu_add_sub fas(sign_A, sign_B, exp_A, exp_B, sig_A, sig_B, isZeroA, isZeroB, isInfA, isInfB, isNaNA, isNaNB, isSignaling, sub_op, rounding_mode, overflow_add, underflow_add, invalid_add, inexact_add, add_sub_out);


// MUL-DIV-SQRT signals


wire        mds_done;
wire        overflow_mds;
wire        underflow_mds;
wire        invalid_mds;
wire        inexact_mds;
wire        div_by_zero_mds;
wire [31:0] mds_out;
wire [1:0]  mds_op;

assign mds_op = op[3:0] == 4'b1011 ? 2'b10 : // sqrt
                op[3:0] == 4'b0011 ? 2'b01 : // div
                                     2'b00 ; // mul


fpu_mds_top fpu_mds_top(clk, start, reset, rounding_mode, isSubnormalA, isZeroA, isZeroB, isInfA, isInfB, isNaNA, isNaNB, isSignaling, sign_A, sign_B, exp_A, exp_B, sig_A, sig_B, mds_op, mds_out, mds_done, overflow_mds, underflow_mds, invalid_mds, inexact_mds, div_by_zero_mds);


// FPU-COMPARE signals
wire comp_out;
wire invalid_comp;
// rounding_mode[1:0] is used  for compare function

fpu_compare fpu_compare(rounding_mode[1:0], sign_A, sign_B, exp_A, exp_B, sig_A, sig_B, isNaNA, isNaNB, isSignaling, comp_out, invalid_comp);

//FPU-MIN_MAX signals
wire [31:0] min_max_out;
wire invalid_min_max;
// rounding mode's lsb is determine min or max operatin


fpu_min_max fpu_min_max(rounding_mode[0], sign_A, sign_B, exp_A, exp_B, sig_A, sig_B, isNaNA, isNaNB, isSignaling, min_max_out, invalid_min_max); 

//FPU-SIGN INJECTION signals
wire sign_O_inj;
// rounding mode's lsb is determine injection operation

fpu_sign_inj fpu_sign_inj(rounding_mode[1:0], sign_A, sign_B, sign_O_inj);

//FPU-CONVERT TO INTEGER signals
wire is_exp_neg;
wire [31:0] cvt_to_int_out;
wire overflow_cvt_to_int;
assign is_exp_neg = exp_A[7] ? 1'b0 : (&exp_A[6:0] ? 1'b1 : 1'b0);
fpu_cvt_to_int fpu_cvt_to_int(rs2_lsb, is_exp_neg, rounding_mode, isNaNA, isInfA, sign_A, exp_A, sig_A, cvt_to_int_out, overflow_cvt_to_int);


//FPU-CONVERT TO FLOAT signals
wire [31:0] cvt_to_float_out;
fpu_cvt_to_float fpu_cvt_to_float(rs2_lsb, rounding_mode, A, cvt_to_float_out);

//FPU-CLASSIFIER signals
wire[9:0] classifier_out;

fpu_classifier fpu_classifier(sign_A, isSubnormalA, isZeroA, isInfA, isNaNA, isSignalingA, classifier_out);






// exception flag assignments

assign overflow    = op == 5'b00000 | op == 5'b00001                    ? overflow_add          : // add, sub
                     op == 5'b00010 | op == 5'b00011  | op == 5'b01011  ? overflow_mds          : // mul, div, sqrt
                     op == 5'b00100                                     ? 1'b0                  : // sign injection
                     op == 5'b00101                                     ? 1'b0                  : // min, max
                     op == 5'b11000                                     ? overflow_cvt_to_int  : // convert to int
                     op == 5'b10100                                     ? 1'b0                  : // equ, lt, le
                     1'b0;




assign underflow   = op == 5'b00000 | op == 5'b00001                    ? underflow_add : // add, sub
                     op == 5'b00010 | op == 5'b00011  | op == 5'b01011  ? underflow_mds : // mul, div, sqrt
                     op == 5'b00100                                     ? 1'b0          : // sign injection
                     op == 5'b00101                                     ? 1'b0          : // min, max
                     op == 5'b11000                                     ? 1'b0          : // convert to int
                     op == 5'b10100                                     ? 1'b0          : // equ, lt, le
                     1'b0;

assign invalid     = op == 5'b00000 | op == 5'b00001                    ? invalid_add     : // add, sub
                     op == 5'b00010 | op == 5'b00011  | op == 5'b01011  ? invalid_mds     : // mul, div, sqrt
                     op == 5'b00100                                     ? 1'b0            : // sign injection
                     op == 5'b00101                                     ? invalid_min_max : // min, max
                     op == 5'b11000                                     ? 1'b0            : // convert to int
                     op == 5'b10100                                     ? invalid_comp    : // equ, lt, le
                     1'b0;

assign inexact     = op == 5'b00000 | op == 5'b00001                    ? inexact_add : // add, sub
                     op == 5'b00010 | op == 5'b00011  | op == 5'b01011  ? inexact_mds : // mul, div, sqrt
                     op == 5'b00100                                     ? 1'b0        : // sign injection
                     op == 5'b00101                                     ? 1'b0        : // min, max
                     op == 5'b11000                                     ? 1'b0        : // convert to int
                     op == 5'b10100                                     ? 1'b0        : // equ, lt, le
                     1'b0;
                     
assign div_by_zero = op == 5'b00010 | op == 5'b00011  | op == 5'b01011  ? div_by_zero_mds : // mul, div, sqrt
                     1'b0;

// assignment of final output

assign done        = !start ? 1'b0 :
                    op == 5'b00010 | op == 5'b00011  | op == 5'b01011 ? mds_done : // mul, div, sqrt
                    1'b1;



assign fpu_arith_out = op == 5'b00000 | op == 5'b00001                    ? add_sub_out                      : // add, sub
                       op == 5'b00010 | op == 5'b00011  | op == 5'b01011  ? mds_out                          : // mul, div, sqrt
                       op == 5'b00100                                     ? {sign_O_inj, exp_B, sig_B[22:0]} : // sign injection
                       op == 5'b00101                                     ? min_max_out                      : // min, max
                       op == 5'b11000                                     ? cvt_to_int_out                   : // convert to int
                       op == 5'b11010                                     ? cvt_to_float_out                 : // convert to float
                       op == 5'b10100                                     ? {31'b0,comp_out}                 : // equ, lt, le
                       op == 5'b11100 & rounding_mode[0]                  ? {22'b0,classifier_out}           : // classifier out
                    (op == 5'b11100 | op == 5'b11110) & !(|rounding_mode) ? A                                :
                       32'b0;











endmodule