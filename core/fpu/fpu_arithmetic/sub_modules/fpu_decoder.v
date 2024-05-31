module fpu_decoder(input wire [31:0]  in,
                 output wire         sign_o,
                 output wire  [7:0]  exp_o,
                 output wire [23:0]  sig_o,
                 output wire         isSubnormal,
                 output wire         isZero,
                 output wire         isInf,
                 output wire         isNaN,
                 output wire         isSignaling
                 );
// Exp = 1111_1111 & Significand != 0  -> qNaN or sNaN
// Exp = 1111_1111 & Significand == 0  -> signed infinity 
// 1 <= Exp <= 1111_1110               -> normal number (has a implicit 1)
// Exp = 0000_0000 & Significand != 0  -> subnormal number(has a implicit 0)
// Exp = 0000_0000 & Significand == 0  -> signed zero 
wire        sign;
wire  [7:0] exp;
wire [22:0] fract;

wire        isMaxExp;
wire        isZeroExp;
wire        isZeroFrac;

assign {sign,exp,fract} = in;
assign isMaxExp         = (exp == 8'd255);
assign isZeroExp        = (exp == 8'b0);
assign isZeroFrac       = (fract == 23'b0);


assign isSubnormal      = isZeroExp & !isZeroFrac;
assign isZero           = isZeroExp & isZeroFrac;
assign isInf            = isMaxExp  & isZeroFrac;
assign isNaN            = isMaxExp  & !isZeroFrac;
assign isSignaling      = isMaxExp  & !isZeroFrac & !fract[22];
assign sign_o = sign;
assign exp_o  = isSubnormal ? 8'd1 : exp;
assign sig_o  = {!isSubnormal, fract};

endmodule




