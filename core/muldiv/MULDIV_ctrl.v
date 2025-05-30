module MULDIV_ctrl (
    input clk,
    input start,
    input reset,
    input muldiv_sel,
    input [5:0] AB_status,
    input div_rdy,
    input [1:0] op_mul,
    input [1:0] op_div,
    input [31:0] A,
    input [31:0] B,
    input [31:0] A_2C,
	input [31:0] B_2C,
    output reg div_start,
    output reg reg_AB_en,
    output reg reg_muldiv_en,
    output reg mux_muldiv_sel,
    output reg mux_muldiv_out_sel,
    output reg mux_fastres_sel,
    output reg [31:0] fastres,
    output reg muldiv_done
	);

    wire Am1, Bm1, A0, B0, A1, B1;
    parameter IDLE = 3'd0, DIV = 3'd1, DIV_out = 3'd2,
    MUL1 = 3'd3, MUL2 = 3'd4, MUL_out = 3'd5;

    reg [2:0] current_state , next_state;
    reg mux_fastres_sel_temp;

    //assign {Bm1, B0, B1, Am1, A0, A1} = AB_status; //The order doesn't match with the values below!!
    assign {Bm1, B1, B0, Am1, A1, A0} = AB_status;

always @ (posedge clk or negedge reset) begin
    if(!reset)
        current_state <= IDLE;
    else
        current_state <= next_state;
end

always @*
    mux_fastres_sel = mux_fastres_sel_temp;

always @*
begin
    casez(AB_status)
        // A = 0 cases
        6'b???001: begin
        	// if(AB_status[5:3] == 3'b000 ||
        	//    AB_status[5:3] == 3'b001 ||
        	//    AB_status[5:3] == 3'b010 ||
        	//    AB_status[5:3] == 3'b100) begin

            // 0/0 edge case is missing, let's add it here
        	if(AB_status[5:3] == 3'b000 ||
        	   AB_status[5:3] == 3'b010 ||
        	   AB_status[5:3] == 3'b100) begin

            	fastres = 32'd0;
           		mux_fastres_sel_temp = 1'b1;
       		end
            else if(AB_status[5:3] == 3'b001) begin //A=0 AND B=0 cases
                mux_fastres_sel_temp = 1'b1;

                if(muldiv_sel == 1'b0) fastres = 32'd0; //0*0=0 still holds
                else begin
                    if(op_div[1])
                    fastres = 32'h0; // REM(x/0) = x, so REM(0/0) = 0
                    else
                    fastres = 32'hFFFFFFFF; // 0/0=-1
                end
            end
           	else begin
		       	fastres = 32'd0;
		        mux_fastres_sel_temp = 1'b1;
          	end

        end

        // A = 1 cases
        6'b000010: begin
            if(muldiv_sel == 1'b0) begin
                if(op_mul == 2'b00)
                    fastres = B;
                else if(op_mul == 2'b01) //For MULH, result is the sign bit of B, repeated.
                    fastres = {32{B[31]}};
                else
                    fastres = 32'd0;
            end
            else begin
                if (op_div[1] == 1'b0)
                    fastres = 32'd0;
                else
                    fastres = 32'd1;
            end
            mux_fastres_sel_temp = 1'b1;
        end

        // A = -1 cases
        6'b000100: begin
            if(muldiv_sel == 1'b0) begin
				if(op_mul == 2'b00) //00: MUL, 01:MULH, 10:MULHSU, 11:MULHU
                    fastres = B_2C;
                else if(op_mul == 2'b01) //For MULH, result is the sign bit of the 2's comp of B, repeated.
                    if(B_2C == B) fastres = 32'd0; //For the edge case 0x80000000, 2's comp of B is equal to itself, but its upper bits should be zero.
                    else fastres = {32{B_2C[31]}};
                else
                    fastres = 32'hffffffff; //Not used for MULHU, but used for MULHSU
			end
			else begin
				if (op_div[1] == 1'b0)
					fastres = 32'd0;
				else
					fastres = 32'hffffffff;
			end
            //If operation is MULHU, or DIVU, this case can't be skipped with a fastres
            if((muldiv_sel == 1'b0 && op_mul == 2'b11) || (muldiv_sel == 1'b1 && op_div[0] == 1'b1))
			    mux_fastres_sel_temp = 1'b0;
            else
			    mux_fastres_sel_temp = 1'b1;

        end

        // A = 1 and B = 1 case
        6'b010010: begin
            if(muldiv_sel == 1'b0) begin
                if(op_mul == 2'b00)
                    fastres = 32'd1;
                else
                    fastres = 32'd0;
            end
            else begin
                if (op_div[1] == 1'b0)
                    fastres = 32'd1;
                else
                    fastres = 32'd0;
            end
            mux_fastres_sel_temp = 1'b1;
        end

        // A = 1 and B = -1 case
        6'b100010: begin
			if(muldiv_sel == 1'b0) begin
				fastres = 32'hffffffff;
			end
			else begin
				if (op_div[1] == 1'b0)
					fastres = 32'hffffffff;
				else
					fastres = 32'd0;
			end
            mux_fastres_sel_temp = 1'b1;
        end

        // A = -1 and B = 1 case
       	6'b010100: begin
			if(muldiv_sel == 1'b0) begin
				fastres = 32'hffffffff;
			end
			else begin
				if (op_div[1] == 1'b0)
					fastres = 32'hffffffff;
				else
					fastres = 32'd0;
			end
            mux_fastres_sel_temp = 1'b1;
        end

        // A = -1 and B = -1 cases
        6'b100100: begin
			if(muldiv_sel == 1'b0) begin
				if(op_mul == 2'b00)
                    fastres = 32'd1;
                else
                    fastres = 32'd0;
			end
			else begin
				if (op_div[1] == 1'b0)
					fastres = 32'd1;
				else
					fastres = 32'd0;
			end
			mux_fastres_sel_temp = 1'b1;
        end

        // B = 1 case
        6'b010000: begin
            if(muldiv_sel == 1'b0) begin
                if(op_mul == 2'b00)
                    fastres = A;
                else if((op_mul == 2'b01) || (op_mul == 2'b10))
                    //For MULH, result is the sign bit of A, repeated.
                    fastres = {32{A[31]}}; //For MULHSU operations, carry the sign bit to the result, but only since A is signed, so we don't repeat this for A=1 case
                else
                    fastres = 32'd0;
            end
            else begin
                if (op_div[1] == 1'b0)
                    fastres = A;
                else
                    fastres = 32'd0;
            end
            mux_fastres_sel_temp = 1'b1;
        end

        // B = -1 case
        6'b100000: begin
            if(muldiv_sel == 1'b0) begin
				if(op_mul == 2'b00) //00: MUL, 01:MULH, 10:MULHSU, 11:MULHU
                    fastres = A_2C;
                else if(op_mul == 2'b01) //For MULH, result is the sign bit of the 2's comp of B, repeated.
                    if(A_2C == A) fastres = 32'd0; //For the edge case 0x80000000, 2's comp of B is equal to itself, but its upper bits should be zero.
                    else fastres = {32{A_2C[31]}};
                else
                    fastres = 32'hffffffff; //Not used for MULHU, but used for MULHSU
			end
			else begin
				if (op_div[1] == 1'b0)
					fastres = A_2C;
				else
					fastres = 32'd0;
			end
            //If operation is MULHU, or DIVU, this case can't be skipped with a fastres
            if((muldiv_sel == 1'b0 && op_mul == 2'b11) || (muldiv_sel == 1'b1 && op_div[0] == 1'b1))
			    mux_fastres_sel_temp = 1'b0;
            else
			    mux_fastres_sel_temp = 1'b1;

        end

        // B = 0 cases
        6'b001??0: begin
        	if(AB_status[2:1] != 2'b11) begin

		        if(muldiv_sel == 1'b0)
		            fastres = 32'd0;
		        else begin
		            if (op_div[1] == 1'b0)
		                fastres = 32'hffffffff;
		            else
		                fastres = A;
		        end
		        mux_fastres_sel_temp = 1'b1;
		  	end

		  	else begin
		  		fastres = 32'd0;
		  		mux_fastres_sel_temp = 1'b0;
		  	end
        end

        // non-fast result case
        6'b000000: begin
            mux_fastres_sel_temp = 1'b0;
            fastres = 32'd0;
        end

        // impossible cases
		default: begin
            mux_fastres_sel_temp = 1'b1;
            fastres = 32'd0;
        end
    endcase
end

always @*
begin
case(current_state)

    IDLE: begin
        if(start == 1'b1) begin
            div_start = 1'b0;
            reg_muldiv_en = 1'b0;
            mux_muldiv_sel = 1'b0;
            mux_muldiv_out_sel = 1'b0;

            if(mux_fastres_sel_temp) begin
                reg_AB_en = 1'b0;
                muldiv_done = 1'b1;
                next_state = IDLE;
            end

            else begin
                reg_AB_en = 1'b1;
                muldiv_done = 1'b0;
                if(muldiv_sel)
                    next_state = DIV;
                else
                    next_state = MUL1;
            end
        end

        else begin
            div_start = 1'b0;
            reg_AB_en = 1'b0;
            reg_muldiv_en = 1'b0;
            mux_muldiv_sel = 1'b0;
            mux_muldiv_out_sel = 1'b0;
            muldiv_done = 1'b0;
            next_state = IDLE;
       end
    end
    DIV: begin
        reg_AB_en = 1'b0;
        mux_muldiv_sel = 1'b1;
        mux_muldiv_out_sel = 1'b0;
        muldiv_done = 1'b0;

        if (div_rdy == 1'b1) begin
            div_start = 1'b0;
            reg_muldiv_en = 1'b1;
            next_state = DIV_out;
        end

        else begin
            div_start = 1'b1;
            reg_muldiv_en = 1'b0;
            next_state = DIV;
        end
    end

    DIV_out: begin
        div_start = 1'b0;
        reg_AB_en = 1'b0;
        reg_muldiv_en = 1'b0;
        mux_muldiv_sel = 1'b0;
        mux_muldiv_out_sel = 1'b1;
        muldiv_done = 1'b1;
        next_state = IDLE;
    end

    MUL1: begin
        div_start = 1'b0;
        reg_AB_en = 1'b0;
        reg_muldiv_en = 1'b0;
        mux_muldiv_sel = 1'b0;
        mux_muldiv_out_sel = 1'b0;
        muldiv_done = 1'b0;
        next_state = MUL2;
    end

    MUL2: begin
        div_start = 1'b0;
        reg_AB_en = 1'b0;
        reg_muldiv_en = 1'b1;
        mux_muldiv_sel = 1'b0;
        mux_muldiv_out_sel = 1'b0;
        muldiv_done = 1'b0;
        next_state = MUL_out;
    end

    MUL_out: begin
        div_start = 1'b0;
        reg_AB_en = 1'b0;
        reg_muldiv_en = 1'b1;
        mux_muldiv_sel = 1'b0;
        mux_muldiv_out_sel = 1'b0;
        muldiv_done = 1'b1;
        next_state = IDLE;
    end

    default: begin
        div_start = 1'b0;
        reg_AB_en = 1'b0;
        reg_muldiv_en = 1'b0;
        mux_muldiv_sel = 1'b0;
        mux_muldiv_out_sel = 1'b0;
        muldiv_done = 1'b0;
        next_state = IDLE;
    end
    endcase
end

endmodule
