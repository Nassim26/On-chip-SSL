
module MAC # (
	parameter DATA_RES = 8, 
	parameter WEIGHT_RES = 8, 
	parameter KERNEL_SIZE = 9,
	parameter LINE_WIDTH = 28
) ( 
	input wire clk_i, 
	input wire resetn_i, 
	input wire [DATA_RES*KERNEL_SIZE-1:0] pixel_grid_i, 
	input wire data_valid_i, 
	input wire signed [WEIGHT_RES*(KERNEL_SIZE+1)-1:0] kernel_i, 
	
	output reg [DATA_RES-1:0] pixel_o, 
	output reg pixel_valid_o 
); 

	localparam FxP_RES = 4; // Fixed-Point Resolution of weights, e.g. FxP_RES = 4 <=> minimum res. of 1/16 

	integer i;
	reg signed [WEIGHT_RES-1:0] kernel [KERNEL_SIZE-1:0];
	reg signed [2*WEIGHT_RES-1:0] mul [KERNEL_SIZE-1:0];
	reg signed [19:0] acc_comb; 
	reg signed [19:0] acc; 
	reg mul_valid; 
	reg acc_valid; 
	
	always @(posedge clk_i, posedge resetn_i) begin 
		if (!resetn_i) begin 
			for (i=0;i<9;i=i+1) begin 
				kernel[i] <= 'd0; 
			end 
		end else begin 
			for (i=0;i<9;i=i+1) begin 
				kernel[i] <= kernel_i[i*WEIGHT_RES+:WEIGHT_RES]; 
			end 
		end 
	end 
	
	always @(posedge clk_i, posedge resetn_i) begin 
		if (!resetn_i) begin 
			for (i=0;i<9;i=i+1) begin  
				mul[i] <= 'd0;  
			end 
			mul_valid <= 1'b0;
		end else begin 
			for (i=0;i<9;i=i+1) begin  
				mul[i] <= $signed(kernel[i])*$signed({1'b0, pixel_grid_i[i*DATA_RES+:DATA_RES]}); 
			end 
			mul_valid <= data_valid_i; 
		end 
	end 	
	
	always @(*) begin 
		acc_comb = 0; 
		for (i=0;i<9;i=i+1) begin 
			acc_comb = acc_comb + mul[i];
		end 
	end 
	
	always @(posedge clk_i, posedge resetn_i) begin 
		if (!resetn_i) begin 
			acc <= 'd0; 
			acc_valid <= 1'b0; 
		end else begin 
			acc <= acc_comb >>> FxP_RES; // Acc = Sum over (kernel*pixel_grid) divided by 2^(FxP_RES) <=> arithmetic shift of FxP_RES
			acc_valid <= mul_valid;
		end 
	end 
	
	always @(posedge clk_i, posedge resetn_i) begin 
		if (!resetn_i) begin 
			pixel_o <= 'd0; 
			pixel_valid_o <= 1'b0; 
		end else begin 
			pixel_o <= ((acc > 'd255) ? 'd255 : ((acc < 0) ? 0 : acc)); // Apply "clipped ReLU" nonlinearity
			pixel_valid_o <= acc_valid; 
		end 
	end 

endmodule 
