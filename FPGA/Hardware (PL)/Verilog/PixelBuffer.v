
module PixelBuffer # (
	parameter DATA_RES = 8,
	parameter KERNEL_WIDTH = 3,
	parameter LINE_WIDTH = 28 // e.g., 28 for MNIST	& variants	
) (
	input wire clk_i, 
	input wire resetn_i, 
	input wire [DATA_RES-1:0] pixel_i, 
	input wire data_valid_i,
	// input wire read_buff_i, 
	input wire [4:0] read_address,
	
	output wire [KERNEL_WIDTH*DATA_RES-1:0] pixels_o
); 
	
	integer i;
	reg [DATA_RES-1:0] buffer [LINE_WIDTH-1:0]; 
	reg [$clog2(LINE_WIDTH)-1:0] write_address; // ceil[log2(LINE_WIDTH)], log2(28) ~= 4.8 bits 
	// reg [$clog2(LINE_WIDTH)-1:0] read_address; 
	
	always @(posedge clk_i, posedge resetn_i) begin 
		if (!resetn_i) begin 
		    for(i=0;i<LINE_WIDTH;i=i+1) begin 
				buffer[i] <= 0; 
			end 
			write_address <= 'd0; 
		end else if (data_valid_i) begin 
			buffer[write_address] <= pixel_i; 
			write_address <= (write_address == LINE_WIDTH-1) ? 'd0 : write_address + 1;
		end 
	end 
	
	// always @(posedge clk_i, posedge resetn_i) begin 
	// 	if (!resetn_i) begin 
	// 		read_address <= 'd0; 
	// 	end else if (read_buff_i) begin 
	// 		read_address <= (read_address == LINE_WIDTH-1) ? 'd0 : read_address + 1;
	// 	end else begin
	// 		read_address <= read_address;
	// 	end 
	// end 
	
	assign pixels_o = {buffer[read_address], buffer[read_address+1], buffer[read_address+2]}; 	

endmodule 