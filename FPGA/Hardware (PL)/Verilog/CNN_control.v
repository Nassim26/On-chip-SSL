
module CNN_control # (
	parameter DATA_RES = 8, 
	parameter WEIGHT_RES = 8, 
	parameter IM_DIM = 28, 
	parameter KERNEL_WIDTH = 3,
	parameter KERNEL_SIZE = 9
) ( 
	input wire clk_i, 
	input wire resetn_i, 
	input wire [DATA_RES-1:0] pixel_i, 
	input wire data_valid_i, 
	input wire signed [WEIGHT_RES*(KERNEL_SIZE+1)-1:0] kernel_i,

	output wire [DATA_RES-1:0] pixel_o,
	output wire data_valid_o
); 
	
	reg [$clog2(IM_DIM*IM_DIM)-1:0] total_pix_counter; // [clog2(IM_DIM^2)-1:0] 
	reg [$clog2(IM_DIM)-1:0] row_pix_counter;  // [clog2(IM_DIM)-1:0] 
	reg [1:0] write_state;		// Tracks which buffer is being written to 
	reg [3:0] pixel_buffer_valid;  // One-hot encoding of write_state, tracks buffer receiving valid bits
	wire [KERNEL_WIDTH*DATA_RES-1:0] p_buffers [3:0]; // The actual {KERNEL_WIDTH}-pixel wide buffers 
	
	reg [$clog2(IM_DIM)-1:0] read_row_pix_counter; // (left) position of the CNN kernel
	reg [$clog2(IM_DIM-KERNEL_WIDTH)-1:0] read_row_counter; // How many lines have been processed 
	reg read_buffers;               // High when the convolutional kernel is active 
	reg [1:0] read_state;			// Tracks which buffers are currently being read from 
	// reg [3:0] read_pixel_buffer;    // Inverse one-hot encoding (one-cold?) of read_state, selects appropriate buffers to read from 
		
	reg [KERNEL_SIZE*DATA_RES-1:0] pixel_grid;

	reg reset_buffers;
	
	// States 
	reg [1:0] State;
	localparam awaitBuffers = 2'b00;
	localparam doConv = 2'b01;
	localparam stallConv = 2'b10; 
	localparam imageDone = 2'b11;
	// End states 
	
	always @(posedge clk_i, posedge resetn_i) begin // Write counters
		if (!resetn_i) begin 
			total_pix_counter <= 'd0;
			row_pix_counter <= 'd0; 
		end else begin 
			if (data_valid_i) begin 
				total_pix_counter <= (total_pix_counter == IM_DIM*IM_DIM-1) ? 'd0 : total_pix_counter + 1;
				row_pix_counter <= (row_pix_counter == IM_DIM-1) ? 'd0 : row_pix_counter + 1; 
			end 
		end 
	end 

	
	always @(posedge clk_i, posedge resetn_i) begin // Read counters
		if (!resetn_i) begin 
			read_row_pix_counter <= 'd0; 
			read_row_counter <= 'd0; 
		end else if (read_buffers) begin 
			if (read_row_pix_counter == IM_DIM-KERNEL_WIDTH) begin 
				read_row_pix_counter <= 'd0; 
				read_row_counter <= read_row_counter + 1; 
			end else begin 
				read_row_pix_counter <= read_row_pix_counter + 1; 
				read_row_counter <= read_row_counter;
			end 
		end 
	end 
	
	// FSM
	always @(posedge clk_i, posedge resetn_i) begin 
		if (!resetn_i) begin 
			State <= awaitBuffers; 
			read_buffers <= 1'b0;
			reset_buffers <= 1'b0; 
		end else begin  
			case (State) 
				awaitBuffers: begin 
					reset_buffers <= 1'b0; 
					read_buffers <= 1'b0; 
					if (total_pix_counter == KERNEL_WIDTH*IM_DIM-1) begin 	
						State <= doConv; 
					end else begin 
						State <= awaitBuffers;
					end 
				end 
				
				doConv: begin 
					reset_buffers <= 1'b0; 
					read_buffers <= 1'b1;
					if (read_row_counter == (IM_DIM-KERNEL_WIDTH) && read_row_pix_counter == (IM_DIM-KERNEL_WIDTH-1)) begin 
						State <= imageDone; 
					end else if (read_row_pix_counter == IM_DIM-KERNEL_WIDTH-1) begin 
						State <= stallConv; 
					end else begin 
						State <= doConv; 
					end  
				end 
				
				stallConv: begin 
					reset_buffers <= 1'b0; 
					read_buffers <= 1'b0; 
					if (row_pix_counter == IM_DIM-1 || !data_valid_i && read_row_pix_counter == 'd0) begin 
						State <= doConv; 
					end else begin 
						State <= stallConv;
					end 
				end 
				
				imageDone: begin 
					reset_buffers <= 1'b1; 
					read_buffers <= 1'b0; 
					State <= awaitBuffers; 
				end 
			endcase 
		end 
	end 
	
	always @(posedge clk_i, posedge resetn_i) begin 
		if (!resetn_i) begin 
			write_state <= 'd0; 
		end else begin 
			if (row_pix_counter == IM_DIM-1 && data_valid_i) begin 
				write_state <= write_state + 1;
			end 
		end 
	end 
	
	always @(*) begin 
		case (write_state) // One-hot encoding of write_state
			2'd0: pixel_buffer_valid = 4'b0001; 
			2'd1: pixel_buffer_valid = 4'b0010; 
			2'd2: pixel_buffer_valid = 4'b0100; 
			2'd3: pixel_buffer_valid = 4'b1000; 
		endcase
	end 	
	
	always @(posedge clk_i, posedge resetn_i) begin 
		if (!resetn_i) begin 
			read_state <= 'd0; 
		end else if (read_row_pix_counter == IM_DIM-KERNEL_WIDTH && read_buffers) begin 
			read_state <= read_state + 1;
		end 
	end 
	
	always @(*) begin // Select appropriate buffer-data for pixel grid
        case (read_state) 
            2'd0: begin 
				// read_pixel_buffer = {{3{read_buffers}}, 1'b0}; 
				pixel_grid = {p_buffers[2], p_buffers[1], p_buffers[0]}; 
			end 
            2'd1: begin 
				// read_pixel_buffer = {1'b0, {3{read_buffers}}};
				pixel_grid = {p_buffers[3], p_buffers[2], p_buffers[1]}; 
			end 
			2'd2: begin 
				// read_pixel_buffer = {read_buffers, 1'b0, {2{read_buffers}}}; 
				pixel_grid = {p_buffers[0], p_buffers[3], p_buffers[2]}; 
			end 
			2'd3: begin 
				// read_pixel_buffer = {{2{read_buffers}}, 1'b0, read_buffers}; 
				pixel_grid = {p_buffers[1], p_buffers[0], p_buffers[3]}; 
			end 
		endcase
    end

	// Module declarations

	MAC # (
		.DATA_RES(DATA_RES), 
		.WEIGHT_RES(WEIGHT_RES), 
		.KERNEL_SIZE(KERNEL_SIZE), 
		.LINE_WIDTH(IM_DIM)
	) MAC (
		.clk_i(clk_i), 
		.resetn_i(resetn_i), 
		.pixel_grid_i(pixel_grid),
		.data_valid_i(read_buffers), 
		.kernel_i(kernel_i),
		.pixel_o(pixel_o), 
		.pixel_valid_o(data_valid_o)
	); 

	generate 
		genvar i; 
		for (i = 0; i < 4; i = i + 1) begin : p
			PixelBuffer # (
				.DATA_RES(DATA_RES),
				.KERNEL_WIDTH(KERNEL_WIDTH),
				.LINE_WIDTH(IM_DIM)
			) PixelBuffer (
				.clk_i(clk_i),
				.resetn_i(resetn_i), 
				.pixel_i(pixel_i), 
				.data_valid_i(pixel_buffer_valid[i] & data_valid_i), 
				.read_address(read_row_pix_counter),
				// .read_buff_i(read_pixel_buffer[i]), 
				.pixels_o(p_buffers[i])
			);
		end 
	endgenerate 
		
endmodule 

