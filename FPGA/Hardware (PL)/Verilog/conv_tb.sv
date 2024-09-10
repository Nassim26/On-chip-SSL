module conv_tb;
  logic [7:0] io_pixel_i;
  logic io_pix_data_valid;
  logic [11:0]   io_kernel_0;
  logic [11:0]   io_kernel_1;
  logic [11:0]   io_kernel_2;
  logic [11:0]   io_kernel_3;
  logic [11:0]   io_kernel_4;
  logic [11:0]   io_kernel_5;
  logic [11:0]   io_kernel_6;
  logic [11:0]   io_kernel_7;
  logic [11:0]   io_kernel_8;
  logic [7:0]    io_pixel_o;
  logic          io_image_finished;
  logic			 conv_finished;
  logic          clk;
  logic          reset;
  logic			 waiter;
    
  FilterTop CNN (
  .io_pixel_i(io_pixel_i),
  .io_pix_data_valid(io_pix_data_valid),
  .io_kernel_0(io_kernel_0),
  .io_kernel_1(io_kernel_1),
  .io_kernel_2(io_kernel_2),
  .io_kernel_3(io_kernel_3),
  .io_kernel_4(io_kernel_4),
  .io_kernel_5(io_kernel_5),
  .io_kernel_6(io_kernel_6),
  .io_kernel_7(io_kernel_7),
  .io_kernel_8(io_kernel_8),
  .io_pixel_o(io_pixel_o),
  .io_image_finished(io_image_finished),
  .io_output_valid(conv_finished),
  .clk(clk),
  .reset(reset)
  );

  always #50 clk = ~clk;   

  initial begin
    reset = 1; 
	clk = 0;
	io_pix_data_valid = 0;
	io_pixel_i = '0; 
	{io_kernel_0, io_kernel_1, io_kernel_2} = {12'd2, 12'd3, 12'd7};
	{io_kernel_3, io_kernel_4, io_kernel_5} = {12'd2, 12'd0, 12'd1};
	{io_kernel_6, io_kernel_7, io_kernel_8} = {12'd2, 12'd8, 12'd1};
    // k_val = 108'b000000000001000000000010000000000001000000000010000000000100000000000010000000000001000000000010000000000001;
    #110 reset = 0; 
  end

  integer data_file;
  integer scan_file; 
  logic [7:0] captured_data;
  integer out_file;
 
  initial begin
      data_file = $fopen("output.txt", "r");
      out_file = $fopen("pic_out.txt", "w");
      if (data_file == 0) begin
        $display("data_file handle was NULL");
        $finish;
      end
  end

    always @(posedge clk or posedge reset) begin
      if (~reset) begin
	      io_pix_data_valid <= 1'b1;
          scan_file = $fscanf(data_file, "%d\n", captured_data); 
          if (!$feof(data_file)) begin
            //use captured_data as you would any other wire or reg value;
			waiter <= 1'b1;
            io_pixel_i <= captured_data; 
          end else begin
			waiter <= 1'b0;
		    io_pix_data_valid <= waiter;
		  end
		end else begin 
		  waiter <= 1'b0;
		  io_pix_data_valid <= 1'b0;
		end 

    end
  
  always @(posedge clk) begin
     if (conv_finished == 1) begin
        $fwrite(out_file, "%d\n", io_pixel_o);         
     end
  end
  
endmodule : conv_tb