module CNN_tb;
  logic [7:0] io_pixel_i;
  logic io_pix_data_valid;
  logic [7:0]    io_pixel_o;
  logic			     output_valid;
  logic          m_tlast;
  logic          clk;
  logic          reset;
  logic			     waiter;
  logic          tlast;
  logic          tready;
    
  CNN_AXIS CNN (
  .s_axis_tdata(io_pixel_i),
  .s_axis_tlast(tlast),
  .s_axis_tready(tready),
  .s_axis_tvalid(io_pix_data_valid),
  .m_axis_tdata(io_pixel_o),
  .m_axis_tready(1'b1),
  .m_axis_tvalid(output_valid),
  .m_axis_tlast(m_tlast),
  .clk_i(clk),
  .resetn_i(!reset)
  );

  always #50 clk = ~clk;   

  initial begin
  $display("Hey there! Starting simulation :D");
  tlast = 0;
  reset = 1; 
	clk = 0;
	io_pix_data_valid = 0;
	io_pixel_i = '0; 
  // k_val = 108'b000000000001000000000010000000000001000000000010000000000100000000000010000000000001000000000010000000000001;
  #110 reset = 0; 
  end

  integer data_file;
  integer scan_file; 
  logic [7:0] captured_data;
  integer out_file;
 
  initial begin
      data_file = $fopen("stim.txt", "r");
      out_file = $fopen("picout.txt", "w");
      if (data_file == 0) begin
        $display("data_file handle was NULL");
        $finish;
      end
  end

    always @(posedge clk) begin
    if (!reset) begin
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
     if (output_valid == 1) begin
        $fwrite(out_file, "%d\n", io_pixel_o);         
     end
  end
  
endmodule : CNN_tb