// Generator : SpinalHDL v1.10.2a    git head : a348a60b7e8b6a455c72e1536ec3d74a2ea16935
// Component : FilterTop

module FilterTop (
  input  wire [7:0]    io_pixel_i,
  input  wire          io_pix_data_valid,
  input  wire [11:0]   io_kernel_0,
  input  wire [11:0]   io_kernel_1,
  input  wire [11:0]   io_kernel_2,
  input  wire [11:0]   io_kernel_3,
  input  wire [11:0]   io_kernel_4,
  input  wire [11:0]   io_kernel_5,
  input  wire [11:0]   io_kernel_6,
  input  wire [11:0]   io_kernel_7,
  input  wire [11:0]   io_kernel_8,
  output wire [7:0]    io_pixel_o,
  output wire          io_image_finished,
  output wire          io_output_valid,
  input  wire          clk,
  input  wire          reset
);

  wire                P1_io_data_valid;
  wire                P2_io_data_valid;
  wire                P3_io_data_valid;
  wire                P4_io_data_valid;
  wire       [7:0]    FiltConUnit_io_pixel_o;
  wire       [4:0]    FiltConUnit_io_read_counter;
  wire       [3:0]    FiltConUnit_io_buff_valid;
  wire                FiltConUnit_io_output_valid;
  wire                FiltConUnit_io_image_finished;
  wire       [7:0]    P1_io_pix_buffer_0;
  wire       [7:0]    P1_io_pix_buffer_1;
  wire       [7:0]    P1_io_pix_buffer_2;
  wire       [7:0]    P2_io_pix_buffer_0;
  wire       [7:0]    P2_io_pix_buffer_1;
  wire       [7:0]    P2_io_pix_buffer_2;
  wire       [7:0]    P3_io_pix_buffer_0;
  wire       [7:0]    P3_io_pix_buffer_1;
  wire       [7:0]    P3_io_pix_buffer_2;
  wire       [7:0]    P4_io_pix_buffer_0;
  wire       [7:0]    P4_io_pix_buffer_1;
  wire       [7:0]    P4_io_pix_buffer_2;

  FCU FiltConUnit (
    .io_pix_buffer_1_0 (P1_io_pix_buffer_0[7:0]         ), //i
    .io_pix_buffer_1_1 (P1_io_pix_buffer_1[7:0]         ), //i
    .io_pix_buffer_1_2 (P1_io_pix_buffer_2[7:0]         ), //i
    .io_pix_buffer_2_0 (P2_io_pix_buffer_0[7:0]         ), //i
    .io_pix_buffer_2_1 (P2_io_pix_buffer_1[7:0]         ), //i
    .io_pix_buffer_2_2 (P2_io_pix_buffer_2[7:0]         ), //i
    .io_pix_buffer_3_0 (P3_io_pix_buffer_0[7:0]         ), //i
    .io_pix_buffer_3_1 (P3_io_pix_buffer_1[7:0]         ), //i
    .io_pix_buffer_3_2 (P3_io_pix_buffer_2[7:0]         ), //i
    .io_pix_buffer_4_0 (P4_io_pix_buffer_0[7:0]         ), //i
    .io_pix_buffer_4_1 (P4_io_pix_buffer_1[7:0]         ), //i
    .io_pix_buffer_4_2 (P4_io_pix_buffer_2[7:0]         ), //i
    .io_pix_data_valid (io_pix_data_valid               ), //i
    .io_kernel_0       (io_kernel_0[11:0]               ), //i
    .io_kernel_1       (io_kernel_1[11:0]               ), //i
    .io_kernel_2       (io_kernel_2[11:0]               ), //i
    .io_kernel_3       (io_kernel_3[11:0]               ), //i
    .io_kernel_4       (io_kernel_4[11:0]               ), //i
    .io_kernel_5       (io_kernel_5[11:0]               ), //i
    .io_kernel_6       (io_kernel_6[11:0]               ), //i
    .io_kernel_7       (io_kernel_7[11:0]               ), //i
    .io_kernel_8       (io_kernel_8[11:0]               ), //i
    .io_pixel_o        (FiltConUnit_io_pixel_o[7:0]     ), //o
    .io_read_counter   (FiltConUnit_io_read_counter[4:0]), //o
    .io_buff_valid     (FiltConUnit_io_buff_valid[3:0]  ), //o
    .io_output_valid   (FiltConUnit_io_output_valid     ), //o
    .io_image_finished (FiltConUnit_io_image_finished   ), //o
    .clk               (clk                             ), //i
    .reset             (reset                           )  //i
  );
  PixelBuffer P1 (
    .io_pixel_i      (io_pixel_i[7:0]                 ), //i
    .io_data_valid   (P1_io_data_valid                ), //i
    .io_read_address (FiltConUnit_io_read_counter[4:0]), //i
    .io_pix_buffer_0 (P1_io_pix_buffer_0[7:0]         ), //o
    .io_pix_buffer_1 (P1_io_pix_buffer_1[7:0]         ), //o
    .io_pix_buffer_2 (P1_io_pix_buffer_2[7:0]         ), //o
    .clk             (clk                             ), //i
    .reset           (reset                           )  //i
  );
  PixelBuffer P2 (
    .io_pixel_i      (io_pixel_i[7:0]                 ), //i
    .io_data_valid   (P2_io_data_valid                ), //i
    .io_read_address (FiltConUnit_io_read_counter[4:0]), //i
    .io_pix_buffer_0 (P2_io_pix_buffer_0[7:0]         ), //o
    .io_pix_buffer_1 (P2_io_pix_buffer_1[7:0]         ), //o
    .io_pix_buffer_2 (P2_io_pix_buffer_2[7:0]         ), //o
    .clk             (clk                             ), //i
    .reset           (reset                           )  //i
  );
  PixelBuffer P3 (
    .io_pixel_i      (io_pixel_i[7:0]                 ), //i
    .io_data_valid   (P3_io_data_valid                ), //i
    .io_read_address (FiltConUnit_io_read_counter[4:0]), //i
    .io_pix_buffer_0 (P3_io_pix_buffer_0[7:0]         ), //o
    .io_pix_buffer_1 (P3_io_pix_buffer_1[7:0]         ), //o
    .io_pix_buffer_2 (P3_io_pix_buffer_2[7:0]         ), //o
    .clk             (clk                             ), //i
    .reset           (reset                           )  //i
  );
  PixelBuffer P4 (
    .io_pixel_i      (io_pixel_i[7:0]                 ), //i
    .io_data_valid   (P4_io_data_valid                ), //i
    .io_read_address (FiltConUnit_io_read_counter[4:0]), //i
    .io_pix_buffer_0 (P4_io_pix_buffer_0[7:0]         ), //o
    .io_pix_buffer_1 (P4_io_pix_buffer_1[7:0]         ), //o
    .io_pix_buffer_2 (P4_io_pix_buffer_2[7:0]         ), //o
    .clk             (clk                             ), //i
    .reset           (reset                           )  //i
  );
  assign P1_io_data_valid = (FiltConUnit_io_buff_valid[0] && io_pix_data_valid);
  assign P2_io_data_valid = (FiltConUnit_io_buff_valid[1] && io_pix_data_valid);
  assign P3_io_data_valid = (FiltConUnit_io_buff_valid[2] && io_pix_data_valid);
  assign P4_io_data_valid = (FiltConUnit_io_buff_valid[3] && io_pix_data_valid);
  assign io_pixel_o = FiltConUnit_io_pixel_o;
  assign io_image_finished = FiltConUnit_io_image_finished;
  assign io_output_valid = FiltConUnit_io_output_valid;

endmodule

//PixelBuffer_3 replaced by PixelBuffer

//PixelBuffer_2 replaced by PixelBuffer

//PixelBuffer_1 replaced by PixelBuffer

module PixelBuffer (
  input  wire [7:0]    io_pixel_i,
  input  wire          io_data_valid,
  input  wire [4:0]    io_read_address,
  output wire [7:0]    io_pix_buffer_0,
  output wire [7:0]    io_pix_buffer_1,
  output wire [7:0]    io_pix_buffer_2,
  input  wire          clk,
  input  wire          reset
);

  wire       [4:0]    _zz_write_address;
  reg        [7:0]    _zz_io_pix_buffer_0;
  wire       [4:0]    _zz_io_pix_buffer_0_1;
  wire       [4:0]    _zz_io_pix_buffer_0_2;
  reg        [7:0]    _zz_io_pix_buffer_1;
  wire       [4:0]    _zz_io_pix_buffer_1_1;
  wire       [4:0]    _zz_io_pix_buffer_1_2;
  reg        [7:0]    _zz_io_pix_buffer_2;
  wire       [4:0]    _zz_io_pix_buffer_2_1;
  wire       [4:0]    _zz_io_pix_buffer_2_2;
  reg        [4:0]    write_address;
  reg        [7:0]    line_data_0;
  reg        [7:0]    line_data_1;
  reg        [7:0]    line_data_2;
  reg        [7:0]    line_data_3;
  reg        [7:0]    line_data_4;
  reg        [7:0]    line_data_5;
  reg        [7:0]    line_data_6;
  reg        [7:0]    line_data_7;
  reg        [7:0]    line_data_8;
  reg        [7:0]    line_data_9;
  reg        [7:0]    line_data_10;
  reg        [7:0]    line_data_11;
  reg        [7:0]    line_data_12;
  reg        [7:0]    line_data_13;
  reg        [7:0]    line_data_14;
  reg        [7:0]    line_data_15;
  reg        [7:0]    line_data_16;
  reg        [7:0]    line_data_17;
  reg        [7:0]    line_data_18;
  reg        [7:0]    line_data_19;
  reg        [7:0]    line_data_20;
  reg        [7:0]    line_data_21;
  reg        [7:0]    line_data_22;
  reg        [7:0]    line_data_23;
  reg        [7:0]    line_data_24;
  reg        [7:0]    line_data_25;
  reg        [7:0]    line_data_26;
  reg        [7:0]    line_data_27;
  wire       [31:0]   _zz_1;

  assign _zz_write_address = (write_address + 5'h01);
  assign _zz_io_pix_buffer_0_1 = (_zz_io_pix_buffer_0_2 - 5'h0);
  assign _zz_io_pix_buffer_0_2 = (io_read_address + 5'h02);
  assign _zz_io_pix_buffer_1_1 = (_zz_io_pix_buffer_1_2 - 5'h01);
  assign _zz_io_pix_buffer_1_2 = (io_read_address + 5'h02);
  assign _zz_io_pix_buffer_2_1 = (_zz_io_pix_buffer_2_2 - 5'h02);
  assign _zz_io_pix_buffer_2_2 = (io_read_address + 5'h02);
  always @(*) begin
    case(_zz_io_pix_buffer_0_1)
      5'b00000 : _zz_io_pix_buffer_0 = line_data_0;
      5'b00001 : _zz_io_pix_buffer_0 = line_data_1;
      5'b00010 : _zz_io_pix_buffer_0 = line_data_2;
      5'b00011 : _zz_io_pix_buffer_0 = line_data_3;
      5'b00100 : _zz_io_pix_buffer_0 = line_data_4;
      5'b00101 : _zz_io_pix_buffer_0 = line_data_5;
      5'b00110 : _zz_io_pix_buffer_0 = line_data_6;
      5'b00111 : _zz_io_pix_buffer_0 = line_data_7;
      5'b01000 : _zz_io_pix_buffer_0 = line_data_8;
      5'b01001 : _zz_io_pix_buffer_0 = line_data_9;
      5'b01010 : _zz_io_pix_buffer_0 = line_data_10;
      5'b01011 : _zz_io_pix_buffer_0 = line_data_11;
      5'b01100 : _zz_io_pix_buffer_0 = line_data_12;
      5'b01101 : _zz_io_pix_buffer_0 = line_data_13;
      5'b01110 : _zz_io_pix_buffer_0 = line_data_14;
      5'b01111 : _zz_io_pix_buffer_0 = line_data_15;
      5'b10000 : _zz_io_pix_buffer_0 = line_data_16;
      5'b10001 : _zz_io_pix_buffer_0 = line_data_17;
      5'b10010 : _zz_io_pix_buffer_0 = line_data_18;
      5'b10011 : _zz_io_pix_buffer_0 = line_data_19;
      5'b10100 : _zz_io_pix_buffer_0 = line_data_20;
      5'b10101 : _zz_io_pix_buffer_0 = line_data_21;
      5'b10110 : _zz_io_pix_buffer_0 = line_data_22;
      5'b10111 : _zz_io_pix_buffer_0 = line_data_23;
      5'b11000 : _zz_io_pix_buffer_0 = line_data_24;
      5'b11001 : _zz_io_pix_buffer_0 = line_data_25;
      5'b11010 : _zz_io_pix_buffer_0 = line_data_26;
      default : _zz_io_pix_buffer_0 = line_data_27;
    endcase
  end

  always @(*) begin
    case(_zz_io_pix_buffer_1_1)
      5'b00000 : _zz_io_pix_buffer_1 = line_data_0;
      5'b00001 : _zz_io_pix_buffer_1 = line_data_1;
      5'b00010 : _zz_io_pix_buffer_1 = line_data_2;
      5'b00011 : _zz_io_pix_buffer_1 = line_data_3;
      5'b00100 : _zz_io_pix_buffer_1 = line_data_4;
      5'b00101 : _zz_io_pix_buffer_1 = line_data_5;
      5'b00110 : _zz_io_pix_buffer_1 = line_data_6;
      5'b00111 : _zz_io_pix_buffer_1 = line_data_7;
      5'b01000 : _zz_io_pix_buffer_1 = line_data_8;
      5'b01001 : _zz_io_pix_buffer_1 = line_data_9;
      5'b01010 : _zz_io_pix_buffer_1 = line_data_10;
      5'b01011 : _zz_io_pix_buffer_1 = line_data_11;
      5'b01100 : _zz_io_pix_buffer_1 = line_data_12;
      5'b01101 : _zz_io_pix_buffer_1 = line_data_13;
      5'b01110 : _zz_io_pix_buffer_1 = line_data_14;
      5'b01111 : _zz_io_pix_buffer_1 = line_data_15;
      5'b10000 : _zz_io_pix_buffer_1 = line_data_16;
      5'b10001 : _zz_io_pix_buffer_1 = line_data_17;
      5'b10010 : _zz_io_pix_buffer_1 = line_data_18;
      5'b10011 : _zz_io_pix_buffer_1 = line_data_19;
      5'b10100 : _zz_io_pix_buffer_1 = line_data_20;
      5'b10101 : _zz_io_pix_buffer_1 = line_data_21;
      5'b10110 : _zz_io_pix_buffer_1 = line_data_22;
      5'b10111 : _zz_io_pix_buffer_1 = line_data_23;
      5'b11000 : _zz_io_pix_buffer_1 = line_data_24;
      5'b11001 : _zz_io_pix_buffer_1 = line_data_25;
      5'b11010 : _zz_io_pix_buffer_1 = line_data_26;
      default : _zz_io_pix_buffer_1 = line_data_27;
    endcase
  end

  always @(*) begin
    case(_zz_io_pix_buffer_2_1)
      5'b00000 : _zz_io_pix_buffer_2 = line_data_0;
      5'b00001 : _zz_io_pix_buffer_2 = line_data_1;
      5'b00010 : _zz_io_pix_buffer_2 = line_data_2;
      5'b00011 : _zz_io_pix_buffer_2 = line_data_3;
      5'b00100 : _zz_io_pix_buffer_2 = line_data_4;
      5'b00101 : _zz_io_pix_buffer_2 = line_data_5;
      5'b00110 : _zz_io_pix_buffer_2 = line_data_6;
      5'b00111 : _zz_io_pix_buffer_2 = line_data_7;
      5'b01000 : _zz_io_pix_buffer_2 = line_data_8;
      5'b01001 : _zz_io_pix_buffer_2 = line_data_9;
      5'b01010 : _zz_io_pix_buffer_2 = line_data_10;
      5'b01011 : _zz_io_pix_buffer_2 = line_data_11;
      5'b01100 : _zz_io_pix_buffer_2 = line_data_12;
      5'b01101 : _zz_io_pix_buffer_2 = line_data_13;
      5'b01110 : _zz_io_pix_buffer_2 = line_data_14;
      5'b01111 : _zz_io_pix_buffer_2 = line_data_15;
      5'b10000 : _zz_io_pix_buffer_2 = line_data_16;
      5'b10001 : _zz_io_pix_buffer_2 = line_data_17;
      5'b10010 : _zz_io_pix_buffer_2 = line_data_18;
      5'b10011 : _zz_io_pix_buffer_2 = line_data_19;
      5'b10100 : _zz_io_pix_buffer_2 = line_data_20;
      5'b10101 : _zz_io_pix_buffer_2 = line_data_21;
      5'b10110 : _zz_io_pix_buffer_2 = line_data_22;
      5'b10111 : _zz_io_pix_buffer_2 = line_data_23;
      5'b11000 : _zz_io_pix_buffer_2 = line_data_24;
      5'b11001 : _zz_io_pix_buffer_2 = line_data_25;
      5'b11010 : _zz_io_pix_buffer_2 = line_data_26;
      default : _zz_io_pix_buffer_2 = line_data_27;
    endcase
  end

  assign _zz_1 = ({31'd0,1'b1} <<< write_address);
  assign io_pix_buffer_0 = _zz_io_pix_buffer_0;
  assign io_pix_buffer_1 = _zz_io_pix_buffer_1;
  assign io_pix_buffer_2 = _zz_io_pix_buffer_2;
  always @(posedge clk or posedge reset) begin
    if(reset) begin
      write_address <= 5'h0;
      line_data_0 <= 8'h0;
      line_data_1 <= 8'h0;
      line_data_2 <= 8'h0;
      line_data_3 <= 8'h0;
      line_data_4 <= 8'h0;
      line_data_5 <= 8'h0;
      line_data_6 <= 8'h0;
      line_data_7 <= 8'h0;
      line_data_8 <= 8'h0;
      line_data_9 <= 8'h0;
      line_data_10 <= 8'h0;
      line_data_11 <= 8'h0;
      line_data_12 <= 8'h0;
      line_data_13 <= 8'h0;
      line_data_14 <= 8'h0;
      line_data_15 <= 8'h0;
      line_data_16 <= 8'h0;
      line_data_17 <= 8'h0;
      line_data_18 <= 8'h0;
      line_data_19 <= 8'h0;
      line_data_20 <= 8'h0;
      line_data_21 <= 8'h0;
      line_data_22 <= 8'h0;
      line_data_23 <= 8'h0;
      line_data_24 <= 8'h0;
      line_data_25 <= 8'h0;
      line_data_26 <= 8'h0;
      line_data_27 <= 8'h0;
    end else begin
      if(io_data_valid) begin
        if(_zz_1[0]) begin
          line_data_0 <= io_pixel_i;
        end
        if(_zz_1[1]) begin
          line_data_1 <= io_pixel_i;
        end
        if(_zz_1[2]) begin
          line_data_2 <= io_pixel_i;
        end
        if(_zz_1[3]) begin
          line_data_3 <= io_pixel_i;
        end
        if(_zz_1[4]) begin
          line_data_4 <= io_pixel_i;
        end
        if(_zz_1[5]) begin
          line_data_5 <= io_pixel_i;
        end
        if(_zz_1[6]) begin
          line_data_6 <= io_pixel_i;
        end
        if(_zz_1[7]) begin
          line_data_7 <= io_pixel_i;
        end
        if(_zz_1[8]) begin
          line_data_8 <= io_pixel_i;
        end
        if(_zz_1[9]) begin
          line_data_9 <= io_pixel_i;
        end
        if(_zz_1[10]) begin
          line_data_10 <= io_pixel_i;
        end
        if(_zz_1[11]) begin
          line_data_11 <= io_pixel_i;
        end
        if(_zz_1[12]) begin
          line_data_12 <= io_pixel_i;
        end
        if(_zz_1[13]) begin
          line_data_13 <= io_pixel_i;
        end
        if(_zz_1[14]) begin
          line_data_14 <= io_pixel_i;
        end
        if(_zz_1[15]) begin
          line_data_15 <= io_pixel_i;
        end
        if(_zz_1[16]) begin
          line_data_16 <= io_pixel_i;
        end
        if(_zz_1[17]) begin
          line_data_17 <= io_pixel_i;
        end
        if(_zz_1[18]) begin
          line_data_18 <= io_pixel_i;
        end
        if(_zz_1[19]) begin
          line_data_19 <= io_pixel_i;
        end
        if(_zz_1[20]) begin
          line_data_20 <= io_pixel_i;
        end
        if(_zz_1[21]) begin
          line_data_21 <= io_pixel_i;
        end
        if(_zz_1[22]) begin
          line_data_22 <= io_pixel_i;
        end
        if(_zz_1[23]) begin
          line_data_23 <= io_pixel_i;
        end
        if(_zz_1[24]) begin
          line_data_24 <= io_pixel_i;
        end
        if(_zz_1[25]) begin
          line_data_25 <= io_pixel_i;
        end
        if(_zz_1[26]) begin
          line_data_26 <= io_pixel_i;
        end
        if(_zz_1[27]) begin
          line_data_27 <= io_pixel_i;
        end
        write_address <= ((write_address == 5'h1b) ? 5'h0 : _zz_write_address);
      end
    end
  end


endmodule

module FCU (
  input  wire [7:0]    io_pix_buffer_1_0,
  input  wire [7:0]    io_pix_buffer_1_1,
  input  wire [7:0]    io_pix_buffer_1_2,
  input  wire [7:0]    io_pix_buffer_2_0,
  input  wire [7:0]    io_pix_buffer_2_1,
  input  wire [7:0]    io_pix_buffer_2_2,
  input  wire [7:0]    io_pix_buffer_3_0,
  input  wire [7:0]    io_pix_buffer_3_1,
  input  wire [7:0]    io_pix_buffer_3_2,
  input  wire [7:0]    io_pix_buffer_4_0,
  input  wire [7:0]    io_pix_buffer_4_1,
  input  wire [7:0]    io_pix_buffer_4_2,
  input  wire          io_pix_data_valid,
  input  wire [11:0]   io_kernel_0,
  input  wire [11:0]   io_kernel_1,
  input  wire [11:0]   io_kernel_2,
  input  wire [11:0]   io_kernel_3,
  input  wire [11:0]   io_kernel_4,
  input  wire [11:0]   io_kernel_5,
  input  wire [11:0]   io_kernel_6,
  input  wire [11:0]   io_kernel_7,
  input  wire [11:0]   io_kernel_8,
  output wire [7:0]    io_pixel_o,
  output wire [4:0]    io_read_counter,
  output wire [3:0]    io_buff_valid,
  output wire          io_output_valid,
  output wire          io_image_finished,
  input  wire          clk,
  input  wire          reset
);
  localparam conv_fsm_enumDef_BOOT = 3'd0;
  localparam conv_fsm_enumDef_awaitBuffers = 3'd1;
  localparam conv_fsm_enumDef_convolve = 3'd2;
  localparam conv_fsm_enumDef_imageDone = 3'd3;
  localparam conv_fsm_enumDef_stallConv = 3'd4;

  wire       [7:0]    MAC_unit_io_pixel_o;
  wire                MAC_unit_io_output_valid;
  wire       [1:0]    _zz_active_write_buffer;
  reg        [7:0]    _zz_pixel_grid_0;
  wire       [1:0]    _zz_pixel_grid_0_1;
  wire       [1:0]    _zz_pixel_grid_0_2;
  wire       [1:0]    _zz_pixel_grid_0_3;
  reg        [7:0]    _zz_pixel_grid_1;
  wire       [1:0]    _zz_pixel_grid_1_1;
  wire       [1:0]    _zz_pixel_grid_1_2;
  wire       [1:0]    _zz_pixel_grid_1_3;
  reg        [7:0]    _zz_pixel_grid_2;
  wire       [1:0]    _zz_pixel_grid_2_1;
  wire       [1:0]    _zz_pixel_grid_2_2;
  wire       [1:0]    _zz_pixel_grid_2_3;
  reg        [7:0]    _zz_pixel_grid_3;
  wire       [1:0]    _zz_pixel_grid_3_1;
  wire       [1:0]    _zz_pixel_grid_3_2;
  wire       [1:0]    _zz_pixel_grid_3_3;
  reg        [7:0]    _zz_pixel_grid_4;
  wire       [1:0]    _zz_pixel_grid_4_1;
  wire       [1:0]    _zz_pixel_grid_4_2;
  wire       [1:0]    _zz_pixel_grid_4_3;
  reg        [7:0]    _zz_pixel_grid_5;
  wire       [1:0]    _zz_pixel_grid_5_1;
  wire       [1:0]    _zz_pixel_grid_5_2;
  wire       [1:0]    _zz_pixel_grid_5_3;
  reg        [7:0]    _zz_pixel_grid_6;
  wire       [1:0]    _zz_pixel_grid_6_1;
  wire       [1:0]    _zz_pixel_grid_6_2;
  wire       [1:0]    _zz_pixel_grid_6_3;
  reg        [7:0]    _zz_pixel_grid_7;
  wire       [1:0]    _zz_pixel_grid_7_1;
  wire       [1:0]    _zz_pixel_grid_7_2;
  wire       [1:0]    _zz_pixel_grid_7_3;
  reg        [7:0]    _zz_pixel_grid_8;
  wire       [1:0]    _zz_pixel_grid_8_1;
  wire       [1:0]    _zz_pixel_grid_8_2;
  wire       [1:0]    _zz_pixel_grid_8_3;
  wire       [1:0]    _zz_active_read_buffer;
  reg        [9:0]    pix_counter;
  reg        [4:0]    row_pix_counter;
  reg        [4:0]    read_pix_counter;
  reg        [4:0]    read_row_counter;
  wire       [7:0]    p_buffers_0_0;
  wire       [7:0]    p_buffers_0_1;
  wire       [7:0]    p_buffers_0_2;
  wire       [7:0]    p_buffers_1_0;
  wire       [7:0]    p_buffers_1_1;
  wire       [7:0]    p_buffers_1_2;
  wire       [7:0]    p_buffers_2_0;
  wire       [7:0]    p_buffers_2_1;
  wire       [7:0]    p_buffers_2_2;
  wire       [7:0]    p_buffers_3_0;
  wire       [7:0]    p_buffers_3_1;
  wire       [7:0]    p_buffers_3_2;
  reg        [1:0]    active_write_buffer;
  reg        [1:0]    active_read_buffer;
  wire       [7:0]    pixel_grid_0;
  wire       [7:0]    pixel_grid_1;
  wire       [7:0]    pixel_grid_2;
  wire       [7:0]    pixel_grid_3;
  wire       [7:0]    pixel_grid_4;
  wire       [7:0]    pixel_grid_5;
  wire       [7:0]    pixel_grid_6;
  wire       [7:0]    pixel_grid_7;
  wire       [7:0]    pixel_grid_8;
  reg                 reset_buffers;
  wire                reset_write_buffers;
  wire                when_FCU_l53;
  wire                when_FCU_l57;
  wire                conv_fsm_wantExit;
  reg                 conv_fsm_wantStart;
  wire                conv_fsm_wantKill;
  reg        [1:0]    _zz_when_State_l238;
  reg                 conv_fsm_conv_data_valid;
  reg                 conv_fsm_finished_image;
  reg                 conv_fsm_increment_read_buffer;
  wire                when_FCU_l129;
  reg        [2:0]    conv_fsm_stateReg;
  reg        [2:0]    conv_fsm_stateNext;
  wire                when_FCU_l80;
  wire                when_FCU_l89;
  wire                when_FCU_l91;
  wire                when_State_l238;
  wire                when_StateMachine_l253;
  `ifndef SYNTHESIS
  reg [95:0] conv_fsm_stateReg_string;
  reg [95:0] conv_fsm_stateNext_string;
  `endif


  assign _zz_active_write_buffer = (active_write_buffer + 2'b01);
  assign _zz_pixel_grid_0_1 = (_zz_pixel_grid_0_2 % 3'b100);
  assign _zz_pixel_grid_0_2 = (_zz_pixel_grid_0_3 - 2'b00);
  assign _zz_pixel_grid_0_3 = (2'b10 + active_read_buffer);
  assign _zz_pixel_grid_1_1 = (_zz_pixel_grid_1_2 % 3'b100);
  assign _zz_pixel_grid_1_2 = (_zz_pixel_grid_1_3 - 2'b00);
  assign _zz_pixel_grid_1_3 = (2'b10 + active_read_buffer);
  assign _zz_pixel_grid_2_1 = (_zz_pixel_grid_2_2 % 3'b100);
  assign _zz_pixel_grid_2_2 = (_zz_pixel_grid_2_3 - 2'b00);
  assign _zz_pixel_grid_2_3 = (2'b10 + active_read_buffer);
  assign _zz_pixel_grid_3_1 = (_zz_pixel_grid_3_2 % 3'b100);
  assign _zz_pixel_grid_3_2 = (_zz_pixel_grid_3_3 - 2'b01);
  assign _zz_pixel_grid_3_3 = (2'b10 + active_read_buffer);
  assign _zz_pixel_grid_4_1 = (_zz_pixel_grid_4_2 % 3'b100);
  assign _zz_pixel_grid_4_2 = (_zz_pixel_grid_4_3 - 2'b01);
  assign _zz_pixel_grid_4_3 = (2'b10 + active_read_buffer);
  assign _zz_pixel_grid_5_1 = (_zz_pixel_grid_5_2 % 3'b100);
  assign _zz_pixel_grid_5_2 = (_zz_pixel_grid_5_3 - 2'b01);
  assign _zz_pixel_grid_5_3 = (2'b10 + active_read_buffer);
  assign _zz_pixel_grid_6_1 = (_zz_pixel_grid_6_2 % 3'b100);
  assign _zz_pixel_grid_6_2 = (_zz_pixel_grid_6_3 - 2'b10);
  assign _zz_pixel_grid_6_3 = (2'b10 + active_read_buffer);
  assign _zz_pixel_grid_7_1 = (_zz_pixel_grid_7_2 % 3'b100);
  assign _zz_pixel_grid_7_2 = (_zz_pixel_grid_7_3 - 2'b10);
  assign _zz_pixel_grid_7_3 = (2'b10 + active_read_buffer);
  assign _zz_pixel_grid_8_1 = (_zz_pixel_grid_8_2 % 3'b100);
  assign _zz_pixel_grid_8_2 = (_zz_pixel_grid_8_3 - 2'b10);
  assign _zz_pixel_grid_8_3 = (2'b10 + active_read_buffer);
  assign _zz_active_read_buffer = (active_read_buffer + 2'b01);
  MAC MAC_unit (
    .io_pixel_grid_i_0 (pixel_grid_0[7:0]       ), //i
    .io_pixel_grid_i_1 (pixel_grid_1[7:0]       ), //i
    .io_pixel_grid_i_2 (pixel_grid_2[7:0]       ), //i
    .io_pixel_grid_i_3 (pixel_grid_3[7:0]       ), //i
    .io_pixel_grid_i_4 (pixel_grid_4[7:0]       ), //i
    .io_pixel_grid_i_5 (pixel_grid_5[7:0]       ), //i
    .io_pixel_grid_i_6 (pixel_grid_6[7:0]       ), //i
    .io_pixel_grid_i_7 (pixel_grid_7[7:0]       ), //i
    .io_pixel_grid_i_8 (pixel_grid_8[7:0]       ), //i
    .io_kernel_0       (io_kernel_0[11:0]       ), //i
    .io_kernel_1       (io_kernel_1[11:0]       ), //i
    .io_kernel_2       (io_kernel_2[11:0]       ), //i
    .io_kernel_3       (io_kernel_3[11:0]       ), //i
    .io_kernel_4       (io_kernel_4[11:0]       ), //i
    .io_kernel_5       (io_kernel_5[11:0]       ), //i
    .io_kernel_6       (io_kernel_6[11:0]       ), //i
    .io_kernel_7       (io_kernel_7[11:0]       ), //i
    .io_kernel_8       (io_kernel_8[11:0]       ), //i
    .io_data_valid     (conv_fsm_conv_data_valid), //i
    .io_pixel_o        (MAC_unit_io_pixel_o[7:0]), //o
    .io_output_valid   (MAC_unit_io_output_valid), //o
    .clk               (clk                     ), //i
    .reset             (reset                   )  //i
  );
  always @(*) begin
    case(_zz_pixel_grid_0_1)
      2'b00 : _zz_pixel_grid_0 = p_buffers_0_0;
      2'b01 : _zz_pixel_grid_0 = p_buffers_1_0;
      2'b10 : _zz_pixel_grid_0 = p_buffers_2_0;
      default : _zz_pixel_grid_0 = p_buffers_3_0;
    endcase
  end

  always @(*) begin
    case(_zz_pixel_grid_1_1)
      2'b00 : _zz_pixel_grid_1 = p_buffers_0_1;
      2'b01 : _zz_pixel_grid_1 = p_buffers_1_1;
      2'b10 : _zz_pixel_grid_1 = p_buffers_2_1;
      default : _zz_pixel_grid_1 = p_buffers_3_1;
    endcase
  end

  always @(*) begin
    case(_zz_pixel_grid_2_1)
      2'b00 : _zz_pixel_grid_2 = p_buffers_0_2;
      2'b01 : _zz_pixel_grid_2 = p_buffers_1_2;
      2'b10 : _zz_pixel_grid_2 = p_buffers_2_2;
      default : _zz_pixel_grid_2 = p_buffers_3_2;
    endcase
  end

  always @(*) begin
    case(_zz_pixel_grid_3_1)
      2'b00 : _zz_pixel_grid_3 = p_buffers_0_0;
      2'b01 : _zz_pixel_grid_3 = p_buffers_1_0;
      2'b10 : _zz_pixel_grid_3 = p_buffers_2_0;
      default : _zz_pixel_grid_3 = p_buffers_3_0;
    endcase
  end

  always @(*) begin
    case(_zz_pixel_grid_4_1)
      2'b00 : _zz_pixel_grid_4 = p_buffers_0_1;
      2'b01 : _zz_pixel_grid_4 = p_buffers_1_1;
      2'b10 : _zz_pixel_grid_4 = p_buffers_2_1;
      default : _zz_pixel_grid_4 = p_buffers_3_1;
    endcase
  end

  always @(*) begin
    case(_zz_pixel_grid_5_1)
      2'b00 : _zz_pixel_grid_5 = p_buffers_0_2;
      2'b01 : _zz_pixel_grid_5 = p_buffers_1_2;
      2'b10 : _zz_pixel_grid_5 = p_buffers_2_2;
      default : _zz_pixel_grid_5 = p_buffers_3_2;
    endcase
  end

  always @(*) begin
    case(_zz_pixel_grid_6_1)
      2'b00 : _zz_pixel_grid_6 = p_buffers_0_0;
      2'b01 : _zz_pixel_grid_6 = p_buffers_1_0;
      2'b10 : _zz_pixel_grid_6 = p_buffers_2_0;
      default : _zz_pixel_grid_6 = p_buffers_3_0;
    endcase
  end

  always @(*) begin
    case(_zz_pixel_grid_7_1)
      2'b00 : _zz_pixel_grid_7 = p_buffers_0_1;
      2'b01 : _zz_pixel_grid_7 = p_buffers_1_1;
      2'b10 : _zz_pixel_grid_7 = p_buffers_2_1;
      default : _zz_pixel_grid_7 = p_buffers_3_1;
    endcase
  end

  always @(*) begin
    case(_zz_pixel_grid_8_1)
      2'b00 : _zz_pixel_grid_8 = p_buffers_0_2;
      2'b01 : _zz_pixel_grid_8 = p_buffers_1_2;
      2'b10 : _zz_pixel_grid_8 = p_buffers_2_2;
      default : _zz_pixel_grid_8 = p_buffers_3_2;
    endcase
  end

  `ifndef SYNTHESIS
  always @(*) begin
    case(conv_fsm_stateReg)
      conv_fsm_enumDef_BOOT : conv_fsm_stateReg_string = "BOOT        ";
      conv_fsm_enumDef_awaitBuffers : conv_fsm_stateReg_string = "awaitBuffers";
      conv_fsm_enumDef_convolve : conv_fsm_stateReg_string = "convolve    ";
      conv_fsm_enumDef_imageDone : conv_fsm_stateReg_string = "imageDone   ";
      conv_fsm_enumDef_stallConv : conv_fsm_stateReg_string = "stallConv   ";
      default : conv_fsm_stateReg_string = "????????????";
    endcase
  end
  always @(*) begin
    case(conv_fsm_stateNext)
      conv_fsm_enumDef_BOOT : conv_fsm_stateNext_string = "BOOT        ";
      conv_fsm_enumDef_awaitBuffers : conv_fsm_stateNext_string = "awaitBuffers";
      conv_fsm_enumDef_convolve : conv_fsm_stateNext_string = "convolve    ";
      conv_fsm_enumDef_imageDone : conv_fsm_stateNext_string = "imageDone   ";
      conv_fsm_enumDef_stallConv : conv_fsm_stateNext_string = "stallConv   ";
      default : conv_fsm_stateNext_string = "????????????";
    endcase
  end
  `endif

  assign reset_write_buffers = 1'b0;
  assign p_buffers_0_0 = io_pix_buffer_1_0;
  assign p_buffers_0_1 = io_pix_buffer_1_1;
  assign p_buffers_0_2 = io_pix_buffer_1_2;
  assign p_buffers_1_0 = io_pix_buffer_2_0;
  assign p_buffers_1_1 = io_pix_buffer_2_1;
  assign p_buffers_1_2 = io_pix_buffer_2_2;
  assign p_buffers_2_0 = io_pix_buffer_3_0;
  assign p_buffers_2_1 = io_pix_buffer_3_1;
  assign p_buffers_2_2 = io_pix_buffer_3_2;
  assign p_buffers_3_0 = io_pix_buffer_4_0;
  assign p_buffers_3_1 = io_pix_buffer_4_1;
  assign p_buffers_3_2 = io_pix_buffer_4_2;
  assign io_buff_valid = ((active_write_buffer == 2'b00) ? 4'b0001 : ((active_write_buffer == 2'b01) ? 4'b0010 : ((active_write_buffer == 2'b10) ? 4'b0100 : 4'b1000)));
  assign io_read_counter = read_pix_counter;
  assign when_FCU_l53 = (io_pix_data_valid && (pix_counter == 10'h30f));
  assign when_FCU_l57 = (io_pix_data_valid && (row_pix_counter == 5'h1b));
  assign pixel_grid_0 = _zz_pixel_grid_0;
  assign pixel_grid_1 = _zz_pixel_grid_1;
  assign pixel_grid_2 = _zz_pixel_grid_2;
  assign pixel_grid_3 = _zz_pixel_grid_3;
  assign pixel_grid_4 = _zz_pixel_grid_4;
  assign pixel_grid_5 = _zz_pixel_grid_5;
  assign pixel_grid_6 = _zz_pixel_grid_6;
  assign pixel_grid_7 = _zz_pixel_grid_7;
  assign pixel_grid_8 = _zz_pixel_grid_8;
  assign conv_fsm_wantExit = 1'b0;
  always @(*) begin
    conv_fsm_wantStart = 1'b0;
    case(conv_fsm_stateReg)
      conv_fsm_enumDef_awaitBuffers : begin
      end
      conv_fsm_enumDef_convolve : begin
      end
      conv_fsm_enumDef_imageDone : begin
      end
      conv_fsm_enumDef_stallConv : begin
      end
      default : begin
        conv_fsm_wantStart = 1'b1;
      end
    endcase
  end

  assign conv_fsm_wantKill = 1'b0;
  assign io_image_finished = conv_fsm_finished_image;
  assign when_FCU_l129 = (conv_fsm_conv_data_valid && (read_pix_counter == 5'h19));
  assign io_pixel_o = MAC_unit_io_pixel_o;
  assign io_output_valid = MAC_unit_io_output_valid;
  always @(*) begin
    conv_fsm_stateNext = conv_fsm_stateReg;
    case(conv_fsm_stateReg)
      conv_fsm_enumDef_awaitBuffers : begin
        if(when_FCU_l80) begin
          conv_fsm_stateNext = conv_fsm_enumDef_convolve;
        end
      end
      conv_fsm_enumDef_convolve : begin
        if(when_FCU_l89) begin
          conv_fsm_stateNext = conv_fsm_enumDef_imageDone;
        end else begin
          if(when_FCU_l91) begin
            conv_fsm_stateNext = conv_fsm_enumDef_stallConv;
          end else begin
            conv_fsm_stateNext = conv_fsm_enumDef_convolve;
          end
        end
      end
      conv_fsm_enumDef_imageDone : begin
        conv_fsm_stateNext = conv_fsm_enumDef_awaitBuffers;
      end
      conv_fsm_enumDef_stallConv : begin
        if(when_State_l238) begin
          conv_fsm_stateNext = conv_fsm_enumDef_convolve;
        end
      end
      default : begin
      end
    endcase
    if(conv_fsm_wantStart) begin
      conv_fsm_stateNext = conv_fsm_enumDef_awaitBuffers;
    end
    if(conv_fsm_wantKill) begin
      conv_fsm_stateNext = conv_fsm_enumDef_BOOT;
    end
  end

  assign when_FCU_l80 = (pix_counter == 10'h053);
  assign when_FCU_l89 = ((read_row_counter == 5'h19) && (read_pix_counter == 5'h18));
  assign when_FCU_l91 = (read_pix_counter == 5'h18);
  assign when_State_l238 = (_zz_when_State_l238 <= 2'b01);
  assign when_StateMachine_l253 = ((! (conv_fsm_stateReg == conv_fsm_enumDef_stallConv)) && (conv_fsm_stateNext == conv_fsm_enumDef_stallConv));
  always @(posedge clk or posedge reset) begin
    if(reset) begin
      pix_counter <= 10'h0;
      row_pix_counter <= 5'h0;
      read_pix_counter <= 5'h0;
      read_row_counter <= 5'h0;
      active_write_buffer <= 2'b00;
      active_read_buffer <= 2'b00;
      reset_buffers <= 1'b0;
      conv_fsm_conv_data_valid <= 1'b0;
      conv_fsm_finished_image <= 1'b0;
      conv_fsm_increment_read_buffer <= 1'b0;
      conv_fsm_stateReg <= conv_fsm_enumDef_BOOT;
    end else begin
      active_write_buffer <= (((row_pix_counter == 5'h1b) && io_pix_data_valid) ? _zz_active_write_buffer : active_write_buffer);
      if(when_FCU_l53) begin
        pix_counter <= 10'h001;
      end else begin
        if(io_pix_data_valid) begin
          pix_counter <= (pix_counter + 10'h001);
        end
      end
      if(when_FCU_l57) begin
        row_pix_counter <= 5'h0;
      end else begin
        if(io_pix_data_valid) begin
          row_pix_counter <= (row_pix_counter + 5'h01);
        end
      end
      if(reset_buffers) begin
        active_read_buffer <= 2'b00;
      end else begin
        active_read_buffer <= (conv_fsm_increment_read_buffer ? _zz_active_read_buffer : active_read_buffer);
      end
      if(reset_buffers) begin
        read_pix_counter <= 5'h0;
        read_row_counter <= 5'h0;
      end else begin
        if(when_FCU_l129) begin
          read_pix_counter <= 5'h0;
          read_row_counter <= (read_row_counter + 5'h01);
        end else begin
          if(conv_fsm_conv_data_valid) begin
            read_pix_counter <= (read_pix_counter + 5'h01);
          end
        end
      end
      conv_fsm_stateReg <= conv_fsm_stateNext;
      case(conv_fsm_stateReg)
        conv_fsm_enumDef_awaitBuffers : begin
          reset_buffers <= 1'b0;
          conv_fsm_finished_image <= 1'b0;
          conv_fsm_conv_data_valid <= 1'b0;
        end
        conv_fsm_enumDef_convolve : begin
          conv_fsm_finished_image <= 1'b0;
          conv_fsm_conv_data_valid <= 1'b1;
          conv_fsm_increment_read_buffer <= 1'b0;
        end
        conv_fsm_enumDef_imageDone : begin
          reset_buffers <= 1'b1;
          conv_fsm_finished_image <= 1'b1;
          conv_fsm_conv_data_valid <= 1'b0;
        end
        conv_fsm_enumDef_stallConv : begin
          conv_fsm_finished_image <= 1'b0;
          conv_fsm_conv_data_valid <= 1'b0;
          if(when_State_l238) begin
            conv_fsm_increment_read_buffer <= 1'b1;
          end
        end
        default : begin
        end
      endcase
    end
  end

  always @(posedge clk) begin
    case(conv_fsm_stateReg)
      conv_fsm_enumDef_awaitBuffers : begin
      end
      conv_fsm_enumDef_convolve : begin
      end
      conv_fsm_enumDef_imageDone : begin
      end
      conv_fsm_enumDef_stallConv : begin
        _zz_when_State_l238 <= (_zz_when_State_l238 - 2'b01);
      end
      default : begin
      end
    endcase
    if(when_StateMachine_l253) begin
      _zz_when_State_l238 <= 2'b10;
    end
  end


endmodule

module MAC (
  input  wire [7:0]    io_pixel_grid_i_0,
  input  wire [7:0]    io_pixel_grid_i_1,
  input  wire [7:0]    io_pixel_grid_i_2,
  input  wire [7:0]    io_pixel_grid_i_3,
  input  wire [7:0]    io_pixel_grid_i_4,
  input  wire [7:0]    io_pixel_grid_i_5,
  input  wire [7:0]    io_pixel_grid_i_6,
  input  wire [7:0]    io_pixel_grid_i_7,
  input  wire [7:0]    io_pixel_grid_i_8,
  input  wire [11:0]   io_kernel_0,
  input  wire [11:0]   io_kernel_1,
  input  wire [11:0]   io_kernel_2,
  input  wire [11:0]   io_kernel_3,
  input  wire [11:0]   io_kernel_4,
  input  wire [11:0]   io_kernel_5,
  input  wire [11:0]   io_kernel_6,
  input  wire [11:0]   io_kernel_7,
  input  wire [11:0]   io_kernel_8,
  input  wire          io_data_valid,
  output wire [7:0]    io_pixel_o,
  output wire          io_output_valid,
  input  wire          clk,
  input  wire          reset
);

  wire       [20:0]   _zz_mul_0_0;
  wire       [8:0]    _zz_mul_0_0_1;
  wire       [20:0]   _zz_mul_0_1;
  wire       [8:0]    _zz_mul_0_1_1;
  wire       [20:0]   _zz_mul_0_2;
  wire       [8:0]    _zz_mul_0_2_1;
  wire       [20:0]   _zz_mul_1_0;
  wire       [8:0]    _zz_mul_1_0_1;
  wire       [20:0]   _zz_mul_1_1;
  wire       [8:0]    _zz_mul_1_1_1;
  wire       [20:0]   _zz_mul_1_2;
  wire       [8:0]    _zz_mul_1_2_1;
  wire       [20:0]   _zz_mul_2_0;
  wire       [8:0]    _zz_mul_2_0_1;
  wire       [20:0]   _zz_mul_2_1;
  wire       [8:0]    _zz_mul_2_1_1;
  wire       [20:0]   _zz_mul_2_2;
  wire       [8:0]    _zz_mul_2_2_1;
  wire       [23:0]   _zz_acc;
  wire       [23:0]   _zz_acc_1;
  wire       [23:0]   _zz_acc_2;
  wire       [23:0]   _zz_acc_3;
  wire       [23:0]   _zz_acc_4;
  wire       [23:0]   _zz_acc_5;
  wire       [23:0]   _zz_acc_6;
  wire       [23:0]   _zz_acc_7;
  wire       [23:0]   _zz_acc_8;
  wire       [23:0]   _zz_acc_9;
  wire       [23:0]   _zz_acc_10;
  wire       [23:0]   _zz_acc_11;
  wire       [23:0]   _zz_acc_12;
  wire       [23:0]   _zz_acc_13;
  wire       [23:0]   _zz_acc_14;
  wire       [23:0]   _zz_acc_15;
  wire       [7:0]    _zz_io_pixel_o;
  wire       [19:0]   _zz_io_pixel_o_1;
  reg        [19:0]   mul_0_0;
  reg        [19:0]   mul_0_1;
  reg        [19:0]   mul_0_2;
  reg        [19:0]   mul_1_0;
  reg        [19:0]   mul_1_1;
  reg        [19:0]   mul_1_2;
  reg        [19:0]   mul_2_0;
  reg        [19:0]   mul_2_1;
  reg        [19:0]   mul_2_2;
  reg                 mul_data_valid;
  reg        [23:0]   acc;
  wire       [19:0]   shifted_acc;
  reg                 mul_data_valid_regNext;

  assign _zz_mul_0_0 = ($signed(io_kernel_0) * $signed(_zz_mul_0_0_1));
  assign _zz_mul_0_0_1 = {1'b0,io_pixel_grid_i_0};
  assign _zz_mul_0_1 = ($signed(io_kernel_1) * $signed(_zz_mul_0_1_1));
  assign _zz_mul_0_1_1 = {1'b0,io_pixel_grid_i_1};
  assign _zz_mul_0_2 = ($signed(io_kernel_2) * $signed(_zz_mul_0_2_1));
  assign _zz_mul_0_2_1 = {1'b0,io_pixel_grid_i_2};
  assign _zz_mul_1_0 = ($signed(io_kernel_3) * $signed(_zz_mul_1_0_1));
  assign _zz_mul_1_0_1 = {1'b0,io_pixel_grid_i_3};
  assign _zz_mul_1_1 = ($signed(io_kernel_4) * $signed(_zz_mul_1_1_1));
  assign _zz_mul_1_1_1 = {1'b0,io_pixel_grid_i_4};
  assign _zz_mul_1_2 = ($signed(io_kernel_5) * $signed(_zz_mul_1_2_1));
  assign _zz_mul_1_2_1 = {1'b0,io_pixel_grid_i_5};
  assign _zz_mul_2_0 = ($signed(io_kernel_6) * $signed(_zz_mul_2_0_1));
  assign _zz_mul_2_0_1 = {1'b0,io_pixel_grid_i_6};
  assign _zz_mul_2_1 = ($signed(io_kernel_7) * $signed(_zz_mul_2_1_1));
  assign _zz_mul_2_1_1 = {1'b0,io_pixel_grid_i_7};
  assign _zz_mul_2_2 = ($signed(io_kernel_8) * $signed(_zz_mul_2_2_1));
  assign _zz_mul_2_2_1 = {1'b0,io_pixel_grid_i_8};
  assign _zz_acc = ($signed(_zz_acc_1) + $signed(_zz_acc_8));
  assign _zz_acc_1 = ($signed(_zz_acc_2) + $signed(_zz_acc_5));
  assign _zz_acc_2 = ($signed(_zz_acc_3) + $signed(_zz_acc_4));
  assign _zz_acc_3 = {{4{mul_0_0[19]}}, mul_0_0};
  assign _zz_acc_4 = {{4{mul_0_1[19]}}, mul_0_1};
  assign _zz_acc_5 = ($signed(_zz_acc_6) + $signed(_zz_acc_7));
  assign _zz_acc_6 = {{4{mul_0_2[19]}}, mul_0_2};
  assign _zz_acc_7 = {{4{mul_1_0[19]}}, mul_1_0};
  assign _zz_acc_8 = ($signed(_zz_acc_9) + $signed(_zz_acc_12));
  assign _zz_acc_9 = ($signed(_zz_acc_10) + $signed(_zz_acc_11));
  assign _zz_acc_10 = {{4{mul_1_1[19]}}, mul_1_1};
  assign _zz_acc_11 = {{4{mul_1_2[19]}}, mul_1_2};
  assign _zz_acc_12 = ($signed(_zz_acc_13) + $signed(_zz_acc_14));
  assign _zz_acc_13 = {{4{mul_2_0[19]}}, mul_2_0};
  assign _zz_acc_14 = {{4{mul_2_1[19]}}, mul_2_1};
  assign _zz_acc_15 = {{4{mul_2_2[19]}}, mul_2_2};
  assign _zz_io_pixel_o_1 = shifted_acc;
  assign _zz_io_pixel_o = _zz_io_pixel_o_1[7:0];
  assign shifted_acc = (acc >>> 3'd4);
  assign io_pixel_o = (($signed(shifted_acc) < $signed(20'h0)) ? 8'h0 : (($signed(20'h000ff) < $signed(shifted_acc)) ? 8'hff : _zz_io_pixel_o));
  assign io_output_valid = mul_data_valid_regNext;
  always @(posedge clk) begin
    mul_data_valid <= io_data_valid;
    mul_0_0 <= _zz_mul_0_0[19 : 0];
    mul_0_1 <= _zz_mul_0_1[19 : 0];
    mul_0_2 <= _zz_mul_0_2[19 : 0];
    mul_1_0 <= _zz_mul_1_0[19 : 0];
    mul_1_1 <= _zz_mul_1_1[19 : 0];
    mul_1_2 <= _zz_mul_1_2[19 : 0];
    mul_2_0 <= _zz_mul_2_0[19 : 0];
    mul_2_1 <= _zz_mul_2_1[19 : 0];
    mul_2_2 <= _zz_mul_2_2[19 : 0];
    mul_data_valid_regNext <= mul_data_valid;
  end

  always @(posedge clk or posedge reset) begin
    if(reset) begin
      acc <= 24'h0;
    end else begin
      acc <= ($signed(_zz_acc) + $signed(_zz_acc_15));
    end
  end


endmodule
