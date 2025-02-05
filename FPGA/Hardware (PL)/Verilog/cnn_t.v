
`timescale 1 ns / 1 ps

module CNN_t #
	(
	    parameter DATA_RES = 8, 
	    parameter WEIGHT_RES = 8,
	    parameter LINE_WIDTH = 28, 
	    parameter KERNEL_WIDTH = 3,
	    parameter KERNEL_SIZE = 9,
		// Parameters of Axi Slave Bus Interface S_AXIS
		parameter integer S_AXIS_TDATA_WIDTH	= 8,

		// Parameters of Axi Master Bus Interface M_AXIS
		parameter integer M_AXIS_TDATA_WIDTH	= 8

	)
	(

		// Ports of Axi Slave Bus Interface S_AXIS
		input wire  s_axis_aclk,
		input wire  s_axis_aresetn,
		output wire  s_axis_tready,
		input wire [S_AXIS_TDATA_WIDTH-1 : 0] s_axis_tdata,
		input wire  s_axis_tlast,
		input wire  s_axis_tvalid,

		// Ports of Axi Master Bus Interface M_AXIS
		input wire  m_axis_aclk,   
        input wire  m_axis_aresetn,
		output wire  m_axis_tvalid,
		output wire [M_AXIS_TDATA_WIDTH-1 : 0] m_axis_tdata,
		output wire  m_axis_tlast,
		input wire  m_axis_tready

	);
	
	// wire [KERNEL_SIZE*DATA_RES-1:0] pixel_grid; 
    wire [DATA_RES-1:0] conv_data;
    wire mac_data_valid;
    wire intr;
    wire output_pixel;
    wire conv_data_valid; 
    wire axis_prog_full;
    
    assign s_axis_tready = !axis_prog_full;
    assign interrupt = intr;
     
    reg [WEIGHT_RES*KERNEL_SIZE-1:0] kernel;
    
    initial begin 
         // Gaussian blur filter, for testing purposes:
        kernel = 72'b000000010000001000000001000000100000010000000010000000010000001000000001;
    end 
    
    CNN_control # (
        .DATA_RES(DATA_RES), 
        .WEIGHT_RES(WEIGHT_RES),
        .LINE_WIDTH(LINE_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE)
    ) controller (
        .clk_i(s_axis_aclk),
        .resetn_i(s_axis_aresetn), 
        .pixel_i(s_axis_tdata), 
        .data_valid_i(s_axis_tvalid),
        .kernel_i(kernel),
        .pixel_o(conv_data), 
        .data_valid_o(mac_data_valid),
        .pixel_valid_o(conv_data_valid)
    ); 
    
    
    output_fifo output_buffer (
        .s_aclk(s_axis_aclk),                 
        .s_aresetn(s_axis_aresetn),            
        .s_axis_tvalid(conv_data_valid),  
        .s_axis_tready(),    
        .s_axis_tdata(conv_data),      
        .m_axis_tvalid(m_axis_tvalid),    
        .m_axis_tready(m_axis_tready),    
        .m_axis_tdata(m_axis_tdata),      
        .axis_prog_full(axis_prog_full)  
    );

endmodule
