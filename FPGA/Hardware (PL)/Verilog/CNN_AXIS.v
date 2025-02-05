
module CNN_AXIS #
(
    // Parameters of Axi Master Bus Interface M_AXIS
    parameter integer M_AXIS_TDATA_WIDTH	= 8,
    // Parameters of Axi Slave Bus Interface S_AXIS
    parameter integer S_AXIS_TDATA_WIDTH	= 8,

    // CNN-bound parameters
    parameter DATA_RES = 8, // Should match master & slave tdata widths
    parameter WEIGHT_RES = 8, 
    parameter IM_DIM = 28, 
    parameter KERNEL_WIDTH = 3, 
    parameter KERNEL_SIZE = 9
)
(
    input wire  clk_i,
    input wire  resetn_i,
    
    // Ports of Axi Master Bus Interface M_AXIS
    output wire  m_axis_tvalid,
    output wire [M_AXIS_TDATA_WIDTH-1 : 0] m_axis_tdata,
    output wire  m_axis_tlast,
    input wire  m_axis_tready,

    // Ports of Axi Slave Bus Interface S_AXIS
    output wire  s_axis_tready,
    input wire [S_AXIS_TDATA_WIDTH-1 : 0] s_axis_tdata,
    input wire  s_axis_tlast,
    input wire  s_axis_tvalid
);

    wire [M_AXIS_TDATA_WIDTH-1:0] CNN_out;
    wire out_val;
    reg [79:0] kernel_reg;

    always @(posedge clk_i, posedge resetn_i) begin // For testing purposes, Gaussian blur kernel
        if (!resetn_i) begin 
            kernel_reg <= 80'd0;
        end else begin 
            kernel_reg <= 80'b00000000000000010000001000000001000000100000010000000010000000010000001000000001;
        end 
    end 

    CNN_control # (
        .DATA_RES(DATA_RES),
        .WEIGHT_RES(WEIGHT_RES),
        .IM_DIM(IM_DIM),
        .KERNEL_WIDTH(KERNEL_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE)
    ) CNN (
        .clk_i(clk_i), 
        .resetn_i(resetn_i), 
        .pixel_i(s_axis_tdata),
        .data_valid_i(s_axis_tvalid & s_axis_tready),
        .kernel_i(kernel_reg), 
        .pixel_o(CNN_out), //o 
        .data_valid_o(out_val) //o
    );

    // Internal signal for the data
    reg [M_AXIS_TDATA_WIDTH-1:0]    tdata_reg;
    reg                             tvalid_reg;
    reg                             tlast_reg;
    // reg					  in_valid;

    // States
    reg [1:0] State;
    localparam Init = 2'b00; 
    localparam Reading = 2'b01; 
    localparam Streaming = 2'b10;
    // End states
        
    reg [9:0] nr_of_writes;
    integer OUTPUT_WORDS;

    initial begin 
        OUTPUT_WORDS = (IM_DIM-KERNEL_WIDTH+1)*(IM_DIM-KERNEL_WIDTH+1); 
    end 

    // Handshaking (ready/valid mechanism)
    assign s_axis_tready = m_axis_tready & s_axis_tvalid;  // Slave ready if master is ready
    assign m_axis_tvalid = tvalid_reg;     // Valid output data when tvalid_reg is set
    assign m_axis_tdata  = tdata_reg;      // Output the processed data
    assign m_axis_tlast  = tlast_reg;      // Propagate tlast signal

    always @(posedge clk_i, posedge resetn_i) begin 
        if (!resetn_i) begin 
            State <= Init; 
        end 
    end 

    // Reading FSM
    always @(posedge clk_i) begin 
        // tvalid_reg <= out_val && m_axis_tready;
        case(State) 
        Init: begin 
            tlast_reg <= 1'b0; 
            nr_of_writes <= OUTPUT_WORDS;
            tdata_reg <= 'd0;	
            tvalid_reg <= 1'b0;
            if (s_axis_tvalid && s_axis_tready) begin 
                State <= Reading;			
            end else begin 
                State <= Init; 
            end 
        end 
        
        Reading: begin 
            tlast_reg <= 1'b0; 
            if (out_val && m_axis_tready) begin  
                tvalid_reg <= 1'b1;
                nr_of_writes <= OUTPUT_WORDS - 1;
                tdata_reg <= CNN_out;
                State <= Streaming; 
            end else begin 
                tvalid_reg <= 1'b0;
                nr_of_writes <= OUTPUT_WORDS;
                tdata_reg <= 'd0;
                State <= Reading; 
            end 
        end  
        
        Streaming: begin 
            if (out_val && m_axis_tready) begin
                tvalid_reg <= 1'b1;
                nr_of_writes <= nr_of_writes - 1; 
                tdata_reg <= CNN_out;
            end else begin 
                tvalid_reg <= 1'b0;
                nr_of_writes <= nr_of_writes;
                tdata_reg <= tdata_reg;
            end 
            
            if (nr_of_writes == 'd1) begin 
                tlast_reg <= 1'b1; 
                State <= Streaming; 
            end else if (nr_of_writes == 'd0) begin 
                tlast_reg <= 1'b0; 
                State <= Init;
            end else begin
                tlast_reg <= 1'b0; 
                State <= Streaming;
            end 
        end 
        endcase 
    end 

endmodule
