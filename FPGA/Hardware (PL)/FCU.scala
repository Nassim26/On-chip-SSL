package hdlproj

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._

// Hardware definition of Filter Control Unit (FCU)
case class FCU(input_res : Int, weight_res : Int, channel_dim: Int, kernel_size : Int) extends Component {
  val kernel_area = kernel_size * kernel_size
  val io = new Bundle {
    val pix_buffer_1 = in Vec(UInt(input_res bits), kernel_size)
    val pix_buffer_2 = in Vec(UInt(input_res bits), kernel_size)
    val pix_buffer_3 = in Vec(UInt(input_res bits), kernel_size)
    val pix_buffer_4 = in Vec(UInt(input_res bits), kernel_size)
    val pix_data_valid = in Bool()
    val kernel = in Vec(SInt(weight_res bits), kernel_area)

    val pixel_o = out UInt(input_res bits)
    val read_counter = out UInt(log2Up(channel_dim) bits)
    val buff_valid = out UInt(4 bits)
    val output_valid = out Bool()
    val image_finished = out Bool()
  }

  // Counter-registers
  val pix_counter = Reg(UInt(log2Up(channel_dim*channel_dim) bits)) init(0)   // Counts number of received pixels
  val row_pix_counter = Reg(UInt(log2Up(channel_dim) bits)) init(0)           // Tracks the row of the incoming pixels
  val read_pix_counter = Reg(UInt(log2Up(channel_dim) bits)) init(0)          // Tracks x-position of convolutional kernel (upper-left corner)
  val read_row_counter = Reg(UInt(log2Up(channel_dim) bits)) init(0)          // Tracks y-position of convolutional kernel (upper-left corner)
  
  // Buffer signals
  val p_buffers = Vec(Vec(UInt(input_res bits), kernel_size), 4)              // The actual pixel (line) buffers
  val active_write_buffer = Reg(UInt(2 bits)) init(0)                         // Track which buffer is being written to
  val active_read_buffer = Reg(UInt(2 bits)) init(0)                          // Track which buffer is being read from (for the pixelgrid output)
  val pixel_grid = Vec(UInt(input_res bits), kernel_area)                     // The 3x3-pixelgrid fed to the MAC-unit
  val reset_buffers = Reg(Bool()) init(False)                                 // Universal reset on all buffers (triggered whenever an image is finished)
  val reset_write_buffers = Reg(Bool()) init(False)

  active_write_buffer := Mux(row_pix_counter === (channel_dim - 1) && io.pix_data_valid, active_write_buffer + 1, active_write_buffer) 
  // active_read_buffer defined below, due to Scala variable ordering requirements

  p_buffers(0) := io.pix_buffer_1
  p_buffers(1) := io.pix_buffer_2
  p_buffers(2) := io.pix_buffer_3
  p_buffers(3) := io.pix_buffer_4

  io.buff_valid := Mux(active_write_buffer === 0, U"0001", 
                  Mux(active_write_buffer === 1, U"0010", 
                  Mux(active_write_buffer === 2, U"0100", U"1000")))          // One-hot encodes which buffer is being written to

  io.read_counter := read_pix_counter

  when (io.pix_data_valid && pix_counter === channel_dim*channel_dim - 1) { 
    pix_counter := U(1, pix_counter.getWidth bits)
  } elsewhen(io.pix_data_valid) { pix_counter := pix_counter + 1 }
  
  when (io.pix_data_valid && row_pix_counter === channel_dim - 1) {
    row_pix_counter := 0
  } elsewhen (io.pix_data_valid) {
    row_pix_counter := row_pix_counter + 1 
  }

  for (i <- 0 until kernel_area) {
    pixel_grid(i) := p_buffers(((kernel_size - 1) + active_read_buffer - Math.floorDiv(i, kernel_size)) % 4)(i % kernel_size)
  }

  val conv_fsm = new StateMachine {
    val awaitBuffers = new State with EntryPoint
    val convolve, imageDone = new State 
    val stallConv = new StateDelay(cyclesCount = kernel_size - 1)
    val conv_data_valid = Reg(Bool()) init(False)
    val finished_image = Reg(Bool()) init(False)
    val increment_read_buffer = Reg(Bool()) init(False)

    awaitBuffers
      .whenIsActive {
        reset_buffers := False
        finished_image := False 
        conv_data_valid := False
        when (pix_counter === (channel_dim * kernel_size - 1)) {
          goto(convolve)
        }
      }

    convolve.whenIsActive {
      finished_image := False 
      conv_data_valid := True
      increment_read_buffer := False
      when (read_row_counter === (channel_dim - kernel_size) && read_pix_counter === (channel_dim - kernel_size - 1)) {
        goto(imageDone)
      } elsewhen (read_pix_counter === channel_dim - kernel_size - 1) {
        goto(stallConv)
      } otherwise {
        goto(convolve)
      }
    }

    stallConv
      .whenIsActive {
        finished_image := False 
        conv_data_valid := False
      }
      .whenCompleted {
        increment_read_buffer := True
        goto(convolve)
      }
    
    imageDone
      .whenIsActive {
        reset_buffers := True
        finished_image := True
        conv_data_valid := False
        goto(awaitBuffers)
      }
  }

  io.image_finished := conv_fsm.finished_image

  when (reset_buffers) {
    active_read_buffer := 0
  } otherwise {
    active_read_buffer := Mux(conv_fsm.increment_read_buffer, active_read_buffer + 1, active_read_buffer)
  } 
  
  // Reading buffer logic 
  when (reset_buffers) {
    read_pix_counter := 0
    read_row_counter := 0
  } elsewhen (conv_fsm.conv_data_valid && read_pix_counter === channel_dim - kernel_size) {
    read_pix_counter := 0
    read_row_counter := read_row_counter + 1
  } elsewhen (conv_fsm.conv_data_valid) {
    read_pix_counter := read_pix_counter + 1
  } 

  // Internal component declaration of Multiply-and-Accumulate (MAC) unit
  val MAC_unit = MAC(input_res, weight_res, kernel_size)
  MAC_unit.io.pixel_grid_i := pixel_grid
  MAC_unit.io.kernel := io.kernel
  MAC_unit.io.data_valid := conv_fsm.conv_data_valid
  io.pixel_o := MAC_unit.io.pixel_o
  val mac_data_valid = MAC_unit.io.output_valid

  io.output_valid := mac_data_valid
}

object CompileFCU extends App {
  Config.spinal.generateVerilog(FCU(8, 12, 28, 3))
}