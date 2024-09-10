package hdlproj

import spinal.core._
import spinal.lib._

// Hardware definition
case class FilterTop(input_res: Int, weight_res: Int, channel_dim: Int, kernel_size: Int) extends Component {
  val kernel_area = kernel_size*kernel_size
  val io = new Bundle {
    val pixel_i = in UInt(input_res bits)
    val pix_data_valid = in Bool()
    val kernel = in Vec(SInt(weight_res bits), kernel_area)

    val pixel_o = out UInt(input_res bits)
    val image_finished = out Bool()
    val output_valid = out Bool()
  }

  // Filter Control Unit declaration
  val FiltConUnit = FCU(input_res, weight_res, channel_dim, kernel_size)

  // Pixel buffer 1-4 declarations
  val P1 = PixelBuffer(input_res, kernel_size, channel_dim)
  val P2 = PixelBuffer(input_res, kernel_size, channel_dim)
  val P3 = PixelBuffer(input_res, kernel_size, channel_dim)
  val P4 = PixelBuffer(input_res, kernel_size, channel_dim)

  // Component interconnect
  FiltConUnit.io.pix_buffer_1 := P1.io.pix_buffer
  FiltConUnit.io.pix_buffer_2 := P2.io.pix_buffer
  FiltConUnit.io.pix_buffer_3 := P3.io.pix_buffer
  FiltConUnit.io.pix_buffer_4 := P4.io.pix_buffer
  FiltConUnit.io.pix_data_valid := io.pix_data_valid
  FiltConUnit.io.kernel := io.kernel

  P1.io.pixel_i := io.pixel_i
  P2.io.pixel_i := io.pixel_i
  P3.io.pixel_i := io.pixel_i
  P4.io.pixel_i := io.pixel_i
  P1.io.data_valid := FiltConUnit.io.buff_valid(0) && io.pix_data_valid
  P2.io.data_valid := FiltConUnit.io.buff_valid(1) && io.pix_data_valid
  P3.io.data_valid := FiltConUnit.io.buff_valid(2) && io.pix_data_valid
  P4.io.data_valid := FiltConUnit.io.buff_valid(3) && io.pix_data_valid
  P1.io.read_address := FiltConUnit.io.read_counter
  P2.io.read_address := FiltConUnit.io.read_counter
  P3.io.read_address := FiltConUnit.io.read_counter
  P4.io.read_address := FiltConUnit.io.read_counter

  io.pixel_o := FiltConUnit.io.pixel_o
  io.image_finished := FiltConUnit.io.image_finished
  io.output_valid := FiltConUnit.io.output_valid
}

object CompileFilterTop extends App {
  Config.spinal.generateVerilog(FilterTop(8, 12, 28, 3))
}

