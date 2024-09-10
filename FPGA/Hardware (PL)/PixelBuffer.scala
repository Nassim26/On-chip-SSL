package hdlproj

import spinal.core._
import spinal.lib._

// Hardware definition
case class PixelBuffer(input_res: Int, kernel_size: Int, buffer_width: Int) extends Component {
  val io = new Bundle {
    val pixel_i = in UInt(input_res bits)
    val data_valid = in Bool()
    val read_address = in UInt(log2Up(buffer_width) bits)

    val pix_buffer = out Vec(UInt(input_res bits), 3)
  }

  val write_address = Reg(UInt(log2Up(buffer_width) bits)) init(0)
  val line_data = Vec(Reg(UInt(input_res bits)) init (0), buffer_width)

  when(io.data_valid) {
    line_data(write_address) := io.pixel_i
    write_address := Mux(write_address === (buffer_width - 1), U(0, log2Up(buffer_width) bits), write_address + 1)
  }

  for (i <- 0 until kernel_size) {
    io.pix_buffer(i) := line_data(io.read_address + (kernel_size - 1) - i)
  }

}

object CompilePixelBuffer extends App {
  Config.spinal.generateVerilog(PixelBuffer(8, 3, 28))
}
