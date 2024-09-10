package hdlproj

import spinal.core._
import spinal.lib._

// Hardware definition
case class MAC(input_res: Int, weight_res: Int, kernel_size: Int) extends Component {
  val kernel_area = kernel_size * kernel_size
  val io = new Bundle {
    val pixel_grid_i = in Vec(UInt(input_res bits), kernel_area)
    val kernel = in Vec(SInt(weight_res bits), kernel_area)
    val data_valid = in Bool()

    val pixel_o = out UInt(input_res bits)
    val output_valid = out Bool()
  }

  val MUL_WID = 20
  val ACC_WID = 24
  val FxP_RES = 4

  val mul = Reg(Vec(Vec(SInt(MUL_WID bits), kernel_size), kernel_size))  
  val mul_data_valid = RegNext(io.data_valid)

  for (i <- 0 until kernel_area) {
    mul(Math.floorDiv(i, kernel_size))(i % kernel_size) := (io.kernel(i) * Cat(U"0", io.pixel_grid_i(i)).asSInt).trim(1)
  }

  val acc = Reg(SInt(ACC_WID bits)) init(0)
  acc := mul.toList.flatten.map(_.resize(ACC_WID)).reduceBalancedTree(_ + _)

  val shifted_acc  = SInt(ACC_WID - FxP_RES bits) 
  shifted_acc := acc >> FxP_RES;

  io.pixel_o := Mux(shifted_acc < 0, U(0, input_res bits), 
                Mux(shifted_acc > 255, U(255, input_res bits), 
                shifted_acc.asUInt.resized))
  
  io.output_valid := RegNext(mul_data_valid)
}

object CompileMAC extends App {
  Config.spinal.generateVerilog(MAC(8, 12, 3))
}