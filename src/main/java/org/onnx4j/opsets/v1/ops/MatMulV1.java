package org.onnx4j.opsets.v1.ops;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * MatMul
 * 
 * Matrix product that behaves like numpy.matmul:
 * https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
 * @version This version of the operator has been available since version 9 of
 *          the default ONNX operator set.
 *
 */
public interface MatMulV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "MatMul";

	/**
	 * @constraints tensor(float16), tensor(float), tensor(double)
	 * @param x0
	 *            N-dimensional matrix A
	 * @param x1
	 *            N-dimensional matrix B
	 * @return Matrix multiply results from A * B
	 */
	public abstract T_TENSOR matmul(T_TENSOR x0, T_TENSOR x1);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Input[] inputArray = inputs.get();
		return Outputs.wrap(
				node,
				this.matmul(inputArray[0].getTensor(),
						inputArray[1].getTensor()));
	}

}
