package org.onnx4j.opsets.v1.ops;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Abs-1
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Acos
 * @version This version of the operator has been available since version 6 of
 *          the default ONNX operator set.
 *
 */
public interface AbsV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "Abs";

	/**
	 * Absolute takes one input data (Tensor) and produces one output data (Tensor)
	 * where the absolute is, y = abs(x), is applied to the tensor elementwise.
	 * 
	 * @param x0
	 * @return
	 */
	public T_TENSOR abs(T_TENSOR x0);

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Input[] inputArray = inputs.get();
		return Outputs.wrap(node, this.abs(inputArray[0].getTensor()));
	}

}
