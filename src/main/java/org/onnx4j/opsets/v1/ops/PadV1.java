package org.onnx4j.opsets.v1.ops;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

public interface PadV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "Pad";

	public abstract T_TENSOR pad(T_TENSOR x0);
	
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
		return Outputs.wrap(node, this.pad(inputArray[0].getTensor()));
	}

}
