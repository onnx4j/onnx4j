package org.onnx4j.opsets.v1.ops;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Identity
 * 
 * Identity operator
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Identity
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
public interface IdentityV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "Identity";

	public abstract T_TENSOR identity(T_TENSOR x0);
	
	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.allTypes();
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
		return Outputs.wrap(node, this.identity(inputArray[0].getTensor()));
	}

}
