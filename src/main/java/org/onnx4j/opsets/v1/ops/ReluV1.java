package org.onnx4j.opsets.v1.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Relu-1
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Relu-1
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
public interface ReluV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "Relu";

	public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

	/**
	 * Relu takes one input data (Tensor) and produces one output data (Tensor)
	 * where the rectified linear function, y = max(0, x), is applied to the
	 * tensor elementwise.
	 * 
	 * @param x
	 *            Input tensor
	 * @param consumed_inputs
	 *            legacy optimization attribute
	 * @return Output tensor
	 */
	public abstract T_TENSOR relu(T_TENSOR x, List<Long> consumed_inputs);

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
		Attributes attrs = node.getAttrs();

		//
		// list of ints
		//
		List<Long> consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS,
				IntsAttribute.class, null);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node,
				this.relu(inputArray[0].getTensor(), consumedInputs));
	}

}
