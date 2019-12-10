package org.onnx4j.opsets.v1.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * LeakyRelu-1
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#LeakyRelu-1
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
public interface LeakyReluV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "LeakyRelu";

	public static final String ATTR_ALPHA = "alpha";

	public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

	/**
	 * LeakyRelu takes input data (Tensor) and an argument alpha, and produces
	 * one output data (Tensor) where the function f(x) = alpha * x for x < 0,
	 * f(x) = x for x >= 0, is applied to the data tensor elementwise.
	 * 
	 * @param x
	 *            Input tensor
	 * @param alpha
	 *            Coefficient of leakage default to 0.01.
	 * @param consumedInputs
	 *            legacy optimization attribute.
	 * @return
	 */
	public abstract T_TENSOR leakyRelu(T_TENSOR x, Float alpha, List<Long> consumedInputs);

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
		// Coefficient of leakage default to 0.01.
		//
		Float alpha = attrs.getAttrValue(ATTR_ALPHA, FloatAttribute.class, 0.01f);

		//
		// Legacy optimization attribute.
		//
		List<Long> consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node, this.leakyRelu(inputArray[0].getTensor(), alpha, consumedInputs));
	}

}
