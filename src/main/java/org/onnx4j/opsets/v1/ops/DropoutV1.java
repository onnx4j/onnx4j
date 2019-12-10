package org.onnx4j.opsets.v1.ops;

import java.util.LinkedList;
import java.util.List;
import java.util.Optional;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Dropout-1
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Dropout-1
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
public interface DropoutV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "Dropout";

	public static final String ATTR_RATIO = "ratio";

	public static final String ATTR_IS_TEST = "is_test";

	public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

	/**
	 * Dropout takes one input data (Tensor) and produces two Tensor outputs,
	 * output (Tensor) and mask (Tensor). Depending on whether it is in test
	 * mode or not, the output Y will either be a random dropout, or a simple
	 * copy of the input. Note that our implementation of Dropout does scaling
	 * in the training phase, so during testing nothing needs to be done.
	 * 
	 * @param data
	 *            The input data as Tensor.
	 * @param isTest
	 *            (int, default 0) if nonzero, run dropout in test mode where
	 *            the output is simply Y = X.
	 * @param ratio
	 *            (float, default 0.5) the ratio of random dropout
	 * @param consumedInputs
	 *            legacy optimization attribute.
	 * @return output : T The output. mask (optional) : T The output mask. If
	 *         is_test is nonzero, this output is not filled.
	 */
	public abstract List<T_TENSOR> dropout(T_TENSOR data, Boolean isTest, Float ratio, List<Long> consumedInputs);

	public default List<T_TENSOR> wrapMultiOutputs(Optional<T_TENSOR> data, Optional<T_TENSOR> mask) {
		List<T_TENSOR> outputs = new LinkedList<T_TENSOR>();

		outputs.add(data.orElseThrow(
				() -> new RuntimeException(String.format("[%s] The field named \"data\" can not be null", OP_TYPE))));
		mask.ifPresent((value) -> outputs.add(value));
		return outputs;
	}

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
		// Legacy optimization attribute.
		//
		List<Long> consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);

		//
		// (float, default 0.5) the ratio of random dropout
		//
		Float ratio = attrs.getAttrValue(ATTR_RATIO, FloatAttribute.class, 0.5f);

		//
		// (int, default 0) if nonzero, run dropout in test mode where the
		// output is simply Y = X.
		//
		Boolean isTest = attrs.getAttrValue(ATTR_IS_TEST, IntAttribute.class, 0L).intValue() != 0 ? true : false;

		Input[] inputArray = inputs.get();
		List<T_TENSOR> outputs = this.dropout(inputArray[0].getTensor(), isTest, ratio, consumedInputs);
		return Outputs.wrap(node, outputs);
	}

}
