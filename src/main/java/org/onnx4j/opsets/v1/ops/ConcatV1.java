package org.onnx4j.opsets.v1.ops;

import java.util.LinkedList;
import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Concat-1
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
public interface ConcatV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "Concat";

	public static final String ATTR_AXIS = "axis";

	/**
	 * Concatenate a list of tensors into a single tensor
	 * 
	 * @param inputs
	 *            List of tensors for concatenation
	 * @return Concatenated tensor
	 */
	public abstract T_TENSOR concat(List<T_TENSOR> inputs, Long axis);

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
		// Which axis to concat on. Default value is 1.
		//
		Long axis = attrs.getAttrValue(ATTR_AXIS, IntAttribute.class, 1L);
		
		Input[] inputArray = inputs.get();
		List<T_TENSOR> inputList = new LinkedList<>();
		for (Input input : inputArray) {
			T_TENSOR tensor = input.getTensor();
			inputList.add(tensor);
		}
		
		return Outputs.wrap(node, this.concat(inputList, axis));
	}

}
