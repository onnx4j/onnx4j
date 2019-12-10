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
 * Div-1
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Div-1
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
public interface ReshapeV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "Reshape";

	//
	// New shape
	//
	public static final String ATTR_SHAPE = "shape";

	//
	// Legacy optimization attribute.
	//
	public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

	/**
	 * Reshape the input tensor similar to numpy.reshape. It takes a tensor as
	 * input and an argument shape. It outputs the reshaped tensor. At most one
	 * dimension of the new shape can be -1. In this case, the value is inferred
	 * from the size of the tensor and the remaining dimensions. A dimension
	 * could also be 0, in which case the actual dimension value is unchanged
	 * (i.e. taken from the input tensor).
	 * 
	 * @param data
	 *            An input tensor
	 * @param shape
	 *            New shape
	 * @param consumedInputs
	 *            legacy optimization attribute
	 * @return Reshaped data
	 */
	public abstract T_TENSOR reshape(T_TENSOR data, List<Long> shape, List<Long> consumedInputs);

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

	class ReshapeInputsV1<T_TENSOR> {
		private T_TENSOR data;
		private List<Long> shape;
		private List<Long> consumedInputs;

		public ReshapeInputsV1(Node node, Inputs inputs) {
			super();

			Attributes attrs = node.getAttrs();
			Input[] inputArray = inputs.get();

			this.data = inputArray[0].getTensor();
			this.shape = attrs.getAttrValue(ATTR_SHAPE, IntsAttribute.class, null);
			this.consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);
		}

		public T_TENSOR getData() {
			return data;
		}

		public List<Long> getShape() {
			return shape;
		}

		public List<Long> getConsumedInputs() {
			return consumedInputs;
		}
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		ReshapeInputsV1<T_TENSOR> operatorInputs = new ReshapeInputsV1<T_TENSOR>(node, inputs);
		return Outputs.wrap(node,
				this.reshape(operatorInputs.getData(), operatorInputs.getShape(), operatorInputs.getConsumedInputs()));
	}

}
