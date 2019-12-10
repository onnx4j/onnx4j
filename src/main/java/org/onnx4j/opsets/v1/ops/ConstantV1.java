package org.onnx4j.opsets.v1.ops;

import org.onnx4j.Inputs;
import org.onnx4j.Outputs;
import org.onnx4j.Tensor;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.TensorAttribute;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Constant-1
 * 
 * @author HarryLee
 * 
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *	
 * @param <T_TENSOR>
 */
public interface ConstantV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "Constant";

	public static final String ATTR_VALUE = "value";

	/**
	 * A constant tensor.
	 * 
	 * @param x0
	 *            The value for the elements of the output tensor.
	 * @return Output tensor containing the same value of the provided tensor.
	 */
	public abstract T_TENSOR constant(Tensor x0);

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

		Tensor value = attrs.getAttrValue(ATTR_VALUE, TensorAttribute.class, null);
		return Outputs.wrap(node, this.constant(value));
	}

}
